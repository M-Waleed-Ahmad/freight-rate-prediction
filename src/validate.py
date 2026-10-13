"""Time-based validation of the rate models.

Run from the project root:
    python src/validate.py

Design
* Corrupted labels are flagged once on the full development set (label QA, see
  features.flag_label_outliers). They are dropped from every training fold and excluded
  from the headline holdout metrics; MAE on all holdout rows is reported alongside.
* Main scheme: expanding-window backtest with a 2-month horizon, mirroring the real task
  (train Jan-Oct, predict Nov-Dec). Six folds, test windows Apr-May ... Sep-Oct.
* Random 5-fold CV is run for contrast, to show how optimistic a shuffled split is.
* Unseen-city check: 8 cities are removed from the training fold to mimic the 8 validation
  cities that never appear in train_test.csv.
"""
from __future__ import annotations

import lightgbm as lgb
import matplotlib

matplotlib.use("Agg")
import matplotlib.dates as mdates
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedFormatter, FixedLocator, NullFormatter
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold

from features import ROOT, build_features, daily_market, flag_label_outliers, load_train, load_validation
from model import LGB_PARAMS, NUM_ROUNDS, TwoStageRateModel, market_design

REPORT_DIR = ROOT / "reports"
FIG_DIR = REPORT_DIR / "figures"
BACKTEST_FOLDS = [(pd.Timestamp(f"2025-{month:02d}-01"), pd.Timestamp(f"2025-{month + 2:02d}-01")) for month in range(4, 10)]
PRIMARY_FOLD = BACKTEST_FOLDS[-1]  # train Jan-Aug, test Sep-Oct
UNSEEN_CITY_COUNT = 8


class LaneMedianBaseline:
    """Median rate per mile for the lane and equipment, falling back to equipment median."""

    def fit(self, frame: pd.DataFrame) -> "LaneMedianBaseline":
        rpm = frame.posted_rate / frame.distance
        self.lane = rpm.groupby([frame.pickup, frame.delivery, frame.equipment]).median()
        self.equipment = rpm.groupby(frame.equipment).median()
        return self

    def predict(self, frame: pd.DataFrame) -> np.ndarray:
        keys = pd.MultiIndex.from_arrays([frame.pickup, frame.delivery, frame.equipment])
        rpm = pd.Series(self.lane.reindex(keys).to_numpy(), index=frame.index)
        rpm = rpm.fillna(frame.equipment.map(self.equipment))
        return (rpm * frame.distance).to_numpy()


class LinearBaseline:
    """Log-linear model: distance curve, equipment, weight, coordinates plus the market terms."""

    @staticmethod
    def design(features: pd.DataFrame) -> pd.DataFrame:
        design = market_design(features)
        design["log_distance_sq"] = features.log_distance ** 2
        for code in (1, 2):
            design[f"equipment_{code}"] = (features.equipment == code).astype(float)
        design["weight"] = features.weight.fillna(features.weight.median())
        for column in ("weight_missing", "weight_at_cap", "pickup_lat", "pickup_lon", "delivery_lat", "delivery_lon"):
            design[column] = features[column]
        return design

    def fit(self, features: pd.DataFrame, rate: pd.Series) -> "LinearBaseline":
        self.model = LinearRegression().fit(self.design(features), np.log(rate) - features.log_distance)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.exp(features.log_distance + self.model.predict(self.design(features))).to_numpy()


class SingleLightGBM:
    """One LightGBM over every feature, calendar and quote_signal included."""

    def fit(self, features: pd.DataFrame, rate: pd.Series) -> "SingleLightGBM":
        dataset = lgb.Dataset(features, np.log(rate) - features.log_distance, categorical_feature=["equipment"])
        self.model = lgb.train(LGB_PARAMS, dataset, NUM_ROUNDS)
        return self

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.exp(features.log_distance + self.model.predict(features)).to_numpy()


def fit_predict(name: str, train: pd.DataFrame, test: pd.DataFrame, features: pd.DataFrame) -> np.ndarray:
    train_x, test_x = features.loc[train.index], features.loc[test.index]
    if name == "Lane median":
        return LaneMedianBaseline().fit(train).predict(test)
    if name == "Log-linear":
        return LinearBaseline().fit(train_x, train.posted_rate).predict(test_x)
    if name == "Single LightGBM":
        return SingleLightGBM().fit(train_x, train.posted_rate).predict(test_x)
    if name == "Two-stage (final)":
        return TwoStageRateModel().fit(train_x, train.posted_rate, train.date).predict(test_x)
    raise ValueError(name)


MODELS = ["Lane median", "Log-linear", "Single LightGBM", "Two-stage (final)"]


def metrics(truth: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    error = pred - truth
    return {
        "MAE": float(np.mean(np.abs(error))),
        "RMSE": float(np.sqrt(np.mean(error ** 2))),
        "MAPE_%": float(100 * np.mean(np.abs(error / truth))),
        "bias_%": float(100 * np.mean(np.log(pred / truth))),
        "R2": float(1 - np.sum(error ** 2) / np.sum((truth - truth.mean()) ** 2)),
    }


def score(test: pd.DataFrame, pred: np.ndarray) -> dict[str, float]:
    clean = ~test.is_outlier.to_numpy()
    result = metrics(test.posted_rate.to_numpy()[clean], pred[clean])
    result["MAE_all_rows"] = float(np.mean(np.abs(pred - test.posted_rate.to_numpy())))
    result["n_test"] = int(len(test))
    return result


def run_backtest(dev: pd.DataFrame, features: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    rows, primary_predictions = [], {}
    for start, end in BACKTEST_FOLDS:
        train = dev[(dev.date < start) & ~dev.is_outlier]
        test = dev[(dev.date >= start) & (dev.date < end)]
        last_train_month = (start.to_period("M") - 1).strftime("%b")
        last_test_month = (end.to_period("M") - 1).strftime("%b")
        label = f"train Jan-{last_train_month} | test {start:%b}-{last_test_month}"
        for name in MODELS:
            pred = fit_predict(name, train, test, features)
            rows.append({"scheme": "expanding window", "fold": label, "model": name, **score(test, pred)})
            if (start, end) == PRIMARY_FOLD:
                primary_predictions[name] = pd.Series(pred, index=test.index)
        print(f"  {label} done")
    return pd.DataFrame(rows), primary_predictions


def run_random_kfold(dev: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    rows = []
    folds = KFold(n_splits=5, shuffle=True, random_state=7).split(dev)
    for fold_number, (train_idx, test_idx) in enumerate(folds, start=1):
        train = dev.iloc[train_idx]
        train = train[~train.is_outlier]
        test = dev.iloc[test_idx]
        for name in ("Single LightGBM", "Two-stage (final)"):
            pred = fit_predict(name, train, test, features)
            rows.append({"scheme": "random 5-fold", "fold": f"fold {fold_number}", "model": name, **score(test, pred)})
    return pd.DataFrame(rows)


def run_unseen_cities(dev: pd.DataFrame, features: pd.DataFrame) -> pd.DataFrame:
    start, end = PRIMARY_FOLD
    cities = sorted(set(dev.pickup) | set(dev.delivery))
    held_out = list(np.random.default_rng(7).choice(cities, UNSEEN_CITY_COUNT, replace=False))
    touches = dev.pickup.isin(held_out) | dev.delivery.isin(held_out)
    train = dev[(dev.date < start) & ~dev.is_outlier & ~touches]
    test = dev[(dev.date >= start) & (dev.date < end)]
    pred = pd.Series(fit_predict("Two-stage (final)", train, test, features), index=test.index)
    rows = []
    for group, mask in (("rows touching held-out cities", touches[test.index]), ("rows with seen cities", ~touches[test.index])):
        rows.append({"scheme": "unseen cities", "fold": group, "model": "Two-stage (final)", **score(test[mask], pred[mask].to_numpy())})
    print(f"  held-out cities: {held_out}")
    return pd.DataFrame(rows)


def plot_split_scheme() -> None:
    """Gantt-style picture of the backtest folds and the final fit."""
    rows = [(f"Fold {number}", start, end) for number, (start, end) in enumerate(BACKTEST_FOLDS, start=1)]
    rows.append(("Final model", pd.Timestamp("2025-11-01"), pd.Timestamp("2026-01-01")))
    origin = pd.Timestamp("2025-01-01")
    figure, axis = plt.subplots(figsize=(10, 3.6))
    for position, (label, start, end) in enumerate(reversed(rows)):
        final = label == "Final model"
        axis.barh(position, (start - origin).days, left=0, color="#064A56", alpha=0.9 if final else 0.75, height=0.6)
        axis.barh(position, (end - start).days, left=(start - origin).days, color="#E07B39" if final else "#F2B279", height=0.6)
        axis.text((start - origin).days + 3, position, "predict Nov-Dec" if final else "test", va="center", fontsize=8.5)
    axis.set_yticks(range(len(rows)), [label for label, _, _ in reversed(rows)])
    month_starts = pd.date_range("2025-01-01", "2026-01-01", freq="MS")
    axis.set_xticks([(month - origin).days for month in month_starts], [f"{month:%b}" for month in month_starts])
    axis.set_xlim(0, (pd.Timestamp("2026-01-01") - origin).days)
    axis.set_title("Expanding-window backtest: train on everything before the window (teal), test on the next 2 months", fontsize=10)
    axis.grid(axis="x", alpha=0.3)
    axis.set_axisbelow(True)
    for side in ("top", "right"):
        axis.spines[side].set_visible(False)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "09_split_scheme.png", dpi=150)
    plt.close(figure)


def plot_backtest(results: pd.DataFrame) -> None:
    backtest = results[results.scheme == "expanding window"]
    figure, axis = plt.subplots(figsize=(11, 4.5))
    for name, group in backtest.groupby("model", sort=False):
        axis.plot(range(len(group)), group["MAPE_%"], marker="o", label=name, lw=2.2 if "final" in name else 1.4)
    axis.set_xticks(range(len(BACKTEST_FOLDS)), [label.split("| ")[1] for label in backtest.fold.unique()])
    axis.set_ylabel("MAPE on clean holdout rows (%)")
    axis.set_title("Expanding-window backtest, 2-month horizon")
    axis.set_ylim(bottom=0)
    axis.grid(alpha=0.3)
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIG_DIR / "06_backtest_mape.png", dpi=130)
    plt.close(figure)


def plot_primary_fold(dev: pd.DataFrame, predictions: dict) -> None:
    start, end = PRIMARY_FOLD
    test = dev[(dev.date >= start) & (dev.date < end) & ~dev.is_outlier]
    figure, axes = plt.subplots(1, 2, figsize=(14, 4.8), gridspec_kw={"width_ratios": [1.6, 1]})
    # Daily mean of log(pred / actual): shows whether the market level is tracked over time.
    for name in ("Single LightGBM", "Two-stage (final)"):
        ratio = np.log(predictions[name][test.index] / test.posted_rate) * 100
        axes[0].plot(ratio.groupby(test.date).mean(), label=name, lw=1.6)
    axes[0].axhline(0, color="black", lw=0.8)
    axes[0].xaxis.set_major_locator(mdates.WeekdayLocator(byweekday=mdates.MO, interval=1))
    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%b %d"))
    axes[0].tick_params(axis="x", rotation=45)
    axes[0].set_title("Sep-Oct holdout: daily mean prediction error (%)")
    axes[0].set_ylabel("mean log(pred / actual) x 100")
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    final = predictions["Two-stage (final)"][test.index]
    axes[1].scatter(test.posted_rate, final, s=2, alpha=0.3, color="#064A56")
    limit = [test.posted_rate.min(), test.posted_rate.max()]
    axes[1].plot(limit, limit, color="#C0392B", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    ticks = [200, 500, 1000, 2000, 5000]
    for axis in (axes[1].xaxis, axes[1].yaxis):
        axis.set_major_locator(FixedLocator(ticks))
        axis.set_major_formatter(FixedFormatter([f"${tick:,}" for tick in ticks]))
        axis.set_minor_formatter(NullFormatter())
    axes[1].set_xlabel("actual posted_rate ($)")
    axes[1].set_ylabel("predicted ($)")
    axes[1].set_title("Two-stage: predicted vs actual (Sep-Oct)")
    figure.tight_layout()
    figure.savefig(FIG_DIR / "07_primary_holdout.png", dpi=130)
    plt.close(figure)


def summary_table(results: pd.DataFrame) -> pd.DataFrame:
    columns = ["MAPE_%", "MAE", "RMSE", "bias_%", "R2", "MAE_all_rows"]
    return results.groupby(["scheme", "model"], sort=False)[columns].mean().round(3)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dev = load_train()
    validation = load_validation()
    dev["is_outlier"] = flag_label_outliers(dev)
    features = build_features(dev, daily_market(dev, validation))
    print(f"flagged {dev.is_outlier.sum()} corrupted labels ({dev.is_outlier.mean():.2%})")

    print("expanding-window backtest")
    backtest, primary_predictions = run_backtest(dev, features)
    print("random 5-fold")
    kfold = run_random_kfold(dev, features)
    print("unseen-city check")
    unseen = run_unseen_cities(dev, features)

    results = pd.concat([backtest, kfold, unseen], ignore_index=True)
    results.to_csv(REPORT_DIR / "validation_results.csv", index=False)
    plot_split_scheme()
    plot_backtest(results)
    plot_primary_fold(dev, primary_predictions)

    pd.set_option("display.width", 200)
    print("\nMean over folds (metrics on clean holdout rows; MAE_all_rows includes corrupted labels):")
    print(summary_table(results).to_string())
    print("\nPer fold, two-stage model:")
    final = backtest[backtest.model == "Two-stage (final)"].set_index("fold")
    print(final[["MAPE_%", "MAE", "RMSE", "bias_%", "R2"]].round(3).to_string())
    print("\nUnseen-city check:")
    print(unseen.set_index("fold")[["MAPE_%", "MAE", "bias_%", "n_test"]].round(3).to_string())


if __name__ == "__main__":
    main()
