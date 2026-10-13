"""Exploratory data analysis for the freight rate challenge.

Run from the project root:
    python src/eda.py

Prints the key statistics and writes figures to reports/figures/.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter
import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

ROOT = Path(__file__).resolve().parents[1]
TRAIN_PATH = ROOT / "data" / "train_test.csv"
VALIDATION_PATH = ROOT / "data" / "validation.csv"
FIG_DIR = ROOT / "reports" / "figures"

WEIGHT_FLOOR = 5_000.0
WEIGHT_CAP = 47_500.0
# Clean rows sit within roughly 0.9x-1.12x of a simple baseline; corrupted labels sit
# at 0.17x-0.45x or 2.2x-5.4x. Anything outside [0.61x, 1.65x] is flagged.
OUTLIER_LOG_THRESHOLD = 0.5
VALIDATION_START = pd.Timestamp("2025-11-01")


def section(title: str) -> None:
    print(f"\n{'=' * 8} {title} {'=' * 8}")


def haversine_miles(lat1, lon1, lat2, lon2):
    lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
    h = np.sin((lat2 - lat1) / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin((lon2 - lon1) / 2) ** 2
    return 3958.8 * 2 * np.arcsin(np.sqrt(h))


def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    train = pd.read_csv(TRAIN_PATH, parse_dates=["date"])
    validation = pd.read_csv(VALIDATION_PATH, parse_dates=["date"])
    return train, validation


def baseline_residuals(train: pd.DataFrame) -> pd.DataFrame:
    """Fit a robust log-linear baseline and return train with residual columns.

    The baseline only uses distance, equipment and the daily market index, so the
    residual exposes everything else: label corruption, calendar and location effects.
    """
    frame = train.copy()
    daily_mi = frame.groupby("date").market_index.transform("mean")
    frame["mi_filled"] = frame.market_index.fillna(daily_mi)
    frame["mi_day"] = daily_mi
    features = pd.DataFrame(
        {
            "log_distance": np.log(frame.distance),
            "reefer": (frame.equipment == "Reefer").astype(float),
            "flatbed": (frame.equipment == "Flatbed").astype(float),
            "mi_day": frame.mi_day,
        }
    )
    target = np.log(frame.posted_rate)
    model = HuberRegressor(max_iter=1000).fit(features, target)
    frame["log_resid"] = target - model.predict(features)
    frame["is_outlier"] = frame.log_resid.abs() > OUTLIER_LOG_THRESHOLD
    clean = ~frame.is_outlier
    # Lane-demeaned residual isolates calendar effects from lane mix.
    lane_mean = frame[clean].groupby(["pickup", "delivery"]).log_resid.transform("mean")
    frame.loc[clean, "lane_resid"] = frame.loc[clean, "log_resid"] - lane_mean
    print("baseline coefficients:", dict(zip(features.columns, model.coef_.round(4))))
    return frame


def report_schema(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    section("Shape, coverage, missing values")
    for name, frame in [("train", train), ("validation", validation)]:
        nulls = frame.isna().sum()
        print(
            f"{name}: {frame.shape}, dates {frame.date.min():%Y-%m-%d} -> {frame.date.max():%Y-%m-%d}, "
            f"nulls {nulls[nulls > 0].to_dict()}, duplicate load_id {frame.load_id.duplicated().sum()}"
        )
    train_cities = set(train.pickup) | set(train.delivery)
    val_cities = set(validation.pickup) | set(validation.delivery)
    unseen = sorted(val_cities - train_cities)
    touches_unseen = validation.pickup.isin(unseen) | validation.delivery.isin(unseen)
    print(f"cities: train {len(train_cities)}, validation {len(val_cities)}; unseen in train: {unseen}")
    print(f"validation rows touching an unseen city: {touches_unseen.sum()} ({touches_unseen.mean():.1%})")
    combined = pd.concat([train, validation])
    coord_counts = combined.groupby("pickup")[["pickup_lat", "pickup_lon"]].nunique().max(axis=1)
    print(f"cities with more than one coordinate pair: {(coord_counts > 1).sum()} (coordinates are fixed per city)")


def report_weight(train: pd.DataFrame, validation: pd.DataFrame) -> None:
    section("Weight quality")
    for name, frame in [("train", train), ("validation", validation)]:
        weight = frame.weight
        print(
            f"{name}: missing {weight.isna().sum()}, negative {(weight < 0).sum()}, "
            f"at floor {WEIGHT_FLOOR:g} {(weight.abs() == WEIGHT_FLOOR).sum()}, at cap {WEIGHT_CAP:g} {(weight.abs() == WEIGHT_CAP).sum()}"
        )
    negative = train.weight[train.weight < 0].abs()
    positive = train.weight[train.weight > 0]
    print(f"median |negative weight| {negative.median():,.0f} vs median positive {positive.median():,.0f} -> sign flips")
    print(f"smallest |weight| {train.weight.abs().min():g}: values are clipped to [{WEIGHT_FLOOR:g}, {WEIGHT_CAP:g}]")


def report_distance(train: pd.DataFrame) -> None:
    section("Distance quality")
    road_ratio = train.distance / haversine_miles(
        train.pickup_lat, train.pickup_lon, train.delivery_lat, train.delivery_lon
    )
    lane_ratio = train.distance / train.groupby(["pickup", "delivery"]).distance.transform("median")
    print(f"distance / straight-line miles: median {road_ratio.median():.3f}, p1 {road_ratio.quantile(.01):.3f}, p99 {road_ratio.quantile(.99):.3f}")
    print(f"distance / lane median: p0.1 {lane_ratio.quantile(.001):.3f}, p99.9 {lane_ratio.quantile(.999):.3f} (per-load jitter, no gross errors)")
    print(f"distance floor: min {train.distance.min():g}, rows at 70 mi: {(train.distance == 70).sum()}")


def report_target(frame: pd.DataFrame) -> None:
    section("Target quality")
    rate_per_mile = frame.posted_rate / frame.distance
    print(f"posted_rate: min {frame.posted_rate.min():,.2f}, median {frame.posted_rate.median():,.2f}, max {frame.posted_rate.max():,.2f}")
    print(f"rate per mile: p0.5 {rate_per_mile.quantile(.005):.2f}, median {rate_per_mile.median():.2f}, p99.5 {rate_per_mile.quantile(.995):.2f}")
    ratio = np.exp(frame.log_resid)
    high = frame[frame.log_resid > OUTLIER_LOG_THRESHOLD]
    low = frame[frame.log_resid < -OUTLIER_LOG_THRESHOLD]
    print(f"clean rows: ratio to baseline p1 {ratio[~frame.is_outlier].quantile(.01):.3f}, p99 {ratio[~frame.is_outlier].quantile(.99):.3f}")
    print(f"inflated labels: {len(high)} rows, {np.exp(high.log_resid).min():.2f}x-{np.exp(high.log_resid).max():.2f}x (median {np.exp(high.log_resid).median():.2f}x)")
    print(f"deflated labels: {len(low)} rows, {np.exp(low.log_resid).min():.2f}x-{np.exp(low.log_resid).max():.2f}x (median {np.exp(low.log_resid).median():.2f}x)")
    print(f"total flagged: {frame.is_outlier.sum()} ({frame.is_outlier.mean():.2%}); spread evenly across months: "
          f"{frame[frame.is_outlier].date.dt.month.value_counts().sort_index().to_dict()}")
    print("clean rate per mile by equipment:", (frame[~frame.is_outlier].posted_rate / frame[~frame.is_outlier].distance)
          .groupby(frame.equipment).median().round(3).to_dict())


def report_time(frame: pd.DataFrame, validation: pd.DataFrame) -> None:
    section("Market index and calendar effects")
    combined = pd.concat([frame, validation])
    print("market_index mean by month:", combined.groupby(combined.date.dt.month).market_index.mean().round(3).to_dict())
    print("market_index mean by weekday (0=Mon):", frame.groupby(frame.date.dt.dayofweek).market_index.mean().round(3).to_dict())
    within_day = (frame.market_index - frame.mi_day).std()
    print(f"market_index std: between days {frame.groupby('date').market_index.mean().std():.3f}, within a day {within_day:.3f}")
    clean = frame[~frame.is_outlier].copy()
    clean["quarter"] = clean.date.dt.quarter
    clean["month_in_quarter"] = (clean.date.dt.month - 1) % 3 + 1
    table = (clean.groupby(["quarter", "month_in_quarter"]).lane_resid.mean() * 100).unstack().round(2)
    print("rate residual % after market index, by quarter x month-in-quarter:")
    print(table.to_string())
    weekday = (clean.groupby(clean.date.dt.dayofweek).lane_resid.mean() * 100).round(2).to_dict()
    print("rate residual % after market index, by weekday:", weekday)


def report_location(frame: pd.DataFrame) -> None:
    section("Location effects")
    clean = frame[~frame.is_outlier].copy()
    clean["day_resid"] = clean.log_resid - clean.groupby("date").log_resid.transform("mean")
    for side in ["pickup", "delivery"]:
        effect = clean.groupby(side).day_resid.mean()
        lat = clean.groupby(side)[f"{side}_lat"].first()
        print(f"{side} effect: range {effect.min():+.3f} to {effect.max():+.3f}, corr with latitude {effect.corr(lat):+.2f}; "
              f"highest {effect.idxmax()}, lowest {effect.idxmin()}")


def report_quote_signal(frame: pd.DataFrame) -> None:
    section("quote_signal")
    clean = frame[~frame.is_outlier].copy()
    clean["day_resid"] = clean.log_resid - clean.groupby("date").log_resid.transform("mean")
    print(f"corr with log rate per mile: {np.log(clean.posted_rate / clean.distance).corr(clean.quote_signal):+.3f}")
    print(f"corr with residual after distance/equipment/market/date: {clean.day_resid.corr(clean.quote_signal):+.3f}")


def plot_target(frame: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    rate_per_mile = frame.posted_rate / frame.distance
    axes[0].hist(rate_per_mile, bins=np.logspace(np.log10(0.3), np.log10(15), 120), color="#064A56")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_title("Rate per mile (log-log): clean mass at $1.7-3.2, corrupted tails")
    axes[0].set_xlabel("posted_rate / distance ($/mi)")
    ratio = np.exp(frame.log_resid)
    axes[1].hist(ratio, bins=np.logspace(np.log10(0.15), np.log10(6), 150), color="#064A56")
    for bound in (np.exp(-OUTLIER_LOG_THRESHOLD), np.exp(OUTLIER_LOG_THRESHOLD)):
        axes[1].axvline(bound, color="#C0392B", ls="--", lw=1)
    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    ticks = [0.2, 0.3, 0.5, 1, 2, 3, 5]
    axes[1].set_xticks(ticks, [f"{tick:g}x" for tick in ticks])
    axes[1].xaxis.set_minor_formatter(NullFormatter())
    axes[1].set_title(f"Label / baseline prediction: {frame.is_outlier.sum()} rows flagged outside red lines")
    axes[1].set_xlabel("ratio (log scale)")
    figure.tight_layout()
    figure.savefig(FIG_DIR / "01_target_outliers.png", dpi=130)
    plt.close(figure)


def plot_weight(train: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(10, 3.8))
    axis.hist(train.weight.dropna(), bins=200, color="#064A56")
    axis.set_yscale("log")
    axis.set_title(f"Weight: {(train.weight < 0).sum()} negative (sign flips), clipped to "
                   f"[{WEIGHT_FLOOR:,.0f}, {WEIGHT_CAP:,.0f}] ({(train.weight.abs() == WEIGHT_CAP).sum()} at cap), "
                   f"{train.weight.isna().sum()} missing")
    axis.set_xlabel("weight (lb)")
    figure.tight_layout()
    figure.savefig(FIG_DIR / "02_weight_quality.png", dpi=130)
    plt.close(figure)


def plot_time(frame: pd.DataFrame, validation: pd.DataFrame) -> None:
    combined = pd.concat([frame, validation])
    daily_mi = combined.groupby("date").market_index.mean()
    clean = frame[~frame.is_outlier]
    level = clean.copy()
    # Rate level without market index: residual from a distance + equipment fit, lane-demeaned.
    features = pd.DataFrame(
        {
            "log_distance": np.log(level.distance),
            "reefer": (level.equipment == "Reefer").astype(float),
            "flatbed": (level.equipment == "Flatbed").astype(float),
        }
    )
    model = HuberRegressor(max_iter=1000).fit(features, np.log(level.posted_rate))
    level["raw_resid"] = np.log(level.posted_rate) - model.predict(features)
    level["raw_resid"] -= level.groupby(["pickup", "delivery"]).raw_resid.transform("mean")
    daily_level = level.groupby("date").raw_resid.mean() * 100
    daily_resid = clean.groupby("date").lane_resid.mean() * 100

    figure, axes = plt.subplots(3, 1, figsize=(13, 9.5), sharex=True)
    axes[0].plot(daily_mi.index, daily_mi.values, color="#064A56", lw=1)
    axes[0].axvline(VALIDATION_START, color="#C0392B", ls="--", lw=1)
    axes[0].text(VALIDATION_START, daily_mi.max(), "  validation", color="#C0392B", va="top")
    axes[0].set_title("Daily mean market_index: weekly cycle (Thu peak, Sun trough) on a seasonal level")
    axes[1].plot(daily_level.index, daily_level.values, color="black", lw=1)
    axes[1].set_title("Daily rate level, % (lane and equipment adjusted, market_index NOT used)")
    axes[2].plot(daily_resid.index, daily_resid.values, color="#7D3C98", lw=1)
    for quarter_start in pd.to_datetime(["2025-04-01", "2025-07-01", "2025-10-01"]):
        axes[2].axvline(quarter_start, color="grey", ls=":", lw=1)
    axes[2].set_title("Daily rate residual % AFTER market_index: ramps through each quarter, resets on quarter start")
    for axis in axes:
        axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "03_time_effects.png", dpi=130)
    plt.close(figure)


def plot_location(frame: pd.DataFrame, validation: pd.DataFrame) -> None:
    clean = frame[~frame.is_outlier].copy()
    clean["day_resid"] = clean.log_resid - clean.groupby("date").log_resid.transform("mean")
    effect = clean.groupby("pickup").day_resid.mean() * 100
    coords = pd.concat([frame, validation]).groupby("pickup")[["pickup_lat", "pickup_lon"]].first()
    unseen = sorted((set(validation.pickup) | set(validation.delivery)) - (set(frame.pickup) | set(frame.delivery)))

    figure, axis = plt.subplots(figsize=(11, 5.5))
    seen = coords.loc[effect.index]
    points = axis.scatter(seen.pickup_lon, seen.pickup_lat, c=effect.values, cmap="RdBu_r", vmin=-4, vmax=4, s=70, edgecolor="k", lw=0.4)
    axis.scatter(coords.loc[unseen].pickup_lon, coords.loc[unseen].pickup_lat, marker="x", color="#C0392B", s=70, label="unseen in train")
    for city in unseen:
        axis.annotate(city, coords.loc[city], fontsize=8, color="#C0392B", xytext=(4, 4), textcoords="offset points")
    figure.colorbar(points, label="origin rate effect (%)")
    axis.legend(loc="lower left")
    axis.set_title("Origin rate effect by city coordinate: south/Texas pricier, north-east cheaper")
    axis.set_xlabel("longitude (as given)")
    axis.set_ylabel("latitude (as given)")
    figure.tight_layout()
    figure.savefig(FIG_DIR / "04_location_effects.png", dpi=130)
    plt.close(figure)


def plot_december_lane(frame: pd.DataFrame) -> None:
    lane = frame[(frame.pickup == "Lexington") & (frame.delivery == "Fort Wayne")].sort_values("date")
    figure, axis = plt.subplots(figsize=(10, 3.8))
    for equipment, group in lane.groupby("equipment"):
        axis.plot(group.date, group.posted_rate, marker="o", ls="", label=equipment)
    axis.set_title(f"Lexington -> Fort Wayne history ({len(lane)} loads): the fixed December chart lane")
    axis.set_ylabel("posted_rate ($)")
    axis.legend()
    axis.grid(alpha=0.3)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "05_december_lane_history.png", dpi=130)
    plt.close(figure)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train, validation = load()
    report_schema(train, validation)
    report_weight(train, validation)
    report_distance(train)
    frame = baseline_residuals(train)
    report_target(frame)
    report_time(frame, validation)
    report_location(frame)
    report_quote_signal(frame)
    plot_target(frame)
    plot_weight(train)
    plot_time(frame, validation)
    plot_location(frame, validation)
    plot_december_lane(frame)
    print(f"\nFigures written to {FIG_DIR}")


if __name__ == "__main__":
    main()
