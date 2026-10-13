"""Ablations of the two-stage model on the same 6-fold expanding-window backtest.

Run from the project root:
    python src/ablation.py
"""
from __future__ import annotations

import pandas as pd

from features import build_features, daily_market, flag_label_outliers, load_train, load_validation
from model import LOAD_FEATURES, MARKET_TERMS, TwoStageRateModel
from validate import BACKTEST_FOLDS, REPORT_DIR, score

DAY_OF_WEEK = [f"day_of_week_{day}" for day in range(1, 7)]
VARIANTS = {
    "final": {},
    "market: market_index only": {"market_terms": ["market_index"]},
    "market: market_index + trend (no ramp)": {"market_terms": ["market_index", "trend_days"]},
    "market: + day-of-week dummies": {"market_terms": MARKET_TERMS + DAY_OF_WEEK},
    "market: + quote_signal": {"market_terms": MARKET_TERMS + ["quote_signal"]},
    "trees: + quote_signal within-day deviation": {"load_features": LOAD_FEATURES + ["qs_dev"]},
    "trees: + market_index within-day deviation": {"load_features": LOAD_FEATURES + ["mi_dev"]},
    "no backfitting (trees ignore date effects)": {"backfit_iterations": 1},
}


def main() -> None:
    dev = load_train()
    dev["is_outlier"] = flag_label_outliers(dev)
    features = build_features(dev, daily_market(dev, load_validation()))
    rows = []
    for name, kwargs in VARIANTS.items():
        for start, end in BACKTEST_FOLDS:
            train = dev[(dev.date < start) & ~dev.is_outlier]
            test = dev[(dev.date >= start) & (dev.date < end)]
            model = TwoStageRateModel(**kwargs).fit(features.loc[train.index], train.posted_rate, train.date)
            rows.append({"variant": name, "test_start": f"{start:%b}", **score(test, model.predict(features.loc[test.index]))})
        print(f"  {name} done")
    results = pd.DataFrame(rows)
    results.to_csv(REPORT_DIR / "ablation_results.csv", index=False)
    per_fold = results.pivot(index="variant", columns="test_start", values="MAPE_%").loc[list(VARIANTS)]
    per_fold = per_fold[[f"{start:%b}" for start, _ in BACKTEST_FOLDS]]
    per_fold["mean MAPE %"] = per_fold.mean(axis=1)
    per_fold["mean |bias| %"] = results.groupby("variant")["bias_%"].apply(lambda bias: bias.abs().mean())
    pd.set_option("display.width", 200)
    print("\nMAPE % on clean holdout rows by test-window start month:")
    print(per_fold.round(2).to_string())


if __name__ == "__main__":
    main()
