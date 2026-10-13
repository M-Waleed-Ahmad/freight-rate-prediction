"""Fit the final model on all clean development data and write both prediction files.

Run from the project root:
    python src/train.py

Outputs
* validation_predictions.csv          load_id,predicted_rate for all 12,000 validation loads
* data/december_chart_inputs.csv       predicted_rate column filled in place (input columns untouched)
* reports/figures/08_december_breakdown.png
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from features import (
    DECEMBER_PATH,
    ROOT,
    TEMPLATE_PATH,
    build_features,
    city_coordinates,
    daily_market,
    flag_label_outliers,
    load_december,
    load_train,
    load_validation,
)
from model import TwoStageRateModel, market_design

PREDICTIONS_PATH = ROOT / "validation_predictions.csv"
FIG_DIR = ROOT / "reports" / "figures"


def write_validation_predictions(validation: pd.DataFrame, predictions: np.ndarray) -> None:
    template = pd.read_csv(TEMPLATE_PATH)
    by_id = pd.Series(predictions, index=validation.load_id)
    if set(template.load_id) != set(by_id.index):
        raise ValueError("validation load_ids do not match the template")
    template["predicted_rate"] = template.load_id.map(by_id).round(2)
    template.to_csv(PREDICTIONS_PATH, index=False)


def write_december_predictions(predictions: np.ndarray) -> None:
    raw = pd.read_csv(DECEMBER_PATH)  # re-read so every input column keeps its original formatting
    raw["predicted_rate"] = np.round(predictions, 2)
    raw.to_csv(DECEMBER_PATH, index=False)


def plot_december_breakdown(model: TwoStageRateModel, december: pd.DataFrame, features: pd.DataFrame) -> None:
    """Show how each market term moves the fixed December load's price."""
    coefficients = model.market_coefficients()
    design = market_design(features)
    rate = model.predict(features)
    first_day = design.iloc[0]
    effects = {
        "market index (weekly cycle)": coefficients["market_index"] * (design.market_index - first_day.market_index),
        "trend": coefficients["trend_days"] * (design.trend_days - first_day.trend_days),
        "quarter-end ramp": coefficients["quarter_end_ramp_days"] * design.quarter_end_ramp_days,
    }
    figure, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]})
    axes[0].plot(december.date, rate, marker="o", color="#064A56", lw=2)
    axes[0].set_ylabel("predicted rate ($)")
    axes[0].set_title("Lexington -> Fort Wayne, Dry Van, 360 mi, 32,000 lb: December 2025 prediction")
    for label, effect in effects.items():
        axes[1].plot(december.date, 100 * (np.exp(effect) - 1), marker=".", label=label)
    axes[1].axhline(0, color="black", lw=0.8)
    axes[1].set_ylabel("change vs Dec 1 (%)")
    axes[1].set_title("What moves the price: each market term relative to Dec 1")
    axes[1].legend(loc="upper left")
    for axis in axes:
        axis.grid(alpha=0.3)
    axes[1].tick_params(axis="x", rotation=35)
    figure.tight_layout()
    figure.savefig(FIG_DIR / "08_december_breakdown.png", dpi=130)
    plt.close(figure)


def main() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    dev = load_train()
    validation = load_validation()
    outliers = flag_label_outliers(dev)
    clean = dev[~outliers]
    print(f"training on {len(clean):,} rows ({outliers.sum()} corrupted labels removed)")

    market = daily_market(dev, validation)
    model = TwoStageRateModel().fit(build_features(clean, market), clean.posted_rate, clean.date)
    print("market model:", {name: round(value, 6) for name, value in model.market_coefficients().items()})

    validation_pred = model.predict(build_features(validation, market))
    write_validation_predictions(validation, validation_pred)
    print(f"wrote {PREDICTIONS_PATH.name}: {len(validation_pred):,} rows, "
          f"rate ${validation_pred.min():,.0f}-${validation_pred.max():,.0f}, median ${np.median(validation_pred):,.0f}")

    december = load_december(validation, city_coordinates(dev, validation))
    december_features = build_features(december, market)
    december_pred = model.predict(december_features)
    write_december_predictions(december_pred)
    plot_december_breakdown(model, december, december_features)
    print(f"filled {DECEMBER_PATH.relative_to(ROOT)}: Dec 1 ${december_pred[0]:,.2f}, Dec 31 ${december_pred[-1]:,.2f}, "
          f"range ${december_pred.min():,.2f}-${december_pred.max():,.2f}")


if __name__ == "__main__":
    main()
