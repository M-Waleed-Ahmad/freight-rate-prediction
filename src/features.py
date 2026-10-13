"""Data loading, cleaning and feature engineering shared by validation and training."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import HuberRegressor

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
TRAIN_PATH = DATA_DIR / "train_test.csv"
VALIDATION_PATH = DATA_DIR / "validation.csv"
TEMPLATE_PATH = DATA_DIR / "validation_predictions_template.csv"
DECEMBER_PATH = DATA_DIR / "december_chart_inputs.csv"

EQUIPMENT_CODES = {"Dry Van": 0, "Reefer": 1, "Flatbed": 2}
WEIGHT_CAP = 47_500.0
# Corrupted labels sit at 0.17x-0.45x or 2.2x-5.4x a simple baseline; clean rows sit
# within 0.90x-1.11x. Anything beyond exp(+-0.5) = [0.61x, 1.65x] is treated as corrupt.
OUTLIER_LOG_THRESHOLD = 0.5
TIME_ORIGIN = pd.Timestamp("2025-01-01")


def load_train() -> pd.DataFrame:
    return pd.read_csv(TRAIN_PATH, parse_dates=["date"])


def load_validation() -> pd.DataFrame:
    return pd.read_csv(VALIDATION_PATH, parse_dates=["date"])


def city_coordinates(*frames: pd.DataFrame) -> pd.DataFrame:
    """City -> (lat, lon). Each city has exactly one coordinate pair in the data."""
    parts = []
    for frame in frames:
        for side in ("pickup", "delivery"):
            part = frame[[side, f"{side}_lat", f"{side}_lon"]]
            parts.append(part.set_axis(["city", "lat", "lon"], axis=1))
    return pd.concat(parts).drop_duplicates("city").set_index("city")


def daily_market(*frames: pd.DataFrame) -> pd.DataFrame:
    """Per-date mean of market_index and quote_signal across every load that day.

    market_index is mostly a date-level signal (std 0.167 between days vs 0.025 within
    a day), so the daily mean is a denoised version of it. Uses features only, never labels.
    """
    rows = pd.concat([frame[["date", "market_index", "quote_signal"]] for frame in frames])
    return rows.groupby("date").agg(mi_day=("market_index", "mean"), qs_day=("quote_signal", "mean"))


def load_december(validation: pd.DataFrame, coords: pd.DataFrame) -> pd.DataFrame:
    """December chart inputs enriched with the columns the raw file lacks.

    Coordinates come from the city lookup. market_index/quote_signal come from the daily
    means of the December loads in validation.csv, so the within-day deviation is zero.
    """
    december = pd.read_csv(DECEMBER_PATH, parse_dates=["date"])
    for side in ("pickup", "delivery"):
        december[f"{side}_lat"] = december[side].map(coords.lat)
        december[f"{side}_lon"] = december[side].map(coords.lon)
    market = daily_market(validation)
    december["market_index"] = december.date.map(market.mi_day)
    december["quote_signal"] = december.date.map(market.qs_day)
    if december[["pickup_lat", "delivery_lat", "market_index"]].isna().any().any():
        raise ValueError("could not enrich every December row")
    return december


def build_features(frame: pd.DataFrame, market: pd.DataFrame) -> pd.DataFrame:
    """Model-ready feature matrix. Works for train, validation and December rows."""
    weight = frame.weight.abs()  # negative weights are sign flips
    market_index = frame.market_index.fillna(frame.date.map(market.mi_day))
    mi_day = frame.date.map(market.mi_day)
    qs_day = frame.date.map(market.qs_day)
    quarter_start = frame.date.dt.to_period("Q").dt.start_time
    quarter_end = frame.date.dt.to_period("Q").dt.end_time.dt.normalize()

    return pd.DataFrame(
        {
            "log_distance": np.log(frame.distance),
            "equipment": frame.equipment.map(EQUIPMENT_CODES),
            "weight": weight,
            "weight_missing": weight.isna().astype(int),
            "weight_at_cap": (weight == WEIGHT_CAP).astype(int),
            "pickup_lat": frame.pickup_lat,
            "pickup_lon": frame.pickup_lon,
            "delivery_lat": frame.delivery_lat,
            "delivery_lon": frame.delivery_lon,
            "market_index": market_index,
            "mi_day": mi_day,
            "mi_dev": market_index - mi_day,
            "quote_signal": frame.quote_signal,
            "qs_day": qs_day,
            "qs_dev": frame.quote_signal - qs_day,
            "day_of_week": frame.date.dt.dayofweek,
            "day_of_month": frame.date.dt.day,
            "day_of_quarter": (frame.date - quarter_start).dt.days + 1,
            "days_to_quarter_end": (quarter_end - frame.date).dt.days,
            "month_in_quarter": (frame.date.dt.month - 1) % 3 + 1,
            "days_since_origin": (frame.date - TIME_ORIGIN).dt.days,
        },
        index=frame.index,
    )


def flag_label_outliers(train: pd.DataFrame) -> pd.Series:
    """True for rows whose posted_rate is a corrupted label rather than a market price.

    A robust log-linear baseline (distance, equipment, daily market index) predicts each
    rate; labels more than exp(0.5) away from it in either direction are flagged. The gap
    between clean and corrupted rows is wide, so the threshold is not sensitive.
    """
    market = daily_market(train)
    mi_day = train.date.map(market.mi_day)
    design = pd.DataFrame(
        {
            "log_distance": np.log(train.distance),
            "reefer": (train.equipment == "Reefer").astype(float),
            "flatbed": (train.equipment == "Flatbed").astype(float),
            "mi_day": mi_day,
        }
    )
    target = np.log(train.posted_rate)
    model = HuberRegressor(max_iter=1000).fit(design, target)
    return pd.Series(np.abs(target - model.predict(design)) > OUTLIER_LOG_THRESHOLD, index=train.index)
