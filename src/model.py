"""Two-stage freight rate model.

log(rate) = log(distance) + load_effect(load features) + market_effect(date, market index)

* load_effect: LightGBM on what makes one load cost more than another on the same day
  (distance curve, equipment, weight, origin/destination coordinates).
* market_effect: a linear model for what moves every load's price on a given day: the
  market index, a slow upward trend and the quarter-end ramp. Being linear, it extrapolates
  to market index levels and dates outside the training window, which trees cannot do.

The two parts are separated by backfitting: the trees are fit against per-date fixed
effects, so day-level movements never leak into the load model.
"""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

LOAD_FEATURES = [
    "log_distance",
    "equipment",
    "weight",
    "weight_missing",
    "weight_at_cap",
    "pickup_lat",
    "pickup_lon",
    "delivery_lat",
    "delivery_lon",
]

LGB_PARAMS = {
    "objective": "regression",
    "learning_rate": 0.03,
    "num_leaves": 31,
    "min_data_in_leaf": 40,
    "feature_fraction": 0.9,
    "bagging_fraction": 0.8,
    "bagging_freq": 1,
    "verbose": -1,
    "seed": 7,
}
NUM_ROUNDS = 1200
BACKFIT_ITERATIONS = 3
MARKET_TERMS = ["market_index", "trend_days", "quarter_end_month", "quarter_end_ramp_days"]


def market_design(features: pd.DataFrame, terms: list[str] = MARKET_TERMS) -> pd.DataFrame:
    """Date-level regressors for the linear market model.

    Only MARKET_TERMS are used by the final model; the others exist for ablations.
    """
    quarter_end_month = (features.month_in_quarter == 3).astype(float)
    candidates = {
        "market_index": features.market_index,
        "trend_days": features.days_since_origin,
        "quarter_end_month": quarter_end_month,
        "quarter_end_ramp_days": quarter_end_month * (features.day_of_month - 1),
        "quote_signal": features.quote_signal,
        **{f"day_of_week_{day}": (features.day_of_week == day).astype(float) for day in range(1, 7)},
    }
    return pd.DataFrame({term: candidates[term] for term in terms}, index=features.index)


class TwoStageRateModel:
    def __init__(
        self,
        load_features: list[str] = LOAD_FEATURES,
        market_terms: list[str] = MARKET_TERMS,
        backfit_iterations: int = BACKFIT_ITERATIONS,
    ):
        self.load_features = load_features
        self.market_terms = market_terms
        self.backfit_iterations = backfit_iterations

    def fit(self, features: pd.DataFrame, rate: pd.Series, dates: pd.Series) -> "TwoStageRateModel":
        target = np.log(rate) - features.log_distance  # log rate per mile
        date_effect = pd.Series(0.0, index=features.index)
        for _ in range(self.backfit_iterations):
            dataset = lgb.Dataset(features[self.load_features], target - date_effect, categorical_feature=["equipment"])
            self.load_model = lgb.train(LGB_PARAMS, dataset, NUM_ROUNDS)
            residual = target - self.load_model.predict(features[self.load_features])
            date_effect = residual.groupby(dates).transform("mean")
        self.market_model = LinearRegression().fit(market_design(features, self.market_terms), residual)
        return self

    def components(self, features: pd.DataFrame) -> pd.DataFrame:
        """Additive log-space parts of each prediction."""
        return pd.DataFrame(
            {
                "log_distance": features.log_distance,
                "load_effect": self.load_model.predict(features[self.load_features]),
                "market_effect": self.market_model.predict(market_design(features, self.market_terms)),
            },
            index=features.index,
        )

    def predict(self, features: pd.DataFrame) -> np.ndarray:
        return np.exp(self.components(features).sum(axis=1).to_numpy())

    def market_coefficients(self) -> dict[str, float]:
        names = self.market_model.feature_names_in_
        return {"intercept": float(self.market_model.intercept_), **dict(zip(names, map(float, self.market_model.coef_)))}
