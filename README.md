# Freight Rate Prediction Challenge

Predicts `posted_rate` for truckload freight. The model is trained on Jan–Oct 2025 loads and forecasts the 12,000 Nov–Dec 2025 validation loads plus the fixed December chart lane. Built for Spotter's Machine Learning Engineer assessment; the data files and `score.py` are as provided by Spotter.

**Result:** in an expanding-window backtest with a 2-month horizon, the model's average error (MAPE) is **1.60%** (MAE $38, R² 0.998), with near-zero bias (+0.09%). On the Sep–Oct holdout, the closest stand-in for the real task, it's 1.56%.

## Setup

Python 3.10+ (developed on 3.14).

```bash
python -m venv .venv
.venv/Scripts/activate        # Windows; on macOS/Linux: source .venv/bin/activate
python -m pip install -r requirements.txt
```

## Run

From the project root:

```bash
python src/train.py           # fit on all clean data, write both prediction files
python score.py --predictions validation_predictions.csv --december-predictions data/december_chart_inputs.csv
```

`src/train.py` writes `validation_predictions.csv` and fills the `predicted_rate` column of `data/december_chart_inputs.csv` in place. The scorer validates both files and creates `scorer_results/candidate_december.png`.

Optional analysis scripts:

```bash
python src/eda.py             # data exploration stats + figures 01-05 (~10 s)
python src/validate.py        # backtests, random-CV contrast, unseen-city check + figures 06-07 (~2 min)
python src/ablation.py        # feature / design ablations on the same backtest (~3 min)
```

## Layout

```
data/                      provided inputs (train_test, validation, template, december chart inputs)
src/features.py            loading, cleaning, label-outlier flagging, feature engineering
src/model.py               two-stage model (LightGBM load effect + linear market effect)
src/train.py               final fit and prediction files
src/validate.py            time-based validation and baselines
src/ablation.py            design ablations
src/eda.py                 exploratory analysis
reports/eda_findings.md    data findings and data-quality issues
reports/modeling.md        split/validation approach, results, model choice
reports/figures/           all charts
score.py                   provided scorer (unchanged)
```

## Approach in brief

- **Data quality:**
  - 677 corrupted rate labels (1.41%) sit 2.2–5.4x too high or 0.17–0.45x too low; they're removed from training.
  - Negative weights are sign flips, so `abs()` is used.
  - Missing `market_index` is filled with that day's mean.
  - 8 validation cities never appear in training, so location is modeled through coordinates rather than city IDs.
- **Validation:** the task is a 2-month forward forecast, so the model is validated with an expanding-window backtest (6 folds, 2-month test windows), never with shuffled splits. A random 5-fold split makes a plain LightGBM look best (0.99% MAPE); forward in time it scores 4.70%.
- **Model:** `log(rate) = log(distance) + load_effect + market_effect`.
  - **Load effect:** LightGBM learns what makes one load pricier than another on the same day (distance curve, equipment, weight, coordinates).
  - **Market effect:** a linear model learns what moves all prices on a given day (`market_index`, a slow trend, and a ramp through the last month of each quarter). Because it's linear, it extrapolates to Nov–Dec, which trees cannot do.
  - **Fitting:** backfitting against per-date fixed effects keeps the two parts separate.
- **December chart:** the input rows have no coordinates or `market_index`. Coordinates come from the city lookup. `market_index` is each December day's mean over the validation loads. The forecast rises from $827 to $890: a weekly cycle peaking on Thursdays, on top of the quarter-end ramp.

Details: [reports/eda_findings.md](reports/eda_findings.md) and [reports/modeling.md](reports/modeling.md).

## Submission checklist

- [x] GitHub repository with code, dependencies (`requirements.txt`) and run instructions (this file)
- [x] `validation_predictions.csv` (`load_id,predicted_rate`, 12,000 rows)
- [x] PDF report: [reports/freight_rate_report.pdf](reports/freight_rate_report.pdf) (validation and split approach, December chart)
- [ ] 2–3 minute Loom
