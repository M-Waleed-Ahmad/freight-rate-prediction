# Validation and Modeling

Reproduce with `python src/validate.py` (backtests, writes `reports/validation_results.csv` and figures 06–07) and `python src/train.py` (final fit and predictions). Data findings are in [eda_findings.md](eda_findings.md).

## 1. How the development data was split and validated

**The task is a forecast.** `train_test.csv` covers Jan 1 – Oct 31 2025, and `validation.csv` is the next two months (Nov 1 – Dec 31). The validation scheme reproduces that situation: always train on the past and test on the following two months.

| Step | What | Why |
|---|---|---|
| Label QA | Flag 677 corrupted rates (1.41%), once, over the whole development set. These are rows more than exp(±0.5) away from a robust log-linear baseline. | They form two separate clusters (2.2–5.4x and 0.17–0.45x). They aren't real prices, so a model can't and shouldn't predict them. |
| Training folds | Flagged rows are dropped from every training fold. | Stops 3–5x labels from dragging the fit. |
| Main scheme | **Expanding-window backtest, 2-month horizon, 6 folds**: train Jan→Mar / test Apr–May, …, train Jan→Aug / test Sep–Oct. | Same horizon as the real task, and each fold only sees the past. Six folds cover every season, including all three quarter-end months in the test windows. |
| Primary fold | Train Jan–Aug → test Sep–Oct. | The closest stand-in for the final setup (train Jan–Oct → predict Nov–Dec). |
| Contrast | Random 5-fold CV on the same data. | Shows how optimistic a shuffled split is. |
| New-city check | Remove 8 random cities from the primary fold's training data, then score test rows that touch them. | 12.1% of validation rows involve 8 cities that never appear in `train_test.csv`. |
| Metrics | MAPE, MAE, RMSE, bias and R² on clean holdout rows. MAE on all holdout rows (including corrupted labels) is reported alongside. | Corrupted labels add error no model can remove, so they would hide real differences between models. |

## 2. Results

Mean over the 6 expanding-window folds:

| Model | MAPE % | MAE $ | RMSE $ | Bias % | MAE $ (all rows) |
|---|---|---|---|---|---|
| Lane × equipment median $/mile | 5.01 | 116.5 | 166.8 | −2.89 | 167.9 |
| Single LightGBM, all features | 4.70 | 99.6 | 145.7 | −2.33 | 151.2 |
| Log-linear (same market terms) | 2.26 | 51.9 | 76.3 | +0.04 | 104.2 |
| **Two-stage (final)** | **1.60** | **38.3** | **55.8** | **+0.09** | **90.7** |

Two-stage model per fold: MAPE 1.54, 1.52, 1.55, 1.89, 1.56, 1.56 (primary Sep–Oct fold: **1.56% MAPE, MAE $36.2, R² 0.999**). It's stable across every season.

**Random K-fold is misleading.** On a shuffled 5-fold split, the single LightGBM looks *best* (0.99% MAPE vs 1.43% for the two-stage model). In the forward-in-time backtest it is three times worse (4.70% vs 1.60%), and it reaches 15.6% on the Apr–May fold. With a shuffled split, every test day has training rows from the same day, so the model memorizes daily price levels it will never have for Nov–Dec.

**New cities** (rows touching 8 held-out cities): 1.72% MAPE vs 1.55% for rows with seen cities, so they cost little extra. Location is learned from coordinates, not city names, so it carries over.

**What's left**: if each test day's market level were known exactly, MAPE would be about 1.40–1.50%. So almost all remaining error is load-level noise, and forecasting the daily level adds only about 0.1–0.4 points.

## 3. The model and why

The EDA showed the price is almost exactly multiplicative, with two different kinds of driver:

- **What makes one load cost more than another on the same day**: distance (non-linear), equipment, weight, origin/destination. These are stable over time and interact non-linearly, which suits **gradient-boosted trees**.
- **What moves every load's price on a given day**: `market_index` (weekly cycle plus seasonal level), a slow upward trend, and the quarter-end ramp. These must be **extrapolated**:
  - Nov–Dec dates are beyond the training window.
  - In some folds, `market_index` levels are outside the training range.
  - Trees can't extrapolate; they flatten at the last value they saw. That's why the single LightGBM fails forward in time.

So the model is:

```
log(rate) = log(distance) + load_effect(x) + market_effect(date, market_index)

load_effect   : LightGBM on log rate per mile
                features: log distance, equipment, weight (+ missing / at-cap flags), pickup and delivery lat/lon
market_effect : linear regression
                terms: market_index + trend (days) + quarter-end-month step + quarter-end ramp (days into month 3)
```

**Fitting (backfitting).** The trees are fit against per-date fixed effects. Each pass removes the date's mean residual before refitting, for 3 passes. This stops the trees from absorbing date-level movements. The linear market model is then fit on what remains.

**Final coefficients (trained on Jan–Oct)** — they're stable across all folds:
- `market_index`: 0.141 (folds: 0.116–0.145)
- trend: +0.0197%/day, about +0.6% per month
- quarter-end ramp: +0.135% per day through the third month of each quarter, about +4% by quarter end

The LightGBM settings are fixed (31 leaves, learning rate 0.03, 1,200 rounds, 0.8 row / 0.9 feature subsampling). They were picked from a small manual comparison during development: 31–127 leaves, squared-error vs Huber loss. All landed within 1.60–1.65% mean backtest MAPE, so the model isn't sensitive to these.

### Decisions backed by the backtest

Reproduce with `python src/ablation.py` (writes `reports/ablation_results.csv`). Each row is MAPE % on clean holdout rows, for the same 6 expanding-window folds; the column is the first month of the 2-month test window.

| Variant | Apr | May | Jun | Jul | Aug | Sep | Mean | Mean \|bias\| | Decision |
|---|---|---|---|---|---|---|---|---|---|
| **Final** | 1.54 | 1.52 | 1.55 | 1.89 | 1.56 | 1.56 | **1.60** | **0.63** | |
| Market: `market_index` only | 1.77 | 2.27 | 2.47 | 2.07 | 3.87 | 4.35 | 2.80 | 2.46 | Trend and ramp needed |
| Market: `market_index` + trend, no ramp | 2.96 | 1.66 | 2.01 | 4.39 | 1.86 | 1.90 | 2.46 | 1.51 | Ramp needed: without it, the trend absorbs quarter-end months |
| Market: + day-of-week dummies | 1.59 | 1.55 | 1.50 | 1.86 | 1.45 | 1.60 | 1.59 | 0.62 | No real gain (`market_index` already carries the weekly cycle); kept simpler |
| Market: + `quote_signal` | 1.61 | 1.50 | 1.54 | 1.91 | 1.54 | 1.55 | 1.61 | 0.68 | No gain |
| Trees: + `quote_signal` within-day deviation | 14.48 | 2.06 | 2.04 | 2.85 | 2.42 | 1.28 | 4.19 | 0.83 | **Rejected**: best on Sep–Oct, disastrous on Apr–May |
| Trees: + `market_index` within-day deviation | 1.60 | 1.55 | 1.62 | 2.19 | 1.67 | 1.57 | 1.70 | 0.75 | Rejected: the linear `market_index` term covers it |
| No backfitting (trees fit without removing date effects) | 2.03 | 2.01 | 1.75 | 1.87 | 1.65 | 1.65 | 1.83 | 0.94 | Backfitting matters |

**Why `quote_signal` is excluded.** Its link to price changes sign between periods:
- correlation +0.30 to +0.41 in the second half of Mar, Jun and Sep;
- about 0 from April to August;
- −0.14 in October.

Nov–Dec look most like August, when it carried no signal. A feature whose sign can't be predicted is a liability two months ahead.

## 4. Data-quality handling

| Issue | Handling |
|---|---|
| 677 corrupted rates (1.41%) | Dropped from training; reported separately in evaluation |
| Negative weights (sign flips) | `abs(weight)` |
| Weight clipped at 5,000 / 47,500 lb | Kept; `weight_at_cap` flag; trees handle the flat ends |
| Missing weight (0.6%) | Left as NaN for LightGBM, plus a `weight_missing` flag |
| Missing `market_index` (0.8%) | Filled with that day's mean `market_index` |
| 8 validation cities unseen in training | Location modeled through coordinates only, no city IDs |
| December chart rows have no coordinates, `market_index` or `quote_signal` | Coordinates from the city lookup; `market_index` = that December day's mean over the validation loads (a feature, not a label) |

## 5. Final predictions

- `validation_predictions.csv`: 12,000 rows. Predicted rate per mile follows on sensibly from October (Dry Van median: Oct actual $2.07, Nov predicted $2.05, Dec predicted $2.11).
- December chart, Lexington → Fort Wayne, Dry Van, 360 mi, 32,000 lb: **$827 on Dec 1 → $890 on Dec 31**. The shape comes from three parts (fig 08):
  - a weekly cycle peaking on Thursdays, from `market_index` (about ±1.5%);
  - the quarter-end ramp (+4.1% through December);
  - the trend (+0.6%).

  Training-era Dry Van loads on this lane were $758–$935.
