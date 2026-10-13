# EDA Findings

Reproduce with `python src/eda.py` (prints every number below and writes `reports/figures/`).

## Data at a glance

| | Train (`data/train_test.csv`) | Validation (`data/validation.csv`) |
|---|---|---|
| Rows | 48,000 | 12,000 |
| Dates | 2025-01-01 → 2025-10-31 | 2025-11-01 → 2025-12-31 |
| Cities | 64 | 72 (8 unseen in train) |
| Loads/day | ~157 | ~197 |
| Target | `posted_rate` | — |

Validation is the two months *after* training ends, so this is a forecasting problem, not random interpolation.

## Data-quality issues

| Issue | Train | Validation | Evidence | Proposed fix |
|---|---|---|---|---|
| **Corrupted target labels** | 677 rows (1.41%) | n/a | Clean rows sit within 0.90x–1.11x of a simple baseline. The corrupted rows form two separate clusters at **2.2x–5.4x** (340 rows) and **0.17x–0.45x** (337 rows), with an empty gap in between (fig 01). Spread evenly across months, equipment and lanes. | Drop from training. They are not real market moves, and at 3–5x they would dominate any squared-error loss. |
| **Negative weight** | 292 | 145 | The absolute values have the same distribution as positive weights (median 31.8k vs 31.5k), so these are sign flips. | `abs(weight)` |
| **Weight clipped** | 1,204 at 47,500; 32 at 5,000 | 301 / 7 | Spikes at both ends of an otherwise smooth distribution (fig 02). | Keep, add an `at_cap` flag; trees handle the flat ends. |
| **Missing weight** | 300 | 165 | Random across dates. | Impute the median and add a missing flag (or let LightGBM use NaN). |
| **Missing `market_index`** | 374 | 249 | Random across dates. | Impute that day's mean `market_index` (the index is mostly a date-level signal; see below). |
| **Distance floor** | 48 rows at exactly 70 mi | 21 | Some lanes are physically shorter, but the field is floored at 70. | Leave as is (small). |
| **Coordinates aren't real geography** | — | — | e.g. Boston at lat 39.3, Phoenix at lat 28.8. But each city has exactly one fixed pair, and stated distance ≈ 1.18x the straight-line distance between them (corr 0.9995). | Treat them as consistent synthetic coordinates. They're still useful for location features. |

What is **not** broken: no duplicate rows or IDs. Distance varies only ±3–5% per load around its lane median, with no gross errors, and stated distance predicts rate better than straight-line distance.

## What drives the rate

After cleaning, a simple log-linear model on distance, equipment and the daily market index already leaves only about **3.3%** residual noise. After also accounting for date and origin/destination, about **2.2%** is left. The signal is strong and mostly multiplicative.

1. **Distance** is dominant: `rate ∝ distance^0.87`. Rate per mile falls with distance (median $2.15/mi), with mild curvature at both ends.
2. **Equipment**: Reefer +12.9% and Flatbed +8.2% vs Dry Van (median $/mi: Dry Van 2.05, Flatbed 2.22, Reefer 2.31). Flatbed has the strongest quarter-end ramp.
3. **`market_index` is a daily market signal**:
   - Most of its variance is between days (std 0.167) rather than within a day (0.025).
   - It has a **weekly cycle**: peak Thursday, trough Sunday, ±~0.1.
   - Its **seasonal level** rises Jan→May, plateaus May–Jul, steps down late July and stays low Aug–Dec (~0.93 in Nov/Dec).
   - It fully explains the weekday pattern in rates: after controlling for it, the weekday residual is ±0.2%.
   - The within-day deviation also carries signal (corr +0.14 with the residual).
4. **Quarter-end ramp (not explained by `market_index`)**: after removing the market effect, rates climb steadily through each quarter, accelerate in its third month, and **reset on the first day of the next quarter** (Apr 1, Jul 1, Oct 1) (fig 03, bottom). On top of that there's a slow upward drift of about +1.5% per quarter.

   | Residual % | Month 1 | Month 2 | Month 3 |
   |---|---|---|---|
   | Q1 | −3.0 | −2.6 | −0.2 |
   | Q2 | −1.4 | −0.5 | +1.9 |
   | Q3 | +0.3 | +0.2 | +2.9 |
   | Q4 | +2.3 | — | — |

   Within the third month, the residual rises about 3.5–4% from week 1 to week 5.
5. **Location**: origin and destination effects span about ±3%. They correlate strongly with latitude (−0.86): Texas and the south are pricier (Dallas, Houston, Oklahoma City +2.5–3%), the north-east and Great Lakes are cheaper (Syracuse, Buffalo, Detroit −3%) (fig 04). There's a small extra lane-specific effect (std 1.4%).
6. **Weight**: about +3% per 10k lb, non-linear (light loads under 10k lb cost more than linear predicts).
7. **`quote_signal`** is weak overall: corr +0.06 with the residual after other effects. Its daily mean moves in step-like regimes unrelated to rate level. A closer look during modeling found its link to price **changes sign between regimes**:
   - +0.30 to +0.41 in the second half of quarter-end months;
   - about 0 in Apr–Aug;
   - −0.14 in October.

   It is excluded from the final model (see [modeling.md](modeling.md)).

## Implications for modeling

- **Validation split must be time-based.** The final predictions are a 2-month forward forecast. Hold out the latest data (e.g. Sep–Oct) or use expanding-window folds. A random K-fold would leak the date-level market level and quarter-ramp state.
- **Predict in log space / as rate-per-mile.** The effects are multiplicative and the noise is proportional.
- **Remove corrupted labels before training,** and use a robust loss as a second safeguard.
- **Unseen cities (12.1% of validation rows):** city ID or target encoding alone won't cover Allentown, Charlotte, Chicago, Jackson, Knoxville, Laredo, Norfolk or San Diego. Latitude/longitude features generalize because the location effect is mostly a north–south gradient. Laredo (lat 25.5) is slightly outside the training range.
- **Calendar features** need to carry the quarter ramp: `day_of_quarter`, `month_in_quarter`, `days_to_quarter_end`. December is the third month of Q4 (a ramp month), and Mar/Jun/Sep give three training examples of that phase. The slow year-over-year drift is a trend that tree models cannot extrapolate; it's worth testing a trend term or a log-linear component.

## The December chart

- `data/december_chart_inputs.csv` has only pickup, delivery, distance, equipment, weight and date. It has **no coordinates, `market_index` or `quote_signal`**.
  - Coordinates come from the city lookup: Lexington (36.99, −85.00), Fort Wayne (41.32, −85.36).
  - **December `market_index` is available**: `validation.csv` has about 200 December loads per day with `market_index`. Its daily mean continues the weekly cycle, peaking around 1.02–1.05 on Thursdays and bottoming around 0.83–0.86 on Sundays/Mondays. Use that daily mean as the input for each chart date (and the daily mean of `quote_signal` the same way). These are features, not labels, so there's no leakage.
- **Expected shape**:
  - a weekly sawtooth from the market-index cycle (about ±1.3% in rate);
  - riding on the third-month-of-quarter ramp (about +3–4% from Dec 1 to Dec 31);
  - starting around the training-era Dry Van level for this lane ($760–$935 in history, fig 05).

  A flat line would mean the model ignores the date.
