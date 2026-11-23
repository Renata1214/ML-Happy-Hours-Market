# WS3 — Daily Sales Forecasting: Experimental Results

## 1. Objective

Forecast daily ordered quantities at each pickup point to support operational planning and reduce the risk of unsold food.

The target is the sum of `quantity_ordered` per pickup point and day. It measures ordered quantities, not confirmed fulfilled sales or food waste directly.

## 2. Dataset and evaluation design

- Pickup points: 25 retained in the forecasting panel.
- Calendar period: 1 June 2024 – 31 August 2026.
- Calendar length: 822 days per pickup point.
- Observed evaluation targets: 13,853 across 24 pickup points.
- Evaluation: eight rolling three-month windows from September 2024 to August 2026.
- Forecast horizon: one day ahead.

Missing pickup-point dates are retained as unobserved. They are not automatically interpreted as zero demand.

Models use information available before each forecast. Earlier observed quantities can be used for subsequent one-day-ahead predictions.

Performance is measured using:
- WAPE: total absolute error divided by total actual quantity.
- MAE: mean absolute error.
- RMSE: root mean squared error.
- Bias: mean signed forecast error.

## 3. Main model comparison

The initial all-25 evaluation produced the following results:

| Model | WAPE |
|---|---:|
| Seasonal naive | 21.85% |
| Weekday average (4) | 18.60% |
| Original global ML | 17.84% |
| Normalized global ML | 18.66% |
| Poisson global ML | 17.86% |
| Hybrid | 18.01% |

The original global ML model achieved the lowest WAPE in this evaluation, improving on the weekday-average baseline by 0.76 percentage points.

Its forecast coverage was 13,668 of 13,853 observed evaluation targets (98.66%).

These figures belong to the original model comparison and should not be mixed with the stricter common-row comparison below.

## 4. Baseline tuning and fair ML comparison

We tested weekday averages based on the 2, 4, 6, 8 and 12 most recent observed quantities for the same weekday.

On the common evaluation sample of 11,299 rows:

| Method | WAPE |
|---|---:|
| Weekday average (2) | 19.16% |
| Weekday average (4) | 18.25% |
| Weekday average (6) | 18.10% |
| Weekday average (8) | 18.06% |
| Weekday average (12) | 18.28% |
| Global ML | **17.41%** |

The global ML model improved WAPE by:
- 0.84 percentage points against weekday average (4).
- 0.64 percentage points against weekday average (8).

The global ML model performed better than weekday average (4) on 22 of 23 pickup points represented in this common sample.

The eight-observation baseline had the best WAPE among the tested baseline windows, but its improvement over the four-observation baseline was small. Its selection was also based on the same rolling evaluation windows, so the result should be treated as experimental rather than definitive.

The common-row comparison is stricter than the original ML evaluation because all methods are assessed on exactly the same targets.

## 5. Additional time-series benchmarks

### STL decomposition

STL decomposition combined with an ARIMA forecasting step was tested using uninterrupted observed history.

Results on the 11,960 common forecast rows:

| Method | WAPE |
|---|---:|
| Seasonal naive | 21.64% |
| Weekday average (4) | 18.38% |
| STL-ARIMA | 23.07% |

STL-ARIMA did not outperform the weekday-average baseline in any of the eight evaluation windows. It was not selected.

### SARIMA

SARIMA was evaluated using the same uninterrupted-history approach.

| Method | WAPE |
|---|---:|
| Seasonal naive | 21.64% |
| Weekday average (4) | 18.38% |
| SARIMA | 19.55% |

SARIMA improved on seasonal naive but did not outperform weekday average (4) in any of the eight evaluation windows. It was not selected.

These experiments demonstrate that more complex statistical time-series models do not necessarily improve forecasting performance on this dataset.

## 6. Prediction uncertainty

Prediction intervals were estimated from historical absolute forecast errors pooled across pickup points. We evaluated a fixed historical margin and rolling calibration windows on a holdout period from 1 March to 31 August 2026.

The calibration data contained 7,934 earlier forecast observations. The holdout contained 3,365 observations.

| Calibration method | Empirical coverage | Average margin |
|---|---:|---:|
| Fixed historical margin | 86.27% | ±355.7 items |
| Rolling 3 months | 90.25% | ±398.0 items |
| Rolling 6 months | **90.40%** | ±397.1 items |
| Rolling 12 months | 88.59% | ±376.1 items |

Six-month rolling calibration achieved the highest overall empirical coverage.

However, coverage varied across months and pickup points:
- June 2026 coverage was 87.55% for the six-month method.
- Only 12 of the 20 pickup points with holdout predictions achieved at least 90% coverage using this shared pooled margin. Point-level coverage varied, so this does not establish 90% coverage for every pickup point.

Therefore, these intervals are experimental uncertainty estimates, not guaranteed 90% prediction intervals. They have not been integrated into the production pipeline.

## 7. Category forecasting extension

We also evaluated daily forecasts for eight core product categories across the 25 pickup points.

The category panel contains 135,480 rows, representing 16,935 observed pickup-point days multiplied by eight categories. Missing categories on an otherwise observed pickup-point day are treated as zero; missing pickup-point days remain unobserved.

On the common evaluation sample of 109,352 rows:

| Method | WAPE |
|---|---:|
| Seasonal naive | 34.35% |
| Weekday average (4) | 28.85% |
| Category global ML | **27.01%** |
| Adaptive hybrid | 27.17% |

The category global ML model outperformed the weekday-average baseline across all eight evaluation windows and all eight categories. It achieved lower WAPE on 173 of the 192 pickup-point/category series.

The fallback hybrid achieved 27.01% WAPE on the common rows and produced 24 additional baseline fallback forecasts where ML predictions were unavailable.

These results support category-level forecasting as a useful extension. However, the forecasts cover only eight core categories, and independently modelled category and pickup-point totals do not necessarily reconcile.

## 8. Model selection and operational approach

The global ML model is the strongest candidate in the fair common-row comparison.

The Dagster implementation uses a hybrid forecasting approach:
- Global ML predictions are used where available.
- The weekday-average baseline provides a fallback when ML predictions are unavailable or history is insufficient.

This preserves broad forecast coverage while using the stronger ML model where possible.

The historical Dagster forecast snapshot evaluated for 1 September 2026 uses information through 31 August 2026. It is not a live forecast for the current date.

## 9. Limitations

- Ordered quantity is not the same as fulfilled or collected sales.
- The target is not a direct measurement of food waste.
- Missing pickup-point dates must not be interpreted as zero demand.
- Pickup-point activity and data coverage vary over time.
- Some pickup points have limited or intermittent observed history.
- The fair ML-versus-baseline comparison covers 11,299 rows, not every observed evaluation target.
- Prediction interval coverage is uneven across months and pickup points.
- The category forecasting extension covers eight core product categories, not every product category.
- Independently modelled pickup-point and category totals do not necessarily reconcile.
- The source data ends on 31 August 2026, so forecasts based on this dataset are historical.

## 10. Reproducibility

The main experiment scripts are in `exploration/`.

Important scripts include:
- `build_daily_forecasting_panel_all25.py`
- `evaluate_global_ml_forecasting_all25.py`
- `tune_weekday_baseline_all25.py`
- `compare_tuned_baselines_vs_ml_all25.py`
- `evaluate_stl_forecasting_all25.py`
- `evaluate_sarima_forecasting_all25.py`
- `evaluate_prediction_intervals_all25.py`

The corresponding experiment outputs are stored under `results/`.

Install dependencies with:

    python -m pip install -r requirements-ws3.txt

Build the all-25 panel with:

    python exploration/build_daily_forecasting_panel_all25.py

Run the prediction interval experiment with:

    python exploration/evaluate_prediction_intervals_all25.py

The input dataset and panel paths can be configured using the environment variables documented in `readme.md`.
