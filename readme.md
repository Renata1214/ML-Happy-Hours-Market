# Goals of the Project. 

## 3 Main Objectives 

1. Data quality and availability
USER STORY Asthe operationslead, I want reliable figures and a record of what was on the app each day,so that I
can tell weak demand apart from missing stock.
● Find, fix, and report the data quality issues, and their effect on your results.
● Reconstruct daily product availability
● Reusable tool that can be run every day to clean data
Output: data quality report and daily availability table.

3. Daily sales forecasting
USER STORY Asthe operationslead, I want to know what willsell tomorrow at each pickup point,so that products
go where customers are.
● At which granularity are daily sales forecasts most reliable, and how do pickup points differ?
● Do hot deals lift sales, and would a better ordering than alphabetical help?
Output: daily forecast by pickup point and a conversion view.

5. Customer behaviour
USER STORY Asthe marketing lead, I want to know what each customer will buy next and who is about to leave,
so that I can send the right offer.
● Predict next purchases from purchase history.
● Segment customers, identify churn signals, and check how customers and revenue split across
pickup points.
Output: customer segments, next-purchase model and churn indicators.
---

# WS3 — Daily Sales Forecasting

## Objective

Forecast daily ordered quantities for each pickup point to support operational planning.

The forecasting target is the daily sum of `quantity_ordered` per pickup point. It represents ordered quantities, not confirmed fulfilled sales.

## Data preparation

The all-25 forecasting panel is built from the pickup-point orders dataset.

- Pickup points: all 25 pickup points are retained.
- Calendar period: 1 June 2024 to 31 August 2026.
- Calendar length: 822 days per pickup point.
- Missing dates are retained as unobserved rather than automatically treated as zero demand.
- The panel builder is `exploration/build_daily_forecasting_panel_all25.py`.

The source dataset is expected to be named:

`pickup_point_orders_2024-01_2026-08.parquet`

The generated panel is:

`hhm_daily_forecasting_panel_all25.parquet`

## Forecasting methodology

The evaluation uses rolling one-day-ahead forecasts across eight consecutive three-month evaluation windows, from September 2024 to August 2026.

For each evaluation month:
- Models are trained using information available before the forecast date.
- Predictions are evaluated against observed daily quantities.
- Actual observations earlier in the same month can be used as history for later one-day-ahead forecasts.

The principal baseline is `weekday_average_4`, which averages the four most recent observed quantities for the same weekday.

The global machine-learning model uses a `HistGradientBoostingRegressor` with squared-error loss, categorical pickup-point and weekday features, and historical demand and calendar features.

Performance is evaluated using MAE, RMSE, WAPE and bias. WAPE is calculated as total absolute error divided by total actual quantity.


## Experimental results

The original global ML model achieved the best overall offline result among the tested approaches.

| Model | MAE | RMSE | WAPE |
|---|---:|---:|---:|
| Seasonal naive | 200.00 | 267.82 | 21.85% |
| Weekday average (4) | 170.25 | 231.26 | 18.60% |
| Original global ML | 163.28 | 219.39 | 17.84% |
| Normalized global ML | 170.82 | 230.21 | 18.66% |
| Poisson global ML | 163.48 | 220.75 | 17.86% |
| Hybrid | 164.84 | 222.35 | 18.01% |

The original global ML model reduced WAPE by 0.76 percentage points compared with the weekday-average baseline.

Forecast coverage was 13,668 out of 13,853 eligible observed evaluation targets (98.66%). Coverage is below 100% because some targets lack sufficient preceding history.

### Additional experiments

- **Normalized global ML:** scaled demand by historical pickup-point demand levels. It did not improve overall performance over the original global ML model or the weekday-average baseline.
- **Poisson global ML:** tested Poisson loss instead of squared-error loss. It performed almost identically to, but slightly worse than, the original global ML model.
- **Hybrid:** selected between the baseline and global ML using historical performance for each pickup point. It improved over the baseline but did not outperform the original global ML model overall.

These alternatives remain experimental comparisons rather than selected models.

## Model decision and deployment status

The original global ML model is the strongest offline experimental model based on overall WAPE.

The current Dagster forecasting implementation uses `weekday_average_4` as its operational baseline. The global ML model has not yet been integrated into Dagster.

The offline evaluation ends on 31 August 2026. Therefore, its final one-day-ahead forecast date is 1 September 2026; these results should not be interpreted as live forecasts.

## Reproducing the experiments

### 1. Install dependencies

From the project root:

    python -m pip install -r requirements-ws3.txt

The tested environment used Python 3.13.13 with the package versions specified in `requirements-ws3.txt`.

### 2. Configure the data directory

By default, scripts look for input data in `~/Downloads`.

To use another directory, set:

    export HHM_DATA_DIR="$HOME/data/hhm"

The source dataset should be available in that directory.

Alternatively, configure the source and panel paths explicitly:

    export HHM_SOURCE_FILE="/path/to/pickup_point_orders_2024-01_2026-08.parquet"
    export HHM_PANEL_FILE="/path/to/hhm_daily_forecasting_panel_all25.parquet"

### 3. Build the all-25 panel

    python exploration/build_daily_forecasting_panel_all25.py

### 4. Run a pilot evaluation

The `--pilot` option runs a smaller evaluation to check the setup before a full run.

    python exploration/evaluate_global_ml_forecasting_all25.py --pilot

### 5. Run the full evaluations

    python exploration/evaluate_global_ml_forecasting_all25.py
    python exploration/evaluate_normalized_global_ml_all25.py
    python exploration/evaluate_poisson_global_ml_all25.py
    python exploration/evaluate_hybrid_forecasting_all25.py

### 6. Inspect results

Each evaluator writes its outputs to a separate directory:

- `results/all25/`
- `results/normalized_all25/`
- `results/poisson_all25/`
- `results/hybrid_all25/`

The `*_overall.csv` files contain aggregate model metrics, while the `*_metrics_by_window.csv` files contain performance by evaluation window.

The hybrid experiment also writes `hybrid_selection_log.csv`, documenting which model was selected for each pickup-point/month decision.

## Important limitations

- The target measures ordered quantities, not confirmed fulfilled sales.
- Missing dates are not automatically equivalent to zero demand.
- Forecast coverage is below 100% because some observations lack sufficient history.
- Results are historical offline evaluations, not evidence of current live forecasting performance.
- Dagster uses a hybrid forecasting approach: global ML predictions where available, with the weekday-average baseline as a fallback.

## Detailed experimental results

For the complete WS3 evaluation, including baseline tuning, fair ML comparisons, STL and SARIMA benchmarks, prediction uncertainty, category forecasting, limitations, and reproducibility details, see:

- [WS3 Experimental Results](WS3_RESULTS.md)
