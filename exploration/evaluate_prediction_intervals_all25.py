from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("results/baseline_vs_ml_all25/comparison_predictions.csv")
OUTPUT_DIR = Path("results/uncertainty_all25")
TEST_START = pd.Timestamp("2026-03-01")
MIN_HISTORY_ROWS = 100
WINDOWS = [3, 6, 12]

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(INPUT)
df["order_date"] = pd.to_datetime(df["order_date"])
df["abs_error"] = (df["actual"] - df["global_ml"]).abs()

calibration = df[df["order_date"] < TEST_START]
test = df[df["order_date"] >= TEST_START].copy()

fixed_margin = calibration["abs_error"].quantile(
    0.90, interpolation="higher"
)

records = []

for row in test.itertuples(index=False):
    record = {
        "order_date": row.order_date,
        "pickup_point": row.pickup_point,
        "actual": row.actual,
        "forecast": row.global_ml,
    }

    margins = {"fixed": fixed_margin}

    for months in WINDOWS:
        history = df[
            (df["order_date"] < row.order_date)
            & (
                df["order_date"]
                >= row.order_date - pd.DateOffset(months=months)
            )
        ]

        margins[f"rolling_{months}m"] = (
            history["abs_error"].quantile(0.90, interpolation="higher")
            if len(history) >= MIN_HISTORY_ROWS
            else np.nan
        )

    for method, margin in margins.items():
        record[f"{method}_margin"] = margin

        if pd.isna(margin):
            record[f"{method}_lower"] = np.nan
            record[f"{method}_upper"] = np.nan
            record[f"{method}_covered"] = np.nan
        else:
            lower = max(0, row.global_ml - margin)
            upper = row.global_ml + margin

            record[f"{method}_lower"] = lower
            record[f"{method}_upper"] = upper
            record[f"{method}_covered"] = (
                lower <= row.actual <= upper
            )

    records.append(record)

predictions = pd.DataFrame(records)
predictions.to_csv(
    OUTPUT_DIR / "prediction_interval_predictions.csv",
    index=False,
)


def summarize(group):
    result = {"rows": len(group)}

    for method in ["fixed", "rolling_3m", "rolling_6m", "rolling_12m"]:
        covered = group[f"{method}_covered"].dropna()
        margins = group[f"{method}_margin"].dropna()

        result[f"{method}_coverage_pct"] = (
            covered.mean() * 100 if len(covered) else np.nan
        )
        result[f"{method}_avg_margin"] = (
            margins.mean() if len(margins) else np.nan
        )
        result[f"{method}_avg_width"] = (
            2 * margins.mean() if len(margins) else np.nan
        )

    return pd.Series(result)


overall = summarize(predictions)
overall.to_frame("value").to_csv(
    OUTPUT_DIR / "interval_metrics_overall.csv"
)

predictions["month"] = predictions["order_date"].dt.to_period("M").astype(str)
monthly = (
    predictions.groupby("month", sort=True)
    .apply(summarize, include_groups=False)
    .reset_index()
)
monthly.to_csv(
    OUTPUT_DIR / "interval_metrics_by_month.csv",
    index=False,
)

by_point = (
    predictions.groupby("pickup_point", sort=True)
    .apply(summarize, include_groups=False)
    .reset_index()
)
by_point.to_csv(
    OUTPUT_DIR / "interval_metrics_by_pickup_point.csv",
    index=False,
)

print("PREDICTION INTERVAL EXPERIMENT COMPLETE")
print("Calibration rows:", len(calibration))
print("Holdout rows:", len(predictions))
print(
    "Holdout dates:",
    predictions["order_date"].min().date(),
    "to",
    predictions["order_date"].max().date(),
)
print("Fixed 90% margin:", round(fixed_margin, 1), "items")

print("\nOVERALL RESULTS")
print(overall.round(2).to_string())

print("\nMONTHLY COVERAGE (%)")
print(
    monthly[
        [
            "month",
            "fixed_coverage_pct",
            "rolling_3m_coverage_pct",
            "rolling_6m_coverage_pct",
            "rolling_12m_coverage_pct",
        ]
    ].round(2).to_string(index=False)
)

print("\nPICKUP-POINT COVERAGE (%)")
print(
    by_point[
        [
            "pickup_point",
            "rows",
            "fixed_coverage_pct",
            "rolling_3m_coverage_pct",
            "rolling_6m_coverage_pct",
            "rolling_12m_coverage_pct",
        ]
    ].round(2).to_string(index=False)
)

print("\nSaved files in:", OUTPUT_DIR)
