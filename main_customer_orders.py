
import json
from pathlib import Path
from src.clean_customer_orders import clean_customer_orders

# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

CUSTOMER_ORDERS_PATH = (
    BASE_DIR / "data/raw/customer_orders.parquet"
)

OUTPUT_DIR = (
    BASE_DIR / "data/processed/customer_orders"
)


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    cleaned_df,
    report,
    anomalies,
    output_dir,
):
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    # Save cleaned customer orders.
    cleaned_df.to_parquet(
        output_dir / "customer_orders_clean.parquet",
        index=False,
    )

    # Save anomaly reports.
    for name, anomaly_df in anomalies.items():

        anomaly_df.to_csv(
            output_dir / f"{name}.csv",
            index=False,
        )

    # Save validation summary.
    with open(
        output_dir / "cleaning_report.json",
        "w",
        encoding="utf-8",
    ) as file:

        json.dump(
            report,
            file,
            indent=4,
            ensure_ascii=False,
        )


# ============================================================
# RUN PIPELINE
# ============================================================

def main():

    cleaned, report, anomalies = clean_customer_orders(
        source=CUSTOMER_ORDERS_PATH,
    )

    # Ensure expected key uniqueness.
    if not report["primary_key_unique"]:
        raise ValueError(
            "Customer orders contain duplicate "
            "(order_id, sku) keys. Inspect key_conflicts."
        )

    save_results(
        cleaned_df=cleaned,
        report=report,
        anomalies=anomalies,
        output_dir=OUTPUT_DIR,
    )

    print("\nCUSTOMER ORDERS CLEANING COMPLETE")
    print("=" * 45)

    print(f"Rows before: {report['rows_before']:,}")
    print(f"Rows after: {report['rows_after']:,}")

    print(
        f"Exact duplicates removed: "
        f"{report['exact_duplicates_removed']:,}"
    )

    print(
        f"Primary key unique: "
        f"{report['primary_key_unique']}"
    )

    print(
        f"Unique orders before: "
        f"{report['unique_orders_before']:,}"
    )

    print(
        f"Unique orders after: "
        f"{report['unique_orders_after']:,}"
    )

    print(
        f"Repeated order/EAN rows: "
        f"{report['repeated_order_ean_rows']:,}"
    )

    print(
        f"Numerical anomaly rows: "
        f"{report['numerical_anomaly_rows']:,}"
    )

    print("\nTOTALS")
    print(
        f"Quantity before: "
        f"{report['totals_before']['quantity_total']:,}"
    )

    print(
        f"Quantity after: "
        f"{report['totals_after']['quantity_total']:,}"
    )

    print(
        f"Sales value before: "
        f"{report['totals_before']['sales_value']:.2f}"
    )

    print(
        f"Sales value after: "
        f"{report['totals_after']['sales_value']:.2f}"
    )

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()




# A point about sales value
# The code calculates:
# \[
# \text{Sales value}=\sum_i \text{quantity}_i\times\text{new\_price}_i
# \]

# This assumes new_price is a per-unit selling price. If HHM's price values have a different business definition, we should adjust that calculation.

# The numeric conversion preserves floating-point values without intentional rounding, but binary float64 is not exact decimal arithmetic. For financial reconciliation requiring exact cents, use a decimal or integer-cent representation.
# 7. What we're deliberately leaving for feature engineering
# Your cleaned table will be ready for joining to the other datasets, but no model-specific categorical encoding is applied.

# For example, your timeslot values will still look like:

# 17:00 - 20:00

# Later, we could extract timeslot_start_hour, timeslot_end_hour, and timeslot_duration_minutes, which may provide more useful information to a forecasting model than an arbitrary category number.

# Similarly, regular_client remains a boolean, and date_type remains DDM or DLC. Their final representation will depend on the model we select.

# One important modeling caution: features derived from order records—such as regular_client or attributes of an individual purchase—may not be known when predicting tomorrow's sales. We should only use information available at prediction time, or derive appropriate historical aggregates.

# The resulting customer_orders_clean.parquet is therefore a standardized, validated transactional dataset. Its original observational grain and identifiers are preserved, which is exactly what we want before building the shared product mappings and ML feature tables.
