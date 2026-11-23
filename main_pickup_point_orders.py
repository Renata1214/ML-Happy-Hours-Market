
import json
from pathlib import Path
from src.clean_pickup_point_orders import (
    clean_pickup_point_orders,
)


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PICKUP_ORDERS_PATH = (
    BASE_DIR / "data/raw/pickup_point_orders.parquet"
)

SNAPSHOT_PATH = (
    BASE_DIR
    / "data/processed/product_snapshots"
    / "product_snapshots_clean.parquet"
)

OUTPUT_DIR = (
    BASE_DIR / "data/processed/pickup_point_orders"
)

# Enable corrections only for the original source files.
ORIGINAL_FILES = True




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

    # Save cleaned dataset.
    cleaned_df.to_parquet(
        output_dir / "pickup_point_orders_clean.parquet",
        index=False,
    )

    # Save anomaly reports.
    for name, anomaly_df in anomalies.items():
        anomaly_df.to_csv(
            output_dir / f"{name}.csv",
            index=False,
        )

    # Save cleaning report.
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

    cleaned, report, anomalies = clean_pickup_point_orders(
        source=PICKUP_ORDERS_PATH,
        snapshot_source=(
            SNAPSHOT_PATH if ORIGINAL_FILES else None
        ),
        original_files=ORIGINAL_FILES,
    )

    save_results(
        cleaned_df=cleaned,
        report=report,
        anomalies=anomalies,
        output_dir=OUTPUT_DIR,
    )

    print("\nPICKUP POINT ORDERS CLEANING COMPLETE")
    print("=" * 48)

    print(f"Rows before: {report['rows_before']:,}")
    print(f"Rows after: {report['rows_after']:,}")

    print(
        "Exact duplicates removed:",
        report["exact_duplicates_removed"],
    )

    print(
        "Primary key unique:",
        report["primary_key_unique"],
    )

    print(
        "Duplicate-key group rows:",
        report["key_conflict_rows"],
    )

    print(
        "EAN rows corrected:",
        report["ean_rows_corrected"],
    )

    print(
        "SKUs with multiple EANs:",
        report["skus_with_multiple_eans"],
    )

    print(
        "Numerical anomaly rows:",
        report["numerical_anomaly_rows"],
    )

    print("\nQUANTITY VALIDATION")

    print(
        "Quantity difference:",
        report["quantity_difference"],
    )

    print(
        "Number of orders difference:",
        report["n_orders_difference"],
    )

    print(f"\nResults saved to: {OUTPUT_DIR}")

    # Do not silently accept an invalid primary key.
    if not report["primary_key_unique"]:
        print(
            "\nWARNING: Duplicate primary keys remain."
            "\nReview key_conflicts.csv before using "
            "this dataset for ML joins."
        )


if __name__ == "__main__":
    main()
