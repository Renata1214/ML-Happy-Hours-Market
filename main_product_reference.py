"""Run HHM product-reference cleaning and enrichment with cleaned snapshots.

Run main_product_snapshots.py first. This runner writes the single authoritative
reference dataset to data/processed/product_reference/product_reference_clean.parquet.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.clean_product_reference import clean_product_reference


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

# Adjust this path if your raw reference filename differs.
REFERENCE_PATH = BASE_DIR / "data" / "raw" / "product_reference.parquet"

# Must be the output of main_product_snapshots.py, not the raw snapshots.
SNAPSHOT_PATH = (
    BASE_DIR / "data" / "processed" / "product_snapshots"
    / "product_snapshots_clean.parquet"
)

OUTPUT_DIR = BASE_DIR / "data" / "processed" / "product_reference"


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    cleaned_reference: pd.DataFrame,
    report: dict,
    anomalies: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    """Save authoritative reference, report, and diagnostic CSVs."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    reference_path = output_dir / "product_reference_clean.parquet"
    cleaned_reference.to_parquet(reference_path, index=False)

    for name, anomaly_df in anomalies.items():
        if not isinstance(anomaly_df, pd.DataFrame):
            raise TypeError(f"Anomaly '{name}' must be a pandas DataFrame")
        anomaly_df.to_csv(output_dir / f"{name}.csv", index=False)

    report_path = output_dir / "cleaning_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"Cleaned reference: {reference_path}")
    print(f"Cleaning report:   {report_path}")
    print(f"Anomaly reports:   {output_dir}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not REFERENCE_PATH.is_file():
        raise FileNotFoundError(
            f"Original reference not found: {REFERENCE_PATH}\n"
            "Update REFERENCE_PATH to the actual raw reference file."
        )

    if not SNAPSHOT_PATH.is_file():
        raise FileNotFoundError(
            f"Cleaned snapshots not found: {SNAPSHOT_PATH}\n"
            "Run main_product_snapshots.py before this script."
        )

    cleaned_reference, report, anomalies = clean_product_reference(
        source=REFERENCE_PATH,
        snapshot_source=SNAPSHOT_PATH,
    )

    
    cleaned_reference,corrected_snapshots,report,anomalies, = clean_product_reference(
        source=REFERENCE_PATH,
        snapshot_source=SNAPSHOT_PATH,
    )

    # This is a master table: downstream joins require one row per EAN.
    if not report["primary_key_unique"]:
        raise ValueError(
            "Product reference is not unique by ean_code. "
            "Review duplicate_ean_conflicts before exporting."
        )

    if report["category_reconciliation"] is None:
        raise ValueError("Reference enrichment did not run as expected.")

    save_results(
        cleaned_reference=cleaned_reference,
        report=report,
        anomalies=anomalies,
        output_dir=OUTPUT_DIR,
    )

    # Save snapshot categories aligned to product reference.

    if corrected_snapshots is not None:

        # Ensure snapshot primary key remains valid.
        if corrected_snapshots.duplicated(
            ["snapshot_date", "sku"]
        ).any():
            raise ValueError(
                "Corrected snapshots contain duplicate primary keys."
            )

        # Save back to the cleaned snapshots file.
        corrected_snapshots.to_parquet(
            SNAPSHOT_PATH,
            index=False,
        )

        print(
            "Snapshot categories aligned:",
            report["snapshot_category_alignment"]["rows_corrected"]
        )

    reconciliation = report["category_reconciliation"]
    print("\nCLEANING SUMMARY")
    print(f"Rows before:                {report['rows_before']:,}")
    print(f"Rows after:                 {report['rows_after']:,}")
    print(f"Unique EAN primary key:     {report['primary_key_unique']}")
    print(f"Exact duplicates removed:   {report['exact_duplicates_removed']:,}")
    print(f"Compatible EANs merged:     {report['duplicate_ean_groups_merged']:,}")
    print(f"Missing EAN rows removed:   {report['missing_ean_rows_removed']:,}")
    print(f"New snapshot EANs added:    {reconciliation['new_eans_added']:,}")
    print(f"Categories imputed:         {reconciliation['existing_categories_imputed']:,}")
    print(f"Category mismatches:        {reconciliation['category_mismatches']:,}")
    print(f"Ambiguous snapshot EANs:    {reconciliation['ambiguous_snapshot_eans']:,}")
    print(f"Missing categories after:   {reconciliation['missing_categories_after']:,}")


if __name__ == "__main__":
    main()
