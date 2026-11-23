"""Run the independent Happy Hours Market product-snapshot cleaning pipeline.

Expected project layout (adjust INPUT_PATH if your raw filename differs):
    project/
      main_product_snapshots.py
      src/clean_product_snapshots.py
      data/raw/product_snapshots.parquet

The product-reference dataset is enriched separately by its own pipeline.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from src.clean_product_snapshots import clean_product_snapshots


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
INPUT_PATH = BASE_DIR / "data" / "raw" / "product_snapshots.parquet"
OUTPUT_DIR = BASE_DIR / "data" / "processed" / "product_snapshots"

# True only for original/raw snapshots; False for already corrected inputs.
ORIGINAL_FILES = True


# ============================================================
# SAVE RESULTS
# ============================================================

def save_results(
    cleaned_snapshots: pd.DataFrame,
    report: dict,
    anomalies: dict[str, pd.DataFrame],
    output_dir: Path,
) -> None:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    snapshots_path = output_dir / "product_snapshots_clean.parquet"
    cleaned_snapshots.to_parquet(snapshots_path, index=False)

    for name, anomaly_df in anomalies.items():
        if not isinstance(anomaly_df, pd.DataFrame):
            raise TypeError(f"Anomaly '{name}' must be a pandas DataFrame")
        anomaly_df.to_csv(output_dir / f"{name}.csv", index=False)

    report_path = output_dir / "cleaning_report.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)

    print(f"Cleaned snapshots: {snapshots_path}")
    print(f"Cleaning report:   {report_path}")
    print(f"Anomaly reports:   {output_dir}")


# ============================================================
# MAIN
# ============================================================

def main() -> None:
    if not INPUT_PATH.is_file():
        raise FileNotFoundError(
            f"Snapshot input not found: {INPUT_PATH}\n"
            "Update INPUT_PATH to your actual original snapshot file."
        )

    cleaned_snapshots, report, anomalies = clean_product_snapshots(
        source=INPUT_PATH,
        original_files=ORIGINAL_FILES,
    )

    save_results(
        cleaned_snapshots=cleaned_snapshots,
        report=report,
        anomalies=anomalies,
        output_dir=OUTPUT_DIR,
    )

    print("\nCLEANING SUMMARY")
    print(f"Rows before: {report['rows_before']:,}")
    print(f"Rows after:  {report['rows_after']:,}")
    print(f"Primary key unique: {report['primary_key_unique']}")
    print(
        "Leading-zero EAN rows corrected: "
        f"{report['leading_zero_ean_corrections']['rows_corrected']:,}"
    )
    print(
        "Hardcoded EAN rows corrected: "
        f"{report['known_snapshot_ean_corrections']['rows_corrected']:,}"
    )
    print(f"Remaining SKU/EAN conflicts: {report['skus_with_multiple_eans']:,} SKUs")


if __name__ == "__main__":
    main()
