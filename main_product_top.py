
import json
from pathlib import Path
from src.clean_product_top import clean_product_top


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

PRODUCT_TOP_PATH = (
    BASE_DIR / "data/raw/product_top.parquet"
)

SNAPSHOT_PATH = (
    BASE_DIR
    / "data/processed/product_snapshots"
    / "product_snapshots_clean.parquet"
)

OUTPUT_DIR = (
    BASE_DIR / "data/processed/product_top"
)

# Set False to clean hot deals independently.
VALIDATE_WITH_SNAPSHOTS = True


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

    cleaned_df.to_parquet(
        output_dir / "product_top_clean.parquet",
        index=False,
    )

    for name, anomaly_df in anomalies.items():
        anomaly_df.to_csv(
            output_dir / f"{name}.csv",
            index=False,
        )

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

    snapshot_source = (
        SNAPSHOT_PATH
        if VALIDATE_WITH_SNAPSHOTS
        else None
    )

    cleaned, report, anomalies = clean_product_top(
        source=PRODUCT_TOP_PATH,
        snapshot_source=snapshot_source,
    )

    save_results(
        cleaned_df=cleaned,
        report=report,
        anomalies=anomalies,
        output_dir=OUTPUT_DIR,
    )

    print("\nPRODUCT TOP / HOT DEALS CLEANING COMPLETE")
    print("=" * 50)

    print(f"Rows before: {report['rows_before']:,}")
    print(f"Rows after: {report['rows_after']:,}")

    print(
        "Exact duplicates removed:",
        report["exact_duplicates_removed"],
    )

    print(
        "Missing identifier rows removed:",
        report["missing_identifier_rows_removed"],
    )

    print(
        "Primary key unique:",
        report["primary_key_unique"],
    )

    print(
        "Conflicting key rows:",
        report["key_conflict_rows"],
    )

    print("\nHOT DEAL STATUS")

    for status, count in (
        report["hot_deal_status_after"].items()
    ):
        print(f"{status}: {count:,}")

    print(
        "\nStatus consistent:",
        report["status_consistent"],
    )

    join_report = report["snapshot_join_validation"]

    if join_report is not None:
        print("\nSNAPSHOT JOIN VALIDATION")

        print("Join valid:", join_report["join_valid"])

        if join_report["join_valid"]:
            print(
                "Snapshot rows:",
                f"{join_report['snapshot_rows']:,}",
            )

            print(
                "Matched rows:",
                f"{join_report['matched_snapshot_rows']:,}",
            )

            print(
                "Unmatched rows:",
                f"{join_report['unmatched_snapshot_rows']:,}",
            )

            print(
                "Row count preserved:",
                join_report["row_count_preserved"],
            )

    if not report["primary_key_unique"]:
        print(
            "\nWARNING: Duplicate hot-deal keys remain. "
            "Review key_conflicts.csv before joining."
        )

    print(f"\nResults saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
