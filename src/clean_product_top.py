
from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

PRIMARY_KEY = ["snapshot_date", "ean_code"]

REQUIRED_COLUMNS = [
    "snapshot_date",
    "ean_code",
    "is_top",
]

HOT_DEAL_STATUSES = [
    "highlighted",
    "not_highlighted",
    "unknown_not_in_catalog",
    "missing_top_record",
]



# ============================================================
# STEP 0: LOAD DATA
# ============================================================

def load_table(
    source: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    """
    Accept:
        - pandas DataFrame
        - Parquet file
        - CSV file

    Return a copy without changing the original.
    """

    if isinstance(source, pd.DataFrame):
        return source.copy()

    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)

    if path.suffix.lower() == ".csv":
        return pd.read_csv(
            path,
            dtype={"ean_code": "string"},
        )

    raise ValueError(
        f"Unsupported file format: {path.suffix}"
    )


def validate_schema(df: pd.DataFrame) -> None:
    """Validate required columns."""

    missing = sorted(
        set(REQUIRED_COLUMNS) - set(df.columns)
    )

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


# ============================================================
# STEP 1: STANDARDIZE DATA TYPES
# ============================================================

def parse_nullable_boolean(value):
    """
    Convert known boolean representations.

    Preserve missing values as pd.NA.
    """

    if pd.isna(value):
        return pd.NA

    if isinstance(value, bool):
        return value

    text = str(value).strip().lower()

    if text in {"true", "1"}:
        return True

    if text in {"false", "0"}:
        return False

    if text in {"", "none", "null", "nan", "unknown"}:
        return pd.NA

    raise ValueError(
        f"Unexpected is_top value: {value!r}"
    )


def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize dates, EAN identifiers and nullable booleans.
    """

    df = df.copy()

    # Keep EAN as a string to preserve leading zeros.
    df["ean_code"] = (
        df["ean_code"]
        .astype("string")
        .str.strip()
        .replace("", pd.NA)
    )

    # Calendar date, not timestamp.
    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"],
        errors="raise",
    ).dt.date

    # Preserve True, False and Unknown.
    df["is_top"] = (
        df["is_top"]
        .map(parse_nullable_boolean)
        .astype("boolean")
    )

    return df


# ============================================================
# STEP 2: REMOVE MISSING IDENTIFIERS
# ============================================================

def remove_missing_identifiers(df: pd.DataFrame):
    """
    Drop rows with missing snapshot_date or ean_code.

    Do not drop rows only because is_top is missing.
    """

    df = df.copy()

    missing_mask = df[PRIMARY_KEY].isna().any(axis=1)

    removed = df.loc[missing_mask].copy()

    cleaned = df.loc[~missing_mask].copy()

    return cleaned, removed


# ============================================================
# STEP 3: REMOVE EXACT DUPLICATES
# ============================================================

def remove_exact_duplicates(df: pd.DataFrame):
    """
    Remove fully identical observations.
    """

    df = df.copy()

    n_duplicates = int(df.duplicated().sum())

    df = df.drop_duplicates().copy()

    return df, n_duplicates


# ============================================================
# STEP 4: INVESTIGATE DUPLICATE KEYS
# ============================================================

def investigate_duplicate_keys(df: pd.DataFrame):
    """
    Distinguish between:

    1. Duplicate keys with the same is_top value.
    2. Duplicate keys with conflicting is_top values.

    Preserve conflicting groups for investigation.
    """

    df = df.copy()

    # The current table contains one non-key field: is_top.
    # Exact duplicates have already been removed.
    # Any remaining duplicate keys therefore disagree
    # on their is_top values (including null vs non-null).

    conflict_mask = df.duplicated(
        subset=PRIMARY_KEY,
        keep=False,
    )

    conflicts = (
        df.loc[conflict_mask]
        .sort_values(PRIMARY_KEY)
        .reset_index(drop=True)
    )

    # Preserve unresolved conflicting records.
    return df, conflicts


# ============================================================
# STEP 5: HOT DEAL STATUS PREPARATION
# ============================================================

def prepare_hot_deal_status(df: pd.DataFrame) -> pd.DataFrame:
    """
    Assign hot-deal status to existing product-top records.

    Missing product-top records are added separately
    after comparison with cleaned snapshots.
    """
    df = df.copy()

    df["hot_deal_status"] = pd.Series(
        "unknown_not_in_catalog",
        index=df.index,
        dtype="string",
    )

    true_mask = df["is_top"].eq(True).fillna(False)
    false_mask = df["is_top"].eq(False).fillna(False)

    df.loc[true_mask, "hot_deal_status"] = "highlighted"
    df.loc[false_mask, "hot_deal_status"] = "not_highlighted"

    df["hot_deal_status"] = pd.Categorical(
        df["hot_deal_status"],
        categories=HOT_DEAL_STATUSES,
    )

    return df



# ============================================================
# STEP 6: FLAG DISTRIBUTION
# ============================================================

def count_hot_deal_status(df: pd.DataFrame) -> dict:
    """Count all four hot-deal states."""

    if "hot_deal_status" not in df.columns:
        df = prepare_hot_deal_status(df)

    return {
        status: int(
            df["hot_deal_status"].eq(status).sum()
        )
        for status in HOT_DEAL_STATUSES
    }


# ============================================================
# STEP 7: VALIDATE JOIN WITH SNAPSHOTS
# ============================================================

def validate_snapshot_join(
    top_df: pd.DataFrame,
    snapshot_source: str | Path | pd.DataFrame,
):
    """
    Validate snapshot-to-hot-deal join.

    Does NOT modify or return a merged feature table.

    Requires a unique hot-deal key for many-to-one joins.
    """

    snapshots = load_table(snapshot_source)

    required = set(PRIMARY_KEY)

    if not required.issubset(snapshots.columns):
        raise ValueError(
            "Snapshots require snapshot_date and ean_code."
        )

    snapshots = snapshots.copy()

    snapshots["ean_code"] = (
        snapshots["ean_code"]
        .astype("string")
        .str.strip()
    )

    snapshots["snapshot_date"] = pd.to_datetime(
        snapshots["snapshot_date"],
        errors="raise",
    ).dt.date

    if top_df.duplicated(PRIMARY_KEY).any():
        return {
            "join_valid": False,
            "reason": "Duplicate hot-deal keys remain.",
            "snapshot_rows": int(len(snapshots)),
            "unmatched_snapshot_rows": None,
            "matched_snapshot_rows": None,
            "rows_after_join": None,
        }, pd.DataFrame(columns=PRIMARY_KEY)

    # Use the merge indicator to distinguish unmatched
    # records from matched rows where is_top is null.
    comparison = snapshots[PRIMARY_KEY].merge(
        top_df[PRIMARY_KEY + ["is_top"]],
        on=PRIMARY_KEY,
        how="left",
        validate="many_to_one",
        indicator=True,
    )

    unmatched_mask = comparison["_merge"].eq("left_only")

    unmatched = (
        comparison.loc[
            unmatched_mask, PRIMARY_KEY
        ]
        .drop_duplicates()
        .reset_index(drop=True)
    )

    join_report = {
        "join_valid": True,
        "snapshot_rows": int(len(snapshots)),
        "matched_snapshot_rows": int(
            (~unmatched_mask).sum()
        ),
        "unmatched_snapshot_rows": int(
            unmatched_mask.sum()
        ),
        "unmatched_distinct_keys": int(len(unmatched)),
        "rows_after_join": int(len(comparison)),
        "row_count_preserved": bool(
            len(comparison) == len(snapshots)
        ),
    }

    return join_report, unmatched


# ============================================================
# STEP 8: FINAL VALIDATION
# ============================================================

def validate_cleaning(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    missing_identifier_rows: pd.DataFrame,
    exact_duplicates_removed: int,
    key_conflicts: pd.DataFrame,
    join_report: dict | None,
) -> dict:
    """
    Produce the final data quality summary.
    """

    original_counts = count_hot_deal_status(original)
    cleaned_counts = count_hot_deal_status(cleaned)

    return {
        "rows_before": int(len(original)),
        "rows_after": int(len(cleaned)),

        "exact_duplicates_removed": int(
            exact_duplicates_removed
        ),

        "missing_identifier_rows_removed": int(
            len(missing_identifier_rows)
        ),

        "primary_key": PRIMARY_KEY,

        "primary_key_unique": bool(
            not cleaned.duplicated(
                subset=PRIMARY_KEY
            ).any()
        ),

        "key_conflict_rows": int(
            len(key_conflicts)
        ),

        "hot_deal_status_before": original_counts,
        "hot_deal_status_after": cleaned_counts,

        "unknown_not_in_catalog_after": int(
            cleaned["hot_deal_status"]
            .eq("unknown_not_in_catalog")
            .sum()
        ),

        "missing_top_records_added": int(
            cleaned["hot_deal_status"]
            .eq("missing_top_record")
            .sum()
        ),

        "status_consistent": bool(
            (
                (
                    cleaned["is_top"].eq(True).fillna(False)
                    & cleaned["hot_deal_status"].eq("highlighted")
                    & cleaned["top_record_present"]
                )
                |
                (
                    cleaned["is_top"].eq(False).fillna(False)
                    & cleaned["hot_deal_status"].eq("not_highlighted")
                    & cleaned["top_record_present"]
                )
                |
                (
                    cleaned["is_top"].isna()
                    & cleaned["hot_deal_status"].eq(
                        "unknown_not_in_catalog"
                    )
                    & cleaned["top_record_present"]
                )
                |
                (
                    cleaned["is_top"].isna()
                    & cleaned["hot_deal_status"].eq(
                        "missing_top_record"
                    )
                    & ~cleaned["top_record_present"]
                )
            ).all()
        ),

        "snapshot_join_validation": join_report,

        "decisions": [
            "Removed rows with missing identifiers",
            "Removed exact duplicate records",
            "Preserved unresolved duplicate-key conflicts",
            "Preserved unknown is_top values",
            "Did not convert unknown to False",
            "Created four-state categorical hot deal status",
            "Added missing date/EAN records identified from snapshots",
            "Preserved distinction between missing is_top and missing top records",
            "Added top_record_present to identify synthetic records",
            "Preserved EAN string identifiers",
            "Did not one-hot encode is_top",
            "Validated many-to-one snapshot join when supplied",
        ],
    }


# ============================================================
# MAIN CLEANING FUNCTION
# ============================================================


def clean_product_top(
    source: str | Path | pd.DataFrame,
    snapshot_source: str | Path | pd.DataFrame | None = None,
):
    """
    Clean the HHM daily hot-deal indicator table.

    When cleaned snapshots are supplied, add missing
    date/EAN combinations with status 'missing_top_record'.

    Returns:
        cleaned_df
        report
        anomalies
    """

    # STEP 0: LOAD
    df = load_table(source)
    validate_schema(df)
    df = standardize_types(df)

    original = df.copy()

    # STEP 1: REMOVE MISSING IDENTIFIERS
    df, missing_identifier_rows = (
        remove_missing_identifiers(df)
    )

    # STEP 2: REMOVE EXACT DUPLICATES
    df, n_removed = remove_exact_duplicates(df)

    # STEP 3: INVESTIGATE DUPLICATE KEYS
    df, key_conflicts = investigate_duplicate_keys(df)

    # STEP 4: PREPARE HOT DEAL STATUS
    # Existing product-top records receive one of:
    # highlighted, not_highlighted, unknown_not_in_catalog.
    df = prepare_hot_deal_status(df)

    # STEP 5: OPTIONAL SNAPSHOT JOIN VALIDATION
    join_report = None

    unmatched_snapshot_keys = pd.DataFrame(
        columns=PRIMARY_KEY
    )

    if snapshot_source is not None:

        (
            join_report,
            unmatched_snapshot_keys,
        ) = validate_snapshot_join(
            top_df=df,
            snapshot_source=snapshot_source,
        )

        if not join_report["join_valid"]:
            raise ValueError(
                "Cannot add missing top records because "
                "product-top keys are not unique."
            )

        # STEP 6: ADD MISSING TOP RECORDS
        if not unmatched_snapshot_keys.empty:

            missing_records = unmatched_snapshot_keys.copy()

            # No original is_top observation exists.
            missing_records["is_top"] = pd.Series(
                pd.NA,
                index=missing_records.index,
                dtype="boolean",
            )

            missing_records["hot_deal_status"] = (
                "missing_top_record"
            )

            df = pd.concat(
                [df, missing_records],
                ignore_index=True,
            )

    # STEP 7: ENSURE ALL FOUR STATUS LABELS
    df["hot_deal_status"] = pd.Categorical(
        df["hot_deal_status"],
        categories=HOT_DEAL_STATUSES,
    )

    # Identify whether each record originally existed
    # in product_top or was created from snapshots.
    df["top_record_present"] = (
        df["hot_deal_status"] != "missing_top_record"
    )

    # STEP 8: VALIDATE PRIMARY KEY
    if df.duplicated(PRIMARY_KEY).any():
        raise ValueError(
            "Duplicate product-top primary keys remain."
        )

    # STEP 9: FINAL VALIDATION REPORT
    report = validate_cleaning(
        original=original,
        cleaned=df,
        missing_identifier_rows=missing_identifier_rows,
        exact_duplicates_removed=n_removed,
        key_conflicts=key_conflicts,
        join_report=join_report,
    )

    # STEP 10: COLLECT ANOMALIES
    anomalies = {
        "missing_identifier_rows": missing_identifier_rows,
        "key_conflicts": key_conflicts,
        "unmatched_snapshot_keys": unmatched_snapshot_keys,
    }

    return df, report, anomalies
