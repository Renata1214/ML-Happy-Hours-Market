
from __future__ import annotations
from pathlib import Path
import pandas as pd

# ============================================================
# CONFIGURATION
# ============================================================

PRIMARY_KEY = ["order_date", "pickup_point", "sku"]

IDENTIFIER_COLUMNS = [
    "pickup_point",
    "sku",
    "ean_code",
]

INTEGER_COLUMNS = [
    "quantity_ordered",
    "n_orders",
]

REQUIRED_COLUMNS = [
    "order_date",
    "pickup_point",
    "sku",
    "ean_code",
    "quantity_ordered",
    "new_price",
    "n_orders",
]



# ============================================================
# STEP 0: LOAD DATA
# ============================================================

def load_table(
    source: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    """Load a Parquet file, CSV file, or DataFrame."""

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
            dtype={
                column: "string"
                for column in IDENTIFIER_COLUMNS
            },
        )

    raise ValueError(
        f"Unsupported file format: {path.suffix}"
    )


def validate_schema(df: pd.DataFrame) -> None:
    """Check that all required columns exist."""

    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))

    if missing:
        raise ValueError(
            f"Missing required columns: {missing}"
        )


# ============================================================
# STEP 1: STANDARDIZE DATA TYPES
# ============================================================


def standardize_types(
    df: pd.DataFrame,
    price_decimals: int = 2,
) -> pd.DataFrame:
    """
    Standardize data types and numerical precision.

    - Preserve identifiers as strings.
    - Convert order_date to a calendar date.
    - Preserve integer quantities without truncation.
    - Round new_price to a consistent decimal precision.
    """

    df = df.copy()

    # STEP 1: STANDARDIZE IDENTIFIERS
    for column in IDENTIFIER_COLUMNS:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # STEP 2: STANDARDIZE DATE
    df["order_date"] = pd.to_datetime(
        df["order_date"],
        errors="raise",
    ).dt.date

    # STEP 3: STANDARDIZE INTEGER COLUMNS
    for column in INTEGER_COLUMNS:
        values = pd.to_numeric(
            df[column],
            errors="raise",
        )

        # Do not silently truncate fractional quantities.
        invalid = (
            values.notna()
            & (values % 1 != 0)
        )

        if invalid.any():
            raise ValueError(
                f"{column} contains fractional values."
            )

        df[column] = values.astype("Int64")

    # STEP 4: STANDARDIZE PRICE PRECISION
    df["new_price"] = (
        pd.to_numeric(
            df["new_price"],
            errors="raise",
        )
        .astype("float64")
        .round(price_decimals)
    )

    return df

# ============================================================
# STEP 2: REMOVE EXACT DUPLICATES
# ============================================================

def remove_exact_duplicates(df: pd.DataFrame):
    """
    Remove only identical records.

    Do not automatically remove duplicate composite keys.
    """

    df = df.copy()

    n_duplicates = int(df.duplicated().sum())

    df = df.drop_duplicates().copy()

    return df, n_duplicates


# ============================================================
# STEP 3: INVESTIGATE DUPLICATE KEYS
# ============================================================


def resolve_pickup_key_conflicts(
    df: pd.DataFrame,
):
    """
    Resolve duplicate pickup-point primary keys.

    Key:
        (order_date, pickup_point, sku)

    Rules:
        1. Save every row belonging to a conflicting key.
        2. Keep the row with maximum quantity_ordered.
        3. If quantities tie, prefer higher n_orders.
        4. If still tied, keep the earliest input row.

    Returns:
        cleaned_df
        key_conflicts
        duplicate_rows_removed
    """

    df = df.copy()

    # Identify all rows belonging to conflicting keys.
    duplicate_mask = df.duplicated(
        subset=PRIMARY_KEY,
        keep=False,
    )

    key_conflicts = df.loc[duplicate_mask].copy()

    # Preserve original input order for deterministic ties.
    df["_original_row_order"] = range(len(df))

    # Select the best record for each key.
    df = df.sort_values(
        by=[
            *PRIMARY_KEY,
            "quantity_ordered",
            "n_orders",
            "_original_row_order",
        ],
        ascending=[
            *([True] * len(PRIMARY_KEY)),
            False,
            False,
            True,
        ],
        na_position="last",
        kind="stable",
    )

    # Retain the highest-ranked row for each key.
    df = df.drop_duplicates(
        subset=PRIMARY_KEY,
        keep="first",
    )

    # Restore the original input order.
    df = df.sort_values(
        "_original_row_order",
        kind="stable",
    )

    df = (
        df.drop(columns="_original_row_order")
        .reset_index(drop=True)
    )

    duplicate_rows_removed = (
        len(key_conflicts)
        - key_conflicts[PRIMARY_KEY]
          .drop_duplicates()
          .shape[0]
    )

    assert not df.duplicated(PRIMARY_KEY).any(), (
        "Duplicate primary keys remain after resolution."
    )

    return (
        df,
        key_conflicts.reset_index(drop=True),
        duplicate_rows_removed,
    )



def find_repeated_eans(df: pd.DataFrame) -> pd.DataFrame:
    """
    Diagnostic only.

    Repeated EANs within the same pickup point and date
    are not necessarily errors.
    """

    keys = ["order_date", "pickup_point", "ean_code"]

    repeated = df.loc[
        df.duplicated(
            subset=keys,
            keep=False,
        )
    ].copy()

    return repeated.sort_values(
        keys
    ).reset_index(drop=True)


# ============================================================
# STEP 4: BUILD SNAPSHOT SKU/EAN MAPPING
# ============================================================

def build_snapshot_sku_mapping(
    snapshot_source: str | Path | pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a mapping of SKUs to EANs from cleaned snapshots.

    Only SKUs with exactly one observed EAN are accepted.
    Ambiguous SKUs are excluded.
    """

    snap = load_table(snapshot_source)

    required = {"sku", "ean_code"}

    if not required.issubset(snap.columns):
        raise ValueError(
            "Snapshot table requires sku and ean_code."
        )

    snap = snap[["sku", "ean_code"]].copy()

    for column in ["sku", "ean_code"]:
        snap[column] = (
            snap[column].astype("string").str.strip()
        )

    snap = snap.dropna().drop_duplicates()

    ean_counts = (
        snap.groupby("sku")["ean_code"]
        .nunique()
    )

    valid_skus = ean_counts[
        ean_counts == 1
    ].index

    mapping = (
        snap[snap["sku"].isin(valid_skus)]
        .rename(
            columns={"ean_code": "snapshot_ean_code"}
        )
    )

    return mapping.reset_index(drop=True)


# ============================================================
# STEP 5: CORRECT KNOWN ORIGINAL-FILE ERRORS
# ============================================================


def correct_pickup_ean_errors(
    df: pd.DataFrame,
    snapshot_mapping: pd.DataFrame,
):
    """
    Correct pickup-order EAN mismatches using cleaned snapshots.

    Correction conditions:
        1. SKU exists in cleaned snapshots.
        2. SKU has exactly one EAN in cleaned snapshots.
        3. Pickup EAN differs from snapshot EAN.

    No date restrictions or expected correction counts.

    Returns:
        corrected_df
        corrections_report
    """

    df = df.copy()

    # STEP 1: Attach the validated snapshot EAN.
    comparison = df.merge(
        snapshot_mapping,
        on="sku",
        how="left",
        validate="many_to_one",
        sort=False,
    )

    # STEP 2: Identify rows with a valid snapshot mapping.
    has_reference = comparison["snapshot_ean_code"].notna()

    # STEP 3: Identify EAN mismatches.
    mismatch = (
        comparison["ean_code"]
        != comparison["snapshot_ean_code"]
    ).fillna(False)

    # STEP 4: Select rows eligible for correction.
    correction_mask = (
        has_reference
        & mismatch
    )

    # STEP 5: Record corrections before changing values.
    corrections = comparison.loc[
        correction_mask,
        [
            *PRIMARY_KEY,
            "ean_code",
            "snapshot_ean_code",
        ],
    ].copy()

    corrections = corrections.rename(
        columns={
            "ean_code": "old_ean",
            "snapshot_ean_code": "corrected_ean",
        }
    )

    corrections["reason"] = (
        "SKU/EAN mismatch corrected using cleaned snapshots"
    )

    # STEP 6: Apply corrections.
    comparison.loc[
        correction_mask, "ean_code"
    ] = comparison.loc[
        correction_mask, "snapshot_ean_code"
    ].to_numpy()

    # STEP 7: Validate corrections.
    assert len(comparison) == len(df), (
        "EAN correction changed the number of rows."
    )

    remaining_mismatches = (
        comparison.loc[correction_mask, "ean_code"]
        != comparison.loc[
            correction_mask, "snapshot_ean_code"
        ]
    ).fillna(False)

    assert not remaining_mismatches.any(), (
        "Some corrected EANs still differ from snapshots."
    )

    assert len(comparison) == len(df), (
        "EAN correction unexpectedly changed the number of rows."
    )

    remaining_mismatches = (
        comparison.loc[correction_mask, "ean_code"]
        != comparison.loc[correction_mask, "snapshot_ean_code"]
    ).fillna(False)

    assert not remaining_mismatches.any(), (
        "Some corrected EANs still differ from snapshots."
    )

    # STEP 8: Remove temporary mapping column.
    comparison = comparison.drop(
        columns=["snapshot_ean_code"]
    )

    # STEP 9: Restore original index.
    comparison.index = df.index

    return comparison, corrections.reset_index(drop=True)



# ============================================================
# STEP 6: MISSING VALUES
# ============================================================

def inspect_nulls(df: pd.DataFrame) -> dict:
    """Record null counts without modifying them."""

    return {
        column: int(count)
        for column, count in df.isna().sum().items()
        if count > 0
    }


# ============================================================
# STEP 7: NUMERICAL VALIDATION
# ============================================================

def find_numerical_anomalies(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Flag suspicious values without correcting them."""

    checks = pd.DataFrame(index=df.index)

    checks["negative_quantity"] = (
        df["quantity_ordered"] < 0
    )

    checks["negative_n_orders"] = (
        df["n_orders"] < 0
    )

    checks["negative_price"] = (
        df["new_price"] < 0
    )

    # Diagnostic; interpretation depends on HHM definitions.
    checks["orders_exceed_quantity"] = (
        df["n_orders"] > df["quantity_ordered"]
    )

    checks = checks.fillna(False).astype(bool)

    affected = checks.any(axis=1)

    anomalies = df.loc[affected].copy()

    for column in checks.columns:
        anomalies[column] = checks.loc[
            affected, column
        ]

    return anomalies


# ============================================================
# STEP 8: REMAINING SKU/EAN CONFLICTS
# ============================================================

def find_sku_ean_conflicts(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Find SKUs still associated with multiple EANs."""

    counts = (
        df.groupby("sku")["ean_code"]
        .nunique()
    )

    conflicting_skus = counts[
        counts > 1
    ].index

    return (
        df.loc[
            df["sku"].isin(conflicting_skus),
            ["sku", "ean_code"],
        ]
        .drop_duplicates()
        .sort_values(["sku", "ean_code"])
        .reset_index(drop=True)
    )


# ============================================================
# STEP 9: VALIDATION REPORT
# ============================================================

def calculate_totals(df: pd.DataFrame) -> dict:
    """Calculate quantities and order totals."""

    return {
        "quantity_ordered": int(
            df["quantity_ordered"].sum()
        ),
        "n_orders": int(
            df["n_orders"].sum()
        ),
    }


def validate_cleaning(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    exact_duplicates_removed: int,
    key_conflicts: pd.DataFrame,
    repeated_eans: pd.DataFrame,
    sku_ean_conflicts: pd.DataFrame,
    ean_corrections: pd.DataFrame,
    numerical_anomalies: pd.DataFrame,
    original_files: bool,
) -> dict:

    before = calculate_totals(original)
    after = calculate_totals(cleaned)

    return {
        "original_files": original_files,

        "rows_before": int(len(original)),
        "rows_after": int(len(cleaned)),

        "exact_duplicates_removed": (
            exact_duplicates_removed
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

        "repeated_ean_rows": int(
            len(repeated_eans)
        ),

        "skus_with_multiple_eans": int(
            sku_ean_conflicts["sku"].nunique()
        ),

        "ean_rows_corrected": int(
            len(ean_corrections)
        ),

        "numerical_anomaly_rows": int(
            len(numerical_anomalies)
        ),

        "null_counts_after": inspect_nulls(cleaned),

        "totals_before": before,
        "totals_after": after,

        "quantity_difference": (
            after["quantity_ordered"]
            - before["quantity_ordered"]
        ),

        "n_orders_difference": (
            after["n_orders"]
            - before["n_orders"]
        ),

        "decisions": [
            "Standardized monetary prices to two decimal places",
    "Removed exact duplicate rows",
    "Resolved duplicate primary keys using maximum quantity_ordered",
    "Preserved all conflicting-key rows in the audit CSV",
    "Removed numerical anomalies from cleaned data",
    "Preserved numerical anomalies in the audit CSV",
    "Corrected EAN mismatches using cleaned snapshots when enabled",
    "Preserved daily pickup-point SKU granularity",
    "Preserved EAN identifiers as strings",
    "Did not aggregate different SKUs into EANs",
    "Did not encode categorical identifiers",
        ],
    }


# ============================================================
# MAIN CLEANING FUNCTION
# ============================================================

def clean_pickup_point_orders(
    source: str | Path | pd.DataFrame,
    snapshot_source: str | Path | pd.DataFrame | None = None,
    original_files: bool = False,
    expected_ean_corrections: int | None = 29,
):
    """
    Clean pickup point orders.

    Returns:
        cleaned_df
        report
        anomalies
    """

    # STEP 0: LOAD AND VALIDATE
    df = load_table(source)
    validate_schema(df)
    df = standardize_types(df)

    original = df.copy()

    # STEP 1: REMOVE EXACT DUPLICATES
    df, exact_duplicates_removed = (
        remove_exact_duplicates(df)
    )

    # STEP 2: OPTIONAL EAN CORRECTION
    ean_corrections = pd.DataFrame(
        columns=[
            *PRIMARY_KEY,
            "old_ean",
            "corrected_ean",
            "reason",
        ]
    )

    if original_files:
        if snapshot_source is None:
            raise ValueError(
                "snapshot_source is required when original_files=True."
            )

        snapshot_mapping = build_snapshot_sku_mapping(
            snapshot_source
        )

        df, ean_corrections = correct_pickup_ean_errors(
            df,
            snapshot_mapping,
        )

    # STEP 3: DETECT DUPLICATE KEYS
    df, key_conflicts, duplicate_rows_removed = resolve_pickup_key_conflicts(df)

    # STEP 4: INSPECT REPEATED EANS
    repeated_eans = find_repeated_eans(df)

    # STEP 5: INSPECT REMAINING SKU/EAN CONFLICTS
    sku_ean_conflicts = find_sku_ean_conflicts(df)

    # STEP 6: INSPECT NUMERICAL ANOMALIES
    numerical_anomalies = find_numerical_anomalies(df)

    if not numerical_anomalies.index.isin(df.index).all():
        raise ValueError(
            "Numerical anomaly indices do not match input rows."
        )

    # Exclude detected anomalies from the cleaned data.
    df = df.drop(
        index=numerical_anomalies.index
    ).reset_index(drop=True)

    # Final key validation.
    if df.duplicated(PRIMARY_KEY).any():
        raise ValueError(
            "Pickup-point primary key is not unique."
        )

    # STEP 7: VALIDATE
    report = validate_cleaning(
        original=original,
        cleaned=df,
        exact_duplicates_removed=exact_duplicates_removed,
        key_conflicts=key_conflicts,
        repeated_eans=repeated_eans,
        sku_ean_conflicts=sku_ean_conflicts,
        ean_corrections=ean_corrections,
        numerical_anomalies=numerical_anomalies,
        original_files=original_files,
    )

    report["duplicate_key_rows_removed"] = int(
    duplicate_rows_removed
    )

    report["numerical_anomaly_rows_removed"] = int(
        len(numerical_anomalies)
    )

    report["numerical_precision"] = {
        "new_price_decimal_places": 2,
        "quantity_ordered": "integer",
        "n_orders": "integer",
    }

    # STEP 8: COLLECT ANOMALY TABLES
    anomalies = {
        "key_conflicts": key_conflicts,
        "repeated_eans": repeated_eans,
        "sku_ean_conflicts": sku_ean_conflicts,
        "ean_corrections": ean_corrections,
        "numerical_anomalies": numerical_anomalies,
    }

    return df, report, anomalies
