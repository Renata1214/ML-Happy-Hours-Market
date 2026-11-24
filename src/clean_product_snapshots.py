"""Clean Happy Hours Market product snapshots without enriching product reference.

Workflow:
    1. Validate schema and standardize types.
    2. Drop exact duplicate rows (preserve non-exact key conflicts).
    3. Optionally correct leading-zero and three known SKU/EAN mistakes.
    4. Flag remaining SKU/EAN conflicts.
    5. Create normalized product-name and category columns.
    6. Flag suspicious inventory values without changing quantities.
    7. Validate and return data, report, and anomaly tables.

The caller is responsible for saving outputs. The reference cleaner performs
all reference reconciliation and enrichment in a separate pipeline.
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd
import numpy as np
from itertools import combinations

# ============================================================
# CONFIGURATION
# ============================================================

PRIMARY_KEY = ["snapshot_date", "sku"]
IDENTIFIER_COLUMNS = ["sku", "ean_code", "shop_branch"]
INTEGER_COLUMNS = [
    "quantity_sold",
    "quantity_end_sales",
    "origin_quantity",
    "quantity_put_on_sale_start_of_day",
]
FLOAT_COLUMNS = ["unit_weight", "new_price", "old_price"]
REQUIRED_COLUMNS = [
    *PRIMARY_KEY,
    "ean_code",
    "product_name",
    "category",
    "date_type",
    "shop_branch",
    "scan_date",
    "expiry_date",
    *INTEGER_COLUMNS,
    *FLOAT_COLUMNS,
]


# ============================================================
# UTILITIES
# ============================================================

def load_table(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Load a DataFrame, Parquet file, or CSV without modifying the input."""
    if isinstance(source, pd.DataFrame):
        return source.copy()

    path = Path(source)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    suffix = path.suffix.lower()
    if suffix == ".parquet":
        return pd.read_parquet(path)
    if suffix == ".csv":
        # EANs and other identifiers must remain strings (including zeros).
        return pd.read_csv(
            path,
            dtype={
                "sku": "string",
                "ean_code": "string",
                "shop_branch": "string",
            },
        )
    raise ValueError(f"Unsupported file type: {suffix}")


def validate_schema(df: pd.DataFrame) -> None:
    """Make sure the columns required by the cleaning pipeline exist."""
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def normalize_text(value: object) -> object:
    """Normalize text for comparisons, without semantic product merging.

    Preserve numbers, package sizes, flavors, and meaningful identifiers.
    The original source strings remain in their own columns.
    """
    if pd.isna(value):
        return pd.NA

    text = str(value).strip().casefold()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[|;,_]+", " ", text)
    text = re.sub(r"\s*([/().-])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    """Convert source columns to consistent logical types."""
    df = df.copy()

    for column in IDENTIFIER_COLUMNS:
        df[column] = df[column].astype("string").str.strip()

    df["product_name"] = df["product_name"].astype("string")
    df["category"] = df["category"].astype("string")
    df["date_type"] = (
        df["date_type"].astype("string").str.strip().str.upper()
    )

    # Preserve previous pipeline behavior: Python calendar dates.
    df["snapshot_date"] = pd.to_datetime(
        df["snapshot_date"], errors="raise"
    ).dt.date

    for column in ["scan_date", "expiry_date"]:
        df[column] = pd.to_datetime(df[column], errors="raise")

    for column in INTEGER_COLUMNS:
        values = pd.to_numeric(df[column], errors="raise")
        invalid_integer = values.notna() & (values % 1 != 0)
        if invalid_integer.any():
            raise ValueError(f"{column} contains non-integer quantities.")
        df[column] = values.astype("Int64")

    for column in FLOAT_COLUMNS:
        df[column] = pd.to_numeric(
            df[column], errors="raise"
        ).astype("float64")

    return df

def normalize_category(value):
    """
    Normalize category labels consistently with product reference.

    - Convert to lowercase
    - Remove accents
    - Treat punctuation and separators as spaces
    - Normalize whitespace
    - Preserve actual words
    """

    if pd.isna(value):
        return pd.NA

    text = unicodedata.normalize(
        "NFKD",
        str(value).strip().casefold()
    )

    # Remove accents.
    text = "".join(
        char for char in text
        if not unicodedata.combining(char)
    )

    # Replace punctuation with spaces.
    text = re.sub(r"[^\w\s]", " ", text)

    # Normalize repeated whitespace.
    text = re.sub(r"\s+", " ", text).strip()

    return text if text else pd.NA



# ============================================================
# DUPLICATE HANDLING
# ============================================================

def clean_duplicates(df: pd.DataFrame):
    """Remove exact duplicates, but only report duplicate primary keys."""
    df = df.copy()
    exact_duplicate_count = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()
    key_conflicts = df.loc[
        df.duplicated(subset=PRIMARY_KEY, keep=False)
    ].copy()
    return df, exact_duplicate_count, key_conflicts


# ============================================================
# ORIGINAL-FILE EAN CORRECTIONS
# ============================================================

def correct_leading_zero_eans(df: pd.DataFrame):
    """Unify a SKU's EANs only if they differ solely by leading zeros.

    Always choose the longest observed string, which keeps the most
    leading zeros. Do not alter SKUs with genuinely different EANs.

    Returns (corrected_df, corrections_report).
    """
    df = df.copy()
    ean_counts = df.groupby("sku")["ean_code"].nunique()
    conflicting_skus = ean_counts[ean_counts > 1].index
    corrections = []

    for sku in conflicting_skus:
        sku_mask = (df["sku"] == sku).fillna(False)
        eans = df.loc[sku_mask, "ean_code"].dropna().unique().tolist()

        if not eans or not all(ean.isdigit() for ean in eans):
            continue

        normalized = {ean.lstrip("0") or "0" for ean in eans}
        if len(normalized) != 1:
            continue

        canonical_ean = max(eans, key=len)
        for old_ean in eans:
            if old_ean == canonical_ean:
                continue

            affected_mask = sku_mask & (
                df["ean_code"] == old_ean
            ).fillna(False)
            n_rows = int(affected_mask.sum())
            df.loc[affected_mask, "ean_code"] = canonical_ean
            corrections.append(
                {
                    "sku": sku,
                    "old_ean": old_ean,
                    "corrected_ean": canonical_ean,
                    "rows_corrected": n_rows,
                }
            )

    return df, pd.DataFrame(
        corrections,
        columns=["sku", "old_ean", "corrected_ean", "rows_corrected"],
    )


def correct_known_snapshot_eans(
    df: pd.DataFrame,
    original_files: bool = False,
):
    """Correct three confirmed erroneous (SKU, old EAN) combinations.

    Match *both* SKU and original EAN; do not globally replace an EAN.
    Apply only if original_files=True.
    """
    df = df.copy()
    corrections_map = {
        ("ZasELirVqScsHzHXz9z3", "3760207115019"): "3760207115033",
        ("b8f87d0f-b57e-4c0a-8ff5-0501cbce4277", "H1556794"): "1556794",
        ("fb84e253-caa6-4e94-b9ad-4a91bf33ad31", "8714811175483"): "87148111754830",
    }
    report_columns = ["sku", "old_ean", "corrected_ean", "rows_corrected"]
    corrections = []

    if not original_files:
        return df, pd.DataFrame(columns=report_columns)

    for (sku, old_ean), corrected_ean in corrections_map.items():
        mask = (
            (df["sku"] == sku) & (df["ean_code"] == old_ean)
        ).fillna(False)
        n_rows = int(mask.sum())
        if n_rows == 0:
            continue

        df.loc[mask, "ean_code"] = corrected_ean
        corrections.append(
            {
                "sku": sku,
                "old_ean": old_ean,
                "corrected_ean": corrected_ean,
                "rows_corrected": n_rows,
            }
        )

    return df, pd.DataFrame(corrections, columns=report_columns)


# ============================================================
# TEXT NORMALIZATION AND IDENTITY CHECKS
# ============================================================

def find_sku_ean_conflicts(df: pd.DataFrame) -> pd.DataFrame:
    """Report SKUs that still map to more than one distinct EAN."""
    ean_counts = df.groupby("sku")["ean_code"].nunique()
    conflicting_skus = ean_counts[ean_counts > 1].index

    conflicts = (
        df.loc[
            df["sku"].isin(conflicting_skus),
            ["sku", "ean_code", "product_name"],
        ]
        .drop_duplicates()
        .sort_values(["sku", "ean_code", "product_name"])
    )
    return conflicts.reset_index(drop=True)


def standardize_product_names(df: pd.DataFrame) -> pd.DataFrame:
    """Preserve product_name and add product_name_normalized."""
    df = df.copy()
    df["product_name_normalized"] = (
        df["product_name"].map(normalize_text).astype("string")
    )
    return df


def standardize_categories(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preserve original categories and create normalized
    categories using the same rules as product reference.
    """

    df = df.copy()

    df["category_normalized"] = (
        df["category"]
        .map(normalize_category)
        .astype("string")
    )

    return df


# ============================================================
# MISSING VALUES AND INVENTORY ANOMALIES
# ============================================================

def inspect_nulls(df: pd.DataFrame) -> dict:
    """Report nonzero missing-value counts; do not impute them."""
    return {
        column: int(count)
        for column, count in df.isna().sum().items()
        if count > 0
    }



def find_inventory_anomalies(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify inventory anomalies and assign categorical labels
    to every snapshot record.

    A row can have one or multiple anomaly types.

    If there are no anomalies:
        inventory_anomaly_label = pd.NA

    If there are anomalies:
        inventory_anomaly_label = a categorical label
        representing the exact combination.

    Returns:
        cleaned_df_with_labels
        inventory_anomalies_report

    Inventory values are never modified.
    """

    df = df.copy()

    # STEP 1: DEFINE THE ANOMALY CONDITIONS
    checks = pd.DataFrame(index=df.index)

    checks["negative_quantity"] = (
        df[INTEGER_COLUMNS].lt(0).any(axis=1)
    )

    checks["negative_price_or_weight"] = (
        df[FLOAT_COLUMNS].lt(0).any(axis=1)
    )

    checks["sold_exceeds_start_of_day"] = (
        df["quantity_sold"]
        > df["quantity_put_on_sale_start_of_day"]
    )

    checks["end_exceeds_start_of_day"] = (
        df["quantity_end_sales"]
        > df["quantity_put_on_sale_start_of_day"]
    )

    checks = checks.fillna(False).astype(bool)

    # STEP 2: DEFINE ALL POSSIBLE COMBINATIONS
    anomaly_types = list(checks.columns)

    # Each combination is a predefined categorical label.
    # Bitmasks ensure deterministic combinations.
    labels = [
        " + ".join(
            anomaly_types[i]
            for i in range(len(anomaly_types))
            if code & (1 << i)
        )
        for code in range(1, 2 ** len(anomaly_types))
    ]

    # STEP 3: IDENTIFY ACTIVE CONDITIONS PER ROW
    # This is vectorized for large datasets.

    mask_codes = np.zeros(
        len(df),
        dtype=np.uint8,
    )

    for i, column in enumerate(anomaly_types):
        mask_codes |= (
            checks[column].to_numpy(dtype=np.uint8)
            << i
        )

    # STEP 4: CREATE CATEGORICAL LABEL COLUMN
    # 0 means no anomaly.
    # Pandas categorical code -1 represents missing.
    df["inventory_anomaly_label"] = pd.Categorical.from_codes(
        codes=mask_codes.astype(np.int16) - 1,
        categories=labels,
    )

    # STEP 5: EXTRACT ANOMALOUS RECORDS
    affected = mask_codes != 0

    inventory_anomalies = df.loc[affected].copy()

    # Preserve individual Boolean flags in the audit CSV.
    for column in anomaly_types:
        inventory_anomalies[column] = (
            checks.loc[affected, column]
        )

    return df, inventory_anomalies.reset_index(drop=True)


# ============================================================
# VALIDATION AND REPORTING
# ============================================================

def quantity_totals(df: pd.DataFrame) -> dict:
    return {column: int(df[column].sum()) for column in INTEGER_COLUMNS}


def daily_counts(df: pd.DataFrame) -> pd.Series:
    return df.groupby("snapshot_date").size()


def validate_cleaning(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    exact_duplicates_removed: int,
    key_conflicts: pd.DataFrame,
    sku_ean_conflicts: pd.DataFrame,
    inventory_anomalies: pd.DataFrame,
) -> dict:
    original_totals = quantity_totals(original)
    cleaned_totals = quantity_totals(cleaned)
    original_daily = daily_counts(original)
    cleaned_daily = daily_counts(cleaned)

    return {
        "rows_before": int(len(original)),
        "rows_after": int(len(cleaned)),
        "exact_duplicates_removed": exact_duplicates_removed,
        "primary_key": PRIMARY_KEY,
        "primary_key_unique": bool(
            not cleaned.duplicated(subset=PRIMARY_KEY).any()
        ),
        "key_conflict_rows": int(len(key_conflicts)),
        "skus_with_multiple_eans": int(
            sku_ean_conflicts["sku"].nunique()
        ),
        "sku_ean_conflict_rows": int(len(sku_ean_conflicts)),
        "inventory_anomaly_rows": int(len(inventory_anomalies)),
        "null_counts_after": inspect_nulls(cleaned),
        "quantity_totals_before": original_totals,
        "quantity_totals_after": cleaned_totals,
        "quantity_total_differences": {
            col: cleaned_totals[col] - original_totals[col]
            for col in INTEGER_COLUMNS
        },
        "daily_counts_changed": bool(
            not original_daily.equals(cleaned_daily)
        ),
        "days_before": int(original_daily.size),
        "days_after": int(cleaned_daily.size),
        "decisions": [
            "Removed exact duplicates only",
            "Applied SKU/EAN corrections when original_files=True",
            "Preserved SKUs and did not aggregate SKUs into EANs",
            "Flagged remaining multiple-EAN SKUs without merging",
            "Preserved original product names and categories",
            "Created normalized comparison columns",
            "Did not impute missing snapshot values",
            "Flagged suspicious inventory values without changing them",
            "Did not enrich product reference in snapshot cleaning",
        ],
    }


# ============================================================
# MAIN REUSABLE CLEANING FUNCTION
# ============================================================

def clean_product_snapshots(
    source: str | Path | pd.DataFrame,
    original_files: bool = False,
):
    """Clean snapshots without loading or enriching product reference.

    original_files=True enables the leading-zero and three hardcoded
    SKU/EAN corrections for the original source files.

    Returns:
        cleaned_snapshots: pd.DataFrame
        report: dict
        anomalies: dict[str, pd.DataFrame]
    """
    df = load_table(source)
    validate_schema(df)
    df = standardize_types(df)
    original = df.copy()

    df, n_removed, _ = clean_duplicates(df)

    ean_corrections = pd.DataFrame(
        columns=["sku", "old_ean", "corrected_ean", "rows_corrected"]
    )
    if original_files:
        df, ean_corrections = correct_leading_zero_eans(df)

    df, known_ean_corrections = correct_known_snapshot_eans(
        df, original_files=original_files
    )

    sku_ean_conflicts = find_sku_ean_conflicts(df)
    df = standardize_product_names(df)
    df = standardize_categories(df)
    df, inventory_anomalies = find_inventory_anomalies(df)

    # Re-check final key after applying corrections.
    key_conflicts = df.loc[
        df.duplicated(subset=PRIMARY_KEY, keep=False)
    ].copy()

    report = validate_cleaning(
        original=original,
        cleaned=df,
        exact_duplicates_removed=n_removed,
        key_conflicts=key_conflicts,
        sku_ean_conflicts=sku_ean_conflicts,
        inventory_anomalies=inventory_anomalies,
    )
    report["original_files"] = original_files
    report["leading_zero_ean_corrections"] = {
        "skus_corrected": int(ean_corrections["sku"].nunique()),
        "rows_corrected": int(ean_corrections["rows_corrected"].sum()),
    }
    report["known_snapshot_ean_corrections"] = {
        "sku_ean_combinations_corrected": int(len(known_ean_corrections)),
        "rows_corrected": int(known_ean_corrections["rows_corrected"].sum()),
    }

    anomalies = {
        "sku_ean_conflicts": sku_ean_conflicts,
        "key_conflicts": key_conflicts,
        "inventory_anomalies": inventory_anomalies,
        "ean_corrections": ean_corrections,
        "known_ean_corrections": known_ean_corrections,
    }
    return df, report, anomalies
