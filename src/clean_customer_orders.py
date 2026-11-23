
from __future__ import annotations
import re
import unicodedata
from pathlib import Path
import pandas as pd

PRIMARY_KEY = ["order_id", "sku"]

IDENTIFIER_COLUMNS = [
    "user_id",
    "order_id",
    "pickup_point",
    "sku",
    "ean_code",
    "shop_branch",
]

INTEGER_COLUMNS = ["quantity"]

FLOAT_COLUMNS = [
    "old_price",
    "new_price",
]

REQUIRED_COLUMNS = [
    "user_id",
    "order_id",
    "order_datetime",
    "pick_up_date",
    "timeslot",
    "pickup_point",
    "sku",
    "ean_code",
    "product_name",
    "quantity",
    "old_price",
    "new_price",
    "date_type",
    "expiry_date",
    "shop_branch",
    "regular_client",
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
    Returns a copy without modifying the original source.
    """

    if isinstance(source, pd.DataFrame):
        return source.copy()

    path = Path(source)

    if not path.exists():
        raise FileNotFoundError(
            f"File not found: {path}"
        )

    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return pd.read_parquet(path)

    if suffix == ".csv":
        return pd.read_csv(
            path,
            dtype={
                column: "string"
                for column in IDENTIFIER_COLUMNS
            },
        )

    raise ValueError(
        f"Unsupported file format: {suffix}"
    )


def validate_schema(df: pd.DataFrame) -> None:
    """
    Verify that all expected columns exist.
    """

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

def parse_boolean(value):
    """
    Convert valid boolean representations safely.
    Do not interpret the string 'False' as True.
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

    raise ValueError(
        f"Invalid boolean value: {value!r}"
    )


def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Standardize data types while preserving original
    identifiers and numeric values.
    """

    df = df.copy()

    # IDENTIFIERS
    for column in IDENTIFIER_COLUMNS:
        df[column] = (
            df[column]
            .astype("string")
            .str.strip()
        )

    # DATES
    # Order datetime represents a local/naive timestamp.
    df["order_datetime"] = pd.to_datetime(
        df["order_datetime"],
        errors="raise",
    ).dt.date

    # Keep pickup date as a calendar date.
    df["pick_up_date"] = pd.to_datetime(
        df["pick_up_date"],
        errors="raise",
    ).dt.date

    # The supplied expiry_date is UTC-aware.
    # Naive input timestamps would be interpreted as UTC.
    df["expiry_date"] = pd.to_datetime(
        df["expiry_date"],
        errors="raise",
        utc=True,
    ).dt.date

    # TEXT AND CATEGORIES
    df["product_name"] = (
        df["product_name"].astype("string")
    )

    df["timeslot"] = (
        df["timeslot"]
        .astype("string")
        .str.strip()
        .str.replace(r"\s*-\s*", " - ", regex=True)
        .str.replace(r"\s+", " ", regex=True)
    )

    df["date_type"] = (
        df["date_type"]
        .astype("string")
        .str.strip()
        .str.upper()
    )

    # BOOLEAN
    df["regular_client"] = (
        df["regular_client"]
        .map(parse_boolean)
        .astype("boolean")
    )

    # INTEGER VALUES
    for column in INTEGER_COLUMNS:

        values = pd.to_numeric(
            df[column],
            errors="raise",
        )

        invalid_integer = (
            values.notna()
            & (values % 1 != 0)
        )

        if invalid_integer.any():
            raise ValueError(
                f"{column} contains fractional quantities."
            )

        df[column] = values.astype("Int64")

    # FLOAT VALUES
    for column in FLOAT_COLUMNS:
        df[column] = pd.to_numeric(
            df[column],
            errors="raise",
        ).astype("float64")

    return df


# ============================================================
# STEP 2: REMOVE DUPLICATES
# ============================================================

def clean_duplicates(df: pd.DataFrame):
    """
    Remove exact duplicates only.

    Preserve different SKUs sharing the same EAN.
    """

    df = df.copy()

    exact_duplicate_count = int(
        df.duplicated().sum()
    )

    # Remove complete duplicate rows.
    df = df.drop_duplicates().copy()

    # Check candidate primary key.
    key_conflicts = df.loc[
        df.duplicated(
            subset=PRIMARY_KEY,
            keep=False,
        )
    ].copy()

    # Diagnostic only:
    # Repeated EANs within an order are not removed.
    repeated_order_eans = df.loc[
        df.duplicated(
            subset=["order_id", "ean_code"],
            keep=False,
        )
    ].copy()

    return (
        df,
        exact_duplicate_count,
        key_conflicts,
        repeated_order_eans,
    )


# ============================================================
# STEP 3: MISSING VALUES
# ============================================================

def inspect_nulls(df: pd.DataFrame) -> dict:
    """
    Report null values without imputing or dropping them.
    """

    return {
        column: int(count)
        for column, count in df.isna().sum().items()
        if count > 0
    }


# ============================================================
# STEP 4: PRODUCT NAME NORMALIZATION
# ============================================================

def normalize_text(value):
    """
    Normalize text for later comparison.

    Preserve product quantities, flavors,
    brands, and other meaningful information.
    """

    if pd.isna(value):
        return pd.NA

    text = str(value).strip().casefold()

    # Unicode normalization.
    text = unicodedata.normalize(
        "NFKC",
        text,
    )

    # Normalize separators.
    text = re.sub(
        r"[|;,_]+",
        " ",
        text,
    )

    # Normalize punctuation spacing.
    text = re.sub(
        r"\s*([/().-])\s*",
        r"\1",
        text,
    )

    # Remove repeated whitespace.
    text = re.sub(
        r"\s+",
        " ",
        text,
    ).strip()

    return text


def standardize_product_names(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """Preserve original name and add a normalized column.
    """

    df = df.copy()

    df["product_name_normalized"] = (
        df["product_name"]
        .map(normalize_text)
        .astype("string")
    )

    return df


# ============================================================
# STEP 5: NUMERICAL VALIDATION
# ============================================================

def find_numerical_anomalies(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Flag suspicious values without modifying them.
    """

    checks = pd.DataFrame(index=df.index)

    checks["nonpositive_quantity"] = (
        df["quantity"] <= 0
    )

    checks["negative_old_price"] = (
        df["old_price"] < 0
    )

    checks["negative_new_price"] = (
        df["new_price"] < 0
    )

    # Diagnostic: may be legitimate in some cases.
    checks["new_price_exceeds_old_price"] = (
        df["new_price"] > df["old_price"]
    )

    checks = checks.fillna(False).astype(bool)

    affected_rows = checks.any(axis=1)

    anomalies = df.loc[
        affected_rows
    ].copy()

    for column in checks.columns:
        anomalies[column] = checks.loc[
            affected_rows, column
        ]

    return anomalies


# ============================================================
# STEP 6: CATEGORICAL PREPARATION
# ============================================================

def prepare_categorical_features(
    df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep identifiers and categories in consistent formats.

    Do not apply model-specific encoding yet.
    """

    df = df.copy()

    # Convert stable categorical attributes.
    # Keep identifiers as strings for cross-table joins.
    df["date_type"] = df["date_type"].astype("string")
    df["timeslot"] = df["timeslot"].astype("string")

    # regular_client remains a nullable boolean.
    # sku, ean_code, pickup_point and shop_branch
    # remain string identifiers.

    return df


# ============================================================
# STEP 7: VALIDATION REPORT
# ============================================================

def calculate_totals(df: pd.DataFrame) -> dict:
    """
    Calculate transaction totals.

    Assumes old_price and new_price are unit prices.
    """

    quantity_total = df["quantity"].sum()

    sales_value = (
        df["quantity"] * df["new_price"]
    ).sum()

    original_value = (
        df["quantity"] * df["old_price"]
    ).sum()

    return {
        "quantity_total": int(quantity_total),
        "sales_value": float(sales_value),
        "original_value": float(original_value),
    }


def validate_cleaning(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    exact_duplicates_removed: int,
    key_conflicts: pd.DataFrame,
    repeated_order_eans: pd.DataFrame,
    numerical_anomalies: pd.DataFrame,
) -> dict:

    totals_before = calculate_totals(original)
    totals_after = calculate_totals(cleaned)

    original_orders = original["order_id"].nunique()
    cleaned_orders = cleaned["order_id"].nunique()

    report = {
        "rows_before": int(len(original)),
        "rows_after": int(len(cleaned)),

        "exact_duplicates_removed": int(
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

        "repeated_order_ean_rows": int(
            len(repeated_order_eans)
        ),

        "numerical_anomaly_rows": int(
            len(numerical_anomalies)
        ),

        "null_counts_after": inspect_nulls(cleaned),

        "unique_orders_before": int(original_orders),
        "unique_orders_after": int(cleaned_orders),

        "totals_before": totals_before,
        "totals_after": totals_after,

        "quantity_difference": int(
            totals_after["quantity_total"]
            - totals_before["quantity_total"]
        ),

        "sales_value_difference": float(
            totals_after["sales_value"]
            - totals_before["sales_value"]
        ),

        "decisions": [
            "Removed exact duplicates only",
            "Preserved repeated EANs within orders",
            "Preserved original SKU and EAN identifiers",
            "Preserved original product names",
            "Created normalized product name column",
            "Standardized dates and categorical values",
            "Did not impute missing values",
            "Did not encode categorical identifiers",
            "Flagged suspicious numerical values",
        ],
    }

    return report


# ============================================================
# MAIN CLEANING FUNCTION
# ============================================================

def clean_customer_orders(
    source: str | Path | pd.DataFrame,
):
    """
    Run the Customer Orders cleaning pipeline.

    Returns
    -------
    cleaned_df: pd.DataFrame
    report: dict
    anomalies: dict[str, pd.DataFrame]
    """

    # STEP 0: LOAD DATA
    df = load_table(source)
    validate_schema(df)

    # STEP 1: STANDARDIZE DATA TYPES
    df = standardize_types(df)

    original = df.copy()

    # STEP 2: REMOVE EXACT DUPLICATES
    (
        df,
        n_removed,
        key_conflicts,
        repeated_order_eans,
    ) = clean_duplicates(df)

    # STEP 3: MISSING VALUES
    # No automatic imputation.

    # STEP 4: PRODUCT NAME NORMALIZATION
    df = standardize_product_names(df)

    # STEP 5: NUMERICAL VALIDATION
    numerical_anomalies = find_numerical_anomalies(df)

    # STEP 6: CATEGORICAL PREPARATION
    df = prepare_categorical_features(df)

    # STEP 7: FINAL VALIDATION
    report = validate_cleaning(
        original=original,
        cleaned=df,
        exact_duplicates_removed=n_removed,
        key_conflicts=key_conflicts,
        repeated_order_eans=repeated_order_eans,
        numerical_anomalies=numerical_anomalies,
    )

    # COLLECT ANOMALY TABLES
    anomalies = {
        "key_conflicts": key_conflicts,
        "repeated_order_eans": repeated_order_eans,
        "numerical_anomalies": numerical_anomalies,
    }

    return df, report, anomalies
