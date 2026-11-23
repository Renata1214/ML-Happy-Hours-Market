"""Clean and enrich HHM product reference using ALREADY-CLEANED snapshots.

This is the sole owner of reference enrichment. Run the snapshot cleaner first.
The caller saves the three returned objects (reference, report, anomalies).
"""

from __future__ import annotations

import re
import unicodedata
from pathlib import Path

import pandas as pd

from src.clean_product_snapshots import load_table

PRIMARY_KEY = ["ean_code"]
REQUIRED_COLUMNS = [
    "ean_code", "product_concept", "main_category", "bio", "unit", "unit_quantity",
]


def validate_schema(df: pd.DataFrame) -> None:
    missing = sorted(set(REQUIRED_COLUMNS) - set(df.columns))
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def parse_bio(value):
    """Retain True, False and missing; reject unexpected values."""
    if pd.isna(value):
        return pd.NA
    if isinstance(value, bool):
        return value
    value = str(value).strip().lower()
    if value in {"true", "1"}:
        return True
    if value in {"false", "0"}:
        return False
    raise ValueError(f"Unexpected bio value: {value!r}")


def standardize_types(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ean_code"] = df["ean_code"].astype("string").str.strip()
    for column in ["product_concept", "main_category", "unit"]:
        df[column] = df[column].astype("string").str.strip().replace("", pd.NA)
    df["bio"] = df["bio"].map(parse_bio).astype("boolean")
    df["unit_quantity"] = pd.to_numeric(
        df["unit_quantity"], errors="raise"
    ).astype("float64")
    return df


def clean_ean_duplicates(df: pd.DataFrame):
    """Discard missing IDs and exact duplicates, merge complementary EAN records.

    Non-identical records with conflicting non-null attributes are preserved
    and reported; reference enrichment subsequently refuses ambiguous keys.
    """
    df = df.copy()
    missing_ean = df["ean_code"].isna() | df["ean_code"].eq("").fillna(False)
    missing_ean_rows = df.loc[missing_ean].copy()
    df = df.loc[~missing_ean].copy()

    exact_duplicates_removed = int(df.duplicated().sum())
    df = df.drop_duplicates().copy()

    duplicate_mask = df.duplicated(subset=PRIMARY_KEY, keep=False)
    duplicate_groups = df.loc[duplicate_mask]
    unique_records = df.loc[~duplicate_mask].copy()

    merged_records = []
    conflicting_records = []
    attributes = [col for col in REQUIRED_COLUMNS if col != "ean_code"]

    for _, group in duplicate_groups.groupby("ean_code"):
        has_conflict = any(
            group[column].dropna().nunique() > 1 for column in attributes
        )
        if has_conflict:
            conflicting_records.append(group)
            continue
        merged = group.iloc[0].copy()
        for column in attributes:
            nonnull = group[column].dropna()
            if not nonnull.empty:
                merged[column] = nonnull.iloc[0]
        merged_records.append(merged)

    if merged_records:
        df = pd.concat(
            [unique_records, pd.DataFrame(merged_records)], ignore_index=True
        )
    else:
        df = unique_records.reset_index(drop=True)

    if conflicting_records:
        conflicts = pd.concat(conflicting_records, ignore_index=True)
        df = pd.concat([df, conflicts], ignore_index=True)
    else:
        conflicts = pd.DataFrame(columns=REQUIRED_COLUMNS)

    return (
        df,
        missing_ean_rows,
        exact_duplicates_removed,
        conflicts,
        len(merged_records),
    )


def normalize_text(value):
    """Normalize names/units without merging distinct product descriptions."""
    if pd.isna(value):
        return pd.NA
    text = unicodedata.normalize("NFKC", str(value))
    text = text.strip().casefold()
    text = re.sub(r"[|;,_]+", " ", text)
    text = re.sub(r"\s*([/().-])\s*", r"\1", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or pd.NA


def normalize_category(value):
    """Make category labels comparable across reference and snapshots."""
    if pd.isna(value):
        return pd.NA
    text = unicodedata.normalize("NFKD", str(value).strip().casefold())
    text = "".join(char for char in text if not unicodedata.combining(char))
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text or pd.NA


def reconcile_reference_categories(
    product_reference: pd.DataFrame,
    product_snapshots: pd.DataFrame,
):
    """Fill unambiguous missing categories; append EANs missing from reference.

    Keep existing non-null master categories unchanged, flag disagreements,
    and preserve missing non-category attributes of newly inserted EANs.
    """
    ref = product_reference.copy()
    snap = product_snapshots.copy()

    required_snap = {"ean_code", "category"}
    missing_snap = sorted(required_snap - set(snap.columns))
    if missing_snap:
        raise ValueError(f"Missing snapshot columns for enrichment: {missing_snap}")

    for table in (ref, snap):
        table["ean_code"] = table["ean_code"].astype("string").str.strip()

    if ref["ean_code"].duplicated().any():
        raise ValueError("Duplicate EANs found in product reference.")

    ref["main_category"] = ref["main_category"].astype("string")
    snap["category"] = snap["category"].astype("string")
    ref["_category_norm"] = ref["main_category"].map(normalize_category).astype("string")
    snap["_category_norm"] = snap["category"].map(normalize_category).astype("string")

    valid_categories = (
        snap.loc[
            snap["ean_code"].notna() & snap["_category_norm"].notna(),
            ["ean_code", "_category_norm"],
        ]
        .drop_duplicates()
    )

    category_counts = (
        valid_categories.groupby("ean_code")["_category_norm"]
        .nunique()
        .rename("n_snapshot_categories")
    )

    unique_categories = valid_categories.loc[
        valid_categories["ean_code"].isin(
            category_counts[category_counts == 1].index
        )
    ].rename(columns={"_category_norm": "snapshot_category"})

    snapshot_eans = snap[["ean_code"]].dropna().drop_duplicates()
    missing_eans = snapshot_eans.loc[
        ~snapshot_eans["ean_code"].isin(ref["ean_code"])
    ].copy()

    new_records = pd.DataFrame(
        pd.NA,
        index=range(len(missing_eans)),
        columns=product_reference.columns,
    )
    new_records["ean_code"] = missing_eans["ean_code"].reset_index(drop=True)
    ref["reference_record_added"] = False
    new_records["reference_record_added"] = True
    ref = pd.concat([ref, new_records], ignore_index=True)

    ref = ref.merge(
        category_counts,
        on="ean_code", how="left", validate="one_to_one",
    )
    ref = ref.merge(
        unique_categories,
        on="ean_code", how="left", validate="one_to_one",
    )
    ref["n_snapshot_categories"] = (
        ref["n_snapshot_categories"].fillna(0).astype(int)
    )

    missing_category = ref["main_category"].isna()
    can_impute = (
        missing_category
        & ref["snapshot_category"].notna()
        & (ref["n_snapshot_categories"] == 1)
    )
    ref["category_imputed"] = can_impute
    ref.loc[can_impute, "main_category"] = ref.loc[
        can_impute, "snapshot_category"
    ]

    mismatch = (
        ~missing_category
        & ref["snapshot_category"].notna()
        & (ref["_category_norm"] != ref["snapshot_category"])
    ).fillna(False)
    ambiguous = ref["n_snapshot_categories"] > 1
    ref["category_conflict"] = mismatch | ambiguous

    category_conflicts = ref.loc[
        ref["category_conflict"],
        [
            "ean_code", "main_category", "snapshot_category",
            "n_snapshot_categories", "reference_record_added",
        ],
    ].copy()
    category_conflicts["conflict_type"] = "category_mismatch"
    category_conflicts.loc[
        category_conflicts["n_snapshot_categories"] > 1,
        "conflict_type",
    ] = "multiple_snapshot_categories"

    category_report = {
        "reference_rows_before": int(len(product_reference)),
        "reference_rows_after": int(len(ref)),
        "new_eans_added": int(len(missing_eans)),
        "existing_categories_imputed": int(
            (can_impute & ~ref["reference_record_added"]).sum()
        ),
        "new_eans_with_category": int(
            (can_impute & ref["reference_record_added"]).sum()
        ),
        "missing_categories_after": int(ref["main_category"].isna().sum()),
        "category_mismatches": int(mismatch.sum()),
        "ambiguous_snapshot_eans": int(ambiguous.sum()),
    }

    ref = ref.drop(
        columns=["_category_norm", "snapshot_category", "n_snapshot_categories"]
    )
    return ref, category_report, category_conflicts




def align_snapshot_categories_to_reference(
    snapshots: pd.DataFrame,
    reference: pd.DataFrame,
):
    """
    Resolve snapshot category conflicts using product reference
    as the authoritative category source.

    Only updates snapshot rows when:
        1. The EAN exists in reference.
        2. Reference has category_conflict=True.
        3. Both categories are non-null.
        4. Normalized categories differ.

    Preserves original snapshot categories for corrected rows.

    Returns:
        corrected_snapshots
        corrections_report
    """

    snapshots = snapshots.copy()
    reference = reference.copy()

    if not reference["ean_code"].is_unique:
        raise ValueError(
            "Reference EAN must be unique before alignment."
        )

    # STEP 1: Build EAN-based lookup tables.
    ref_lookup = reference.set_index("ean_code")

    reference_category = snapshots["ean_code"].map(
        ref_lookup["main_category"]
    ).astype("string")

    reference_conflict = snapshots["ean_code"].map(
        ref_lookup["category_conflict"]
    ).fillna(False).astype(bool)

    # STEP 2: Normalize both categories for comparison.
    snapshot_category_norm = (
        snapshots["category"]
        .map(normalize_category)
        .astype("string")
    )

    reference_category_norm = (
        reference_category
        .map(normalize_category)
        .astype("string")
    )

    # STEP 3: Identify actual category disagreements.
    correction_mask = (
        reference_conflict
        & snapshot_category_norm.notna()
        & reference_category_norm.notna()
        & (
            snapshot_category_norm
            != reference_category_norm
        ).fillna(False)
    )

    # STEP 4: Record corrections before modifying data.
    corrections = snapshots.loc[
        correction_mask,
        ["snapshot_date", "sku", "ean_code", "category"],
    ].copy()

    corrections = corrections.rename(
        columns={"category": "old_snapshot_category"}
    )

    corrections["reference_category"] = (
        reference_category.loc[correction_mask].to_numpy()
    )

    corrections["reason"] = (
        "Snapshot category aligned with authoritative reference"
    )

    # STEP 5: Preserve original category for changed rows.
    if "category_before_reference_alignment" not in snapshots:
        snapshots["category_before_reference_alignment"] = (
            pd.Series(
                pd.NA,
                index=snapshots.index,
                dtype="string",
            )
        )

    if "category_replaced_from_reference" not in snapshots:
        snapshots["category_replaced_from_reference"] = False

    # Only store the original value if not recorded previously.
    save_original_mask = (
        correction_mask
        & snapshots[
            "category_before_reference_alignment"
        ].isna()
    )

    snapshots.loc[
        save_original_mask,
        "category_before_reference_alignment",
    ] = snapshots.loc[
        save_original_mask,
        "category",
    ]

    # STEP 6: Replace category with reference category.
    snapshots.loc[
        correction_mask,
        "category",
    ] = reference_category.loc[
        correction_mask
    ].to_numpy()

    snapshots.loc[
        correction_mask,
        "category_replaced_from_reference",
    ] = True

    # STEP 7: Refresh normalized category column.
    snapshots["category_normalized"] = (
        snapshots["category"]
        .map(normalize_category)
        .astype("string")
    )

    # STEP 8: Validate corrections.
    if len(snapshots) != len(
        snapshots.index
    ):
        raise ValueError("Unexpected row-count inconsistency.")

    remaining_mismatch = (
        snapshots.loc[
            correction_mask, "category_normalized"
        ]
        != reference_category_norm.loc[correction_mask]
    ).fillna(False)

    if remaining_mismatch.any():
        raise ValueError(
            "Category alignment failed for some snapshot rows."
        )

    return snapshots, corrections.reset_index(drop=True)


def normalize_reference_attributes(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["product_concept_normalized"] = (
        df["product_concept"].map(normalize_text).astype("string")
    )
    # The same category normalization rules used by reconciliation.
    df["main_category_normalized"] = (
        df["main_category"].map(normalize_category).astype("string")
    )
    df["unit_normalized"] = df["unit"].map(normalize_text).astype("string")
    return df


def prepare_missing_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["main_category_for_ml"] = df["main_category_normalized"].fillna("unknown")
    df["unit_for_ml"] = df["unit_normalized"].fillna("unknown")
    # bio remains nullable boolean, unit_quantity remains float with NaN.
    return df


def inspect_nulls(df: pd.DataFrame) -> dict:
    return {
        col: int(count)
        for col, count in df[REQUIRED_COLUMNS].isna().sum().items()
    }


def validate_cleaning(
    original: pd.DataFrame,
    cleaned: pd.DataFrame,
    exact_duplicates_removed: int,
    removed_missing_eans: int,
    merged_duplicate_eans: int,
    conflicting_duplicate_eans: int,
    category_report: dict | None,
) -> dict:
    return {
        "rows_before": int(len(original)),
        "rows_after": int(len(cleaned)),
        "unique_eans_before": int(original["ean_code"].nunique()),
        "unique_eans_after": int(cleaned["ean_code"].nunique()),
        "primary_key_unique": bool(cleaned["ean_code"].is_unique),
        "exact_duplicates_removed": exact_duplicates_removed,
        "missing_ean_rows_removed": removed_missing_eans,
        "duplicate_ean_groups_merged": merged_duplicate_eans,
        "conflicting_duplicate_eans": conflicting_duplicate_eans,
        "missing_values_before": inspect_nulls(original),
        "missing_values_after": inspect_nulls(cleaned),
        "product_concepts_before": int(original["product_concept"].nunique()),
        "product_concepts_normalized": int(
            cleaned["product_concept_normalized"].nunique()
        ),
        "categories_before": int(original["main_category"].nunique()),
        "categories_normalized": int(
            cleaned["main_category_normalized"].nunique()
        ),
        "category_reconciliation": category_report,
        "decisions": [
            "Removed rows with missing EAN",
            "Removed exact duplicate records",
            "Merged compatible duplicate EAN records",
            "Preserved unresolved conflicting EAN records",
            "Preserved original product concepts",
            "Created normalized categorical attributes",
            "Preserved unknown bio values",
            "Preserved missing unit quantities",
            "Added Unknown modeling-category columns",
            "Did not apply final ML encoding",
        ],
    }


def clean_product_reference(
    source: str | Path | pd.DataFrame,
    snapshot_source: str | Path | pd.DataFrame | None = None,
):
    """Clean product reference, then optionally enrich from cleaned snapshots.

    Returns: cleaned_reference, cleaning_report, anomaly_tables
    """
    df = load_table(source)
    validate_schema(df)
    df = standardize_types(df)
    original = df.copy()

    (
        df,
        missing_ean_rows,
        exact_duplicates_removed,
        duplicate_ean_conflicts,
        merged_duplicate_eans,
    ) = clean_ean_duplicates(df)

    category_report = None
    category_conflicts = pd.DataFrame()

    corrected_snapshots = None
    snapshot_category_corrections = pd.DataFrame()
    snapshots = None

    if snapshot_source is not None:
        if not df["ean_code"].is_unique:
            raise ValueError(
                "Reference contains unresolved duplicate EANs. "
                "Resolve these conflicts before enrichment."
            )
        snapshots = load_table(snapshot_source)
        df, category_report, category_conflicts = reconcile_reference_categories(
            product_reference=df,
            product_snapshots=snapshots,
        )
        df = standardize_types(df)

    # Derived columns must be computed AFTER category enrichment.
    df = normalize_reference_attributes(df)
    df = prepare_missing_features(df)

    # STEP: ALIGN SNAPSHOT CATEGORIES TO REFERENCE

    if snapshots is not None:
        (
            corrected_snapshots,
            snapshot_category_corrections,
        ) = align_snapshot_categories_to_reference(
            snapshots=snapshots,
            reference=df,
        )

    report = validate_cleaning(
        original=original,
        cleaned=df,
        exact_duplicates_removed=exact_duplicates_removed,
        removed_missing_eans=len(missing_ean_rows),
        merged_duplicate_eans=merged_duplicate_eans,
        conflicting_duplicate_eans=int(
            duplicate_ean_conflicts["ean_code"].nunique()
        ),
        category_report=category_report,
    )
    report["snapshot_category_alignment"] = {
        "rows_corrected": int(
            len(snapshot_category_corrections)
        ),
        "unique_eans_corrected": int(
            snapshot_category_corrections["ean_code"].nunique()
        ),
        "unique_skus_corrected": int(
            snapshot_category_corrections["sku"].nunique()
        ),
    }
    anomalies = {
        "missing_ean_rows": missing_ean_rows,
        "duplicate_ean_conflicts": duplicate_ean_conflicts,
        "category_conflicts": category_conflicts,
        "snapshot_category_corrections": snapshot_category_corrections,
    }
    return df, corrected_snapshots, report, anomalies
