"""Shared order preparation, RFM segmentation, and pickup-point analysis."""

from pathlib import Path
import os

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "outputs" / "question5" / "dashboard.sqlite"
TRAIN_START = pd.Timestamp("2025-03-01")
VALIDATION_START = pd.Timestamp("2026-02-01")
TEST_START = pd.Timestamp("2026-05-01")
TEST_END = pd.Timestamp("2026-09-01")
SPLITS = {
    "train": (TRAIN_START, VALIDATION_START),
    "validation": (VALIDATION_START, TEST_START),
    "test": (TEST_START, TEST_END),
}
RFM_COLUMNS = ["recency", "frequency", "monetary"]
SEGMENT_REPORT_GROUPS = ["scope", "pickup_point", "segment"]


def load_basket_lines():
    # Step 1: Locate the customer orders and product reference files
    order_path = os.getenv("HHM_CUSTOMER_ORDERS")
    if order_path is None:
        candidates = sorted((ROOT / "data").glob("customer_orders*.parquet"))
        if not candidates:
            raise FileNotFoundError("Set HHM_CUSTOMER_ORDERS to a customer orders parquet file.")
        order_path = candidates[-1]
    reference_path = os.getenv("HHM_PRODUCT_REFERENCE", ROOT / "data" / "product_reference.parquet")

    # Step 2: Read the basket lines and check that the required values are usable
    fields = ["order_id", "user_id", "order_datetime", "pickup_point", "ean_code", "quantity", "new_price"]
    lines = pd.read_parquet(order_path, columns=fields)
    lines["order_datetime"] = pd.to_datetime(lines["order_datetime"])
    if lines.empty or lines[fields].isna().any().any():
        raise ValueError("Customer orders must contain complete required fields and at least one row.")
    if (lines["quantity"] <= 0).any() or (lines["new_price"] < 0).any():
        raise ValueError("Check nonpositive quantities or negative selling prices before aggregating.")

    # Step 3: Use the EAN to attach a product concept where the reference has a match
    reference = pd.read_parquet(reference_path, columns=["ean_code", "product_concept"])
    reference = reference.dropna().drop_duplicates()
    lines = lines.merge(reference, on="ean_code", how="left", validate="many_to_one")

    # Step 4: Estimate sales for each basket line from quantity and unit price
    lines["sales"] = lines["quantity"] * lines["new_price"]
    return lines


def aggregate_orders(lines, *, include_pickup=True):
    # Step 1: Check that order details agree across basket lines
    groups = lines.groupby("order_id", sort=False)
    attributes = ["user_id", "order_datetime"]
    if include_pickup:
        attributes.append("pickup_point")
    if groups[attributes].nunique().gt(1).any().any():
        raise ValueError("An order ID has conflicting customer, timestamp, or pickup values.")

    # Step 2: Combine the basket lines into one row per order and sum their sales
    order_details = {attribute: (attribute, "first") for attribute in attributes}
    orders = groups.agg(
        **order_details,
        sales=("sales", "sum"),
    ).reset_index()

    # Step 3: Sort orders by time, using the order ID to break ties
    return orders.sort_values(["order_datetime", "order_id"], ignore_index=True)


def observation_end(orders):
    """Return the exclusive end of the complete observation window."""
    # Step 1: Use the last order date unless the extract declares a later complete day
    last_order = orders["order_datetime"].max()
    complete_day = pd.Timestamp(os.getenv("HHM_OBSERVED_THROUGH", last_order.date()))

    # Step 2: Use midnight after the complete day as the exclusive cutoff
    end = complete_day.normalize() + pd.Timedelta(days=1)

    # Step 3: Check that the cutoff includes every order in the extract
    if last_order >= end:
        raise ValueError("HHM_OBSERVED_THROUGH must include every supplied order date.")
    return end


def segment_customers(orders, *, reference_date=None):
    # Step 1: Calculate recency, frequency, and monetary value for each customer
    rfm = orders.groupby("user_id").agg(
        last_purchase=("order_datetime", "max"),
        frequency=("order_id", "size"),
        monetary=("sales", "sum"),
    )
    if reference_date is None:
        reference_date = orders["order_datetime"].max() + pd.Timedelta(days=1)
    rfm["recency"] = (reference_date - rfm["last_purchase"]).dt.days

    # Step 2: Apply log1p and standardize the three features before clustering
    values = StandardScaler().fit_transform(np.log1p(rfm[RFM_COLUMNS]))

    # Step 3: Fit K-means and assign each customer to one of three clusters
    if len(np.unique(values, axis=0)) < 3:
        raise ValueError("The three-segment RFM model requires at least three distinct customer profiles.")
    rfm["cluster"] = KMeans(n_clusters=3, random_state=42, n_init=20).fit_predict(values)

    # Step 4: Name the active cluster by recency and the other two by spending
    profile = rfm.groupby("cluster")[RFM_COLUMNS].median()
    active = profile.sort_values(["recency", "monetary"], ascending=[True, False]).index[0]
    moderate, inactive = profile.drop(active).sort_values("monetary", ascending=False).index
    rfm["segment"] = rfm["cluster"].map({
        active: "Active High-Value",
        moderate: "Less-Recent Moderate-Value",
        inactive: "Long-Inactive Low-Value",
    })

    # Step 5: Summarize the size, typical RFM values, and sales share of each segment
    profiles = rfm.groupby("segment", as_index=False).agg(
        customers=("recency", "size"),
        median_recency=("recency", "median"),
        median_frequency=("frequency", "median"),
        median_monetary=("monetary", "median"),
        total_monetary=("monetary", "sum"),
    )
    profiles["customer_share_pct"] = 100 * profiles["customers"] / profiles["customers"].sum()
    total = profiles["total_monetary"].sum()
    profiles["revenue_share_pct"] = 100 * profiles["total_monetary"] / total if total else 0.0
    return rfm.reset_index()[["user_id", "segment", *RFM_COLUMNS]], profiles


def segment_customers_as_of(orders, as_of_date):
    """Fit RFM using one row per order and history through the selected Brussels date."""
    # Step 1: Use Brussels calendar dates for timezone-aware timestamps
    as_of = pd.Timestamp(as_of_date).normalize()
    if orders["order_datetime"].dt.tz is not None:
        local_times = orders["order_datetime"].dt.tz_convert("Europe/Brussels").dt.tz_localize(None)
        orders = orders.assign(order_datetime=local_times)

    # Step 2: Keep all orders before the midnight after the selected day
    cutoff = as_of + pd.Timedelta(days=1)
    history = orders.loc[orders["order_datetime"].lt(cutoff)]
    if history.empty:
        raise ValueError(f"No customer orders are available through {as_of.date()}.")

    # Step 3: Match the notebook's recency calculation and advance it on days without orders
    last_order = history["order_datetime"].max()
    reference_date = last_order + pd.Timedelta(days=1) + (as_of - last_order.normalize())
    customers, profiles = segment_customers(history, reference_date=reference_date)
    return customers.assign(as_of_date=as_of), profiles.assign(as_of_date=as_of)


def build_customer_profiles(order_rows, as_of_date):
    """Build the customer profiles asset from silver orders or basket lines in Pandas."""
    # Step 1: Select the fields used for RFM and get the sales value for each row
    fields = ["order_id", "user_id", "order_datetime"]
    sales = order_rows["sales"] if "sales" in order_rows else order_rows["quantity"] * order_rows["new_price"]
    lines = order_rows[fields].assign(
        order_datetime=pd.to_datetime(order_rows["order_datetime"]),
        sales=sales,
    )

    # Step 2: Aggregate by order so a multi-product basket counts as one purchase
    orders = aggregate_orders(lines, include_pickup=False)

    # Step 3: Fit RFM through the selected date and return one row per customer
    customers, _ = segment_customers_as_of(orders, as_of_date)
    return customers


def transform_customer_profiles(historical_orders, as_of_date):
    """Adapt the team's Polars silver table to the customer profiles asset."""
    import polars as pl

    # Step 1: Convert the silver orders to Pandas for the existing RFM code
    order_rows = historical_orders.to_pandas()

    # Step 2: Compute the customer segments for the selected date
    customers = build_customer_profiles(order_rows, as_of_date)

    # Step 3: Return the profiles to Dagster as a Polars table
    return pl.from_pandas(customers, schema_overrides={"as_of_date": pl.Date})


def summarize_pickup_points(orders, customers):
    """Summarize the supplied order window using the latest assigned segments."""

    # Step 1: Count each customer's activity at each pickup point
    visits = orders.groupby(["user_id", "pickup_point"], as_index=False).agg(
        orders=("order_id", "size"),
        last_visit=("order_datetime", "max"),
        sales=("sales", "sum"),
    )

    # Step 2: Identify each customer's primary pickup point
    primary = (visits.sort_values(
        ["orders", "last_visit", "pickup_point"], ascending=[False, False, True]
    ).drop_duplicates("user_id"))

    # Step 3: Calculate pickup-point performance
    summary = visits.groupby("pickup_point", as_index=False).agg(
        customers_served=("user_id", "size"), orders=("orders", "sum"), sales=("sales", "sum")
    )

    # Step 4: Count primary customers
    primary_counts = primary.groupby("pickup_point").size()
    summary["primary_customers"] = summary["pickup_point"].map(primary_counts).fillna(0).astype(int)
    
    # Step 5: Calculate percentages
    summary["primary_customer_share_pct"] = 100 * summary["primary_customers"] / len(primary)
    total_sales = summary["sales"].sum()
    summary["sales_share_pct"] = 100 * summary["sales"] / total_sales if total_sales else 0.0

    # Step 6: Calculate customer segment mix
    mix = (
        visits.merge(customers[["user_id", "segment"]], on="user_id", validate="many_to_one")
        .groupby(["pickup_point", "segment"], as_index=False)
        .agg(customers_served=("user_id", "size"), sales=("sales", "sum"))
    )

    return summary.sort_values("sales", ascending=False), mix, primary


def _segment_report_data(rows, customers, report_date, fields, start_date):
    """Select one period and prepare rows for overall and pickup-point summaries."""
    # Step 1: Set the reporting dates and include the full last day
    report_date = pd.Timestamp(report_date).normalize()
    first = rows["order_datetime"].min() if start_date is None else start_date
    window_start = pd.Timestamp(first).normalize()
    end = report_date + pd.Timedelta(days=1)

    # Step 2: Keep rows in the period and attach each customer's segment
    window = rows.loc[
        rows["order_datetime"].between(window_start, end, inclusive="left"),
        ["user_id", "pickup_point", *fields],
    ].merge(customers[["user_id", "segment"]], on="user_id", validate="many_to_one")

    # Step 3: Prepare one copy for the pickup-point view and one for the overall view
    scoped = pd.concat([
        window.assign(scope="pickup_point"),
        window.assign(scope="all", pickup_point=pd.NA),
    ], ignore_index=True)

    # Step 4: Record the reporting dates so both output tables use the same period
    period = {
        "report_date": report_date,
        "window_start": window_start,
        "window_days": (end - window_start).days,
    }
    return scoped, period


def summarize_segment_statistics(orders, customers, report_date, *, start_date=None):
    """Summarize distinct buyers and sales over the selected reporting period."""
    # Step 1: Prepare the orders and segment assignments for the period
    groups = SEGMENT_REPORT_GROUPS
    window, period = _segment_report_data(orders, customers, report_date, ["sales"], start_date)

    # Step 2: Count distinct customers and sum sales for each segment and location
    counts = window.groupby(groups, as_index=False, dropna=False).agg(
        num_customers=("user_id", "nunique"),
        provisional_sales=("sales", "sum"),
    )

    # Step 3: Show zero customers and sales for segments with no orders at a location
    locations = window.loc[window["scope"].eq("pickup_point"), groups[:2]].drop_duplicates()
    locations = pd.concat([
        locations, pd.DataFrame({"scope": ["all"], "pickup_point": [pd.NA]}),
    ], ignore_index=True)
    segment_names = customers[["segment"]].drop_duplicates()
    stats = locations.merge(segment_names, how="cross").merge(counts, on=groups, how="left")
    stats["num_customers"] = stats["num_customers"].fillna(0).astype(int)
    stats["provisional_sales"] = stats["provisional_sales"].fillna(0.0)

    # Step 4: Divide each segment's count by the customer total for the same view
    totals = stats.groupby(groups[:2], dropna=False)["num_customers"].transform("sum")
    stats["customer_share_pct"] = (100 * stats["num_customers"] / totals.where(totals.gt(0))).fillna(0)
    columns = [
        *period, *groups, "num_customers", "customer_share_pct", "provisional_sales",
    ]
    return stats.assign(**period)[columns].sort_values(groups, ignore_index=True).astype({
        "pickup_point": "string",
    })


def summarize_top_products(lines, customers, report_date, top_n=10, *, start_date=None):
    """Rank past purchases using customer segments assigned as of report_date."""
    # Step 1: Prepare the basket lines and product identifiers for the period
    fields = ["order_id", "product_concept", "ean_code"]
    groups = SEGMENT_REPORT_GROUPS
    window, period = _segment_report_data(lines, customers, report_date, fields, start_date)
    concept_by_ean = window.drop_duplicates("ean_code").set_index("ean_code")["product_concept"]

    # Step 2: Rank product concepts and individual EANs separately
    rankings = []
    for level, product in (("concept", "product_concept"), ("ean", "ean_code")):
        # Step 3: Count distinct customers and orders for each product in each group
        counts = window.dropna(subset=[product]).groupby([*groups, product], as_index=False, dropna=False).agg(
            num_customers=("user_id", "nunique"),
            num_orders=("order_id", "nunique"),
        )

        # Step 4: Rank by customers first, then orders and product name or barcode
        counts = counts.sort_values(
            [*groups, "num_customers", "num_orders", product],
            ascending=[True] * len(groups) + [False, False, True],
        )
        counts["rank"] = counts.groupby(groups, dropna=False).cumcount() + 1
        counts = counts.loc[counts["rank"].le(top_n)].copy()

        # Step 5: Keep the concept name on EAN rows for context
        if level == "ean":
            counts["product_concept"] = counts["ean_code"].map(concept_by_ean)
        else:
            counts["ean_code"] = pd.NA
        rankings.append(counts.assign(**period, product_level=level))

    # Step 6: Combine the two rankings and keep EANs as text
    columns = [
        *period, *groups, "product_level", "rank", "product_concept", "ean_code",
        "num_customers", "num_orders",
    ]
    return pd.concat(rankings, ignore_index=True)[columns].astype({
        "pickup_point": "string", "product_concept": "string", "ean_code": "string",
    })
