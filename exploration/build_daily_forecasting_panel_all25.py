import os
from pathlib import Path

import duckdb


DATA_DIR = Path(
    os.environ.get("HHM_DATA_DIR", str(Path.home() / "Downloads"))
)

SOURCE_FILE = Path(
    os.environ.get(
        "HHM_SOURCE_FILE",
        str(DATA_DIR / "pickup_point_orders_2024-01_2026-08.parquet"),
    )
)

OUTPUT_FILE = Path(
    os.environ.get(
        "HHM_PANEL_FILE",
        str(DATA_DIR / "hhm_daily_forecasting_panel_all25.parquet"),
    )
)

START_DATE = "2024-06-01"
END_DATE = "2026-08-31"

if not SOURCE_FILE.exists():
    raise FileNotFoundError(SOURCE_FILE)


def sql_path(path: Path) -> str:
    return "'" + str(path).replace("'", "''") + "'"


con = duckdb.connect()

con.execute(f"""
    CREATE VIEW pickup_point_orders AS
    SELECT *
    FROM read_parquet({sql_path(SOURCE_FILE)})
""")


# Include every pickup point found in the source.
locations = con.execute("""
    SELECT DISTINCT pickup_point
    FROM pickup_point_orders
    WHERE pickup_point IS NOT NULL
    ORDER BY pickup_point
""").df()

if locations.empty:
    raise RuntimeError("No pickup points found.")

print("\n========== PICKUP POINTS INCLUDED ==========")
print(f"Total pickup points: {len(locations)}")
print(locations.to_string(index=False))


# Build a complete calendar for every pickup point.
query = f"""
COPY (
    WITH all_locations AS (
        SELECT DISTINCT pickup_point
        FROM pickup_point_orders
        WHERE pickup_point IS NOT NULL
    ),

    calendar AS (
        SELECT
            l.pickup_point,
            CAST(d.day AS DATE) AS order_date
        FROM all_locations l
        CROSS JOIN generate_series(
            DATE '{START_DATE}',
            DATE '{END_DATE}',
            INTERVAL 1 DAY
        ) AS d(day)
    ),

    daily_observations AS (
        SELECT
            order_date,
            pickup_point,
            SUM(quantity_ordered) AS daily_quantity,
            COUNT(*) AS product_rows,
            COUNT(DISTINCT ean_code) AS distinct_ean_codes
        FROM pickup_point_orders
        WHERE order_date BETWEEN DATE '{START_DATE}' AND DATE '{END_DATE}'
          AND pickup_point IS NOT NULL
        GROUP BY order_date, pickup_point
    )

    SELECT
        c.order_date,
        c.pickup_point,
        d.daily_quantity,
        d.product_rows,
        d.distinct_ean_codes,
        (d.order_date IS NOT NULL) AS is_observed,
        EXTRACT(DOW FROM c.order_date) AS day_of_week,
        EXTRACT(MONTH FROM c.order_date) AS month,
        EXTRACT(YEAR FROM c.order_date) AS year,
        EXTRACT(DOW FROM c.order_date) = 0 AS is_sunday
    FROM calendar c
    LEFT JOIN daily_observations d
      ON c.order_date = d.order_date
     AND c.pickup_point = d.pickup_point
    ORDER BY c.pickup_point, c.order_date
)
TO {sql_path(OUTPUT_FILE)}
(FORMAT PARQUET, COMPRESSION ZSTD)
"""

con.execute(query)


print("\n========== PANEL SUMMARY ==========")

summary = con.execute(f"""
    SELECT
        COUNT(*) AS total_point_days,
        COUNT(DISTINCT pickup_point) AS pickup_points,
        COUNT(*) FILTER (WHERE is_observed) AS observed_days,
        COUNT(*) FILTER (WHERE NOT is_observed) AS missing_days,
        ROUND(
            100.0 * COUNT(*) FILTER (WHERE is_observed) / COUNT(*),
            2
        ) AS coverage_pct,
        MIN(order_date) AS first_date,
        MAX(order_date) AS last_date,
        SUM(daily_quantity) FILTER (WHERE is_observed)
            AS total_observed_quantity
    FROM read_parquet({sql_path(OUTPUT_FILE)})
""").df()

print(summary.to_string(index=False))


print("\n========== COVERAGE BY PICKUP POINT ==========")

by_point = con.execute(f"""
    SELECT
        pickup_point,
        COUNT(*) AS calendar_days,
        COUNT(*) FILTER (WHERE is_observed) AS observed_days,
        COUNT(*) FILTER (WHERE NOT is_observed) AS missing_days,
        ROUND(
            100.0 * COUNT(*) FILTER (WHERE is_observed) / COUNT(*),
            2
        ) AS coverage_pct,
        MIN(order_date) FILTER (WHERE is_observed) AS first_observed_date,
        MAX(order_date) FILTER (WHERE is_observed) AS last_observed_date,
        ROUND(AVG(daily_quantity) FILTER (WHERE is_observed), 1)
            AS mean_daily_quantity
    FROM read_parquet({sql_path(OUTPUT_FILE)})
    GROUP BY pickup_point
    ORDER BY observed_days DESC, pickup_point
""").df()

print(by_point.to_string(index=False))


print(f"\nNew panel saved to: {OUTPUT_FILE}")

con.close()
