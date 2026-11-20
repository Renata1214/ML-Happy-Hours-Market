import os
from pathlib import Path
import argparse

import duckdb
import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


DATA_DIR = Path(
    os.environ.get("HHM_DATA_DIR", str(Path.home() / "Downloads"))
)

PANEL_FILE = Path(
    os.environ.get(
        "HHM_PANEL_FILE",
        str(DATA_DIR / "hhm_daily_forecasting_panel_all25.parquet"),
    )
)
OUTPUT_DIR = Path("results/hybrid_all25")

EVALUATION_WINDOWS = [
    ("Sep-Nov 2024", "2024-09-01", "2024-11-30"),
    ("Dec 2024-Feb 2025", "2024-12-01", "2025-02-28"),
    ("Mar-May 2025", "2025-03-01", "2025-05-31"),
    ("Jun-Aug 2025", "2025-06-01", "2025-08-31"),
    ("Sep-Nov 2025", "2025-09-01", "2025-11-30"),
    ("Dec 2025-Feb 2026", "2025-12-01", "2026-02-28"),
    ("Mar-May 2026", "2026-03-01", "2026-05-31"),
    ("Jun-Aug 2026", "2026-06-01", "2026-08-31"),
]

CATEGORICAL_FEATURES = [
    "pickup_point",
    "weekday",
]

NUMERIC_FEATURES = [
    "lag_1_observed",
    "lag_7_observed",
    "same_weekday_lag_1",
    "same_weekday_lag_2",
    "same_weekday_lag_3",
    "same_weekday_lag_4",
    "same_weekday_mean_4",
    "same_weekday_std_4",
    "recent_mean_7",
    "recent_mean_28",
    "recent_std_28",
    "recent_trend",
    "days_since_last_observation",
    "weekday_sin",
    "weekday_cos",
    "month_sin",
    "month_cos",
    "day_of_year_sin",
    "day_of_year_cos",
    "is_weekend",
]

FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


def build_features(observed):
    """Build each row using only observations before its target date."""

    feature_rows = []

    for point, group in observed.groupby("pickup_point"):
        group = group.sort_values("order_date").reset_index(drop=True)

        dates = group["order_date"].to_numpy()
        quantities = group["daily_quantity"].to_numpy(dtype=float)
        weekdays = group["weekday_number"].to_numpy(dtype=int)

        # Require 28 previous observed days before creating a row.
        for i in range(28, len(group)):
            target_date = pd.Timestamp(dates[i])
            history = quantities[:i]
            history_weekdays = weekdays[:i]

            same_weekday_indices = np.flatnonzero(
                history_weekdays == target_date.dayofweek
            )[-4:]

            if len(same_weekday_indices) < 4:
                continue

            same_weekday_values = history[same_weekday_indices]

            recent_7 = history[-7:]
            recent_28 = history[-28:]

            weekday_number = target_date.dayofweek
            month_number = target_date.month
            day_of_year = target_date.dayofyear

            feature_rows.append({
                "order_date": target_date,
                "pickup_point": point,
                "weekday": str(weekday_number),

                # Historical demand features
                "lag_1_observed": history[-1],
                "lag_7_observed": history[-7],

                "same_weekday_lag_1": same_weekday_values[-1],
                "same_weekday_lag_2": same_weekday_values[-2],
                "same_weekday_lag_3": same_weekday_values[-3],
                "same_weekday_lag_4": same_weekday_values[-4],
                "same_weekday_mean_4": same_weekday_values.mean(),
                "same_weekday_std_4": same_weekday_values.std(),

                "recent_mean_7": recent_7.mean(),
                "recent_mean_28": recent_28.mean(),
                "recent_std_28": recent_28.std(),
                "recent_trend": recent_7.mean() - recent_28.mean(),

                "days_since_last_observation": (
                    target_date - pd.Timestamp(dates[i - 1])
                ).days,

                # Calendar features
                "weekday_sin": np.sin(
                    2 * np.pi * weekday_number / 7
                ),
                "weekday_cos": np.cos(
                    2 * np.pi * weekday_number / 7
                ),
                "month_sin": np.sin(
                    2 * np.pi * month_number / 12
                ),
                "month_cos": np.cos(
                    2 * np.pi * month_number / 12
                ),
                "day_of_year_sin": np.sin(
                    2 * np.pi * day_of_year / 365.25
                ),
                "day_of_year_cos": np.cos(
                    2 * np.pi * day_of_year / 365.25
                ),
                "is_weekend": int(weekday_number >= 5),

                # Actual target and benchmark predictions
                "actual": quantities[i],
                "seasonal_naive": same_weekday_values[-1],
                "weekday_average_4": same_weekday_values.mean(),
            })

    return pd.DataFrame(feature_rows)


def make_model():
    preprocessing = ColumnTransformer(
        transformers=[
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
            ("numeric", "passthrough", NUMERIC_FEATURES),
        ]
    )

    regressor = HistGradientBoostingRegressor(
        loss="squared_error",
        learning_rate=0.05,
        max_iter=150,
        max_leaf_nodes=15,
        min_samples_leaf=20,
        l2_regularization=2.0,
        early_stopping=False,
        random_state=42,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("regressor", regressor),
    ])


def calculate_metrics(data, method):
    actual = data["actual"].to_numpy(dtype=float)
    predicted = data[method].to_numpy(dtype=float)
    error = predicted - actual

    return {
        "n_forecasts": len(data),
        "MAE": np.mean(np.abs(error)),
        "RMSE": np.sqrt(np.mean(error ** 2)),
        "WAPE_pct": (
            np.sum(np.abs(error)) / np.sum(actual) * 100
        ),
        "Bias": np.mean(error),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pilot",
        action="store_true",
        help="Evaluate only the first month of the first window.",
    )
    args = parser.parse_args()

    if not PANEL_FILE.exists():
        raise FileNotFoundError(PANEL_FILE)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading daily forecasting panel...")

    df = duckdb.query(f"""
        SELECT
            CAST(order_date AS DATE) AS order_date,
            pickup_point,
            daily_quantity,
            is_observed
        FROM read_parquet('{PANEL_FILE}')
    """).df()

    df["order_date"] = pd.to_datetime(df["order_date"])
    df["daily_quantity"] = pd.to_numeric(
        df["daily_quantity"], errors="coerce"
    )
    df["is_observed"] = df["is_observed"].astype(bool)
    df["weekday_number"] = df["order_date"].dt.dayofweek

    observed = df[
        df["is_observed"] & df["daily_quantity"].notna()
    ].copy()

    print(
        f"Observed rows: {len(observed):,}; "
        f"pickup points: {observed['pickup_point'].nunique()}"
    )

    print("\nBuilding historical features...")
    features = build_features(observed)

    if features.empty:
        raise RuntimeError("No feature rows were generated.")

    print(f"Feature rows: {len(features):,}")

    features["evaluation_window"] = None

    for name, start, end in EVALUATION_WINDOWS:
        mask = (
            (features["order_date"] >= pd.Timestamp(start))
            & (features["order_date"] <= pd.Timestamp(end))
        )
        features.loc[mask, "evaluation_window"] = name

    targets = features[
        features["evaluation_window"].notna()
    ].copy()

    targets = targets.sort_values(
        "order_date"
    ).reset_index(drop=True)

    evaluation_months = sorted(
        targets["order_date"].dt.to_period("M").unique()
    )

    if args.pilot:
        evaluation_months = evaluation_months[:1]
        print("\nPILOT MODE: first evaluation month only")

    prediction_rows = []
    training_log = []
    selection_rows = []

    MIN_HISTORY = 28

    for month in evaluation_months:
        month_start = month.start_time

        train = features[
            features["order_date"] < month_start
        ].copy()

        test = targets[
            targets["order_date"].dt.to_period("M") == month
        ].copy()

        if train.empty or test.empty:
            continue

        # Only predictions from completed earlier evaluation months
        # are available when selecting a model for this month.
        if prediction_rows:
            history = pd.concat(
                prediction_rows,
                ignore_index=True,
            )
        else:
            history = pd.DataFrame()

        selected_by_point = {}

        for point in test["pickup_point"].unique():

            historical = (
                history[
                    history["pickup_point"] == point
                ].copy()
                if not history.empty
                else pd.DataFrame()
            )

            history_n = len(historical)
            baseline_wape = np.nan
            ml_wape = np.nan

            if (
                history_n >= MIN_HISTORY
                and historical["actual"].sum() > 0
            ):
                baseline_wape = (
                    (
                        historical["weekday_average_4"]
                        - historical["actual"]
                    ).abs().sum()
                    / historical["actual"].sum()
                    * 100
                )

                ml_wape = (
                    (
                        historical["global_ml"]
                        - historical["actual"]
                    ).abs().sum()
                    / historical["actual"].sum()
                    * 100
                )

                selected_model = (
                    "global_ml"
                    if ml_wape < baseline_wape
                    else "weekday_average_4"
                )

                reason = "lower_historical_wape"
            else:
                selected_model = "weekday_average_4"
                reason = "insufficient_history"

            selected_by_point[point] = selected_model

            selection_rows.append({
                "month": str(month),
                "pickup_point": point,
                "historical_forecasts": history_n,
                "historical_baseline_WAPE": baseline_wape,
                "historical_global_ml_WAPE": ml_wape,
                "selected_model": selected_model,
                "selection_reason": reason,
            })

        model = make_model()

        model.fit(
            train[FEATURE_COLUMNS],
            train["actual"],
        )

        raw_predictions = model.predict(
            test[FEATURE_COLUMNS]
        )

        predictions = np.maximum(0.0, raw_predictions)

        test = test.copy()
        test["global_ml"] = predictions

        test["selected_model"] = (
            test["pickup_point"].map(selected_by_point)
        )

        test["hybrid"] = np.where(
            test["selected_model"] == "global_ml",
            test["global_ml"],
            test["weekday_average_4"],
        )

        prediction_rows.append(test)

        selected_counts = (
            test.groupby("selected_model")
            .size()
            .to_dict()
        )

        print(
            "  Hybrid selections: "
            + ", ".join(
                f"{name}={count}"
                for name, count in sorted(selected_counts.items())
            )
        )

        training_log.append({
            "month": str(month),
            "training_rows": len(train),
            "test_rows": len(test),
            "training_cutoff": month_start.date(),
        })

        print(
            f"{month}: trained on {len(train):,} rows; "
            f"predicted {len(test):,} rows"
        )

    if not prediction_rows:
        raise RuntimeError("No predictions were generated.")

    predictions = pd.concat(
        prediction_rows,
        ignore_index=True,
    )

    # Measure how many observed target days had enough history
    # to produce a forecast.
    observed_eval = observed.copy()
    observed_eval["evaluation_window"] = None

    for name, start, end in EVALUATION_WINDOWS:
        mask = (
            (observed_eval["order_date"] >= pd.Timestamp(start))
            & (observed_eval["order_date"] <= pd.Timestamp(end))
        )
        observed_eval.loc[mask, "evaluation_window"] = name

    observed_eval = observed_eval[
        observed_eval["evaluation_window"].notna()
    ].copy()

    # In pilot mode, measure coverage only for the months
    # that were actually evaluated.
    observed_eval = observed_eval[
        observed_eval["order_date"].dt.to_period("M").isin(
            evaluation_months
        )
    ].copy()

    expected_by_point = (
        observed_eval.groupby("pickup_point")
        .size()
        .rename("observed_target_days")
    )

    eligible_by_point = (
        predictions.groupby("pickup_point")
        .size()
        .rename("eligible_forecasts")
    )

    coverage_by_point = pd.DataFrame({
        "pickup_point": df["pickup_point"].drop_duplicates()
    })

    coverage_by_point = coverage_by_point.merge(
        expected_by_point,
        on="pickup_point",
        how="left",
    ).merge(
        eligible_by_point,
        on="pickup_point",
        how="left",
    )

    coverage_by_point[
        ["observed_target_days", "eligible_forecasts"]
    ] = coverage_by_point[
        ["observed_target_days", "eligible_forecasts"]
    ].fillna(0).astype(int)

    coverage_by_point["forecast_coverage_pct"] = np.where(
        coverage_by_point["observed_target_days"] > 0,
        100 * coverage_by_point["eligible_forecasts"]
        / coverage_by_point["observed_target_days"],
        np.nan,
    )

    coverage_by_point = coverage_by_point.sort_values(
        "pickup_point"
    )

    expected_by_window_point = (
        observed_eval.groupby(["evaluation_window", "pickup_point"])
        .size()
        .rename("observed_target_days")
    )

    eligible_by_window_point = (
        predictions.groupby(["evaluation_window", "pickup_point"])
        .size()
        .rename("eligible_forecasts")
    )

    coverage_by_window_point = (
        expected_by_window_point.to_frame()
        .join(eligible_by_window_point, how="left")
        .fillna({"eligible_forecasts": 0})
        .reset_index()
    )

    coverage_by_window_point["eligible_forecasts"] = (
        coverage_by_window_point["eligible_forecasts"].astype(int)
    )

    coverage_by_window_point["forecast_coverage_pct"] = (
        100
        * coverage_by_window_point["eligible_forecasts"]
        / coverage_by_window_point["observed_target_days"]
    )

    print("\nFORECAST COVERAGE BY PICKUP POINT")
    print(coverage_by_point.to_string(
        index=False,
        formatters={
            "forecast_coverage_pct": lambda x: (
                f"{x:.2f}" if pd.notna(x) else "N/A"
            ),
        },
    ))

    total_observed_targets = coverage_by_point[
        "observed_target_days"
    ].sum()

    total_eligible_forecasts = coverage_by_point[
        "eligible_forecasts"
    ].sum()

    print(
        f"\nTotal observed evaluation targets: "
        f"{total_observed_targets:,}"
    )
    print(
        f"Total eligible forecasts: "
        f"{total_eligible_forecasts:,}"
    )
    print(
        f"Overall forecast coverage: "
        f"{100 * total_eligible_forecasts / total_observed_targets:.2f}%"
        if total_observed_targets else
        "Overall forecast coverage: N/A"
    )

    methods = [
        "seasonal_naive",
        "weekday_average_4",
        "global_ml",
        "hybrid",
    ]

    common = predictions.dropna(
        subset=methods
    ).copy()

    if common.empty:
        raise RuntimeError("No common forecast rows.")

    print("\nForecast coverage:")
    for method in methods:
        count = predictions[method].notna().sum()
        print(
            f"  {method}: {count:,}/{len(predictions):,}"
        )

    print(
        f"\nCommon rows across all models: "
        f"{len(common):,}/{len(predictions):,}"
    )

    overall_rows = []

    for method in methods:
        overall_rows.append({
            "method": method,
            **calculate_metrics(common, method),
        })

    overall = pd.DataFrame(overall_rows)

    window_rows = []

    for window, group in common.groupby(
        "evaluation_window",
        sort=False,
    ):
        for method in methods:
            window_rows.append({
                "evaluation_window": window,
                "method": method,
                **calculate_metrics(group, method),
            })

    window_metrics = pd.DataFrame(window_rows)

    print("\nOVERALL PERFORMANCE — COMMON ROWS")
    print(overall.to_string(
        index=False,
        formatters={
            "MAE": "{:.2f}".format,
            "RMSE": "{:.2f}".format,
            "WAPE_pct": "{:.2f}".format,
            "Bias": "{:+.2f}".format,
        },
    ))

    print("\nPERFORMANCE BY EVALUATION WINDOW")
    print(window_metrics.to_string(
        index=False,
        formatters={
            "MAE": "{:.2f}".format,
            "RMSE": "{:.2f}".format,
            "WAPE_pct": "{:.2f}".format,
            "Bias": "{:+.2f}".format,
        },
    ))

    suffix = "_pilot" if args.pilot else ""

    predictions.to_csv(
        OUTPUT_DIR / f"hybrid_predictions{suffix}.csv",
        index=False,
    )
    overall.to_csv(
        OUTPUT_DIR / f"hybrid_overall{suffix}.csv",
        index=False,
    )
    window_metrics.to_csv(
        OUTPUT_DIR / f"hybrid_metrics_by_window{suffix}.csv",
        index=False,
    )
    pd.DataFrame(training_log).to_csv(
        OUTPUT_DIR / f"hybrid_training_log{suffix}.csv",
        index=False,
    )

    coverage_by_point.to_csv(
        OUTPUT_DIR / f"hybrid_coverage_by_point{suffix}.csv",
        index=False,
    )

    coverage_by_window_point.to_csv(
        OUTPUT_DIR / f"hybrid_coverage_by_window_point{suffix}.csv",
        index=False,
    )

    pd.DataFrame(selection_rows).to_csv(
        OUTPUT_DIR / f"hybrid_selection_log{suffix}.csv",
        index=False,
    )

    print(f"\nSaved results in {OUTPUT_DIR}/.")


if __name__ == "__main__":
    main()
