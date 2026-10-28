
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/train.csv")
RESULTS_PATH = Path("rolling_backtest_results.csv")
DAILY_RESULTS_PATH = Path("rolling_backtest_daily_results.csv")
WINDOW_RESULTS_PATH = Path("rolling_backtest_window_summary.csv")

MIN_TRAIN_DAYS = 100
N_ESTIMATORS = 100
RANDOM_STATE = 42
HORIZON = 7
N_WINDOWS = 4

FEATURES = [
    "day_of_week",
    "day_of_month",
    "month",
    "year",
    "onpromotion",
    "sales_lag_7",
    "sales_lag_14",
    "sales_avg_7",
]


def calculate_metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return mae, rmse


def percentage_improvement(baseline, model):
    if baseline == 0:
        return np.nan
    return (baseline - model) / baseline * 100


def main():
    print("Loading data...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    required = {
        "date", "store_nbr", "family", "sales", "onpromotion"
    }
    missing = required - set(df.columns)

    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["onpromotion"] = pd.to_numeric(
        df["onpromotion"], errors="coerce"
    ).fillna(0)

    df = df.dropna(
        subset=["date", "store_nbr", "family", "sales"]
    )

    daily = (
        df.groupby(
            ["store_nbr", "family", "date"],
            as_index=False,
        )
        .agg(
            sales=("sales", "sum"),
            onpromotion=("onpromotion", "sum"),
        )
    )

    end_date = daily["date"].max().normalize()

    # Four consecutive, non-overlapping seven-day windows.
    windows = []

    for i in reversed(range(N_WINDOWS)):
        window_end = end_date - pd.Timedelta(days=i * HORIZON)
        window_start = window_end - pd.Timedelta(days=HORIZON - 1)
        windows.append((window_start, window_end))

    combinations = daily[
        ["store_nbr", "family"]
    ].drop_duplicates()

    combination_results = []
    daily_results = []

    print(f"Store-product combinations: {len(combinations)}")
    print(f"Forecast windows: {len(windows)}")

    for window_number, (test_start, test_end) in enumerate(
        windows, start=1
    ):
        print(
            f"\nWindow {window_number}/{N_WINDOWS}: "
            f"{test_start.date()} to {test_end.date()}"
        )

        for store, family in combinations.itertuples(
            index=False, name=None
        ):
            series = daily[
                (daily["store_nbr"] == store)
                & (daily["family"] == family)
            ].copy()

            series = (
                series.set_index("date")
                .sort_index()
                .asfreq("D")
            )

            # Assumption: missing dates mean zero sales.
            # Validate this before using results operationally.
            series["sales"] = series["sales"].fillna(0)
            series["onpromotion"] = (
                series["onpromotion"].fillna(0)
            )

            train = series.loc[
                series.index < test_start
            ].copy()

            test = series.loc[
                (series.index >= test_start)
                & (series.index <= test_end)
            ].copy()

            if len(train) < MIN_TRAIN_DAYS:
                continue

            if len(test) != HORIZON:
                continue

            # Build training features from historical data only.
            feature_frame = pd.DataFrame(index=train.index)
            feature_frame["day_of_week"] = train.index.dayofweek
            feature_frame["day_of_month"] = train.index.day
            feature_frame["month"] = train.index.month
            feature_frame["year"] = train.index.year
            feature_frame["onpromotion"] = train["onpromotion"]

            feature_frame["sales_lag_7"] = (
                train["sales"].shift(7)
            )
            feature_frame["sales_lag_14"] = (
                train["sales"].shift(14)
            )
            feature_frame["sales_avg_7"] = (
                train["sales"].shift(1).rolling(7).mean()
            )

            training = feature_frame.copy()
            training["target"] = train["sales"]
            training = training.dropna()

            if len(training) < MIN_TRAIN_DAYS:
                continue

            model = RandomForestRegressor(
                n_estimators=N_ESTIMATORS,
                random_state=RANDOM_STATE,
                n_jobs=-1,
                min_samples_leaf=2,
            )

            model.fit(
                training[FEATURES],
                training["target"],
            )

            # Recursively forecast seven days.
            # Actual test-window sales are never fed into the model.
            history = train["sales"].tolist()
            predictions = []

            for forecast_date in test.index:
                row = pd.DataFrame([{
                    "day_of_week": forecast_date.dayofweek,
                    "day_of_month": forecast_date.day,
                    "month": forecast_date.month,
                    "year": forecast_date.year,

                    # Assumption: future promotions are zero.
                    "onpromotion": 0,

                    "sales_lag_7": history[-7],
                    "sales_lag_14": history[-14],
                    "sales_avg_7": np.mean(history[-7:]),
                }])

                prediction = max(
                    0.0,
                    float(model.predict(row[FEATURES])[0]),
                )

                predictions.append(prediction)
                history.append(prediction)

            actual = test["sales"].to_numpy()

            # Weekly-naive baseline: same weekday one week earlier.
            baseline_dates = (
                test.index - pd.Timedelta(days=7)
            )
            baseline = series.loc[
                baseline_dates, "sales"
            ].to_numpy()

            rf_mae, rf_rmse = calculate_metrics(
                actual, predictions
            )
            base_mae, base_rmse = calculate_metrics(
                actual, baseline
            )

            mae_improvement = percentage_improvement(
                base_mae, rf_mae
            )
            rmse_improvement = percentage_improvement(
                base_rmse, rf_rmse
            )

            combination_results.append({
                "store": store,
                "family": family,
                "test_start": test_start.date(),
                "test_end": test_end.date(),
                "random_forest_mae": rf_mae,
                "random_forest_rmse": rf_rmse,
                "baseline_mae": base_mae,
                "baseline_rmse": base_rmse,
                "mae_improvement_percent": mae_improvement,
                "rmse_improvement_percent": rmse_improvement,
                "rf_wins": rf_mae < base_mae,
            })

            # Save individual daily predictions so we can calculate
            # genuine overall MAE and RMSE across all observations.
            for forecast_date, observed, rf_pred, base_pred in zip(
                test.index, actual, predictions, baseline
            ):
                daily_results.append({
                    "store": store,
                    "family": family,
                    "date": forecast_date.date(),
                    "actual_sales": float(observed),
                    "random_forest_prediction": float(rf_pred),
                    "baseline_prediction": float(base_pred),
                    "window_start": test_start.date(),
                    "window_end": test_end.date(),
                })

    if not combination_results or not daily_results:
        print("No combinations could be evaluated.")
        return

    # Save results.
    output = pd.DataFrame(combination_results)
    daily_output = pd.DataFrame(daily_results)

    output.to_csv(RESULTS_PATH, index=False)
    daily_output.to_csv(DAILY_RESULTS_PATH, index=False)

    # ---------------------------------------------------------
    # 1. OVERALL METRICS ACROSS ALL DAILY FORECAST OBSERVATIONS
    # ---------------------------------------------------------
    actual_all = daily_output["actual_sales"].to_numpy()
    rf_all = daily_output["random_forest_prediction"].to_numpy()
    baseline_all = daily_output["baseline_prediction"].to_numpy()

    overall_rf_mae, overall_rf_rmse = calculate_metrics(
        actual_all, rf_all
    )
    overall_base_mae, overall_base_rmse = calculate_metrics(
        actual_all, baseline_all
    )

    print("\n" + "=" * 65)
    print("OVERALL ROLLING BACKTEST RESULTS")
    print("=" * 65)

    print(f"Forecast windows: {N_WINDOWS}")
    print(
        f"Forecast dates: {windows[0][0].date()} through "
        f"{windows[-1][1].date()}"
    )
    print(f"Daily forecast observations: {len(daily_output):,}")
    print(f"Store-product-window evaluations: {len(output):,}")

    print("\nOVERALL ERROR METRICS")
    print("-" * 65)
    print(f"{'Metric':<25}{'Random Forest':>18}{'Baseline':>18}")
    print("-" * 65)
    print(
        f"{'MAE':<25}{overall_rf_mae:>18.4f}"
        f"{overall_base_mae:>18.4f}"
    )
    print(
        f"{'RMSE':<25}{overall_rf_rmse:>18.4f}"
        f"{overall_base_rmse:>18.4f}"
    )

    mae_improvement = percentage_improvement(
        overall_base_mae, overall_rf_mae
    )
    rmse_improvement = percentage_improvement(
        overall_base_rmse, overall_rf_rmse
    )

    print(
        f"\nOverall MAE improvement: "
        f"{mae_improvement:.2f}%"
        if not np.isnan(mae_improvement)
        else "\nOverall MAE improvement: undefined (baseline MAE is zero)"
    )
    print(
        f"Overall RMSE improvement: {rmse_improvement:.2f}%"
        if not np.isnan(rmse_improvement)
        else "Overall RMSE improvement: undefined (baseline RMSE is zero)"
    )

    wins = int(output["rf_wins"].sum())
    total = len(output)

    print(
        f"\nRandom Forest lower MAE in {wins}/{total} "
        f"combinations ({wins / total * 100:.2f}%)."
    )

    # ---------------------------------------------------------
    # 2. RESULTS BY FORECAST WINDOW
    # ---------------------------------------------------------
    print("\nRESULTS BY FORECAST WINDOW")

    window_rows = []

    for (start, end), group in daily_output.groupby(
        ["window_start", "window_end"]
    ):
        rf_mae, rf_rmse = calculate_metrics(
            group["actual_sales"],
            group["random_forest_prediction"],
        )
        base_mae, base_rmse = calculate_metrics(
            group["actual_sales"],
            group["baseline_prediction"],
        )

        combo_group = output[
            (output["test_start"] == start)
            & (output["test_end"] == end)
        ]

        window_rows.append({
            "test_start": start,
            "test_end": end,
            "daily_observations": len(group),
            "combinations": len(combo_group),
            "random_forest_mae": rf_mae,
            "baseline_mae": base_mae,
            "mae_improvement_percent": percentage_improvement(
                base_mae, rf_mae
            ),
            "random_forest_rmse": rf_rmse,
            "baseline_rmse": base_rmse,
            "rmse_improvement_percent": percentage_improvement(
                base_rmse, rf_rmse
            ),
            "rf_wins": int(combo_group["rf_wins"].sum()),
            "rf_win_rate_percent": (
                combo_group["rf_wins"].mean() * 100
            ),
        })

    window_summary = pd.DataFrame(window_rows)
    window_summary.to_csv(WINDOW_RESULTS_PATH, index=False)

    print(
        window_summary.round(2).to_string(index=False)
    )

    # ---------------------------------------------------------
    # 3. STORE-PRODUCT COMBINATION SUMMARY
    # ---------------------------------------------------------
    valid_mae = output.dropna(
        subset=["mae_improvement_percent"]
    )
    valid_rmse = output.dropna(
        subset=["rmse_improvement_percent"]
    )

    print("\nSTORE-PRODUCT SUMMARY")
    print("-" * 65)

    print(
        f"Mean combination-level MAE improvement: "
        f"{valid_mae['mae_improvement_percent'].mean():.2f}%"
        if not valid_mae.empty
        else "Mean combination-level MAE improvement: unavailable"
    )

    print(
        f"Median combination-level MAE improvement: "
        f"{valid_mae['mae_improvement_percent'].median():.2f}%"
        if not valid_mae.empty
        else "Median combination-level MAE improvement: unavailable"
    )

    print(
        f"Mean combination-level RMSE improvement: "
        f"{valid_rmse['rmse_improvement_percent'].mean():.2f}%"
        if not valid_rmse.empty
        else "Mean combination-level RMSE improvement: unavailable"
    )

    print("\nFILES SAVED")
    print(f"Combination results: {RESULTS_PATH.resolve()}")
    print(f"Daily predictions: {DAILY_RESULTS_PATH.resolve()}")
    print(f"Window summary: {WINDOW_RESULTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
