
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/train.csv")
RESULTS_PATH = Path("multiple_model_evaluation.csv")

TEST_FRACTION = 0.20
MIN_RECORDS = 100
N_ESTIMATORS = 100
RANDOM_STATE = 42


def calculate_metrics(actual, predictions):
    mae = mean_absolute_error(actual, predictions)
    rmse = np.sqrt(mean_squared_error(actual, predictions))
    return mae, rmse


def main():
    print("Loading retail sales data...")

    if not DATA_PATH.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_PATH.resolve()}"
        )

    df = pd.read_csv(DATA_PATH, parse_dates=["date"])

    required_columns = {
        "date", "store_nbr", "family", "sales", "onpromotion"
    }

    missing = required_columns - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {sorted(missing)}"
        )

    df["sales"] = pd.to_numeric(df["sales"], errors="coerce")
    df["onpromotion"] = pd.to_numeric(
        df["onpromotion"], errors="coerce"
    ).fillna(0)

    df = df.dropna(
        subset=["date", "store_nbr", "family", "sales"]
    )

    # Aggregate daily sales for every store-product combination.
    daily = (
        df.groupby(
            ["store_nbr", "family", "date"],
            as_index=False,
        )
        .agg(
            sales=("sales", "sum"),
            onpromotion=("onpromotion", "sum"),
        )
        .sort_values(["store_nbr", "family", "date"])
        .reset_index(drop=True)
    )

    combinations = daily[["store_nbr", "family"]].drop_duplicates()
    total = len(combinations)
    results = []

    print(f"Found {total} store-product combinations.")

    for number, (store, family) in enumerate(
        combinations.itertuples(index=False, name=None), start=1
    ):
        print(
            f"[{number}/{total}] Store {store} - {family}"
        )

        data = (
            daily[
                (daily["store_nbr"] == store)
                & (daily["family"] == family)
            ]
            .sort_values("date")
            .copy()
            .reset_index(drop=True)
        )

        if len(data) < MIN_RECORDS:
            print("  Skipped: insufficient historical records.")
            continue

        # Calendar features
        data["day_of_week"] = data["date"].dt.day_of_week
        data["day_of_month"] = data["date"].dt.day
        data["month"] = data["date"].dt.month
        data["year"] = data["date"].dt.year

        # Lag and rolling features use only earlier observations.
        data["sales_lag_7"] = data["sales"].shift(7)
        data["sales_lag_14"] = data["sales"].shift(14)
        data["sales_avg_7"] = (
            data["sales"].shift(1).rolling(7).mean()
        )

        # Same weekday in the previous week.
        data["baseline_prediction"] = data["sales"].shift(7)

        data = data.dropna().reset_index(drop=True)

        if len(data) < MIN_RECORDS:
            print("  Skipped after feature creation.")
            continue

        features = [
            "day_of_week",
            "day_of_month",
            "month",
            "year",
            "onpromotion",
            "sales_lag_7",
            "sales_lag_14",
            "sales_avg_7",
        ]

        # Chronological split: never randomly shuffle time-series data.
        split_index = int(len(data) * (1 - TEST_FRACTION))
        train = data.iloc[:split_index]
        test = data.iloc[split_index:].copy()

        if train.empty or test.empty:
            print("  Skipped: empty train or test partition.")
            continue

        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2,
        )

        model.fit(train[features], train["sales"])

        actual = test["sales"].to_numpy()

        rf_predictions = np.maximum(
            model.predict(test[features]), 0
        )

        baseline_predictions = test[
            "baseline_prediction"
        ].to_numpy()

        rf_mae, rf_rmse = calculate_metrics(
            actual, rf_predictions
        )
        baseline_mae, baseline_rmse = calculate_metrics(
            actual, baseline_predictions
        )

        improvement = (
            (baseline_mae - rf_mae) / baseline_mae * 100
            if baseline_mae > 0
            else np.nan
        )

        results.append({
            "store": store,
            "family": family,
            "test_records": len(test),
            "test_start": test["date"].min().date(),
            "test_end": test["date"].max().date(),
            "random_forest_mae": round(rf_mae, 2),
            "random_forest_rmse": round(rf_rmse, 2),
            "baseline_mae": round(baseline_mae, 2),
            "baseline_rmse": round(baseline_rmse, 2),
            "mae_improvement_percent": (
                round(improvement, 2)
                if np.isfinite(improvement)
                else np.nan
            ),
        })

    if not results:
        print("No combinations could be evaluated.")
        return

    results_df = pd.DataFrame(results)
    results_df.to_csv(RESULTS_PATH, index=False)

    valid = results_df.dropna(
        subset=["mae_improvement_percent"]
    )

    better_count = int(
        (valid["mae_improvement_percent"] > 0).sum()
    )
    evaluated_count = len(results_df)

    print("\nMULTIPLE STORE-PRODUCT EVALUATION")
    print(results_df.to_string(index=False))

    
    print("\nSUMMARY")
    print(f"Combinations evaluated: {evaluated_count}")

    rf_better_count = int(
        (
            results_df["random_forest_mae"]
            < results_df["baseline_mae"]
        ).sum()
    )

    print(
        "Random Forest has lower MAE: "
        f"{rf_better_count}/{evaluated_count} "
        f"({rf_better_count / evaluated_count * 100:.1f}%)"
    )

    if not valid.empty:
        print(
            "Mean MAE improvement across combinations: "
            f"{valid['mae_improvement_percent'].mean():.2f}%"
        )
        print(
            "Median MAE improvement across combinations: "
            f"{valid['mae_improvement_percent'].median():.2f}%"
        )

    print(f"\nResults saved to {RESULTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
