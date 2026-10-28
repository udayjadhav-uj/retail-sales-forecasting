
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

DATA_PATH = Path("data/train.csv")
RESULTS_PATH = Path("seven_day_evaluation.csv")

MIN_TRAIN_DAYS = 100
N_ESTIMATORS = 100
RANDOM_STATE = 42
FORECAST_HORIZON = 7


def metrics(actual, predicted):
    return (
        mean_absolute_error(actual, predicted),
        np.sqrt(mean_squared_error(actual, predicted)),
    )


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
            ["store_nbr", "family", "date"], as_index=False
        )
        .agg(
            sales=("sales", "sum"),
            onpromotion=("onpromotion", "sum"),
        )
    )

    # Use the same final seven calendar days for all combinations.
    global_end = daily["date"].max()
    test_start = (
        global_end - pd.Timedelta(days=FORECAST_HORIZON - 1)
    )

    results = []
    combinations = daily[["store_nbr", "family"]].drop_duplicates()

    print(f"Forecast dates: {test_start.date()} to {global_end.date()}")
    print(f"Combinations: {len(combinations)}")

    for number, (store, family) in enumerate(
        combinations.itertuples(index=False, name=None), start=1
    ):
        if number % 100 == 0:
            print(f"Processed {number}/{len(combinations)}")

        series = daily[
            (daily["store_nbr"] == store)
            & (daily["family"] == family)
        ].copy()

        # Build a complete daily series so lags mean calendar days.
        series = (
            series.set_index("date")
            .sort_index()
            .asfreq("D")
        )

        # Missing dates are treated as zero sales only when the
        # series has no recorded row for that date.
        series["sales"] = series["sales"].fillna(0)
        series["onpromotion"] = series["onpromotion"].fillna(0)

        train = series.loc[series.index < test_start].copy()
        test = series.loc[
            (series.index >= test_start)
            & (series.index <= global_end)
        ].copy()

        if len(train) < MIN_TRAIN_DAYS or len(test) != FORECAST_HORIZON:
            continue

        train_features = pd.DataFrame(index=train.index)
        train_features["day_of_week"] = train.index.dayofweek
        train_features["day_of_month"] = train.index.day
        train_features["month"] = train.index.month
        train_features["year"] = train.index.year
        train_features["onpromotion"] = train["onpromotion"]
        train_features["sales_lag_7"] = train["sales"].shift(7)
        train_features["sales_lag_14"] = train["sales"].shift(14)
        train_features["sales_avg_7"] = (
            train["sales"].shift(1).rolling(7).mean()
        )

        training_data = train_features.copy()
        training_data["target"] = train["sales"]
        training_data = training_data.dropna()

        if len(training_data) < MIN_TRAIN_DAYS:
            continue

        model_features = [
            "day_of_week", "day_of_month", "month", "year",
            "onpromotion", "sales_lag_7", "sales_lag_14",
            "sales_avg_7",
        ]

        model = RandomForestRegressor(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            min_samples_leaf=2,
        )

        model.fit(
            training_data[model_features],
            training_data["target"],
        )

        # Recursively predict each test day. Actual test sales are
        # never used as lag or rolling inputs.
        history = train["sales"].tolist()
        predictions = []

        for forecast_date in test.index:
            row = pd.DataFrame([{
                "day_of_week": forecast_date.dayofweek,
                "day_of_month": forecast_date.day,
                "month": forecast_date.month,
                "year": forecast_date.year,
                # Assumption: future promotions are unknown and zero.
                "onpromotion": 0,
                "sales_lag_7": history[-7],
                "sales_lag_14": history[-14],
                "sales_avg_7": np.mean(history[-7:]),
            }])

            prediction = max(
                0.0, float(model.predict(row[model_features])[0])
            )
            predictions.append(prediction)
            history.append(prediction)

        actual = test["sales"].to_numpy()

        # Weekly-naive baseline: same weekday one week earlier.
        baseline = series.loc[
            test.index - pd.Timedelta(days=7), "sales"
        ].to_numpy()

        rf_mae, rf_rmse = metrics(actual, predictions)
        base_mae, base_rmse = metrics(actual, baseline)

        improvement = (
            (base_mae - rf_mae) / base_mae * 100
            if base_mae > 0 else np.nan
        )

        results.append({
            "store": store,
            "family": family,
            "test_start": test.index.min().date(),
            "test_end": test.index.max().date(),
            "random_forest_mae": round(rf_mae, 2),
            "random_forest_rmse": round(rf_rmse, 2),
            "baseline_mae": round(base_mae, 2),
            "baseline_rmse": round(base_rmse, 2),
            "mae_improvement_percent": (
                round(improvement, 2)
                if np.isfinite(improvement) else np.nan
            ),
        })

    if not results:
        print("No combinations met the evaluation requirements.")
        return

    output = pd.DataFrame(results)
    output.to_csv(RESULTS_PATH, index=False)

    valid = output.dropna(subset=["mae_improvement_percent"])
    wins = int(
        (output["random_forest_mae"] < output["baseline_mae"]).sum()
    )

    print("\nSEVEN-DAY FORECAST EVALUATION")
    print(f"Combinations evaluated: {len(output)}")
    print(
        f"Random Forest lower MAE: {wins}/{len(output)} "
        f"({wins / len(output) * 100:.1f}%)"
    )

    if not valid.empty:
        print(
            "Mean MAE improvement: "
            f"{valid['mae_improvement_percent'].mean():.2f}%"
        )
        print(
            "Median MAE improvement: "
            f"{valid['mae_improvement_percent'].median():.2f}%"
        )

    print(f"Results saved to {RESULTS_PATH.resolve()}")


if __name__ == "__main__":
    main()
