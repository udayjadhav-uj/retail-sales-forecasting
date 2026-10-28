
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

DATA_PATH = Path("data/train.csv")
OUTPUT_PATH = Path("future_sales_forecast.csv")

HORIZON = 7
N_ESTIMATORS = 100
MIN_TRAIN_DAYS = 100
RANDOM_STATE = 42

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


def create_features(series):
    """Create historical features for one store-product time series."""
    features = pd.DataFrame(index=series.index)

    features["day_of_week"] = series.index.dayofweek
    features["day_of_month"] = series.index.day
    features["month"] = series.index.month
    features["year"] = series.index.year
    features["onpromotion"] = series["onpromotion"]

    features["sales_lag_7"] = series["sales"].shift(7)
    features["sales_lag_14"] = series["sales"].shift(14)
    features["sales_avg_7"] = (
        series["sales"].shift(1).rolling(7).mean()
    )

    return features


def main():
    print("Loading historical sales data...")

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

    # Aggregate to one row per store, product family, and date.
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

    last_date = daily["date"].max().normalize()
    first_date = daily["date"].min().normalize()

    future_dates = pd.date_range(
        start=last_date + pd.Timedelta(days=1),
        periods=HORIZON,
        freq="D",
    )

    combinations = daily[
        ["store_nbr", "family"]
    ].drop_duplicates()

    total_combinations = len(combinations)
    results = []
    skipped = 0

    print(f"Store-product combinations: {total_combinations}")
    print(f"Historical period: {first_date.date()} to {last_date.date()}")
    print(
        f"Forecast period: {future_dates[0].date()} "
        f"to {future_dates[-1].date()}"
    )
    print("\nTraining models and generating forecasts...")

    # Group once instead of repeatedly filtering the entire dataset.
    grouped = {
        (store, family): group
        for (store, family), group in daily.groupby(
            ["store_nbr", "family"], sort=False
        )
    }

    for number, (store, family) in enumerate(
        combinations.itertuples(index=False, name=None),
        start=1,
    ):
        group = grouped[(store, family)]

        # Build a continuous daily series up to the dataset's last date.
        # Assumption: missing daily records represent zero sales.
        # Verify this assumption before operational use.
        series = (
            group.set_index("date")[["sales", "onpromotion"]]
            .sort_index()
            .asfreq("D")
        )

        series["sales"] = series["sales"].fillna(0)
        series["onpromotion"] = series["onpromotion"].fillna(0)

        # Ensure every combination is aligned to the same forecast origin.
        full_history_index = pd.date_range(
            start=first_date,
            end=last_date,
            freq="D",
        )

        series = series.reindex(full_history_index)
        series["sales"] = series["sales"].fillna(0)
        series["onpromotion"] = series["onpromotion"].fillna(0)

        # Create training features using historical observations only.
        feature_frame = create_features(series)
        training = feature_frame.copy()
        training["target"] = series["sales"]
        training = training.dropna()

        if len(training) < MIN_TRAIN_DAYS:
            skipped += 1
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

        # Recursively forecast the next seven days.
        # Actual future sales are never used.
        history = series["sales"].tolist()

        for forecast_date in future_dates:
            row = pd.DataFrame([{
                "day_of_week": forecast_date.dayofweek,
                "day_of_month": forecast_date.day,
                "month": forecast_date.month,
                "year": forecast_date.year,

                # Future promotions are unknown; assumed to be zero.
                "onpromotion": 0,

                "sales_lag_7": history[-7],
                "sales_lag_14": history[-14],
                "sales_avg_7": float(np.mean(history[-7:])),
            }])

            prediction = max(
                0.0,
                float(model.predict(row[FEATURES])[0]),
            )

            results.append({
                "date": forecast_date.strftime("%Y-%m-%d"),
                "store_nbr": store,
                "family": family,
                "predicted_sales": round(prediction, 2),
            })

            # Feed the prediction into history for the next day.
            history.append(prediction)

        if number % 100 == 0 or number == total_combinations:
            print(
                f"Processed {number}/{total_combinations} "
                f"combinations..."
            )

    if not results:
        raise RuntimeError(
            "No forecasts were generated. Check the dataset "
            "and minimum training history."
        )

    forecast = pd.DataFrame(results)

    forecast = forecast.sort_values(
        ["store_nbr", "family", "date"]
    ).reset_index(drop=True)

    forecast.to_csv(OUTPUT_PATH, index=False)

    print("\nForecast generation completed.")
    print(f"Combinations forecasted: {forecast[['store_nbr', 'family']].drop_duplicates().shape[0]}")
    print(f"Forecast rows generated: {len(forecast):,}")
    print(f"Skipped combinations: {skipped}")
    print(f"Saved to: {OUTPUT_PATH.resolve()}")

    print("\nFirst 10 forecast rows:")
    print(forecast.head(10).to_string(index=False))

    print("\nForecast date range:")
    print(forecast["date"].min(), "to", forecast["date"].max())


if __name__ == "__main__":
    main()
