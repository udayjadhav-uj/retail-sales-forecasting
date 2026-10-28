
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# Load real data
df = pd.read_csv("data/train.csv", parse_dates=["date"])

data = df[
    (df["store_nbr"] == 1)
    & (df["family"] == "BEVERAGES")
].copy()

data = data.sort_values("date").reset_index(drop=True)

# Calendar features
data["day_of_week"] = data["date"].dt.dayofweek
data["day_of_month"] = data["date"].dt.day
data["month"] = data["date"].dt.month
data["year"] = data["date"].dt.year

# Historical sales features
data["sales_lag_7"] = data["sales"].shift(7)
data["sales_lag_14"] = data["sales"].shift(14)
data["sales_avg_7"] = data["sales"].shift(1).rolling(7).mean()

# Weekly baseline
data["baseline_prediction"] = data["sales"].shift(7)

data = data.dropna().reset_index(drop=True)

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

# Chronological split
split = int(len(data) * 0.8)
train = data.iloc[:split]
test = data.iloc[split:].copy()

# Train Random Forest using training dates only
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)

model.fit(train[features], train["sales"])

# Evaluate both models on exactly the same test dates
actual = test["sales"].to_numpy()
rf_pred = np.maximum(model.predict(test[features]), 0)
baseline_pred = test["baseline_prediction"].to_numpy()

print("FAIR MODEL COMPARISON")
print(f"Test records: {len(test)}")
print(f"Test period: {test['date'].min().date()} to {test['date'].max().date()}")

for name, pred in [
    ("Random Forest", rf_pred),
    ("Weekly Baseline", baseline_pred),
]:
    mae = mean_absolute_error(actual, pred)
    rmse = np.sqrt(mean_squared_error(actual, pred))
    print(f"\n{name}")
    print(f"MAE: {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")

results = pd.DataFrame({
    "date": test["date"].values,
    "actual_sales": actual,
    "random_forest_prediction": rf_pred,
    "weekly_baseline_prediction": baseline_pred,
})

results.to_csv("forecast_results.csv", index=False)
print("\nSaved fair comparison to forecast_results.csv")
