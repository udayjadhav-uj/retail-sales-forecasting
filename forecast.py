
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

# Load data
df = pd.read_csv(
    Path("data/train.csv"),
    parse_dates=["date"]
)

# Start with one store and one product family
store_id = 1
family_name = "BEVERAGES"

data = df[
    (df["store_nbr"] == store_id)
    & (df["family"] == family_name)
].copy()

data = data.sort_values("date").reset_index(drop=True)

# Calendar features
data["day_of_week"] = data["date"].dt.dayofweek
data["day_of_month"] = data["date"].dt.day
data["month"] = data["date"].dt.month
data["year"] = data["date"].dt.year

# Past sales features
data["sales_lag_7"] = data["sales"].shift(7)
data["sales_lag_14"] = data["sales"].shift(14)
data["sales_avg_7"] = data["sales"].shift(1).rolling(7).mean()

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

X = data[features]
y = data["sales"]

# Use the latest 20% of dates for testing
split = int(len(data) * 0.8)

X_train = X.iloc[:split]
X_test = X.iloc[split:]
y_train = y.iloc[:split]
y_test = y.iloc[split:]

# Train model
model = RandomForestRegressor(
    n_estimators=100,
    random_state=42,
    n_jobs=-1,
    min_samples_leaf=2
)

model.fit(X_train, y_train)

# Evaluate on later dates
predictions = np.maximum(model.predict(X_test), 0)

mae = mean_absolute_error(y_test, predictions)
rmse = np.sqrt(mean_squared_error(y_test, predictions))

print(f"Store: {store_id}")
print(f"Product family: {family_name}")
print(f"Training records: {len(X_train)}")
print(f"Testing records: {len(X_test)}")
print(f"MAE: {mae:.2f}")
print(f"RMSE: {rmse:.2f}")

# Show sample predictions
results = pd.DataFrame({
    "date": data["date"].iloc[split:].values,
    "actual_sales": y_test.values,
    "predicted_sales": predictions
})

print("\nSample predictions:")
print(results.tail(10).to_string(index=False))

results.to_csv("forecast_results.csv", index=False)
print("\nSaved predictions to forecast_results.csv")
