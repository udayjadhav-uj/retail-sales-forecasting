
import pandas as pd
import matplotlib.pyplot as plt

forecast = pd.read_csv(
    "future_sales_forecast.csv",
    parse_dates=["date"]
)

plt.figure(figsize=(10, 5))

plt.bar(
    forecast["date"].dt.strftime("%d %b"),
    forecast["predicted_sales"]
)

plt.title("7-Day Sales Forecast — Store 1, BEVERAGES")
plt.xlabel("Forecast Date")
plt.ylabel("Predicted Sales")
plt.xticks(rotation=45)
plt.grid(axis="y", alpha=0.3)
plt.tight_layout()

plt.savefig("future_forecast_chart.png", dpi=150)
plt.show()

print("Saved chart as future_forecast_chart.png")
