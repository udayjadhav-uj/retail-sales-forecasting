
import pandas as pd
import matplotlib.pyplot as plt

# Load the comparison results
df = pd.read_csv("forecast_results.csv", parse_dates=["date"])

# Plot actual sales and both predictions
plt.figure(figsize=(14, 6))

plt.plot(
    df["date"],
    df["actual_sales"],
    label="Actual Sales",
    linewidth=2
)

plt.plot(
    df["date"],
    df["random_forest_prediction"],
    label="Random Forest",
    alpha=0.85
)

plt.plot(
    df["date"],
    df["weekly_baseline_prediction"],
    label="Weekly Baseline",
    alpha=0.7
)

plt.title("Actual vs Predicted Sales — Store 1, BEVERAGES")
plt.xlabel("Date")
plt.ylabel("Sales")
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()

plt.savefig("forecast_comparison.png", dpi=150)
plt.show()

print("Chart saved as forecast_comparison.png")
