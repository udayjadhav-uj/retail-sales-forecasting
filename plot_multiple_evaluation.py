
import pandas as pd
import matplotlib.pyplot as plt

RESULTS_PATH = "multiple_model_evaluation.csv"

# Load evaluation results
results = pd.read_csv(RESULTS_PATH)

# Create readable labels
results["combination"] = (
    "Store " + results["store"].astype(str)
    + " - " + results["family"]
)

# Compare MAE
ax = results.plot(
    x="combination",
    y=["random_forest_mae", "baseline_mae"],
    kind="bar",
    figsize=(12, 6),
    width=0.8,
)

ax.set_title("Forecasting Model Comparison Across Store-Product Combinations")
ax.set_xlabel("Store - Product Family")
ax.set_ylabel("Mean Absolute Error (MAE)")
ax.legend(["Random Forest", "Weekly Baseline"])
ax.tick_params(axis="x", rotation=25)

plt.tight_layout()
plt.savefig("multiple_model_comparison.png", dpi=150)
plt.show()

print("Chart saved to multiple_model_comparison.png")

# Display improvement for each combination
print("\nMAE Improvement by Combination (%)")
print(
    results[
        ["combination", "mae_improvement_percent"]
    ].to_string(index=False)
)
