
from pathlib import Path

import pandas as pd

INVENTORY_PATH = Path("inventory.csv")

# These are illustrative sample values, not real store inventory.
inventory = pd.DataFrame({
    "store_nbr": [1, 1, 1, 1, 1],
    "family": [
        "BEVERAGES",
        "GROCERY I",
        "CLEANING",
        "DAIRY",
        "PERSONAL CARE",
    ],
    "current_stock": [500, 300, 80, 120, 60],
    "lead_time_days": [5, 7, 4, 3, 6],
    "safety_stock": [100, 80, 30, 40, 25],
})

# Validate the sample inputs.
if (inventory["current_stock"] < 0).any():
    raise ValueError("Current stock cannot be negative.")

if (inventory["lead_time_days"] < 1).any():
    raise ValueError("Lead time must be at least one day.")

if (inventory["safety_stock"] < 0).any():
    raise ValueError("Safety stock cannot be negative.")

# Save the inventory input file.
inventory.to_csv(INVENTORY_PATH, index=False)

print("Inventory sample created successfully!")
print(f"Saved to: {INVENTORY_PATH.resolve()}")
print("\nInventory inputs:")
print(inventory.to_string(index=False))

print(
    "\nNOTE: These stock quantities, lead times, and safety-stock "
    "values are illustrative. Replace them with verified data "
    "before using the system for actual replenishment decisions."
)
