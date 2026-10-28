
from pathlib import Path

import numpy as np
import pandas as pd

FORECAST_PATH = Path("future_sales_forecast.csv")
INVENTORY_PATH = Path("inventory.csv")
OUTPUT_PATH = Path("inventory_analysis.csv")

REQUIRED_FORECAST_COLUMNS = {
    "date", "store_nbr", "family", "predicted_sales"
}

REQUIRED_INVENTORY_COLUMNS = {
    "store_nbr",
    "family",
    "current_stock",
    "lead_time_days",
    "safety_stock",
}


def main():
    print("Loading forecast and inventory data...")

    if not FORECAST_PATH.exists():
        raise FileNotFoundError(
            f"Forecast file not found: {FORECAST_PATH.resolve()}"
        )

    if not INVENTORY_PATH.exists():
        raise FileNotFoundError(
            f"Inventory file not found: {INVENTORY_PATH.resolve()}"
        )

    forecasts = pd.read_csv(
        FORECAST_PATH,
        parse_dates=["date"],
    )
    inventory = pd.read_csv(INVENTORY_PATH)

    missing_forecast = (
        REQUIRED_FORECAST_COLUMNS - set(forecasts.columns)
    )
    missing_inventory = (
        REQUIRED_INVENTORY_COLUMNS - set(inventory.columns)
    )

    if missing_forecast:
        raise ValueError(
            f"Forecast file missing columns: {sorted(missing_forecast)}"
        )

    if missing_inventory:
        raise ValueError(
            f"Inventory file missing columns: {sorted(missing_inventory)}"
        )

    forecasts["predicted_sales"] = pd.to_numeric(
        forecasts["predicted_sales"], errors="coerce"
    )
    forecasts["store_nbr"] = pd.to_numeric(
        forecasts["store_nbr"], errors="coerce"
    )
    inventory["store_nbr"] = pd.to_numeric(
        inventory["store_nbr"], errors="coerce"
    )

    for column in [
        "current_stock",
        "lead_time_days",
        "safety_stock",
    ]:
        inventory[column] = pd.to_numeric(
            inventory[column], errors="coerce"
        )

    forecasts = forecasts.dropna(
        subset=[
            "date", "store_nbr", "family", "predicted_sales"
        ]
    )

    inventory = inventory.dropna(
        subset=[
            "store_nbr",
            "family",
            "current_stock",
            "lead_time_days",
            "safety_stock",
        ]
    )

    if (forecasts["predicted_sales"] < 0).any():
        raise ValueError("Forecasted sales cannot be negative.")

    if (inventory["current_stock"] < 0).any():
        raise ValueError("Current stock cannot be negative.")

    if (inventory["lead_time_days"] < 1).any():
        raise ValueError("Lead time must be at least one day.")

    if (inventory["safety_stock"] < 0).any():
        raise ValueError("Safety stock cannot be negative.")

    if inventory.duplicated(["store_nbr", "family"]).any():
        raise ValueError(
            "Inventory contains duplicate store-product combinations."
        )

    if forecasts.duplicated(
        ["store_nbr", "family", "date"]
    ).any():
        raise ValueError(
            "Forecasts contain duplicate store-product-date rows."
        )

    forecast_start = forecasts["date"].min().normalize()
    forecast_end = forecasts["date"].max().normalize()

    # Use the forecast dates as the forecast calendar.
    calendar = pd.date_range(
        forecast_start,
        forecast_end,
        freq="D",
    )

    expected_days = len(calendar)

    # Calculate the demand forecast for each inventory item.
    forecast_groups = {
        (store, family): group.sort_values("date")
        for (store, family), group in forecasts.groupby(
            ["store_nbr", "family"]
        )
    }

    results = []

    for row in inventory.itertuples(index=False):
        store = row.store_nbr
        family = row.family
        stock = float(row.current_stock)
        lead_time = int(row.lead_time_days)
        safety_stock = float(row.safety_stock)

        key = (store, family)
        group = forecast_groups.get(key)

        result = {
            "store_nbr": store,
            "family": family,
            "current_stock": stock,
            "lead_time_days": lead_time,
            "safety_stock": safety_stock,
            "forecast_start": forecast_start.date(),
            "forecast_end": forecast_end.date(),
        }

        if group is None:
            result.update({
                "forecast_days_available": 0,
                "lead_time_demand": np.nan,
                "reorder_point": np.nan,
                "stock_after_lead_time": np.nan,
                "stock_status": "NO FORECAST",
                "suggested_replenishment_qty": np.nan,
                "analysis_note": "No matching store-product forecast.",
            })
            results.append(result)
            continue

        # Reindex to the full forecast calendar so missing forecast
        # dates are detected instead of silently treated as zero.
        daily_forecast = (
            group.set_index("date")["predicted_sales"]
            .reindex(calendar)
        )

        available_days = int(daily_forecast.notna().sum())
        result["forecast_days_available"] = available_days

        if available_days != expected_days:
            result.update({
                "lead_time_demand": np.nan,
                "reorder_point": np.nan,
                "stock_after_lead_time": np.nan,
                "stock_status": "INCOMPLETE FORECAST",
                "suggested_replenishment_qty": np.nan,
                "analysis_note": (
                    "Some forecast dates are missing. "
                    "Generate a complete daily forecast before "
                    "calculating replenishment."
                ),
            })
            results.append(result)
            continue

        if lead_time > expected_days:
            result.update({
                "lead_time_demand": np.nan,
                "reorder_point": np.nan,
                "stock_after_lead_time": np.nan,
                "stock_status": "INSUFFICIENT FORECAST",
                "suggested_replenishment_qty": np.nan,
                "analysis_note": (
                    f"Lead time is {lead_time} days but only "
                    f"{expected_days} forecast days are available. "
                    "Extend the forecast horizon."
                ),
            })
            results.append(result)
            continue

        # Sum demand across the lead-time horizon.
        lead_time_demand = float(
            daily_forecast.iloc[:lead_time].sum()
        )

        # Reorder point = lead-time demand + safety stock.
        reorder_point = lead_time_demand + safety_stock

        # Approximate remaining stock if no replenishment arrives
        # during lead time. This does not model stockouts or timing.
        stock_after_lead_time = stock - lead_time_demand

        if stock <= lead_time_demand:
            stock_status = "HIGH STOCKOUT RISK"
        elif stock <= reorder_point:
            stock_status = "REORDER NOW"
        elif stock <= reorder_point * 1.20:
            stock_status = "STOCK WATCH"
        else:
            stock_status = "STOCK OK"

        # Simple illustrative target: lead-time demand plus
        # safety stock. Actual order quantities need a target level,
        # pack-size constraints, open purchase orders, and other rules.
        suggested_qty = max(0.0, reorder_point - stock)

        result.update({
            "lead_time_demand": round(lead_time_demand, 2),
            "reorder_point": round(reorder_point, 2),
            "stock_after_lead_time": round(
                stock_after_lead_time, 2
            ),
            "stock_status": stock_status,
            "suggested_replenishment_qty": round(
                suggested_qty, 2
            ),
            "analysis_note": (
                "Illustrative result. Validate stock, demand units, "
                "lead time, and safety stock before operational use."
            ),
        })

        results.append(result)

    output = pd.DataFrame(results)
    output.to_csv(OUTPUT_PATH, index=False)

    print("\nINVENTORY ANALYSIS")
    print(output.to_string(index=False))

    print("\nSTOCK STATUS COUNTS")
    print(output["stock_status"].value_counts().to_string())

    print(f"\nSaved to: {OUTPUT_PATH.resolve()}")
    print(
        "\nReminder: inventory quantities and safety stock are "
        "illustrative until verified against actual store data."
    )


if __name__ == "__main__":
    main()
