import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

# --------------------------------------------------
# PAGE CONFIGURATION
# --------------------------------------------------
st.set_page_config(
    page_title="Retail Sales Forecasting & Inventory Analytics",
    page_icon="📊",
    layout="wide",
)

st.title("Retail Sales Forecasting & Inventory Analytics")
st.caption(
    "Historical sales analysis, demand forecasting, model evaluation, "
    "and illustrative inventory reorder alerts."
)

DATA_PATH = Path("data/train.csv")
INVENTORY_PATH = Path("inventory.csv")
MULTI_RESULTS_PATH = Path("multiple_model_evaluation.csv")


# --------------------------------------------------
# DATA LOADING AND MODEL TRAINING
# --------------------------------------------------
@st.cache_data
def load_data(path_string):
    data = pd.read_csv(path_string, parse_dates=["date"])
    required_columns = {
        "date", "store_nbr", "family", "sales", "onpromotion"
    }
    missing_columns = required_columns - set(data.columns)

    if missing_columns:
        raise ValueError(
            "The sales dataset is missing required columns: "
            + ", ".join(sorted(missing_columns))
        )

    data["sales"] = pd.to_numeric(data["sales"], errors="coerce")
    data["onpromotion"] = pd.to_numeric(
        data["onpromotion"], errors="coerce"
    ).fillna(0)

    data = data.dropna(
        subset=["date", "store_nbr", "family", "sales"]
    )
    return data.sort_values("date").reset_index(drop=True)


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


def add_features(data):
    """Create time-based features for one daily store-family series."""
    result = data.copy().sort_values("date").reset_index(drop=True)

    result["day_of_week"] = result["date"].dt.day_of_week
    result["day_of_month"] = result["date"].dt.day
    result["month"] = result["date"].dt.month
    result["year"] = result["date"].dt.year
    result["sales_lag_7"] = result["sales"].shift(7)
    result["sales_lag_14"] = result["sales"].shift(14)
    result["sales_avg_7"] = (
        result["sales"].shift(1).rolling(7).mean()
    )
    result["baseline_prediction"] = result["sales"].shift(7)
    return result


@st.cache_resource
def train_model(data):
    featured = add_features(data)
    training = featured.dropna(subset=FEATURES + ["sales"])

    if len(training) < 30:
        raise ValueError("Not enough historical records to train the model.")

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(training[FEATURES], training["sales"])
    return model, FEATURES


def calendar_diagnostics(data):
    """Return daily dates missing between the first and last observed date."""
    if data.empty or "date" not in data.columns:
        return pd.DatetimeIndex([])
    dates = pd.DatetimeIndex(
        pd.to_datetime(data["date"], errors="coerce")
        .dropna()
        .drop_duplicates()
        .sort_values()
    )
    if len(dates) == 0:
        return pd.DatetimeIndex([])
    expected = pd.date_range(dates.min(), dates.max(), freq="D")
    return expected.difference(dates)


def has_continuous_dates(data):
    """True when data contains every calendar date in its observed range."""
    return not data.empty and len(calendar_diagnostics(data)) == 0


def fill_calendar_gaps_with_zero(data):
    """Fill missing calendar dates with zero sales (only when user opts in)."""
    if data.empty:
        return data.copy()
    result = data.copy()
    result["date"] = pd.to_datetime(result["date"], errors="coerce")
    result = result.dropna(subset=["date"])
    result = result.sort_values("date").drop_duplicates("date", keep="last")
    full_dates = pd.date_range(result["date"].min(), result["date"].max(), freq="D")
    result = (
        result.set_index("date")
        .reindex(full_dates)
        .rename_axis("date")
        .reset_index()
    )
    result["sales"] = pd.to_numeric(result["sales"], errors="coerce").fillna(0)
    result["onpromotion"] = pd.to_numeric(
        result["onpromotion"], errors="coerce"
    ).fillna(0)
    return result


# --------------------------------------------------
# MAIN SALES DASHBOARD
# --------------------------------------------------
try:
    df = load_data(str(DATA_PATH))

    st.sidebar.header("Choose your analysis")
    stores = sorted(df["store_nbr"].unique())
    store = st.sidebar.selectbox("Store", stores)

    available_families = sorted(
        df.loc[df["store_nbr"] == store, "family"].unique()
    )
    family = st.sidebar.selectbox("Product family", available_families)

    selected = df[
        (df["store_nbr"] == store)
        & (df["family"] == family)
    ].copy()

    # There should be one record per date for a given store/family.
    # Aggregate defensively in case the source contains duplicate dates.
    daily = (
        selected.groupby("date", as_index=False)
        .agg(
            sales=("sales", "sum"),
            onpromotion=("onpromotion", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    if daily.empty:
        st.warning("No records were found for this store and product family.")
        st.stop()

    # --------------------------------------------------
    # CALENDAR GAP HANDLING (ONE CHECKBOX ONLY)
    # --------------------------------------------------
    missing_dates = calendar_diagnostics(daily)
    if len(missing_dates) > 0:
        st.sidebar.warning(f"{len(missing_dates)} missing calendar date(s) detected.")
        use_zero_fill = st.sidebar.checkbox(
            "Treat missing dates as zero sales",
            value=False,
            key="use_zero_fill",
            help=(
                "Enable only if you have verified that missing dates really mean "
                "zero sales, rather than missing records."
            ),
        )
    else:
        use_zero_fill = False
        st.sidebar.caption("No calendar gaps detected.")

    if use_zero_fill:
        daily = fill_calendar_gaps_with_zero(daily)
        st.warning(
            "Zero-filling is enabled. Missing dates are treated as zero sales. "
            "Verify this assumption before trusting forecasts or evaluation results."
        )
    st.sidebar.caption(
        "Continuous calendar: "
        + ("Yes" if has_continuous_dates(daily) else "No")
    )

    metric1, metric2, metric3 = st.columns(3)
    metric1.metric("Total Historical Sales", f"{daily['sales'].sum():,.0f}")
    metric2.metric("Average Daily Sales", f"{daily['sales'].mean():,.2f}")
    metric3.metric("Historical Days", f"{len(daily):,}")

    st.subheader("Historical Sales Trend")
    st.line_chart(daily.set_index("date")["sales"])

    st.caption(
        f"Historical data range: {daily['date'].min().date()} "
        f"to {daily['date'].max().date()}."
    )

    # --------------------------------------------------
    # SEVEN-DAY FORECAST
    # --------------------------------------------------
    st.divider()
    st.subheader("Generate a 7-Day Forecast")
    st.caption(
        "Forecast dates continue from the last date in the dataset. "
        "Future promotions are assumed to be zero because future promotion "
        "plans are not supplied by this dataset."
    )

    if st.button("Generate Forecast", key="generate_forecast"):
        if not has_continuous_dates(daily):
            st.error(
                "Forecast blocked because calendar dates are missing. "
                "Tick 'Treat missing dates as zero sales' only if you have "
                "verified that missing dates represent zero sales."
            )
        elif len(daily) < 30:
            st.error("Not enough historical data to generate a forecast.")
        else:
            try:
                # Fit against daily aggregated data to match the forecast series.
                model_data = daily.copy()
                model, features = train_model(model_data)

                history = daily["sales"].astype(float).tolist()
                last_date = daily["date"].max()
                forecast_rows = []

                for step in range(1, 8):
                    next_date = last_date + pd.Timedelta(days=step)

                    row = {
                        "day_of_week": next_date.day_of_week,
                        "day_of_month": next_date.day,
                        "month": next_date.month,
                        "year": next_date.year,
                        "onpromotion": 0,
                        "sales_lag_7": history[-7],
                        "sales_lag_14": history[-14],
                        "sales_avg_7": float(np.mean(history[-7:])),
                    }

                    prediction = max(
                        0.0,
                        float(
                            model.predict(
                                pd.DataFrame([row])[features]
                            )[0]
                        ),
                    )

                    forecast_rows.append({
                        "date": next_date,
                        "predicted_sales": round(prediction, 2),
                    })
                    history.append(prediction)

                forecast = pd.DataFrame(forecast_rows)
                st.success("Forecast generated successfully.")
                st.dataframe(
                    forecast, use_container_width=True, hide_index=True
                )

                st.subheader("Forecast Chart")
                st.bar_chart(
                    forecast.set_index("date")["predicted_sales"]
                )

                st.download_button(
                    "Download Forecast CSV",
                    data=forecast.to_csv(index=False).encode("utf-8"),
                    file_name=(
                        f"forecast_store_{store}_{str(family).lower()}.csv"
                    ),
                    mime="text/csv",
                    key="download_forecast",
                )
            except (ValueError, KeyError) as exc:
                st.error(f"Could not generate the forecast: {exc}")

    with st.expander("View Historical Data"):
        st.dataframe(
            selected.head(100), use_container_width=True, hide_index=True
        )

    # --------------------------------------------------
    # MODEL EVALUATION FOR SELECTED STORE AND FAMILY
    # --------------------------------------------------
    st.divider()
    st.header("Forecast Model Evaluation")
    st.write(
        "Compare Random Forest with a weekly-naive baseline on a "
        "chronological holdout period for the selected store and family."
    )

    if st.button("Evaluate Forecast Accuracy", key="evaluate_accuracy"):
        if not has_continuous_dates(daily):
            st.error(
                "Evaluation blocked because calendar dates are missing. "
                "Tick 'Treat missing dates as zero sales' only if verified."
            )
        else:
            evaluation_data = add_features(daily)
            evaluation_data = evaluation_data.dropna(
                subset=FEATURES + ["sales", "baseline_prediction"]
            ).reset_index(drop=True)

            if len(evaluation_data) < 60:
                st.warning(
                    "Not enough usable historical records for evaluation. "
                    "Choose a product family with more history."
                )
            else:
                split_index = int(len(evaluation_data) * 0.8)
                training_data = evaluation_data.iloc[:split_index]
                testing_data = evaluation_data.iloc[split_index:].copy()

                evaluation_model = RandomForestRegressor(
                    n_estimators=100,
                    random_state=42,
                    n_jobs=-1,
                    min_samples_leaf=2,
                )
                evaluation_model.fit(
                    training_data[FEATURES], training_data["sales"]
                )

                actual = testing_data["sales"].to_numpy()
                rf_predictions = np.maximum(
                    evaluation_model.predict(testing_data[FEATURES]), 0
                )
                baseline_predictions = (
                    testing_data["baseline_prediction"].to_numpy()
                )

                rf_mae = mean_absolute_error(actual, rf_predictions)
                rf_rmse = np.sqrt(mean_squared_error(actual, rf_predictions))
                baseline_mae = mean_absolute_error(
                    actual, baseline_predictions
                )
                baseline_rmse = np.sqrt(
                    mean_squared_error(actual, baseline_predictions)
                )

                st.subheader("Evaluation Metrics")
                mae_col1, mae_col2 = st.columns(2)
                mae_col1.metric("Random Forest MAE", f"{rf_mae:,.2f}")
                mae_col2.metric("Weekly Baseline MAE", f"{baseline_mae:,.2f}")

                rmse_col1, rmse_col2 = st.columns(2)
                rmse_col1.metric("Random Forest RMSE", f"{rf_rmse:,.2f}")
                rmse_col2.metric("Weekly Baseline RMSE", f"{baseline_rmse:,.2f}")

                comparison = pd.DataFrame({
                    "Model": ["Random Forest", "Weekly Baseline"],
                    "MAE": [rf_mae, baseline_mae],
                    "RMSE": [rf_rmse, baseline_rmse],
                })

                st.subheader("Selected Product: Model Comparison")
                st.dataframe(
                    comparison.style.format({"MAE": "{:.3f}", "RMSE": "{:.3f}"}),
                    use_container_width=True,
                    hide_index=True,
                )

                # Separate charts prevent MAE and RMSE from being visually
                # combined into a stacked bar and make each metric easier to compare.
                chart_col1, chart_col2 = st.columns(2)
                with chart_col1:
                    st.caption("MAE comparison (lower is better)")
                    st.bar_chart(comparison.set_index("Model")[["MAE"]])
                with chart_col2:
                    st.caption("RMSE comparison (lower is better)")
                    st.bar_chart(comparison.set_index("Model")[["RMSE"]])

                if baseline_mae > 0:
                    improvement = (
                        (baseline_mae - rf_mae) / baseline_mae
                    ) * 100
                    if improvement > 0:
                        st.success(
                            f"Random Forest reduced MAE by {improvement:.1f}% "
                            "compared with the weekly baseline on this test period."
                        )
                    elif improvement < 0:
                        st.warning(
                            f"Random Forest MAE was {abs(improvement):.1f}% "
                            "higher than the weekly baseline on this test period."
                        )
                    else:
                        st.info("Both models achieved the same MAE.")

                evaluation_results = pd.DataFrame({
                    "date": testing_data["date"].values,
                    "actual_sales": actual,
                    "random_forest_prediction": rf_predictions,
                    "weekly_baseline_prediction": baseline_predictions,
                })

                st.download_button(
                    "Download Selected Product Evaluation CSV",
                    data=evaluation_results.to_csv(index=False).encode("utf-8"),
                    file_name=f"evaluation_store_{store}_{str(family).lower()}.csv",
                    mime="text/csv",
                    key="download_selected_evaluation",
                )

    # --------------------------------------------------
    # ROBUST ROLLING-ORIGIN MODEL COMPARISON
    # --------------------------------------------------
    st.divider()
    st.header("Rolling-Origin Model Comparison")
    st.caption(
        "Each fold trains only on earlier dates and evaluates the next forecast "
        "window. This is more realistic than a random train/test split."
    )

    if st.button("Run Rolling Model Comparison", key="rolling_model_comparison"):
        try:
            if not has_continuous_dates(daily):
                st.error(
                    "Rolling comparison blocked because calendar dates are missing. "
                    "Only enable zero-filling if missing dates truly mean zero sales."
                )
            elif len(daily) < 100:
                st.warning(
                    "At least 100 daily records are recommended for rolling-origin "
                    "comparison. Select a series with more history."
                )
            else:
                horizons = [3, 7, 14]
                fold_rows = []
                series = daily.sort_values("date").reset_index(drop=True).copy()
                min_train = max(60, int(len(series) * 0.5))

                for horizon in horizons:
                    fold_number = 0
                    test_start = min_train
                    while test_start + horizon <= len(series):
                        train_part = series.iloc[:test_start].copy().reset_index(drop=True)
                        test_part = series.iloc[test_start:test_start + horizon].copy().reset_index(drop=True)
                        featured_train = add_features(train_part)
                        train_ready = featured_train.dropna(subset=FEATURES + ["sales"])
                        if len(train_ready) < 30:
                            test_start += horizon
                            continue

                        rf = RandomForestRegressor(
                            n_estimators=100,
                            random_state=42,
                            n_jobs=-1,
                            min_samples_leaf=2,
                        )
                        rf.fit(train_ready[FEATURES], train_ready["sales"])

                        # Recursive multi-step forecast: future actual sales are not
                        # used as lag values during this fold's forecast horizon.
                        history = train_part["sales"].astype(float).tolist()
                        predictions = []
                        for _, test_row in test_part.iterrows():
                            dt = pd.Timestamp(test_row["date"])
                            feature_row = {
                                "day_of_week": dt.dayofweek,
                                "day_of_month": dt.day,
                                "month": dt.month,
                                "year": dt.year,
                                "onpromotion": float(test_row.get("onpromotion", 0) or 0),
                                "sales_lag_7": history[-7],
                                "sales_lag_14": history[-14],
                                "sales_avg_7": float(np.mean(history[-7:])),
                            }
                            pred = max(0.0, float(rf.predict(pd.DataFrame([feature_row])[FEATURES])[0]))
                            predictions.append(pred)
                            history.append(pred)

                        actual = test_part["sales"].astype(float).to_numpy()
                        weekly_naive = train_part["sales"].astype(float).tolist()
                        baseline_predictions = []
                        for i in range(len(test_part)):
                            baseline_predictions.append(weekly_naive[-7] if len(weekly_naive) >= 7 else float(np.mean(weekly_naive)))
                            weekly_naive.append(float(test_part.iloc[i]["sales"]))

                        fold_number += 1
                        fold_rows.append({
                            "Horizon (days)": int(horizon),
                            "Fold": int(fold_number),
                            "Model": "Random Forest",
                            "MAE": float(mean_absolute_error(actual, predictions)),
                            "RMSE": float(np.sqrt(mean_squared_error(actual, predictions))),
                        })
                        fold_rows.append({
                            "Horizon (days)": int(horizon),
                            "Fold": int(fold_number),
                            "Model": "Weekly Naive",
                            "MAE": float(mean_absolute_error(actual, baseline_predictions)),
                            "RMSE": float(np.sqrt(mean_squared_error(actual, baseline_predictions))),
                        })
                        test_start += horizon

                if not fold_rows:
                    st.warning("No valid folds could be created for this series.")
                else:
                    fold_results = pd.DataFrame(fold_rows).reset_index(drop=True)
                    st.subheader("Rolling-Origin Metrics by Fold")
                    st.dataframe(fold_results, use_container_width=True, hide_index=True)

                    st.subheader("Average Error by Forecast Horizon")
                    summary = (
                        fold_results.groupby(["Horizon (days)", "Model"], as_index=False)
                        .agg(MAE=("MAE", "mean"), RMSE=("RMSE", "mean"), Folds=("Fold", "nunique"))
                        .reset_index(drop=True)
                    )
                    st.dataframe(summary, use_container_width=True, hide_index=True)

                    st.subheader("MAE Across Individual Folds")
                    # Avoid pivot/MultiIndex input to Streamlit charts. Use a
                    # normal, flat DataFrame with explicit labels instead.
                    fold_chart = fold_results.copy()
                    fold_chart["Fold Label"] = (
                        fold_chart["Horizon (days)"].astype(str)
                        + "-day | Fold " + fold_chart["Fold"].astype(str)
                        + " | " + fold_chart["Model"].astype(str)
                    )
                    chart_data = fold_chart[["Fold Label", "MAE"]].copy()
                    chart_data = chart_data.reset_index(drop=True)
                    chart_data = chart_data.rename(columns={"MAE": "Mean Absolute Error"})
                    st.bar_chart(chart_data, x="Fold Label", y="Mean Absolute Error")

                    st.download_button(
                        "Download Rolling Comparison CSV",
                        data=fold_results.to_csv(index=False).encode("utf-8"),
                        file_name="rolling_model_comparison.csv",
                        mime="text/csv",
                        key="download_rolling_comparison",
                    )
        except Exception as exc:
            st.error(f"Rolling model comparison failed: {exc}")
            with st.expander("Technical details"):
                import traceback
                st.code(traceback.format_exc())

    # --------------------------------------------------
    # MULTIPLE STORE-PRODUCT EVALUATION RESULTS
    # --------------------------------------------------
    st.divider()
    st.header("Performance Across Multiple Products")

    if MULTI_RESULTS_PATH.exists():
        multi_results = pd.read_csv(MULTI_RESULTS_PATH)

        required_result_columns = {
            "store",
            "family",
            "random_forest_mae",
            "baseline_mae",
        }

        if required_result_columns.issubset(multi_results.columns):
            multi_results["combination"] = (
                "Store "
                + multi_results["store"].astype(str)
                + " - "
                + multi_results["family"].astype(str)
            )

            st.write(
                "Saved historical holdout results for the five "
                "store-product combinations evaluated by "
                "`evaluate_multiple.py`."
            )

            display_columns = [
                column for column in [
                    "store",
                    "family",
                    "test_records",
                    "test_start",
                    "test_end",
                    "random_forest_mae",
                    "random_forest_rmse",
                    "baseline_mae",
                    "baseline_rmse",
                    "mae_improvement_percent",
                ]
                if column in multi_results.columns
            ]

            st.dataframe(
                multi_results[display_columns],
                use_container_width=True,
                hide_index=True,
            )

            st.subheader("MAE Comparison Across Combinations")
            st.bar_chart(
                multi_results.set_index("combination")[
                    ["random_forest_mae", "baseline_mae"]
                ]
            )

            st.download_button(
                "Download Multi-Product Evaluation CSV",
                data=multi_results.to_csv(index=False).encode("utf-8"),
                file_name="multiple_model_evaluation.csv",
                mime="text/csv",
                key="download_multi_evaluation",
            )

            if "mae_improvement_percent" in multi_results.columns:
                average_improvement = multi_results[
                    "mae_improvement_percent"
                ].mean()
                st.caption(
                    "Mean of the per-combination MAE improvement percentages: "
                    f"{average_improvement:.2f}%. This is a summary of the "
                    "saved test results, not a guarantee of future accuracy."
                )
        else:
            st.warning(
                "The multi-product evaluation CSV does not contain the "
                "expected columns. Run `python3 evaluate_multiple.py` again."
            )
    else:
        st.info(
            "Multi-product results are not available yet. Run "
            "`python3 evaluate_multiple.py` to create them."
        )

    # --------------------------------------------------
    # INVENTORY MANAGEMENT
    # --------------------------------------------------
    st.divider()
    st.header("Inventory Management & Reorder Alerts")

    if INVENTORY_PATH.exists():
        inventory = pd.read_csv(INVENTORY_PATH)

        required_inventory_columns = {
            "store_nbr",
            "family",
            "current_stock",
            "lead_time_days",
            "safety_stock",
        }

        if not required_inventory_columns.issubset(inventory.columns):
            st.error(
                "inventory.csv is missing one or more required columns: "
                "store_nbr, family, current_stock, lead_time_days, safety_stock."
            )
        else:
            # Use the full dataset to estimate average daily sales per
            # store/family. This is a demonstration estimate, not a current
            # demand or physical-unit estimate.
            average_sales = (
                df.groupby(["store_nbr", "family", "date"], as_index=False)
                ["sales"].sum()
                .groupby(["store_nbr", "family"], as_index=False)["sales"]
                .mean()
                .rename(columns={"sales": "avg_daily_sales"})
            )

            inventory_report = inventory.merge(
                average_sales,
                on=["store_nbr", "family"],
                how="left",
            )

            inventory_report["reorder_level"] = (
                inventory_report["avg_daily_sales"]
                * inventory_report["lead_time_days"]
                + inventory_report["safety_stock"]
            )

            inventory_report["status"] = np.where(
                inventory_report["avg_daily_sales"].isna(),
                "NO SALES MATCH",
                np.where(
                    inventory_report["current_stock"]
                    <= inventory_report["reorder_level"],
                    "REORDER",
                    "Stock OK",
                ),
            )

            inventory_report["avg_daily_sales"] = (
                inventory_report["avg_daily_sales"].round(2)
            )
            inventory_report["reorder_level"] = (
                inventory_report["reorder_level"].round(2)
            )

            reorder_count = int(
                (inventory_report["status"] == "REORDER").sum()
            )
            metric1, metric2 = st.columns(2)
            metric1.metric("Items Flagged for Reorder", reorder_count)
            metric2.metric("Inventory Items Analyzed", len(inventory_report))

            st.subheader("Inventory Status")
            st.dataframe(
                inventory_report,
                use_container_width=True,
                hide_index=True,
            )

            alerts = inventory_report[
                inventory_report["status"] == "REORDER"
            ]

            st.subheader("Reorder Alerts")
            if not alerts.empty:
                st.warning(
                    f"{len(alerts)} item(s) are below the calculated "
                    "demonstration reorder level."
                )
                st.dataframe(
                    alerts[
                        [
                            "store_nbr",
                            "family",
                            "current_stock",
                            "reorder_level",
                            "status",
                        ]
                    ],
                    use_container_width=True,
                    hide_index=True,
                )
            else:
                st.success(
                    "No matched inventory items are currently flagged for reorder."
                )

            st.download_button(
                "Download Inventory Report CSV",
                data=inventory_report.to_csv(index=False).encode("utf-8"),
                file_name="inventory_analysis.csv",
                mime="text/csv",
                key="download_inventory",
            )

            st.warning(
                "Inventory warning: inventory.csv contains illustrative stock "
                "figures. The dataset's sales measure has not been verified as "
                "physical units. Do not use these reorder levels for real "
                "purchasing until stock, demand, lead-time, and safety-stock "
                "units are consistent."
            )
    else:
        st.info(
            "Inventory data not found. Place inventory.csv in the project "
            "root folder to display inventory analysis."
        )

except FileNotFoundError:
    st.error(
        "Could not find data/train.csv. Confirm that the file exists at "
        "`data/train.csv` relative to the project folder."
    )
except pd.errors.ParserError as exc:
    st.error(f"Could not read the sales CSV. Check the file format. Details: {exc}")
except ValueError as exc:
    st.error(f"Data validation error: {exc}")
