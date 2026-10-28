
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Retail Intelligence",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# DARK DASHBOARD CSS
# ============================================================

st.markdown("""
<style>
:root {
    --bg: #041329;
    --panel: #07182d;
    --panel-alt: #0a203c;
    --border: #14335a;
    --text: #dce9ff;
    --muted: #92a9ca;
    --teal: #11c6ad;
    --blue: #438cff;
}

.stApp {
    background: var(--bg);
    color: var(--text);
}

[data-testid="stHeader"] {
    background: rgba(4, 19, 41, 0.96);
}

[data-testid="stSidebar"] {
    background: #061226;
    border-right: 1px solid var(--border);
}

[data-testid="stSidebar"] * {
    color: var(--text);
}

.block-container {
    max-width: 1650px;
    padding-top: 1.3rem;
    padding-bottom: 2rem;
}

h1, h2, h3, h4 {
    color: #e2edff !important;
    letter-spacing: -0.35px;
}

p, .stCaption {
    color: var(--muted);
}

[data-testid="stMetric"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 12px;
    padding: 18px 20px;
    min-height: 105px;
    box-shadow: 0 5px 18px rgba(0, 0, 0, 0.15);
}

[data-testid="stMetricLabel"] {
    color: #9bb2d4 !important;
}

[data-testid="stMetricValue"] {
    color: #e6efff !important;
    font-weight: 750;
}

.stButton > button,
.stDownloadButton > button {
    background: #0c3260;
    color: #e8f1ff;
    border: 1px solid #24518a;
    border-radius: 8px;
    min-height: 2.65rem;
    font-weight: 600;
}

.stButton > button:hover,
.stDownloadButton > button:hover {
    background: #14518a;
    border-color: var(--teal);
    color: white;
}

.stButton > button[kind="primary"] {
    background: #087f8c;
    border-color: var(--teal);
    color: white;
}

[data-baseweb="select"] > div,
[data-baseweb="input"] > div {
    background: #091d38;
    border-color: #24436c;
    border-radius: 8px;
}

[data-testid="stDataFrame"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    overflow: hidden;
}

[data-testid="stTabs"] [data-baseweb="tab-list"] {
    background: var(--panel);
    border: 1px solid var(--border);
    border-radius: 10px;
    gap: 5px;
    padding: 5px;
}

[data-testid="stTabs"] [data-baseweb="tab"] {
    background: transparent;
    border-radius: 7px;
    color: #a9bdd9;
    padding: 10px 14px;
}

[data-testid="stTabs"] [aria-selected="true"] {
    background: #10345e !important;
    color: white !important;
}

[data-testid="stTabs"] [data-baseweb="tab-highlight"] {
    background: var(--teal);
}

.hero {
    padding: 1.6rem 1.8rem;
    border-radius: 14px;
    border: 1px solid var(--border);
    background: linear-gradient(120deg, #0b2342, #07162b 75%);
    margin-bottom: 1.3rem;
    box-shadow: 0 8px 28px rgba(0, 0, 0, 0.18);
}

.hero-kicker {
    text-transform: uppercase;
    letter-spacing: 0.14em;
    font-size: 0.74rem;
    font-weight: 700;
    color: var(--teal);
}

.hero-title {
    color: #e6efff;
    font-size: clamp(1.5rem, 2.4vw, 2.25rem);
    line-height: 1.2;
    font-weight: 750;
    margin: 0.45rem 0;
}

.hero-copy {
    color: #9bb2d4;
    line-height: 1.6;
    margin-bottom: 0;
}

[data-testid="stExpander"] {
    border: 1px solid var(--border);
    border-radius: 10px;
    background: var(--panel);
}

hr {
    border-color: var(--border);
}

@media (max-width: 768px) {
    .block-container {
        padding: 1rem 0.8rem 2rem;
    }
    .hero {
        padding: 1.1rem;
    }
}
</style>
""", unsafe_allow_html=True)

DATA_PATH = Path("data/train.csv")

if not DATA_PATH.exists():
    DATA_PATH = Path("data/train_sample.csv")

INVENTORY_PATH = Path("inventory.csv")
MULTI_RESULTS_PATH = Path("multiple_model_evaluation.csv")
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

# ============================================================
# DATA LOADING AND CLEANING
# ============================================================

@st.cache_data(show_spinner=False)
def load_data(path_string, modified_time):
    data = pd.read_csv(path_string)

    required = {
        "date", "store_nbr", "family", "sales", "onpromotion"
    }
    missing = required - set(data.columns)

    if missing:
        raise ValueError(
            "Dataset is missing columns: "
            + ", ".join(sorted(missing))
        )

    original_rows = len(data)
    data["date"] = pd.to_datetime(data["date"], errors="coerce")
    data["sales"] = pd.to_numeric(data["sales"], errors="coerce")
    data["onpromotion"] = pd.to_numeric(
        data["onpromotion"], errors="coerce"
    )

    quality = {
        "source_rows": original_rows,
        "invalid_dates": int(data["date"].isna().sum()),
        "invalid_sales": int(data["sales"].isna().sum()),
    }

    data = data.dropna(
        subset=["date", "store_nbr", "family", "sales"]
    ).copy()

    if data.empty:
        raise ValueError("No valid data remains after cleaning.")

    if not np.isfinite(data["sales"].to_numpy(dtype=float)).all():
        raise ValueError("Sales contains infinite values.")

    if (data["sales"] < 0).any():
        raise ValueError(
            "Negative sales found. Investigate the source data first."
        )

    bad_promotion = data["onpromotion"].notna() & (
        ~np.isfinite(data["onpromotion"])
        | (data["onpromotion"] < 0)
    )

    quality["invalid_promotions"] = int(bad_promotion.sum())
    data.loc[bad_promotion, "onpromotion"] = np.nan

    quality["missing_promotions_filled"] = int(
        data["onpromotion"].isna().sum()
    )
    data["onpromotion"] = data["onpromotion"].fillna(0.0)

    quality["rows_removed"] = original_rows - len(data)
    quality["duplicate_store_family_dates"] = int(
        data.duplicated(["date", "store_nbr", "family"]).sum()
    )

    data.attrs["quality"] = quality
    return data.sort_values("date").reset_index(drop=True)


def aggregate_daily(data):
    return (
        data.groupby("date", as_index=False)
        .agg(
            sales=("sales", "sum"),
            onpromotion=("onpromotion", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )


def find_missing_dates(data):
    if data.empty:
        return pd.DatetimeIndex([])

    dates = pd.DatetimeIndex(
        pd.to_datetime(data["date"])
        .dt.normalize()
        .dropna()
        .drop_duplicates()
        .sort_values()
    )

    expected = pd.date_range(dates.min(), dates.max(), freq="D")
    return expected.difference(dates)


def prepare_calendar(data, fill_missing=False):
    result = data.copy()
    result["date"] = pd.to_datetime(
        result["date"], errors="coerce"
    ).dt.normalize()
    result = result.dropna(subset=["date"])

    result = (
        result.groupby("date", as_index=False)
        .agg(
            sales=("sales", "sum"),
            onpromotion=("onpromotion", "sum"),
        )
        .sort_values("date")
        .reset_index(drop=True)
    )

    gaps = find_missing_dates(result)

    if len(gaps) and not fill_missing:
        raise ValueError(
            f"{len(gaps)} missing calendar dates detected. "
            "Verify the data or enable zero filling only if missing "
            "dates genuinely represent zero sales."
        )

    if len(gaps):
        full_dates = pd.date_range(
            result["date"].min(),
            result["date"].max(),
            freq="D",
        )
        result = result.set_index("date").reindex(full_dates)
        result.index.name = "date"
        result = result.reset_index()
        result["sales"] = result["sales"].fillna(0.0)
        result["onpromotion"] = result["onpromotion"].fillna(0.0)

    return result

# ============================================================
# MODEL AND FORECASTING
# ============================================================

def add_features(data):
    result = data.sort_values("date").copy().reset_index(drop=True)
    result["date"] = pd.to_datetime(result["date"])

    result["day_of_week"] = result["date"].dt.dayofweek
    result["day_of_month"] = result["date"].dt.day
    result["month"] = result["date"].dt.month
    result["year"] = result["date"].dt.year
    result["sales_lag_7"] = result["sales"].shift(7)
    result["sales_lag_14"] = result["sales"].shift(14)
    result["sales_avg_7"] = (
        result["sales"].shift(1).rolling(7, min_periods=7).mean()
    )
    return result


def train_model(data):
    featured = add_features(data)
    usable = featured.dropna(subset=FEATURES + ["sales"])

    if len(usable) < 60:
        raise ValueError(
            f"Only {len(usable)} usable training records. "
            "At least 60 are required."
        )

    model = RandomForestRegressor(
        n_estimators=250,
        min_samples_leaf=3,
        max_features=0.9,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(usable[FEATURES], usable["sales"])
    return model


def make_future_features(history, future_date, promotion=0.0):
    values = np.asarray(history, dtype=float)

    if len(values) < 14:
        raise ValueError("At least 14 historical days are required.")

    return {
        "day_of_week": future_date.dayofweek,
        "day_of_month": future_date.day,
        "month": future_date.month,
        "year": future_date.year,
        "onpromotion": float(promotion),
        "sales_lag_7": float(values[-7]),
        "sales_lag_14": float(values[-14]),
        "sales_avg_7": float(np.mean(values[-7:])),
    }


def recursive_forecast(data, horizon, promotion=0.0):
    if horizon < 1:
        raise ValueError("Forecast horizon must be positive.")

    if not np.isfinite(promotion) or promotion < 0:
        raise ValueError("Promotion must be finite and nonnegative.")

    history_df = data.copy()
    history_df["date"] = pd.to_datetime(
        history_df["date"], errors="coerce"
    ).dt.normalize()
    history_df["sales"] = pd.to_numeric(
        history_df["sales"], errors="coerce"
    )
    history_df["onpromotion"] = pd.to_numeric(
        history_df["onpromotion"], errors="coerce"
    )
    history_df = history_df.dropna(
        subset=["date", "sales", "onpromotion"]
    )
    history_df = aggregate_daily(history_df)

    if len(find_missing_dates(history_df)):
        raise ValueError(
            "Historical data contains missing dates. "
            "Resolve calendar gaps before forecasting."
        )

    model = train_model(history_df)
    sales_history = history_df["sales"].astype(float).tolist()

    future_dates = pd.date_range(
        history_df["date"].max() + pd.Timedelta(days=1),
        periods=int(horizon),
        freq="D",
    )

    rows = []

    for future_date in future_dates:
        features = make_future_features(
            sales_history, future_date, promotion
        )
        raw_prediction = float(
            model.predict(pd.DataFrame([features], columns=FEATURES))[0]
        )
        prediction = max(0.0, raw_prediction)

        rows.append({
            "date": future_date,
            "predicted_sales": round(prediction, 2),
            "raw_model_prediction": raw_prediction,
        })
        sales_history.append(prediction)

    return pd.DataFrame(rows)

# ============================================================
# EVALUATION
# ============================================================

def calculate_metrics(actual, predicted):
    actual = np.asarray(actual, dtype=float)
    predicted = np.asarray(predicted, dtype=float)
    total_actual = np.abs(actual).sum()

    wape = (
        np.abs(actual - predicted).sum() / total_actual * 100
        if total_actual > 0 else np.nan
    )

    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
        "WAPE (%)": float(wape),
    }


def one_step_backtest(data, test_days):
    if len(data) < max(90, test_days + 30):
        raise ValueError(
            f"At least {max(90, test_days + 30)} daily records "
            "are required for this holdout."
        )

    split = len(data) - test_days
    training = data.iloc[:split].copy()
    testing = data.iloc[split:].copy()

    model = train_model(training)
    combined = pd.concat([training, testing], ignore_index=True)
    featured = add_features(combined).iloc[split:].copy()
    featured = featured.dropna(subset=FEATURES + ["sales"])

    if len(featured) != len(testing):
        raise ValueError("Could not build evaluation features.")

    actual = testing["sales"].to_numpy(dtype=float)
    rf_prediction = np.maximum(
        0.0, model.predict(featured[FEATURES])
    )
    naive_prediction = featured["sales_lag_7"].to_numpy(dtype=float)

    return pd.DataFrame({
        "date": testing["date"].to_numpy(),
        "actual_sales": actual,
        "random_forest_prediction": rf_prediction,
        "weekly_naive_prediction": naive_prediction,
    })


def rolling_origin_backtest(data, horizon=7, max_folds=5):
    minimum_training = 120

    if len(data) < minimum_training + horizon:
        raise ValueError(
            f"At least {minimum_training + horizon} daily records "
            "are required for this backtest."
        )

    step_size = max(7, horizon)
    cutoffs = list(
        range(minimum_training, len(data) - horizon + 1, step_size)
    )[-max_folds:]

    rows = []

    for fold, cutoff in enumerate(cutoffs, start=1):
        training = data.iloc[:cutoff].copy()
        actual_block = data.iloc[cutoff:cutoff + horizon].copy()

        if len(actual_block) != horizon:
            continue

        rf = recursive_forecast(
            training, horizon=horizon, promotion=0.0
        )["predicted_sales"].to_numpy(dtype=float)

        last_week = training["sales"].tail(7).to_numpy(dtype=float)
        seasonal = np.array([
            last_week[i % 7] for i in range(horizon)
        ])
        moving_average = np.full(horizon, np.mean(last_week))
        actual = actual_block["sales"].to_numpy(dtype=float)

        predictions = {
            "Random Forest": rf,
            "Seasonal Naive": seasonal,
            "7-Day Moving Average": moving_average,
        }

        for model_name, predicted in predictions.items():
            for step in range(horizon):
                rows.append({
                    "fold": fold,
                    "cutoff_date": training.iloc[-1]["date"],
                    "date": actual_block.iloc[step]["date"],
                    "horizon_day": step + 1,
                    "model": model_name,
                    "actual_sales": float(actual[step]),
                    "predicted_sales": float(predicted[step]),
                })

    if not rows:
        raise ValueError("No valid backtest predictions were generated.")

    return pd.DataFrame(rows)

# ============================================================
# DASHBOARD HEADER
# ============================================================

st.markdown("""
<div class="hero">
    <div class="hero-kicker">RETAIL INTELLIGENCE PLATFORM</div>
    <div class="hero-title">📊 Retail Sales & Demand Analytics</div>
    <p class="hero-copy">
        Monitor sales performance, forecast future demand,
        evaluate predictive models, and review inventory signals.
    </p>
</div>
""", unsafe_allow_html=True)

# ============================================================
# MAIN APPLICATION
# ============================================================

try:
    if not DATA_PATH.exists():
        st.error(
            "Dataset not found at data/train.csv. Place your "
            "Kaggle train.csv inside a folder named data next to app.py."
        )
        st.stop()

    df = load_data(str(DATA_PATH), DATA_PATH.stat().st_mtime_ns)
    quality = df.attrs.get("quality", {})

    st.sidebar.title("📊 Retail Analytics")
    st.sidebar.caption("Choose your analysis")

    stores = sorted(df["store_nbr"].unique().tolist())
    store = st.sidebar.selectbox("Store", stores)

    families = sorted(
        df.loc[df["store_nbr"] == store, "family"].unique().tolist()
    )
    family = st.sidebar.selectbox("Product family", families)

    selected = df[
        (df["store_nbr"] == store) & (df["family"] == family)
    ].copy()

    daily_raw = aggregate_daily(selected)
    gaps = find_missing_dates(daily_raw)

    st.sidebar.divider()
    st.sidebar.subheader("Calendar quality")

    if len(gaps):
        st.sidebar.warning(f"{len(gaps)} missing date(s) detected")
        fill_gaps = st.sidebar.checkbox(
            "Treat missing dates as zero sales",
            value=False,
            help=(
                "Enable only if you have verified that missing dates "
                "genuinely mean zero sales."
            ),
        )
    else:
        fill_gaps = False
        st.sidebar.success("No calendar gaps detected")

    ready = True
    try:
        daily = prepare_calendar(daily_raw, fill_missing=fill_gaps)
    except ValueError as exc:
        daily = daily_raw.copy()
        ready = False
        st.warning(str(exc))

    # Summary cards stay visible above the tabs.
    st.subheader("Selected Store Overview")
    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Historical Sales", f"{daily['sales'].sum():,.0f}")
    col2.metric("Average Daily Sales", f"{daily['sales'].mean():,.2f}")
    col3.metric("Historical Days", f"{len(daily):,}")
    col4.metric("Calendar Status", "Ready" if ready else "Gaps Found")

    st.caption(
        f"Store: **{store}** · Family: **{family}** · "
        f"{daily['date'].min().date()} to {daily['date'].max().date()}"
    )

    tabs = st.tabs([
        "📈 Overview",
        "🔮 Forecast",
        "🎯 Model Evaluation",
        "🧪 Rolling Backtest",
        "📦 Inventory",
        "🧹 Data Quality",
    ])

    # ========================================================
    # TAB 1: OVERVIEW
    # ========================================================

    with tabs[0]:
        st.subheader("Revenue & Sales Overview")

        left, right = st.columns([2, 1])

        with left:
            st.subheader("Historical Sales Trend")
            st.line_chart(
                daily.set_index("date")[["sales"]],
                height=360,
            )

        with right:
            st.subheader("Recent 14-Day Sales")
            st.bar_chart(
                daily.tail(14).set_index("date")[["sales"]],
                height=360,
            )

        with st.expander("View historical records"):
            st.dataframe(
                daily.tail(100),
                use_container_width=True,
                hide_index=True,
            )

    # ========================================================
    # TAB 2: FORECAST
    # ========================================================

    with tabs[1]:
        st.subheader("Generate a Demand Forecast")
        st.caption(
            "Future promotion input is an assumption, not a known "
            "promotion plan. Use zero if future promotions are unknown."
        )

        col1, col2 = st.columns(2)

        horizon = col1.selectbox(
            "Forecast horizon",
            [3, 7, 14],
            index=1,
        )

        promotion = col2.number_input(
            "Assumed promotion value per future day",
            min_value=0.0,
            value=0.0,
            step=1.0,
        )

        if st.button("Generate Forecast", type="primary",
                     key="generate_forecast"):
            if not ready:
                st.error("Resolve calendar gaps before forecasting.")
            else:
                try:
                    with st.spinner("Generating forecast..."):
                        forecast = recursive_forecast(
                            daily, horizon, promotion
                        )

                    c1, c2, c3 = st.columns(3)
                    c1.metric(
                        "Forecast Total",
                        f"{forecast['predicted_sales'].sum():,.2f}",
                    )
                    c2.metric(
                        "Forecast Daily Average",
                        f"{forecast['predicted_sales'].mean():,.2f}",
                    )
                    c3.metric(
                        "Recent 28-Day Average",
                        f"{daily.tail(28)['sales'].mean():,.2f}",
                    )

                    st.subheader("Forecast Results")
                    st.dataframe(
                        forecast,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.subheader("Forecast Trend")
                    chart_data = forecast.set_index("date")[
                        ["predicted_sales"]
                    ]
                    st.line_chart(chart_data, height=300)

                    st.download_button(
                        "Download Forecast CSV",
                        data=forecast.to_csv(index=False).encode("utf-8"),
                        file_name=f"forecast_store_{store}.csv",
                        mime="text/csv",
                        key="download_forecast",
                    )

                    st.caption(
                        "Forecasts are estimates. Review historical "
                        "backtest performance before making decisions."
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    st.error(f"Forecast failed: {exc}")

    # ========================================================
    # TAB 3: MODEL EVALUATION
    # ========================================================

    with tabs[2]:
        st.subheader("Chronological Holdout Evaluation")
        st.caption(
            "The latest historical block is used as a test period. "
            "This is one-step-ahead evaluation using historical lag "
            "values, not a recursive multi-day forecast."
        )

        test_days = st.selectbox(
            "Holdout length",
            [28, 56, 90],
            index=1,
            key="holdout_days",
        )

        if st.button("Evaluate Forecast Accuracy",
                     key="evaluate_accuracy"):
            if not ready:
                st.error("Resolve calendar gaps before evaluation.")
            else:
                try:
                    with st.spinner("Evaluating models..."):
                        result = one_step_backtest(daily, test_days)

                    actual = result["actual_sales"]
                    rf = result["random_forest_prediction"]
                    naive = result["weekly_naive_prediction"]

                    summary = pd.DataFrame([
                        {"Model": "Random Forest",
                         **calculate_metrics(actual, rf)},
                        {"Model": "Weekly Naive",
                         **calculate_metrics(actual, naive)},
                    ])

                    st.subheader("Accuracy Metrics")
                    st.dataframe(
                        summary.style.format({
                            "MAE": "{:.3f}",
                            "RMSE": "{:.3f}",
                            "WAPE (%)": "{:.2f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.subheader("Actual vs Predicted")
                    st.line_chart(
                        result.set_index("date")[[
                            "actual_sales",
                            "random_forest_prediction",
                            "weekly_naive_prediction",
                        ]],
                        height=350,
                    )

                    naive_mae = calculate_metrics(actual, naive)["MAE"]
                    rf_mae = calculate_metrics(actual, rf)["MAE"]

                    if naive_mae > 0:
                        improvement = (
                            (naive_mae - rf_mae) / naive_mae
                        ) * 100

                        if improvement > 0:
                            st.success(
                                f"Random Forest MAE is {improvement:.1f}% "
                                "lower than the weekly baseline."
                            )
                        elif improvement < 0:
                            st.warning(
                                f"Random Forest MAE is "
                                f"{abs(improvement):.1f}% higher than "
                                "the weekly baseline."
                            )
                        else:
                            st.info("Both models have the same MAE.")

                    st.download_button(
                        "Download Evaluation CSV",
                        data=result.to_csv(index=False).encode("utf-8"),
                        file_name=f"evaluation_store_{store}.csv",
                        mime="text/csv",
                        key="download_evaluation",
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    st.error(f"Evaluation failed: {exc}")

    # ========================================================
    # TAB 4: ROLLING BACKTEST
    # ========================================================

    with tabs[3]:
        st.subheader("Rolling-Origin Model Comparison")
        st.write(
            "Compare Random Forest, Seasonal Naive, and a 7-day "
            "Moving Average over several forecast horizons."
        )

        folds = st.slider(
            "Number of historical folds",
            min_value=3,
            max_value=6,
            value=5,
        )

        if st.button("Run Model Comparison", type="primary",
                     key="run_rolling"):
            if not ready:
                st.error("Resolve calendar gaps before backtesting.")
            else:
                try:
                    results = []

                    with st.spinner("Running rolling backtests..."):
                        for h in [3, 7, 14]:
                            result = rolling_origin_backtest(
                                daily, horizon=h, max_folds=folds
                            )
                            result["horizon"] = h
                            results.append(result)

                    combined = pd.concat(results, ignore_index=True)
                    summaries = []

                    for (h, model_name), group in combined.groupby(
                        ["horizon", "model"]
                    ):
                        summaries.append({
                            "Horizon (days)": h,
                            "Model": model_name,
                            **calculate_metrics(
                                group["actual_sales"],
                                group["predicted_sales"],
                            ),
                        })

                    summary = pd.DataFrame(summaries).sort_values(
                        ["Horizon (days)", "MAE"]
                    )

                    st.subheader("Performance by Horizon")
                    st.dataframe(
                        summary.style.format({
                            "MAE": "{:.3f}",
                            "RMSE": "{:.3f}",
                            "WAPE (%)": "{:.2f}",
                        }),
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.subheader("MAE Comparison")
                    st.line_chart(
                        summary.pivot(
                            index="Horizon (days)",
                            columns="Model",
                            values="MAE",
                        )
                    )

                    st.subheader("Best Model at Each Horizon")
                    for h in [3, 7, 14]:
                        subset = summary[
                            summary["Horizon (days)"] == h
                        ].sort_values("MAE")

                        if not subset.empty:
                            winner = subset.iloc[0]
                            st.write(
                                f"**{h} days:** {winner['Model']} "
                                f"(MAE: {winner['MAE']:.3f})"
                            )

                    st.download_button(
                        "Download Rolling Backtest CSV",
                        data=combined.to_csv(index=False).encode("utf-8"),
                        file_name=f"rolling_backtest_store_{store}.csv",
                        mime="text/csv",
                        key="download_rolling",
                    )
                except (ValueError, KeyError, TypeError) as exc:
                    st.error(f"Backtest failed: {exc}")

    # ========================================================
    # TAB 5: INVENTORY
    # ========================================================

    with tabs[4]:
        st.subheader("Inventory and Reorder Review")
        st.warning(
            "Only calculate reorder levels after confirming that sales "
            "represents physical units sold and inventory quantities "
            "use compatible units."
        )

        units_confirmed = st.checkbox(
            "I verified that sales represents physical units sold",
            value=False,
        )

        if not INVENTORY_PATH.exists():
            st.info(
                "Optional: add inventory.csv in the project root with "
                "store_nbr, family, current_stock, lead_time_days, "
                "and safety_stock."
            )
        else:
            try:
                inventory = pd.read_csv(INVENTORY_PATH)

                required_inventory = {
                    "store_nbr", "family", "current_stock",
                    "lead_time_days", "safety_stock",
                }
                missing_inventory = (
                    required_inventory - set(inventory.columns)
                )

                if missing_inventory:
                    st.error(
                        "inventory.csv is missing: "
                        + ", ".join(sorted(missing_inventory))
                    )
                else:
                    numeric_cols = [
                        "current_stock", "lead_time_days", "safety_stock"
                    ]
                    for column in numeric_cols:
                        inventory[column] = pd.to_numeric(
                            inventory[column], errors="coerce"
                        )

                    invalid = (
                        inventory[numeric_cols].isna().any(axis=1)
                        | (inventory["current_stock"] < 0)
                        | (inventory["lead_time_days"] < 0)
                        | (inventory["safety_stock"] < 0)
                    )

                    if invalid.any():
                        st.warning(
                            f"{int(invalid.sum())} invalid inventory "
                            "row(s) excluded."
                        )

                    valid_inventory = inventory.loc[~invalid].copy()

                    if not units_confirmed:
                        st.info(
                            "Reorder calculations are disabled until "
                            "sales units are confirmed."
                        )
                        st.dataframe(
                            valid_inventory,
                            use_container_width=True,
                            hide_index=True,
                        )
                    else:
                        average_sales = (
                            df.groupby(
                                ["store_nbr", "family", "date"],
                                as_index=False,
                            )["sales"]
                            .sum()
                            .groupby(
                                ["store_nbr", "family"],
                                as_index=False,
                            )["sales"]
                            .mean()
                            .rename(columns={"sales": "avg_daily_units"})
                        )

                        report = valid_inventory.merge(
                            average_sales,
                            on=["store_nbr", "family"],
                            how="left",
                        )

                        report["reorder_level"] = (
                            report["avg_daily_units"]
                            * report["lead_time_days"]
                            + report["safety_stock"]
                        )

                        report["status"] = np.select(
                            [
                                report["avg_daily_units"].isna(),
                                report["current_stock"]
                                <= report["reorder_level"],
                            ],
                            ["NO SALES MATCH", "REORDER REVIEW"],
                            default="Stock above estimated threshold",
                        )

                        st.dataframe(
                            report,
                            use_container_width=True,
                            hide_index=True,
                        )

                        st.download_button(
                            "Download Inventory CSV",
                            data=report.to_csv(index=False).encode("utf-8"),
                            file_name="inventory_analysis.csv",
                            mime="text/csv",
                            key="download_inventory",
                        )

            except (ValueError, pd.errors.ParserError) as exc:
                st.error(f"Could not read inventory.csv: {exc}")

    # ========================================================
    # TAB 6: DATA QUALITY
    # ========================================================

    with tabs[5]:
        st.subheader("Dataset Health")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Source Rows", f"{quality.get('source_rows', len(df)):,}")
        c2.metric("Rows Removed", f"{quality.get('rows_removed', 0):,}")
        c3.metric(
            "Duplicate Keys",
            f"{quality.get('duplicate_store_family_dates', 0):,}",
        )
        c4.metric(
            "Invalid Promotions",
            f"{quality.get('invalid_promotions', 0):,}",
        )

        st.write({
            "Date range start": str(df["date"].min().date()),
            "Date range end": str(df["date"].max().date()),
            "Invalid dates in source": quality.get("invalid_dates", 0),
            "Invalid sales in source": quality.get("invalid_sales", 0),
            "Missing promotions filled with zero": quality.get(
                "missing_promotions_filled", 0
            ),
            "Selected series calendar gaps": int(len(gaps)),
            "Zero filling enabled": bool(fill_gaps),
        })

        with st.expander("Preview cleaned source data"):
            st.dataframe(
                selected.head(100),
                use_container_width=True,
                hide_index=True,
            )

except FileNotFoundError as exc:
    st.error(
        f"File not found: {exc}. Confirm data/train.csv exists "
        "relative to app.py."
    )

except pd.errors.ParserError as exc:
    st.error(f"Could not parse a CSV file: {exc}")

except ValueError as exc:
    st.error(f"Data validation error: {exc}")
