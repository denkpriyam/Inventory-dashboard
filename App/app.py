"""
NorthBay Living - Demand & Inventory Intelligence
Streamlit Dashboard (Workflow Step 10)

Pages: Home, Sales Analytics, Forecast, Inventory Dashboard, Risk Dashboard,
Product Details, Executive Summary - matching the course's Streamlit dashboard flow doc.

Run with:  streamlit run App/app.py
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# --------------------------------------------------------------------------------------
# Palette (validated categorical / status / sequential colors - see dataviz reference)
# --------------------------------------------------------------------------------------
CATEGORICAL = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS = {"good": "#0ca30c", "warning": "#fab219", "serious": "#ec835a", "critical": "#d03b3b"}
SEQUENTIAL_BLUE = "#2a78d6"
GRIDLINE = "#e1e0d9"
MUTED = "#898781"
RISK_COLOR = {"Reorder": STATUS["critical"], "Hold": STATUS["good"], "Clear": STATUS["warning"]}
RISK_ICON = {"Reorder": "🔴", "Hold": "🟢", "Clear": "🟡"}

PLOTLY_LAYOUT = dict(
    template="plotly_white",
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color="#0b0b0b"),
    plot_bgcolor="#fcfcfb",
    paper_bgcolor="#fcfcfb",
    margin=dict(l=10, r=10, t=40, b=10),
)


def style_fig(fig):
    fig.update_layout(**PLOTLY_LAYOUT)
    fig.update_xaxes(showgrid=False, linecolor=GRIDLINE)
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, zeroline=False)
    return fig


# --------------------------------------------------------------------------------------
# Data loading (cached)
# --------------------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent.parent  # repo root
DATA_DIR = BASE_DIR / "Cleaned + Splitted"
OUT_DIR = BASE_DIR / "CSV Outputs"


@st.cache_data
def load_data():
    sales = pd.read_csv(DATA_DIR / "sales_daily_cleaned_v2.csv", parse_dates=["date"])
    inventory = pd.read_csv(DATA_DIR / "inventory_snapshots_cleaned_v2.csv", parse_dates=["date"])
    sku = pd.read_csv(DATA_DIR / "sku_master_cleaned_v2.csv")
    calendar = pd.read_csv(DATA_DIR / "calender_cleaned_v2.csv", parse_dates=["date"])

    weekly = pd.read_csv(OUT_DIR / "weekly_panel_features.csv", parse_dates=["week"])
    baseline_backtest = pd.read_csv(OUT_DIR / "baseline_backtest.csv", parse_dates=["week"])
    ml_holdout = pd.read_csv(OUT_DIR / "ml_forecast_holdout.csv", parse_dates=["week"])
    ml_panel_compare = pd.read_csv(OUT_DIR / "ml_model_comparison_panel.csv")
    risk = pd.read_csv(OUT_DIR / "risk_scoring_results.csv")

    sales = sales.merge(sku, on=["store_id", "product_id"], how="left")
    return {
        "sales": sales,
        "inventory": inventory,
        "sku": sku,
        "calendar": calendar,
        "weekly": weekly,
        "baseline_backtest": baseline_backtest,
        "ml_holdout": ml_holdout,
        "ml_panel_compare": ml_panel_compare,
        "risk": risk,
    }


data = load_data()


# --------------------------------------------------------------------------------------
# Small reusable components
# --------------------------------------------------------------------------------------
def kpi_row(items):
    """items: list of (label, value, help_text)"""
    cols = st.columns(len(items))
    for col, (label, value, help_text) in zip(cols, items):
        with col:
            st.metric(label, value, help=help_text)


def risk_badge(flag):
    return f"{RISK_ICON.get(flag, '⚪')} **{flag}**"


# --------------------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------------------
st.set_page_config(page_title="NorthBay Living - Demand & Inventory Intelligence", layout="wide")

st.sidebar.title("📦 NorthBay Living")
st.sidebar.caption("Demand & Inventory Intelligence")
page = st.sidebar.radio(
    "Navigate",
    ["Home", "Sales Analytics", "Forecast", "Inventory Dashboard", "Risk Dashboard",
     "Product Details", "Executive Summary"],
)

sales, inventory, sku, calendar = data["sales"], data["inventory"], data["sku"], data["calendar"]
weekly, baseline_backtest = data["weekly"], data["baseline_backtest"]
ml_holdout, ml_panel_compare, risk = data["ml_holdout"], data["ml_panel_compare"], data["risk"]

BASELINE_WAPE = float(np.abs(baseline_backtest["units_sold"] - baseline_backtest["baseline_forecast"]).sum()
                      / baseline_backtest["units_sold"].sum())
BEST_MODEL_ROW = ml_panel_compare.sort_values("WAPE").iloc[0]

# ========================================================================================
# PAGE: HOME
# ========================================================================================
if page == "Home":
    st.title("Demand & Inventory Intelligence")
    st.caption("NorthBay Living - 5 stores x 20 products, Jan 2022 - Jan 2024")

    total_units = int(sales["units_sold"].sum())
    total_revenue = float((sales["units_sold"] * sales["price"]).sum())
    reorder_count = int((risk["risk_flag"] == "Reorder").sum())
    clear_count = int((risk["risk_flag"] == "Clear").sum())

    kpi_row([
        ("Total units sold", f"{total_units:,}", "Full 2-year history, all stores/products"),
        ("Total revenue", f"₹{total_revenue:,.0f}", "units_sold x price, full history"),
        ("SKUs needing reorder", f"{reorder_count}", "Stockout risk this week"),
        ("SKUs to clear", f"{clear_count}", "Overstock risk this week"),
        ("Best forecast model", f"{BEST_MODEL_ROW['Model']}", f"WAPE {BEST_MODEL_ROW['WAPE']:.3f} vs baseline {BASELINE_WAPE:.3f}"),
    ])

    st.divider()
    st.subheader("Weekly demand — full portfolio")
    portfolio_weekly = weekly.groupby("week", as_index=False)["units_sold"].sum()
    fig = px.line(portfolio_weekly, x="week", y="units_sold")
    fig.update_traces(line_color=SEQUENTIAL_BLUE, line_width=2)
    fig.update_layout(yaxis_title="Units sold", xaxis_title=None)
    st.plotly_chart(style_fig(fig), width='stretch')

    st.info(
        "Use the sidebar to explore **Sales Analytics** (trends & drivers), **Forecast** "
        "(model comparison), **Inventory** and **Risk** (stockout/overstock), drill into a "
        "single **Product**, or read the **Executive Summary**."
    )

# ========================================================================================
# PAGE: SALES ANALYTICS
# ========================================================================================
elif page == "Sales Analytics":
    st.title("Sales Analytics")

    col1, col2 = st.columns(2)
    with col1:
        store_filter = st.multiselect("Store", sorted(sales["store_id"].unique()), default=[])
    with col2:
        category_filter = st.multiselect("Category", sorted(sales["category"].unique()), default=[])

    filtered = sales.copy()
    if store_filter:
        filtered = filtered[filtered["store_id"].isin(store_filter)]
    if category_filter:
        filtered = filtered[filtered["category"].isin(category_filter)]

    st.subheader("Daily units sold over time")
    daily_trend = filtered.groupby("date", as_index=False)["units_sold"].sum()
    fig = px.line(daily_trend, x="date", y="units_sold")
    fig.update_traces(line_color=SEQUENTIAL_BLUE, line_width=1.5)
    st.plotly_chart(style_fig(fig), width='stretch')

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Avg units sold by category")
        by_cat = filtered.groupby("category", as_index=False)["units_sold"].mean().sort_values("units_sold", ascending=False)
        fig = px.bar(by_cat, x="category", y="units_sold", color="category", color_discrete_sequence=CATEGORICAL)
        fig.update_layout(showlegend=False, yaxis_title="Avg units sold/day", xaxis_title=None)
        st.plotly_chart(style_fig(fig), width='stretch')
    with c2:
        st.subheader("Avg units sold by store")
        by_store = filtered.groupby("store_id", as_index=False)["units_sold"].mean().sort_values("store_id")
        fig = px.bar(by_store, x="store_id", y="units_sold", color="store_id", color_discrete_sequence=CATEGORICAL)
        fig.update_layout(showlegend=False, yaxis_title="Avg units sold/day", xaxis_title=None)
        st.plotly_chart(style_fig(fig), width='stretch')

    c3, c4 = st.columns(2)
    with c3:
        st.subheader("Discount level vs avg units sold")
        by_disc = filtered.groupby("discount", as_index=False)["units_sold"].mean()
        fig = px.bar(by_disc, x="discount", y="units_sold")
        fig.update_traces(marker_color=SEQUENTIAL_BLUE)
        fig.update_layout(yaxis_title="Avg units sold", xaxis_title="Discount %")
        st.plotly_chart(style_fig(fig), width='stretch')
    with c4:
        st.subheader("Promotion off vs on")
        by_promo = filtered.groupby("promotion", as_index=False)["units_sold"].mean()
        by_promo["promotion"] = by_promo["promotion"].map({0: "Off", 1: "On"})
        fig = px.bar(by_promo, x="promotion", y="units_sold")
        fig.update_traces(marker_color=CATEGORICAL[2])
        fig.update_layout(yaxis_title="Avg units sold", xaxis_title=None)
        st.plotly_chart(style_fig(fig), width='stretch')

# ========================================================================================
# PAGE: FORECAST
# ========================================================================================
elif page == "Forecast":
    st.title("Forecast — Model Comparison")

    st.subheader("Holdout WAPE by model (lower is better)")
    cmp = ml_panel_compare.sort_values("WAPE")
    fig = px.bar(cmp, x="Model", y="WAPE", text="WAPE")
    fig.update_traces(marker_color=SEQUENTIAL_BLUE, texttemplate="%{text:.3f}", textposition="outside")
    fig.add_hline(y=BASELINE_WAPE, line_dash="dash", line_color=STATUS["critical"],
                  annotation_text="baseline", annotation_position="top left")
    st.plotly_chart(style_fig(fig), width='stretch')
    st.caption(f"Best model: **{BEST_MODEL_ROW['Model']}** — WAPE {BEST_MODEL_ROW['WAPE']:.3f} "
               f"vs seasonal-naive baseline {BASELINE_WAPE:.3f}.")

    st.divider()
    st.subheader("Actual vs forecast — pick a store/product")
    c1, c2 = st.columns(2)
    with c1:
        store_pick = st.selectbox("Store", sorted(ml_holdout["store_id"].unique()))
    with c2:
        product_pick = st.selectbox("Product", sorted(ml_holdout[ml_holdout["store_id"] == store_pick]["product_id"].unique()))

    hist = weekly[(weekly["store_id"] == store_pick) & (weekly["product_id"] == product_pick)].sort_values("week")
    hold = ml_holdout[(ml_holdout["store_id"] == store_pick) & (ml_holdout["product_id"] == product_pick)].sort_values("week")

    fig = go.Figure()
    fig.add_trace(go.Scatter(x=hist["week"], y=hist["units_sold"], name="Actual (history)",
                              line=dict(color="#0b0b0b", width=2)))
    fig.add_trace(go.Scatter(x=hold["week"], y=hold["units_sold"], name="Actual (holdout)",
                              mode="markers+lines", line=dict(color="#0b0b0b", width=2), marker=dict(size=8)))
    fig.add_trace(go.Scatter(x=hold["week"], y=hold["baseline_forecast"], name="Baseline",
                              line=dict(color=STATUS["critical"], width=2, dash="dash")))
    fig.add_trace(go.Scatter(x=hold["week"], y=hold["rf_forecast"], name="Random Forest",
                              line=dict(color=CATEGORICAL[0], width=2)))
    fig.add_trace(go.Scatter(x=hold["week"], y=hold["xgb_forecast"], name="XGBoost",
                              line=dict(color=CATEGORICAL[1], width=2)))
    fig.add_trace(go.Scatter(x=hold["week"], y=hold["lgbm_forecast"], name="LightGBM",
                              line=dict(color=CATEGORICAL[2], width=2)))
    fig.update_layout(yaxis_title="Units sold (weekly)", xaxis_title=None, legend=dict(orientation="h", y=-0.2))
    st.plotly_chart(style_fig(fig), width='stretch')

# ========================================================================================
# PAGE: INVENTORY DASHBOARD
# ========================================================================================
elif page == "Inventory Dashboard":
    st.title("Inventory Dashboard")

    latest_date = inventory["date"].max()
    latest_inv = inventory[inventory["date"] == latest_date]
    st.caption(f"Latest snapshot: {latest_date.date()}")

    kpi_row([
        ("Total units in stock", f"{int(latest_inv['inventory_level'].sum()):,}", "Latest snapshot, all stores"),
        ("Historical stockout rows", f"{int((inventory['inventory_level']==0).sum()):,}",
         f"{(inventory['inventory_level']==0).mean()*100:.2f}% of all inventory rows"),
        ("Median days of cover (now)", f"{risk['days_of_cover'].median():.1f}", "current inventory / forecast demand"),
    ])

    st.subheader("Days of cover distribution (current)")
    low_thr, high_thr = risk["days_of_cover"].quantile([0.10, 0.90])
    fig = px.histogram(risk, x="days_of_cover", nbins=40)
    fig.update_traces(marker_color=SEQUENTIAL_BLUE)
    fig.add_vline(x=low_thr, line_dash="dash", line_color=STATUS["critical"], annotation_text="Reorder")
    fig.add_vline(x=high_thr, line_dash="dash", line_color=STATUS["warning"], annotation_text="Clear")
    fig.update_layout(xaxis_title="Days of cover", yaxis_title="SKU-stores")
    st.plotly_chart(style_fig(fig), width='stretch')

    st.subheader("Historical stockouts by product")
    stockout_by_product = (inventory[inventory["inventory_level"] == 0]
                            .groupby("product_id").size().sort_values(ascending=False).reset_index(name="stockout_days"))
    fig = px.bar(stockout_by_product, x="product_id", y="stockout_days")
    fig.update_traces(marker_color=CATEGORICAL[7])
    st.plotly_chart(style_fig(fig), width='stretch')

    st.subheader("Inventory level over time — pick a store/product")
    c1, c2 = st.columns(2)
    with c1:
        store_pick = st.selectbox("Store", sorted(inventory["store_id"].unique()), key="inv_store")
    with c2:
        product_pick = st.selectbox("Product", sorted(inventory["product_id"].unique()), key="inv_product")
    series = inventory[(inventory["store_id"] == store_pick) & (inventory["product_id"] == product_pick)].sort_values("date")
    fig = px.line(series, x="date", y="inventory_level")
    fig.update_traces(line_color=SEQUENTIAL_BLUE, line_width=1.5)
    st.plotly_chart(style_fig(fig), width='stretch')

# ========================================================================================
# PAGE: RISK DASHBOARD
# ========================================================================================
elif page == "Risk Dashboard":
    st.title("Risk Dashboard")

    counts = risk["risk_flag"].value_counts()
    kpi_row([
        ("🔴 Reorder", int(counts.get("Reorder", 0)), "Stockout risk this week"),
        ("🟢 Hold", int(counts.get("Hold", 0)), "Healthy stock position"),
        ("🟡 Clear", int(counts.get("Clear", 0)), "Overstock risk"),
        ("Capital locked (Clear)", f"₹{risk['capital_locked'].sum():,.0f}", "Money tied up in slow-moving stock"),
        ("Revenue at risk (Reorder)", f"₹{risk['revenue_at_risk'].sum():,.0f}", "Unmet demand over the next 7 days if not reordered"),
    ])

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Risk flags by category")
        ct = risk.groupby(["category", "risk_flag"]).size().reset_index(name="count")
        fig = px.bar(ct, x="category", y="count", color="risk_flag", color_discrete_map=RISK_COLOR, barmode="stack")
        st.plotly_chart(style_fig(fig), width='stretch')
    with c2:
        st.subheader("Risk flags by store")
        ct = risk.groupby(["store_id", "risk_flag"]).size().reset_index(name="count")
        fig = px.bar(ct, x="store_id", y="count", color="risk_flag", color_discrete_map=RISK_COLOR, barmode="stack")
        st.plotly_chart(style_fig(fig), width='stretch')

    st.subheader("Action list")
    tab1, tab2 = st.tabs(["🔴 Reorder now", "🟡 Clear stock"])
    with tab1:
        cols = ["store_id", "product_id", "category", "inventory_level", "days_of_cover", "revenue_at_risk"]
        st.dataframe(risk[risk["risk_flag"] == "Reorder"][cols].sort_values("days_of_cover"),
                     width='stretch', hide_index=True)
    with tab2:
        cols = ["store_id", "product_id", "category", "inventory_level", "days_of_cover", "capital_locked"]
        st.dataframe(risk[risk["risk_flag"] == "Clear"][cols].sort_values("days_of_cover", ascending=False),
                     width='stretch', hide_index=True)

# ========================================================================================
# PAGE: PRODUCT DETAILS
# ========================================================================================
elif page == "Product Details":
    st.title("Product Details")

    c1, c2 = st.columns(2)
    with c1:
        store_pick = st.selectbox("Store", sorted(sku["store_id"].unique()), key="pd_store")
    with c2:
        product_pick = st.selectbox("Product", sorted(sku[sku["store_id"] == store_pick]["product_id"].unique()), key="pd_product")

    info = sku[(sku["store_id"] == store_pick) & (sku["product_id"] == product_pick)].iloc[0]
    risk_row = risk[(risk["store_id"] == store_pick) & (risk["product_id"] == product_pick)]

    st.subheader(f"{store_pick} / {product_pick} — {info['category']} ({info['region']})")

    if not risk_row.empty:
        r = risk_row.iloc[0]
        kpi_row([
            ("Current inventory", f"{int(r['inventory_level']):,}", None),
            ("Forecasted daily demand", f"{r['avg_daily_demand_forecast']:.1f}", None),
            ("Days of cover", f"{r['days_of_cover']:.1f}", None),
            ("Risk flag", risk_badge(r["risk_flag"]), None),
        ])

    st.subheader("Sales history (daily units sold)")
    hist_sales = sales[(sales["store_id"] == store_pick) & (sales["product_id"] == product_pick)].sort_values("date")
    fig = px.line(hist_sales, x="date", y="units_sold")
    fig.update_traces(line_color=SEQUENTIAL_BLUE, line_width=1.5)
    st.plotly_chart(style_fig(fig), width='stretch')

    st.subheader("Inventory history")
    hist_inv = inventory[(inventory["store_id"] == store_pick) & (inventory["product_id"] == product_pick)].sort_values("date")
    fig = px.line(hist_inv, x="date", y="inventory_level")
    fig.update_traces(line_color=CATEGORICAL[2], line_width=1.5)
    st.plotly_chart(style_fig(fig), width='stretch')

    st.subheader("Forecast vs actual (holdout weeks)")
    hold = ml_holdout[(ml_holdout["store_id"] == store_pick) & (ml_holdout["product_id"] == product_pick)].sort_values("week")
    if not hold.empty:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=hold["week"], y=hold["units_sold"], name="Actual", line=dict(color="#0b0b0b", width=2)))
        fig.add_trace(go.Scatter(x=hold["week"], y=hold["rf_forecast"], name="Random Forest", line=dict(color=SEQUENTIAL_BLUE, width=2)))
        fig.add_trace(go.Scatter(x=hold["week"], y=hold["baseline_forecast"], name="Baseline", line=dict(color=STATUS["critical"], width=2, dash="dash")))
        fig.update_layout(yaxis_title="Units sold (weekly)", xaxis_title=None, legend=dict(orientation="h", y=-0.2))
        st.plotly_chart(style_fig(fig), width='stretch')

# ========================================================================================
# PAGE: EXECUTIVE SUMMARY
# ========================================================================================
elif page == "Executive Summary":
    st.title("Executive Summary")

    st.markdown("""
**Business problem.** NorthBay Living needs to forecast weekly demand per store-product and
flag stockout/overstock risk early enough to act on it.
""")

    kpi_row([
        ("Best model", BEST_MODEL_ROW["Model"], None),
        ("WAPE improvement vs baseline", f"{(1 - BEST_MODEL_ROW['WAPE']/BASELINE_WAPE)*100:.0f}%", None),
        ("SKUs at risk right now", f"{int((risk['risk_flag'] != 'Hold').sum())} / {len(risk)}", None),
        ("Total ₹ at stake", f"₹{(risk['capital_locked'].sum() + risk['revenue_at_risk'].sum()):,.0f}", "Capital locked + revenue at risk"),
    ])

    st.divider()
    st.markdown(f"""
### Key findings

1. **Epidemic periods are the single strongest demand driver found** — average daily units
   sold drop by roughly a third during epidemic periods. Any forecast that ignores this will
   be systematically wrong exactly when planning matters most.
2. **Discount and promotion genuinely move demand** in this dataset (correlation ≈0.18 and
   ≈0.23 with units sold) — both a usable forecast signal and a lever the business can pull.
3. **The best model ({BEST_MODEL_ROW['Model']}) beats the seasonal-naive baseline** —
   WAPE {BEST_MODEL_ROW['WAPE']:.3f} vs {BASELINE_WAPE:.3f} — and beats the existing `demand`
   column's ~32% daily MAPE benchmark.
4. **Risk scoring surfaces concrete, actionable numbers today**: {int((risk['risk_flag']=='Reorder').sum())}
   SKU-stores need reordering (₹{risk['revenue_at_risk'].sum():,.0f} revenue at risk over the
   next week), and {int((risk['risk_flag']=='Clear').sum())} are overstocked
   (₹{risk['capital_locked'].sum():,.0f} of capital that could be freed up).
5. **Category and store differences are real** — reorder/overstock thresholds should vary by
   category rather than using one blanket rule.

### Recommendation

Adopt **{BEST_MODEL_ROW['Model']}** as the production forecast, refresh the risk table weekly
against each new inventory snapshot, and route the **Reorder** and **Clear** lists directly to
the operations team from the Risk Dashboard page.
""")
