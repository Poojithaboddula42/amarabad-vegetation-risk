"""Amrabad Forest Stress Risk dashboard.

Run:   streamlit run app.py
Needs: the dashboard_data/ folder (created by amrabad_pipeline.py) next to this file.
Install: pip install streamlit streamlit-folium folium plotly xgboost pandas numpy
"""
import json
from pathlib import Path

import folium
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import xgboost as xgb
from plotly.subplots import make_subplots
from streamlit_folium import st_folium

D = Path(__file__).parent / "dashboard_data"
st.set_page_config(page_title="Amrabad Forest Stress Risk", page_icon="🌳", layout="wide")

if not (D / "panel.csv.gz").exists():
    st.error("dashboard_data/ not found. Run amrabad_pipeline.py first and keep the folder next to app.py.")
    st.stop()

CLASSES = ["Low", "Moderate", "High", "Very High"]
COLORS = {"Low": "#2e7d32", "Moderate": "#f9a825", "High": "#ef6c00", "Very High": "#c62828"}
NICE = {
    "ndvi": "Greenness (NDVI)", "ndvi_anom": "Greenness vs normal",
    "anom_lag1": "Greenness vs normal, 1 mo ago", "anom_lag2": "Greenness vs normal, 2 mo ago",
    "anom_lag3": "Greenness vs normal, 3 mo ago", "ndwi": "Leaf water (NDWI)",
    "rain": "Rainfall (mm)", "rain_lag1": "Rainfall 1 mo ago", "rain_lag2": "Rainfall 2 mo ago",
    "rain_lag3": "Rainfall 3 mo ago", "rain_3m": "Rainfall, last 3 months",
    "lst": "Land temperature (C)", "lst_lag1": "Temperature 1 mo ago",
    "lst_lag2": "Temperature 2 mo ago", "lst_lag3": "Temperature 3 mo ago",
    "lst_3m": "Temperature, 3-month avg", "elevation": "Elevation (m)", "slope": "Slope (deg)",
    "aspect": "Aspect (deg)", "m_sin": "Season (sin)", "m_cos": "Season (cos)",
}
RAIN_COLS = ["rain", "rain_lag1", "rain_lag2", "rain_lag3", "rain_3m"]
LST_COLS = ["lst", "lst_lag1", "lst_lag2", "lst_lag3", "lst_3m"]
INPUT_FEATS = ["ndvi", "ndvi_anom", "anom_lag1", "anom_lag2", "anom_lag3", "ndwi",
               "rain", "rain_lag1", "rain_lag2", "rain_lag3",
               "lst", "lst_lag1", "lst_lag2", "lst_lag3", "elevation", "slope", "aspect"]
BIG_STEP = {"rain", "rain_lag1", "rain_lag2", "rain_lag3", "lst", "lst_lag1", "lst_lag2",
            "lst_lag3", "elevation", "slope", "aspect"}


@st.cache_data
def load_data():
    panel = pd.read_csv(D / "panel.csv.gz", parse_dates=["date"])
    meta = json.loads((D / "metrics.json").read_text())
    return panel, meta


@st.cache_resource
def load_model():
    m = xgb.XGBClassifier()
    m.load_model(str(D / "model.json"))
    return m


panel, meta = load_data()
model = load_model()
FEATS, THR = meta["features"], meta["thresholds"]
months = list(pd.to_datetime(sorted(panel.date.unique())))
fmt = lambda t: t.strftime("%b %Y")


def classify(r):
    return np.select([r >= THR["very_high"], r >= THR["high"], r >= THR["moderate"]],
                     ["Very High", "High", "Moderate"], default="Low")


def predict(X):
    return model.predict_proba(X[FEATS].astype(float))[:, 1]


def contributions(row_df):
    """Per-feature contribution to the model score (log-odds) for ONE row."""
    c = model.get_booster().predict(xgb.DMatrix(row_df[FEATS].astype(float)), pred_contribs=True)[0]
    return pd.Series(c[:-1], index=FEATS)


def contrib_fig(row_df):
    c = contributions(row_df)
    top = c.reindex(c.abs().sort_values(ascending=False).index).head(8).iloc[::-1]
    fig = go.Figure(go.Bar(
        x=top.values, y=[NICE.get(i, i) for i in top.index], orientation="h",
        marker_color=["#c62828" if v > 0 else "#2e7d32" for v in top.values]))
    fig.update_layout(height=360, margin=dict(l=0, r=0, t=30, b=0), title="Why this score?",
                      xaxis_title="Effect on risk: right = pushes risk up, left = pushes it down")
    return fig, c


def drivers_sentence(c):
    up = c[c > 0].sort_values(ascending=False).head(3)
    if up.empty:
        return "Nothing in the inputs pushes the risk up."
    return "Main factors pushing risk **up**: " + ", ".join(NICE.get(i, i) for i in up.index) + "."


def make_map(d, risk_col="risk", cls_col="cls"):
    m = folium.Map(location=[d.latitude.mean(), d.longitude.mean()], zoom_start=9, tiles="CartoDB positron")
    for r in d.itertuples():
        folium.CircleMarker(
            [r.latitude, r.longitude], radius=4, color=COLORS[getattr(r, cls_col)],
            fill=True, fill_opacity=0.85, weight=0,
            tooltip=f"{r.pixel_id} | risk index {getattr(r, risk_col):.2f} ({getattr(r, cls_col)})").add_to(m)
    return m


def legend():
    st.markdown(" ".join(f"<span style='color:{COLORS[c]}'>&#9679;</span> {c}" for c in CLASSES),
                unsafe_allow_html=True)


def month_slider(label="Month", key=None):
    return st.select_slider(label, options=months, value=months[-1], format_func=fmt, key=key)


SPLIT_NOTE = {
    "train": ("warning", "This month is in the training period: the model has seen these outcomes, so it is not a fair test."),
    "val": ("info", "Validation period (2025)."),
    "test": ("info", "Test period (2026): outcomes were not used in training."),
    "predict": ("success", "Forecast: the outcome has not been observed yet."),
}

# ------------------------------------------------------------------ sidebar
page = st.sidebar.radio("Page", ["🗺️ Risk Map", "🔍 Drivers (SHAP)", "📈 Trends",
                                 "🎛️ What-if & Input", "ℹ️ Method & Limits"])
st.sidebar.caption(f"{meta['n_pixels']} forest pixels (250 m) | data to {meta['last_date']}")

# ------------------------------------------------------------------ 1. MAP
if page.startswith("🗺️"):
    st.title("🌳 Amrabad Tiger Reserve: Vegetation Stress Risk")
    st.caption("Risk index = chance-like score that vegetation will be well below its seasonal normal "
               f"{meta['horizon_months']} months after the selected month. It is an index, not a calibrated probability.")
    mon = month_slider()
    d = panel[panel.date == mon].copy()
    d["cls"] = classify(d.risk.values)
    kind, text = SPLIT_NOTE[d.split.iloc[0]]
    getattr(st, kind)(f"Risk for {fmt(mon + pd.DateOffset(months=meta['horizon_months']))}. {text}")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("High + Very High pixels", int(d.cls.isin(["High", "Very High"]).sum()))
    c2.metric("Mean risk index", f"{d.risk.mean():.2f}")
    c3.metric("Highest risk index", f"{d.risk.max():.2f}")
    if d.split.iloc[0] != "predict":
        c4.metric("Pixels that actually became stressed", int(d.label.sum()))
    else:
        c4.metric("Pixels that actually became stressed", "not yet known")

    left, right = st.columns([3, 1])
    with left:
        st_folium(make_map(d), height=520, use_container_width=True, returned_objects=[])
        legend()
    with right:
        counts = d.cls.value_counts().reindex(CLASSES, fill_value=0)
        fig = go.Figure(go.Bar(x=counts.index, y=counts.values, marker_color=[COLORS[c] for c in counts.index]))
        fig.update_layout(height=300, margin=dict(l=0, r=0, t=30, b=0), title="Pixels per class")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Top 15 hotspots")
    hot = d.sort_values("risk", ascending=False).head(15)[
        ["pixel_id", "latitude", "longitude", "elevation", "risk", "cls"]]
    hot.columns = ["Pixel", "Lat", "Lon", "Elevation (m)", "Risk index", "Class"]
    st.dataframe(hot.round(3), hide_index=True, use_container_width=True)
    st.download_button("Download this month (CSV)", d[["pixel_id", "latitude", "longitude", "risk", "cls"]]
                       .to_csv(index=False), file_name=f"risk_{mon:%Y_%m}.csv")

# ------------------------------------------------------------------ 2. DRIVERS
elif page.startswith("🔍"):
    st.title("What drives the risk?")
    st.caption("SHAP explains the model's behaviour, not the forest itself: these are statistical associations, not proven causes.")
    t1, t2 = st.tabs(["Global importance", "Explain one pixel"])
    with t1:
        a, b = st.columns(2)
        for col, img, cap in [(a, "shap_all.png", "All features"),
                              (b, "shap_external_only.png", "External drivers only (no NDVI features)")]:
            if (D / img).exists():
                col.image(str(D / img), caption=cap, use_container_width=True)
        st.info("NDVI-based features are partly circular because the label itself is built from NDVI. "
                "The right-hand plot (rain, temperature, terrain, season) is the more meaningful story.")
    with t2:
        mon = month_slider(key="drv_month")
        d = panel[panel.date == mon].sort_values("risk", ascending=False)
        pix = st.selectbox("Pixel (highest risk first)", d.pixel_id.head(40).tolist())
        row = d[d.pixel_id == pix]
        r = float(row.risk.iloc[0])
        st.metric("Risk index", f"{r:.2f}", classify(np.array([r]))[0])
        fig, c = contrib_fig(row)
        st.plotly_chart(fig, use_container_width=True)
        st.markdown(drivers_sentence(c))

# ------------------------------------------------------------------ 3. TRENDS
elif page.startswith("📈"):
    st.title("Trends")
    scope = st.radio("Scope", ["Whole reserve (average)", "Single pixel"], horizontal=True)
    if scope == "Single pixel":
        pix = st.selectbox("Pixel", sorted(panel.pixel_id.unique()))
        t = panel[panel.pixel_id == pix].sort_values("date")
    else:
        t = panel.groupby("date")[["ndvi_anom", "rain", "lst", "risk"]].mean().reset_index()
    fig = make_subplots(rows=4, cols=1, shared_xaxes=True, vertical_spacing=0.05,
                        subplot_titles=("Greenness vs seasonal normal (SD)", "Rainfall (mm)",
                                        "Land temperature (C)", "Risk index"))
    fig.add_trace(go.Scatter(x=t.date, y=t.ndvi_anom, line_color="#2e7d32"), 1, 1)
    fig.add_hline(y=meta["threshold_sd"], line_dash="dash", line_color="red", row=1, col=1)
    fig.add_trace(go.Bar(x=t.date, y=t.rain, marker_color="#1e88e5"), 2, 1)
    fig.add_trace(go.Scatter(x=t.date, y=t.lst, line_color="#ef6c00"), 3, 1)
    fig.add_trace(go.Scatter(x=t.date, y=t.risk, line_color="#6a1b9a"), 4, 1)
    fig.update_layout(height=760, showlegend=False, margin=dict(l=0, r=0, t=40, b=0))
    st.plotly_chart(fig, use_container_width=True)
    st.caption("Red dashed line = stress threshold. Risk before 2025 is in-sample (the model was trained on it).")

# ------------------------------------------------------------------ 4. WHAT-IF + INPUT
elif page.startswith("🎛️"):
    st.title("What-if & manual input")
    tab_a, tab_b = st.tabs(["Scenario on the map", "Type your own values"])

    with tab_a:
        st.caption("Change rainfall and temperature for the selected month and its lag features, then see how risk shifts.")
        mon = month_slider(key="wi_month")
        s1, s2 = st.columns(2)
        rain_pct = s1.slider("Rainfall change (%)", -80, 50, 0, 5)
        lst_delta = s2.slider("Temperature change (C)", -3.0, 5.0, 0.0, 0.5)
        base = panel[panel.date == mon].copy()
        scen = base.copy()
        scen[RAIN_COLS] = scen[RAIN_COLS] * (1 + rain_pct / 100)
        scen[LST_COLS] = scen[LST_COLS] + lst_delta
        base["risk"], scen["risk"] = predict(base), predict(scen)
        base["cls"], scen["cls"] = classify(base.risk.values), classify(scen.risk.values)

        m1, m2, m3 = st.columns(3)
        m1.metric("Mean risk index", f"{scen.risk.mean():.2f}", f"{scen.risk.mean() - base.risk.mean():+.3f} vs baseline")
        hi = lambda x: int(x.cls.isin(["High", "Very High"]).sum())
        m2.metric("High + Very High pixels", hi(scen), f"{hi(scen) - hi(base):+d} vs baseline")
        m3.metric("Scenario", f"Rain {rain_pct:+d}%, Temp {lst_delta:+.1f} C")

        left, right = st.columns([3, 1])
        with left:
            st_folium(make_map(scen), height=480, use_container_width=True, returned_objects=[])
            legend()
        with right:
            cb = base.cls.value_counts().reindex(CLASSES, fill_value=0)
            cs = scen.cls.value_counts().reindex(CLASSES, fill_value=0)
            fig = go.Figure([go.Bar(name="Baseline", x=CLASSES, y=cb.values, marker_color="#9e9e9e"),
                             go.Bar(name="Scenario", x=CLASSES, y=cs.values, marker_color="#6a1b9a")])
            fig.update_layout(barmode="group", height=320, margin=dict(l=0, r=0, t=30, b=0),
                              legend=dict(orientation="h"))
            st.plotly_chart(fig, use_container_width=True)
        st.caption("The model saw only ~2 years of training data, so treat extreme scenarios as indicative, not exact.")

    with tab_b:
        st.caption("Start from a real pixel and month, then edit any value. Season and 3-month totals are computed for you.")
        p1, p2 = st.columns(2)
        pix = p1.selectbox("Start from pixel", sorted(panel.pixel_id.unique()), key="in_pix")
        mon_b = p2.selectbox("...and month", months, index=len(months) - 1, format_func=fmt, key="in_mon")
        row = panel[(panel.pixel_id == pix) & (panel.date == mon_b)].iloc[0]

        vals = {}
        cols = st.columns(4)
        for i, f in enumerate(INPUT_FEATS):
            with cols[i % 4]:
                vals[f] = st.number_input(NICE[f], value=float(round(row[f], 3)), format="%.3f",
                                          step=1.0 if f in BIG_STEP else 0.05, key=f"{pix}_{mon_b}_{f}")
        vals["rain_3m"] = vals["rain"] + vals["rain_lag1"] + vals["rain_lag2"]
        vals["lst_3m"] = (vals["lst"] + vals["lst_lag1"] + vals["lst_lag2"]) / 3
        vals["m_sin"] = float(np.sin(2 * np.pi * mon_b.month / 12))
        vals["m_cos"] = float(np.cos(2 * np.pi * mon_b.month / 12))
        x = pd.DataFrame([vals])[FEATS]

        risk = float(predict(x)[0])
        cls = classify(np.array([risk]))[0]
        g1, g2 = st.columns([1, 2])
        with g1:
            gauge = go.Figure(go.Indicator(
                mode="gauge+number", value=risk, number={"valueformat": ".2f"},
                gauge={"axis": {"range": [0, 1]}, "bar": {"color": "#333"}, "steps": [
                    {"range": [0, THR["moderate"]], "color": COLORS["Low"]},
                    {"range": [THR["moderate"], THR["high"]], "color": COLORS["Moderate"]},
                    {"range": [THR["high"], THR["very_high"]], "color": COLORS["High"]},
                    {"range": [THR["very_high"], 1], "color": COLORS["Very High"]}]}))
            gauge.update_layout(height=260, margin=dict(l=10, r=10, t=30, b=0))
            st.plotly_chart(gauge, use_container_width=True)
            st.subheader(f"Class: {cls}")
        with g2:
            fig, c = contrib_fig(x)
            st.plotly_chart(fig, use_container_width=True)
            st.markdown(drivers_sentence(c))

# ------------------------------------------------------------------ 5. METHOD
else:
    st.title("Method & limitations")
    v, t = meta["val"], meta["test"]
    st.subheader("How well does it work?")
    tbl = pd.DataFrame({
        "Period": ["Validation 2025", "Test 2026 (Jan-Jun)"],
        "Base rate of stress": [v["base_rate"], t["base_rate"]],
        "PR-AUC": [v["pr_auc"], t["pr_auc"]],
        "ROC-AUC": [v["roc_auc"], t["roc_auc"]],
        "Hit rate in top 10% of scores": [v["top10_hit_rate"], t["top10_hit_rate"]],
        "Lift vs random": [v["lift_top10"], t["lift_top10"]],
    }).round(3)
    st.dataframe(tbl, hide_index=True, use_container_width=True)
    st.caption(f"Without NDVI features, test PR-AUC = {meta['test_no_ndvi']['pr_auc']:.3f}, "
               f"ROC-AUC = {meta['test_no_ndvi']['roc_auc']:.3f}.")
    st.markdown(f"""
**Task.** For each 250 m forest pixel, predict whether NDVI {meta['horizon_months']} months ahead will be below
{meta['threshold_sd']:+.1f} SD of that calendar month's 2018-2022 average (a same-season anomaly, so normal leaf-fall is not flagged).

**Model.** XGBoost on rainfall, land temperature, leaf water, terrain, season, and 1-3 month lags. Trained on 2023-2024,
validated on 2025, tested on Jan-Jun 2026. Risk classes are relative to the 2025-2026 score distribution.

**Limitations**
- Skill is modest and not consistent across years. Use it to prioritise attention, not as proof of anything.
- It flags vegetation *stress* (drought, fire, leaf-fall timing, clearing), not confirmed deforestation.
- Only two years of training data; the climate signal repeats across pixels, so effective sample size is small.
- Rainfall (5 km) and temperature (1 km) are coarser than the 250 m grid.
- Monsoon gaps (about 4% of values) were filled by interpolation.
- The label comes from satellite data, not field verification. SHAP shows what the model uses, not what causes loss.
""")