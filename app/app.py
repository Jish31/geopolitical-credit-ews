"""
Geopolitical Credit Risk Early Warning System
Streamlit app — Sprint 6 deliverable

Run locally:  streamlit run app/app.py
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
import joblib
from pathlib import Path

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Geopolitical Credit EWS",
    page_icon="📊",
    layout="wide"
)

# ── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_resource
def load_model():
    model_path = Path("outputs/xgb_augmented_model.pkl")
    if model_path.exists():
        return joblib.load(model_path)
    return None

@st.cache_data(ttl=86400)
def load_gpr_history():
    """Load GPR index history — replace with live fetch in Sprint 6."""
    gpr_path = Path("data/processed/gpr_monthly.csv")
    if gpr_path.exists():
        return pd.read_csv(gpr_path, parse_dates=["date"], index_col="date")
    # Placeholder until real data is loaded
    dates = pd.date_range("2015-01", "2026-01", freq="MS")
    np.random.seed(42)
    return pd.DataFrame({"gpr_global": np.random.normal(100, 30, len(dates))}, index=dates)

# ── Sidebar inputs ───────────────────────────────────────────────────────────
st.sidebar.header("Scenario inputs")

sector = st.sidebar.selectbox(
    "Sector (NACE)",
    ["Agri-food (A/C10-12)", "Manufacturing (C)", "Construction (F)", "Services (G-N)"]
)
sector_key = sector.split("(")[0].strip().lower().replace("-", "_").replace("/", "_")

gpr_value = st.sidebar.slider(
    "Current GPR level",
    min_value=50, max_value=300, value=120,
    help="Caldara & Iacoviello GPR index. Historical mean ≈ 100."
)

energy_stress = st.sidebar.slider("Energy price stress index", 0, 100, 40)
fx_volatility = st.sidebar.slider("EUR/USD volatility (annualised %)", 5, 30, 10)
trade_openness = st.sidebar.slider("Trade flow stress", 0, 100, 30)

leverage = st.sidebar.slider("Leverage ratio", 0.0, 5.0, 1.5, step=0.1)
current_ratio = st.sidebar.slider("Current ratio", 0.5, 3.0, 1.2, step=0.1)
roa = st.sidebar.slider("ROA (%)", -20.0, 20.0, 4.0, step=0.5)

# ── Main layout ──────────────────────────────────────────────────────────────
st.title("📊 Geopolitical Credit Early Warning System")
st.caption("Irish corporate credit risk — sector-level PD uplift from geopolitical signals")

col1, col2, col3 = st.columns(3)

# Placeholder PD calculation until model is loaded
base_pd = 0.05 + (leverage - 1.5) * 0.02 - roa * 0.003
geo_uplift = (gpr_value - 100) * 0.0003 + energy_stress * 0.001
total_pd = np.clip(base_pd + geo_uplift, 0.001, 0.999)

with col1:
    st.metric("Predicted PD", f"{total_pd:.1%}",
              delta=f"+{geo_uplift:.1%} geo uplift",
              delta_color="inverse")
with col2:
    risk_band = "Low" if total_pd < 0.05 else "Medium" if total_pd < 0.15 else "High"
    st.metric("Risk band", risk_band)
with col3:
    st.metric("sGPR contribution", f"{geo_uplift/total_pd:.0%} of total PD")

st.divider()

# GPR trend chart
gpr_df = load_gpr_history()
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=gpr_df.index, y=gpr_df["gpr_global"],
    mode="lines", name="GPR (global)",
    line=dict(color="#185FA5", width=1.5)
))
fig.add_hline(y=100, line_dash="dot", line_color="gray",
              annotation_text="Historical mean")
fig.add_hline(y=gpr_value, line_dash="dash", line_color="#E24B4A",
              annotation_text="Current input")
fig.update_layout(
    title="Geopolitical Risk Index — 10-year history",
    xaxis_title=None, yaxis_title="GPR level",
    height=300, margin=dict(t=40, b=20, l=10, r=10),
    plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)"
)
st.plotly_chart(fig, use_container_width=True)

st.info("🔧 **Model not yet loaded** — this app uses placeholder calculations. "
        "Run `src/models/train.py` first to generate `outputs/xgb_augmented_model.pkl`, "
        "then refresh.")
