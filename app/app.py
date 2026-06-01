"""
Geopolitical Credit Risk Early Warning System
==============================================
Sprint 6 — Streamlit Dashboard v3.1
"""
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
import joblib
from pathlib import Path
from datetime import datetime

st.set_page_config(
    page_title="GeoCredit EWS",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Fraunces:ital,opsz,wght@0,9..144,300;0,9..144,400;0,9..144,600;0,9..144,700;1,9..144,300&family=DM+Sans:wght@300;400;500&family=DM+Mono:wght@400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    color: #cbd5e1;
}
.stApp { background-color: #07090f; }
.main .block-container { padding: 2.5rem 3rem; max-width: 1380px; }

/* Sidebar */
div[data-testid="stSidebar"] {
    background: #050710;
    border-right: 1px solid rgba(148,163,184,0.08);
}
div[data-testid="stSidebar"] .block-container { padding: 1.5rem 1.2rem; }

/* Headings */
h1 {
    font-family: 'Fraunces', serif !important;
    font-weight: 700 !important;
    font-size: 3rem !important;
    letter-spacing: -0.03em !important;
    line-height: 1.05 !important;
    color: #f1f5f9 !important;
}
h2 {
    font-family: 'Fraunces', serif !important;
    font-weight: 600 !important;
    font-size: 1.5rem !important;
    color: #e2e8f0 !important;
    letter-spacing: -0.01em !important;
    margin-top: 0.5rem !important;
}
h3 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 1rem !important;
    color: #94a3b8 !important;
    letter-spacing: 0.02em !important;
    text-transform: uppercase !important;
}

/* KPI grid */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 1rem;
    margin: 1.5rem 0 1rem;
}
.kpi-card {
    background: #0d1117;
    border: 1px solid rgba(148,163,184,0.1);
    border-radius: 14px;
    padding: 1.5rem;
    position: relative;
    overflow: hidden;
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0; height: 1px;
    background: linear-gradient(90deg, transparent, rgba(148,163,184,0.15), transparent);
}
.kpi-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    color: #475569;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    margin-bottom: 0.8rem;
}
.kpi-value {
    font-family: 'Fraunces', serif;
    font-size: 2.6rem;
    font-weight: 700;
    line-height: 1;
    margin-bottom: 0.5rem;
    letter-spacing: -0.02em;
}
.kpi-sub {
    font-size: 0.78rem;
    color: #475569;
    line-height: 1.4;
}
.val-high    { color: #f87171; }
.val-medium  { color: #fb923c; }
.val-low     { color: #4ade80; }
.val-neutral { color: #7dd3fc; }

/* Risk pill */
.risk-pill {
    display: inline-block;
    padding: 0.3rem 1rem;
    border-radius: 100px;
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    letter-spacing: 0.1em;
    font-weight: 500;
}
.pill-high   { background: rgba(248,113,113,0.1); color: #f87171; border: 1px solid rgba(248,113,113,0.25); }
.pill-medium { background: rgba(251,146,60,0.1);  color: #fb923c; border: 1px solid rgba(251,146,60,0.25); }
.pill-low    { background: rgba(74,222,128,0.1);  color: #4ade80; border: 1px solid rgba(74,222,128,0.25); }

/* Info boxes */
.info-box {
    background: rgba(125,211,252,0.04);
    border: 1px solid rgba(125,211,252,0.12);
    border-left: 3px solid #38bdf8;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #94a3b8;
    line-height: 1.7;
}
.info-box strong { color: #cbd5e1; }
.warning-box {
    background: rgba(251,146,60,0.04);
    border: 1px solid rgba(251,146,60,0.15);
    border-left: 3px solid #fb923c;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #9a7352;
    line-height: 1.7;
}
.warning-box strong { color: #cbd5e1; }
.success-box {
    background: rgba(74,222,128,0.04);
    border: 1px solid rgba(74,222,128,0.15);
    border-left: 3px solid #4ade80;
    border-radius: 10px;
    padding: 1.1rem 1.3rem;
    margin: 0.8rem 0;
    font-size: 0.9rem;
    color: #6b9e7a;
    line-height: 1.7;
}
.success-box strong { color: #cbd5e1; }

/* Dataset badges */
.dataset-badge {
    display: inline-block;
    font-family: 'DM Mono', monospace;
    font-size: 0.68rem;
    padding: 0.3rem 0.8rem;
    background: rgba(148,163,184,0.06);
    border: 1px solid rgba(148,163,184,0.15);
    border-radius: 100px;
    color: #64748b;
    margin: 0.2rem;
    letter-spacing: 0.04em;
}

/* Section tag */
.section-tag {
    font-family: 'DM Mono', monospace;
    font-size: 0.63rem;
    color: #38bdf8;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    margin-bottom: 0.3rem;
    display: block;
}

/* Sidebar styles */
div[data-testid="stSidebar"] label {
    color: #64748b !important;
    font-size: 0.82rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 400 !important;
}
div[data-testid="stSidebar"] .stSelectbox label { color: #94a3b8 !important; }
div[data-testid="stSidebar"] p {
    font-size: 0.82rem !important;
    color: #475569 !important;
    line-height: 1.6 !important;
}

/* Sidebar section headers */
.sb-section {
    font-family: 'DM Mono', monospace;
    font-size: 0.62rem;
    color: #38bdf8;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    padding-bottom: 0.4rem;
    border-bottom: 1px solid rgba(148,163,184,0.08);
    margin-bottom: 0.6rem;
}
.sb-desc {
    font-size: 0.8rem !important;
    color: #475569 !important;
    line-height: 1.6 !important;
    margin-bottom: 0.8rem !important;
}

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: transparent;
    border-bottom: 1px solid rgba(148,163,184,0.1);
    gap: 0;
}
.stTabs [data-baseweb="tab"] {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.72rem !important;
    letter-spacing: 0.12em !important;
    color: #475569 !important;
    padding: 0.8rem 1.5rem !important;
    border-radius: 0 !important;
    background: transparent !important;
}
.stTabs [aria-selected="true"] {
    color: #7dd3fc !important;
    border-bottom: 2px solid #38bdf8 !important;
    background: transparent !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #64748b !important;
    letter-spacing: 0.08em !important;
}

/* Metric widget */
[data-testid="stMetric"] label {
    font-family: 'DM Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #475569 !important;
    letter-spacing: 0.1em !important;
}
[data-testid="stMetricValue"] {
    font-family: 'Fraunces', serif !important;
    font-size: 1.8rem !important;
    color: #e2e8f0 !important;
}

/* Footer */
.footer {
    font-family: 'DM Mono', monospace;
    font-size: 0.6rem;
    color: #1e293b;
    text-align: center;
    letter-spacing: 0.1em;
    padding: 2.5rem 0 0.5rem;
    border-top: 1px solid rgba(148,163,184,0.06);
    margin-top: 3rem;
    line-height: 2;
}
</style>
""", unsafe_allow_html=True)


# ── Load assets ──────────────────────────────────────────
@st.cache_resource
def load_model():
    p = Path("outputs/xgb_final_v2.pkl")
    return joblib.load(p) if p.exists() else None

@st.cache_data
def load_training_data():
    p = Path("data/processed/final_dataset_v2.csv")
    return pd.read_csv(p) if p.exists() else None

@st.cache_data
def load_sgpr():
    p = Path("data/processed/sgpr_monthly.csv")
    return pd.read_csv(p, parse_dates=['date']) if p.exists() else None

model       = load_model()
df_train    = load_training_data()
sgpr_monthly = load_sgpr()

FIN = ['r11_wm','r12_wm','r14_wm','r21_wm','r22_wm',
       'r25_wm','r31_wm','r32_wm','r51_wm','r52_wm','r53_wm']
GEO = ['sgpr','sgpr_lag1','gpr_global']
MAC = ['crisis_2008','brexit_period','post_covid']
ALL_FEATURES = FIN + GEO + MAC

FIN_LABELS = {
    'r11_wm': 'ROA — Return on Assets (%)',
    'r12_wm': 'ROE — Return on Equity (%)',
    'r14_wm': 'Operating Margin (%)',
    'r21_wm': 'Current Ratio',
    'r22_wm': 'Quick Ratio',
    'r25_wm': 'Debt / EBITDA',
    'r31_wm': 'Interest Coverage',
    'r32_wm': 'Net Debt / Equity',
    'r51_wm': 'Asset Turnover',
    'r52_wm': 'Receivables Days',
    'r53_wm': 'Payables Days',
}
FIN_BOUNDS = {
    'r11_wm': (-30.0, 50.0),  'r12_wm': (-50.0, 80.0),
    'r14_wm': (-20.0, 40.0),  'r21_wm': (0.0,   10.0),
    'r22_wm': (0.0,   10.0),  'r25_wm': (-10.0, 50.0),
    'r31_wm': (-5.0,  40.0),  'r32_wm': (-5.0,  20.0),
    'r51_wm': (0.0,   5.0),   'r52_wm': (0.0,  180.0),
    'r53_wm': (0.0,  180.0),
}
FIN_HELP = {
    'r11_wm': 'Net profit ÷ total assets. Measures how efficiently assets generate profit. Healthy SME range: 2–10%.',
    'r12_wm': 'Net profit ÷ equity. Shows return to owners. Healthy range: 8–20%.',
    'r14_wm': 'Operating profit ÷ revenue. How much of each euro in sales becomes operating profit. Healthy: 5–15%.',
    'r21_wm': 'Current assets ÷ current liabilities. A ratio above 1.0 means the firm can cover its short-term obligations.',
    'r22_wm': 'Like current ratio but excludes slow-moving inventory. More conservative liquidity measure.',
    'r25_wm': 'Years needed to repay total debt using EBITDA. Under 3x is healthy for SMEs.',
    'r31_wm': 'How many times operating profit covers interest payments. Above 2x is considered safe.',
    'r32_wm': 'Net financial debt relative to equity. High positive values indicate heavy leverage.',
    'r51_wm': 'Revenue generated per euro of assets. Higher = more efficient use of the asset base.',
    'r52_wm': 'Average days to collect payment from customers. Lower is better for cash flow.',
    'r53_wm': 'Average days taken to pay suppliers. Too high can signal liquidity strain.',
}

# ── Sidebar ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("""
    <p style='font-family:Fraunces,serif; font-weight:700; font-size:1.3rem;
       color:#f1f5f9; margin:0; letter-spacing:-0.02em'>🌍 GeoCredit EWS</p>
    <p style='font-family:DM Mono,monospace; font-size:0.62rem; color:#334155;
       letter-spacing:0.14em; margin:0.3rem 0 0'>EARLY WARNING SYSTEM v2.0</p>
    """, unsafe_allow_html=True)
    st.divider()

    st.markdown("<p class='sb-section'>01 — Sector</p>", unsafe_allow_html=True)
    st.markdown("<p class='sb-desc'>Choose the industry sector to analyse. Each sector has a different geopolitical risk profile.</p>", unsafe_allow_html=True)

    sector = st.selectbox("Sector", ["construction","manufacturing","services"],
                          format_func=str.title, label_visibility="collapsed")

    SECTOR_DESC = {
        'construction':  '🏗️ Highly exposed to energy prices, raw material costs, and cross-border labour supply chains.',
        'manufacturing': '🏭 Sensitive to trade tariffs, supply chain disruptions, and imported input cost shocks.',
        'services':      '💼 Affected by consumer confidence, regulatory changes, and cross-border talent mobility.',
    }
    st.markdown(f"<div class='info-box' style='margin-top:0.4rem'>{SECTOR_DESC[sector]}</div>",
                unsafe_allow_html=True)

    st.divider()
    st.markdown("<p class='sb-section'>02 — Geopolitical Inputs</p>", unsafe_allow_html=True)
    st.markdown("<p class='sb-desc'>These signals measure global and sector-specific geopolitical tension. Pre-filled with the latest available values.</p>", unsafe_allow_html=True)

    if sgpr_monthly is not None:
        latest       = sgpr_monthly.iloc[-1]
        default_sgpr = float(latest.get(f'sgpr_{sector}', 1.5))
        default_gpr  = float(latest.get('GPR', 120))
    else:
        default_sgpr, default_gpr = 1.5, 120.0

    gpr_input  = st.slider("🌐 Global GPR Level", 50.0, 300.0,
                            float(round(default_gpr, 0)), step=1.0,
                            help="Caldara & Iacoviello GPR Index (Fed). Baseline mean = 100. Values above 150 indicate elevated global conflict risk.")
    sgpr_input = st.slider(f"🎯 Sector sGPR — {sector.title()}", 0.3, 5.0,
                            float(round(default_sgpr, 2)), step=0.05,
                            help="Sector-Weighted GPR: this project's novel index. Weights global GPR by Ireland's bilateral trade exposure per sector.")
    sgpr_lag   = st.slider("⏮ sGPR Prior Year", 0.3, 5.0,
                            float(round(default_sgpr * 0.85, 2)), step=0.05,
                            help="Last year's sector GPR — included because geopolitical stress leads credit events by 6–12 months.")

    st.divider()
    st.markdown("<p class='sb-section'>03 — Financial Ratios</p>", unsafe_allow_html=True)
    st.markdown("<p class='sb-desc'>Key balance sheet health indicators. Defaults are set to historical sector medians. Hover the ⓘ icon for definitions.</p>", unsafe_allow_html=True)

    if df_train is not None:
        sec_data = df_train[df_train['sector_bucket'] == sector]
        defaults = sec_data[FIN].median().to_dict()
    else:
        defaults = {f: 1.0 for f in FIN}

    fin_inputs = {}
    for feat, label in FIN_LABELS.items():
        lo, hi = FIN_BOUNDS[feat]
        val = float(np.clip(defaults.get(feat, (lo+hi)/2), lo, hi))
        fin_inputs[feat] = st.slider(label, lo, hi, val, step=0.1, help=FIN_HELP[feat])

    st.divider()
    st.markdown("<p class='sb-section'>04 — Period Flags</p>", unsafe_allow_html=True)
    st.markdown("<p class='sb-desc'>Binary flags for macro shock periods. These help the model account for extraordinary economic conditions.</p>", unsafe_allow_html=True)

    crisis_2008 = st.checkbox("📉 2008 Financial Crisis period", False,
                               help="Tick for scenarios set during the 2008–2009 global recession.")
    brexit      = st.checkbox("🇬🇧 Brexit period (2016–2019)", False,
                               help="Tick for scenarios during the Brexit uncertainty window.")
    post_covid  = st.checkbox("🦠 Post-COVID era (2020+)", True,
                               help="Tick for scenarios set after the 2020 COVID-19 shock.")


# ── Build feature vector & predict ───────────────────────
feature_vector = pd.DataFrame([{
    **fin_inputs,
    'sgpr': sgpr_input, 'sgpr_lag1': sgpr_lag,
    'gpr_global': gpr_input,
    'crisis_2008': int(crisis_2008),
    'brexit_period': int(brexit),
    'post_covid': int(post_covid),
}])[ALL_FEATURES]

if model is not None:
    pd_score = float(model.predict_proba(feature_vector)[0][1])
    explainer = shap.TreeExplainer(model)
    shap_vals = explainer.shap_values(feature_vector)
else:
    pd_score  = float(np.clip(0.05 + (gpr_input-100)*0.001 + sgpr_input*0.05, 0.01, 0.99))
    shap_vals = None

risk_band = "HIGH" if pd_score > 0.55 else "MEDIUM" if pd_score > 0.35 else "LOW"
rc        = risk_band.lower()

# ── PAGE HEADER ──────────────────────────────────────────
st.markdown("# Geopolitical<br>Credit EWS", unsafe_allow_html=True)
st.markdown(f"""
<p style='font-size:1.05rem; color:#64748b; line-height:1.7; max-width:780px;
   font-family:DM Sans,sans-serif; font-weight:300; margin-bottom:0.5rem'>
An ML-based Early Warning System quantifying geopolitical risk transmission to
corporate distress for European SMEs — powered by XGBoost and SHAP explainability.
&nbsp;&nbsp;
<span class='risk-pill pill-{rc}'>{risk_band} RISK</span>
&nbsp;
<span style='color:#334155; font-family:DM Mono,monospace; font-size:0.75rem'>
SECTOR: {sector.upper()}</span>
</p>
""", unsafe_allow_html=True)

st.markdown("""
<div style='margin:0.6rem 0 1.8rem'>
  <span class='dataset-badge'>BACH / ECCBSO</span>
  <span class='dataset-badge'>Caldara &amp; Iacoviello GPR</span>
  <span class='dataset-badge'>Eurostat Business Demography</span>
  <span class='dataset-badge'>CSO Ireland Trade Stats</span>
  <span class='dataset-badge'>12 Countries · 3 Sectors · 2010–2020</span>
</div>
""", unsafe_allow_html=True)

# ── KPI CARDS ────────────────────────────────────────────
gpr_color = 'high' if gpr_input > 150 else 'medium' if gpr_input > 110 else 'low'
st.markdown(f"""
<div class='kpi-grid'>
  <div class='kpi-card'>
    <div class='kpi-label'>Distress Probability</div>
    <div class='kpi-value val-{rc}'>{pd_score:.1%}</div>
    <div class='kpi-sub'>Predicted probability of sector-level credit distress</div>
  </div>
  <div class='kpi-card'>
    <div class='kpi-label'>Risk Classification</div>
    <div class='kpi-value val-{rc}'>{risk_band}</div>
    <div class='kpi-sub'>&gt;55% High &nbsp;·&nbsp; 35–55% Medium &nbsp;·&nbsp; &lt;35% Low</div>
  </div>
  <div class='kpi-card'>
    <div class='kpi-label'>Global GPR Index</div>
    <div class='kpi-value val-{gpr_color}'>{gpr_input:.0f}</div>
    <div class='kpi-sub'>Historical mean = 100 &nbsp;·&nbsp; Current input = {gpr_input:.0f}</div>
  </div>
  <div class='kpi-card'>
    <div class='kpi-label'>Sector sGPR</div>
    <div class='kpi-value val-neutral'>{sgpr_input:.2f}</div>
    <div class='kpi-sub'>Sector-weighted geopolitical risk index (novel)</div>
  </div>
</div>
""", unsafe_allow_html=True)

# Context callout
if rc == 'high':
    msg = f"""⚠️ <strong>Elevated distress signal.</strong> The combination of GPR={gpr_input:.0f}
    and current sector financial health indicators suggests meaningful credit stress risk for
    <strong>{sector}</strong>. Similar conditions were observed during the 2013–2015 post-GFC
    recovery and the 2018–2019 trade uncertainty period across European SMEs."""
    box_cls = 'warning-box'
elif rc == 'low':
    msg = f"""✅ <strong>Low distress probability.</strong> Current geopolitical and financial
    conditions suggest a relatively stable credit environment for <strong>{sector}</strong>.
    Monitor the sGPR trend — geopolitical signals typically lead credit events by 6–12 months."""
    box_cls = 'success-box'
else:
    msg = f"""ℹ️ <strong>Watch zone.</strong> The scenario sits in the moderate risk band.
    Geopolitical conditions are {'elevated' if gpr_input > 120 else 'moderate'} and financial
    ratios are {'below' if fin_inputs['r11_wm'] < 3 else 'near'} sector median.
    Consider stress-testing with higher sGPR values to assess sensitivity."""
    box_cls = 'info-box'
st.markdown(f"<div class='{box_cls}'>{msg}</div>", unsafe_allow_html=True)

st.markdown("<div style='height:0.5rem'></div>", unsafe_allow_html=True)

# ── TABS ─────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📈  GPR SIGNALS",
    "🔍  SHAP EXPLAINABILITY",
    "📊  HISTORICAL CONTEXT",
    "📚  METHODOLOGY",
])

# ══════════════════════════════════════════════════════════
# TAB 1 — GPR SIGNALS
# ══════════════════════════════════════════════════════════
with tab1:
    st.markdown("## Geopolitical Risk Index — Historical Trend")
    st.markdown("""<div class='info-box'>
    <strong>What am I looking at?</strong> The Caldara &amp; Iacoviello (2022) GPR Index measures
    global conflict and geopolitical tension using automated text analysis of major newspapers.
    Normalised so the 1985–2019 mean = 100. The amber dotted line is the <em>sector-weighted sGPR</em>
    — this project's novel contribution — which weights the global GPR score by Ireland's bilateral
    trade exposure in this specific sector. Key historical events are annotated. Your current
    scenario value is shown as a red dashed line.
    </div>""", unsafe_allow_html=True)

    if sgpr_monthly is not None:
        sgpr_col = f'sgpr_{sector}'
        plot_df  = sgpr_monthly[['date','GPR',sgpr_col]].dropna().tail(240).copy()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=plot_df['date'], y=plot_df['GPR'],
            name='Global GPR', mode='lines',
            line=dict(color='#38bdf8', width=1.8),
            fill='tozeroy', fillcolor='rgba(56,189,248,0.05)'
        ))
        fig.add_trace(go.Scatter(
            x=plot_df['date'], y=plot_df[sgpr_col] * 100,
            name=f'sGPR {sector.title()} (×100)', mode='lines',
            line=dict(color='#fb923c', width=1.5, dash='dot')
        ))

        # Key events — use actual datetime objects to avoid type error
        events = [
            (datetime(2016, 6, 1),  'Brexit Vote',    '#4ade80'),
            (datetime(2020, 3, 1),  'COVID‑19',       '#f87171'),
            (datetime(2022, 2, 1),  'Ukraine War',    '#f87171'),
            (datetime(2025, 3, 1),  'US Tariffs 2025','#fb923c'),
        ]
        for dt, label, color in events:
            if plot_df['date'].min() <= pd.Timestamp(dt) <= plot_df['date'].max():
                fig.add_vline(
                    x=dt.timestamp() * 1000,
                    line_dash='dot', line_color=color, opacity=0.5,
                    annotation_text=label,
                    annotation_font=dict(color=color, size=10),
                    annotation_position='top right'
                )

        fig.add_hline(y=100, line_dash='dot', line_color='#334155',
                      annotation_text='Baseline = 100',
                      annotation_font=dict(color='#475569', size=10))
        fig.add_hline(y=gpr_input, line_dash='dash', line_color='#f87171',
                      annotation_text=f'Scenario ({gpr_input:.0f})',
                      annotation_font=dict(color='#f87171', size=10))

        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#64748b', family='DM Mono', size=11),
            xaxis=dict(gridcolor='rgba(148,163,184,0.06)', title=None,
                       tickfont=dict(size=10)),
            yaxis=dict(gridcolor='rgba(148,163,184,0.06)', title='GPR Level'),
            legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.18,
                        font=dict(size=11)),
            margin=dict(t=20, b=50, l=10, r=10), height=400
        )
        st.plotly_chart(fig, use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        latest_gpr = float(plot_df['GPR'].iloc[-1])
        mean_gpr   = float(plot_df['GPR'].mean())
        with c1: st.metric("Latest GPR",       f"{latest_gpr:.0f}", f"{latest_gpr-mean_gpr:+.0f} vs mean")
        with c2: st.metric("Historical Mean",  f"{mean_gpr:.0f}")
        with c3: st.metric("All-Time Peak",    f"{plot_df['GPR'].max():.0f}")
        with c4: st.metric("Your Scenario",    f"{gpr_input:.0f}",
                            f"{'above' if gpr_input > mean_gpr else 'below'} mean")
    else:
        st.warning("sgpr_monthly.csv not found. Run `src/data/gpr_loader.py` first.")


# ══════════════════════════════════════════════════════════
# TAB 2 — SHAP EXPLAINABILITY
# ══════════════════════════════════════════════════════════
with tab2:
    st.markdown("## What Is Driving This Prediction?")
    st.markdown("""<div class='info-box'>
    <strong>How to read this chart:</strong> SHAP (SHapley Additive exPlanations) values show
    exactly how much each feature pushed the prediction <span style='color:#f87171'>higher
    (red = towards distress)</span> or <span style='color:#4ade80'>lower (green = away from
    distress)</span>. The longer the bar, the more influence that feature had on <em>this specific
    scenario</em>. Unlike a black-box model, you can see precisely why the risk score is what it is.
    </div>""", unsafe_allow_html=True)

    if shap_vals is not None:
        shap_s   = pd.Series(shap_vals[0], index=ALL_FEATURES)
        top12    = shap_s.abs().nlargest(12)
        signed   = shap_s[top12.index]
        bar_clrs = ['#f87171' if v > 0 else '#4ade80' for v in signed.values]
        labels   = [FIN_LABELS.get(f, f.replace('_wm','').replace('_',' ').upper())
                    for f in top12.index]

        fig2 = go.Figure(go.Bar(
            x=signed.values[::-1], y=labels[::-1],
            orientation='h',
            marker_color=bar_clrs[::-1], marker_line_width=0,
            text=[f"{v:+.3f}" for v in signed.values[::-1]],
            textposition='outside',
            textfont=dict(family='DM Mono', size=10, color='#64748b')
        ))
        fig2.add_vline(x=0, line_color='rgba(148,163,184,0.2)', line_width=1)
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#64748b', family='DM Mono', size=11),
            xaxis=dict(gridcolor='rgba(148,163,184,0.06)',
                       title='SHAP value  ·  red = increases distress risk  ·  green = reduces it',
                       title_font=dict(size=10)),
            yaxis=dict(gridcolor='rgba(0,0,0,0)', tickfont=dict(size=11)),
            margin=dict(t=10, b=50, l=10, r=90), height=440
        )
        st.plotly_chart(fig2, use_container_width=True)

        geo_pct = shap_s[GEO].abs().sum() / shap_s.abs().sum() * 100
        fin_pct = shap_s[FIN].abs().sum() / shap_s.abs().sum() * 100
        mac_pct = shap_s[MAC].abs().sum() / shap_s.abs().sum() * 100

        st.markdown("### Feature Group Attribution")
        st.markdown("""<div class='info-box'>
        What proportion of this prediction is driven by geopolitical signals vs financial health vs
        macro period flags? A higher geo share means external tensions are dominating the risk
        signal over internal financial fundamentals.
        </div>""", unsafe_allow_html=True)

        c1, c2, c3 = st.columns(3)
        with c1: st.metric("🌍 Geopolitical", f"{geo_pct:.1f}%", help="sGPR, sGPR lag, global GPR")
        with c2: st.metric("📊 Financial",    f"{fin_pct:.1f}%", help="11 accounting ratios from BACH")
        with c3: st.metric("📅 Macro Flags",  f"{mac_pct:.1f}%", help="Crisis 2008, Brexit, Post-COVID")

        top_feat = shap_s.abs().idxmax()
        top_val  = shap_s[top_feat]
        direction = "increasing" if top_val > 0 else "reducing"
        top_label = FIN_LABELS.get(top_feat, top_feat.replace('_',' ').upper())
        geo_note  = ("This is a geopolitical signal — external macro stress is dominating this prediction."
                     if top_feat in GEO else
                     "This is a financial ratio — the sector's internal health is the primary driver.")
        st.markdown(f"""<div class='info-box'>
        🔎 <strong>Top driver:</strong> <code>{top_label}</code> is the most influential feature,
        {direction} the distress probability by {abs(top_val):.3f} SHAP units. {geo_note}
        </div>""", unsafe_allow_html=True)
    else:
        st.warning("Model not loaded. Ensure `outputs/xgb_final_v2.pkl` exists.")


# ══════════════════════════════════════════════════════════
# TAB 3 — HISTORICAL CONTEXT
# ══════════════════════════════════════════════════════════
with tab3:
    st.markdown("## Historical Distress Rates & Geopolitical Context")
    st.markdown("""<div class='info-box'>
    <strong>What is the distress rate?</strong> Derived from Eurostat's Business Demography dataset
    (indicator V97020 — annual business death rate). A year is labelled "distress" when the sector
    death rate rises abnormally above its rolling 3-year average. The amber line tracks the
    sector-weighted sGPR over the same period — allowing you to visually assess whether geopolitical
    stress preceded credit stress. Blue bars = normal years. Red bars = distress years.
    </div>""", unsafe_allow_html=True)

    if df_train is not None:
        hist = df_train[df_train['sector_bucket']==sector].groupby('year').agg(
            distress_rate=('distress_label','mean'),
            sgpr_mean=('sgpr','mean'),
        ).reset_index()

        bar_clrs = ['#f87171' if r > 0.5 else '#1e3a5f' for r in hist['distress_rate']]
        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=hist['year'], y=hist['distress_rate'],
            name='Distress Rate', marker_color=bar_clrs,
            marker_line_width=0, opacity=0.9,
            hovertemplate='Year: %{x}<br>Distress Rate: %{y:.0%}<extra></extra>'
        ))
        sgpr_norm = hist['sgpr_mean'] / hist['sgpr_mean'].max()
        fig3.add_trace(go.Scatter(
            x=hist['year'], y=sgpr_norm, name='sGPR (normalised)',
            mode='lines+markers', yaxis='y2',
            line=dict(color='#fb923c', width=2),
            marker=dict(size=7, color='#fb923c'),
            hovertemplate='Year: %{x}<br>sGPR (norm): %{y:.2f}<extra></extra>'
        ))
        fig3.add_hline(y=0.5, line_dash='dot', line_color='#f87171',
                       opacity=0.3, annotation_text='50% threshold',
                       annotation_font=dict(color='#f87171', size=10))
        fig3.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#64748b', family='DM Mono', size=11),
            xaxis=dict(gridcolor='rgba(148,163,184,0.06)', tickmode='linear', dtick=1),
            yaxis=dict(gridcolor='rgba(148,163,184,0.06)', title='Distress Rate', tickformat='.0%'),
            yaxis2=dict(overlaying='y', side='right', title='sGPR (norm.)', showgrid=False),
            legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.2, font=dict(size=11)),
            margin=dict(t=10, b=60, l=10, r=10), height=360
        )
        st.plotly_chart(fig3, use_container_width=True)

        st.markdown("### Cross-Sector Comparison")
        st.markdown("""<div class='info-box'>
        Compares distress rates across all three sectors. Sector-specific spikes suggest
        idiosyncratic shocks (e.g. construction crash). Simultaneous spikes suggest economy-wide
        stress. Notice how construction was most affected during the 2013–2015 post-GFC period.
        </div>""", unsafe_allow_html=True)

        fig4 = go.Figure()
        colors_sec = {'construction':'#38bdf8','manufacturing':'#fb923c','services':'#4ade80'}
        for sec in ['construction','manufacturing','services']:
            sh = df_train[df_train['sector_bucket']==sec].groupby('year')['distress_label'].mean().reset_index()
            fig4.add_trace(go.Scatter(
                x=sh['year'], y=sh['distress_label'], name=sec.title(),
                mode='lines+markers',
                line=dict(color=colors_sec[sec], width=2), marker=dict(size=7),
                hovertemplate=f'{sec.title()}: %{{y:.0%}}<extra></extra>'
            ))
        fig4.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font=dict(color='#64748b', family='DM Mono', size=11),
            xaxis=dict(gridcolor='rgba(148,163,184,0.06)', tickmode='linear', dtick=1),
            yaxis=dict(gridcolor='rgba(148,163,184,0.06)', title='Distress Rate', tickformat='.0%'),
            legend=dict(bgcolor='rgba(0,0,0,0)', orientation='h', y=-0.2, font=dict(size=11)),
            margin=dict(t=10, b=60, l=10, r=10), height=300
        )
        st.plotly_chart(fig4, use_container_width=True)


# ══════════════════════════════════════════════════════════
# TAB 4 — METHODOLOGY
# ══════════════════════════════════════════════════════════
with tab4:
    st.markdown("## How This System Works")
    st.markdown("""<div class='info-box'>
    This tool is the output of a personal research project exploring whether geopolitical risk
    signals can improve early warning of corporate credit distress for European SMEs — a question
    directly relevant to credit risk teams at financial institutions. The full pipeline covers
    data acquisition, feature engineering, model training, and this interactive dashboard.
    </div>""", unsafe_allow_html=True)

    st.markdown("## The Research Question")
    st.markdown("""
    > *"Can macroeconomic and geopolitical signals provide meaningful early warnings of credit
    deterioration for unlisted European corporates — and which sectors are most exposed?"*
    """)

    st.markdown("## Data Sources")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("""
**📊 BACH Database (ECCBSO)**
Harmonised annual financial accounts of non-financial corporations from 12 European countries.
Covers balance sheets, income statements, and financial ratios broken down by NACE sector and
size class. Used for: leverage, liquidity, profitability, and efficiency ratios across SME size
classes. Coverage: 2010–2020, 12 countries, 3 sectors after filtering.
[bach.banque-france.fr](https://bach.banque-france.fr)

**📉 Eurostat Business Demography (BD_9AC_L_FORM_R2)**
Annual business birth and death rates by NACE sector and country. The death rate (V97020) is
the distress label — a sector-year is flagged as distress when the death rate rises abnormally
above its rolling 3-year average. This avoids look-ahead bias in labelling.
[ec.europa.eu/eurostat](https://ec.europa.eu/eurostat)
""")
    with c2:
        st.markdown("""
**🌍 Caldara & Iacoviello GPR Index (Federal Reserve)**
Monthly geopolitical risk index built from automated analysis of major newspapers worldwide.
Country-specific sub-indices for 44 countries, going back to 1900. Free, publicly available,
updated monthly around the 10th of each month.
[matteoiacoviello.com/gpr](https://matteoiacoviello.com/gpr.htm)

**🎯 sGPR Index — Novel Contribution**
A sector-weighted version of the GPR index constructed for this research. For each Irish NACE
sector, country-level GPR scores are weighted by Ireland's bilateral trade exposure (from CSO
Ireland). This produces a sector-specific geopolitical signal reflecting how much each sector's
supply chain is exposed to global conflict zones.
[cso.ie](https://www.cso.ie)
""")

    st.markdown("## Model Performance")
    st.markdown("""<div class='info-box'>
    Two XGBoost models are compared using <strong>leave-one-year-out cross-validation</strong> —
    the correct approach for time-series panel data. Training on all years except one, testing
    on the held-out year, rotating through all 11 years. This prevents look-ahead bias and
    simulates real-world deployment where you always predict an unseen future year.
    </div>""", unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    with c1: st.metric("Financial-Only AUC", "0.665", help="XGBoost trained on 11 financial ratios only")
    with c2: st.metric("Augmented AUC",      "0.700", help="XGBoost + geopolitical + macro features")
    with c3: st.metric("AUC Lift",           "+3.5pp", help="Marginal improvement from geo features")
    with c4: st.metric("Geo SHAP Share",     "22.2%", help="Proportion of predictive power from geo signals")

    st.markdown("""<div class='success-box'>
    ✅ <strong>Key finding:</strong> Adding geopolitical features improves AUC from 0.665 to 0.700
    — a +3.5 percentage point lift under rigorous temporal CV. The global GPR index is the #2 most
    important predictor overall, ranked above 9 of 11 financial ratios. Construction shows the
    highest geopolitical sensitivity (26.2% of SHAP attribution), followed by services (20.3%)
    and manufacturing (19.3%).
    </div>""", unsafe_allow_html=True)

    st.markdown("## Important Limitations")
    st.markdown("""<div class='warning-box'>
    ⚠️ <strong>Research prototype — not a production credit tool.</strong>
    The dataset covers 360 country-sector-year observations across 12 EU countries.
    Ireland is not in the BACH database and is proxied by similar small open economies (Belgium,
    Slovakia, Portugal, Croatia). The distress label is an aggregate sector death rate, not
    individual firm default. Results indicate a macro-level geopolitical signal, not firm-level
    prediction capability. Always consult qualified credit risk professionals before making
    any lending or investment decisions.
    </div>""", unsafe_allow_html=True)

    st.markdown("## Publication Path")
    st.markdown("""
- **SSRN Preprint** — first submission to timestamp the contribution and build visibility
- **Primary target:** Journal of Risk and Financial Management (MDPI) — open access, fast turnaround (~8 weeks), strong applied finance track record
- **Secondary target:** Finance Research Letters (Elsevier) — short empirical format, high visibility
    """)


# ── FOOTER ───────────────────────────────────────────────
st.markdown("""
<div class='footer'>
  GEOCREDIT EWS &nbsp;·&nbsp; GPR INDEX: CALDARA &amp; IACOVIELLO (FEDERAL RESERVE, 2022)
  &nbsp;·&nbsp; FINANCIAL DATA: BACH / ECCBSO &nbsp;·&nbsp; BUSINESS DEMOGRAPHY: EUROSTAT
  <br>
  BUILT WITH XGBOOST · SHAP · STREAMLIT &nbsp;·&nbsp; FOR RESEARCH USE ONLY
  &nbsp;·&nbsp; © 2025 JISHNU JANARDHANAN NAIR
</div>
""", unsafe_allow_html=True)
