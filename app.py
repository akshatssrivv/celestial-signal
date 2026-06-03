import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import zipfile
import shutil
import hashlib
import os
from nelson_siegel_fn import plot_ns_animation, nelson_siegel
from ai_explainer_utils import format_bond_diagnostics, generate_ai_explanation
import ast
import re
from datetime import datetime
import boto3
from scipy.interpolate import interp1d
import altair as alt
from curve_trade_agent1 import chat_with_trades, get_system_prompt

# ─────────────────────────────────────────────
# Page config — must be first Streamlit call
# ─────────────────────────────────────────────
st.set_page_config(page_title="Celestial Bond Analytics", layout="wide", initial_sidebar_state="collapsed")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Mono:wght@400;500&family=Syne:wght@400;600;700&display=swap');

/* ── Reset & page shell ── */
html, body, [class*="css"] { font-family: 'Syne', sans-serif !important; }

main .block-container {
    padding: 0.75rem 1.5rem 2rem 1.5rem !important;
    max-width: 100% !important;
}

/* ── Hide default Streamlit chrome ── */
#MainMenu, footer, header { visibility: hidden; }

/* ── Custom top header bar ── */
.cel-header {
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.6rem 0 1rem 0;
    border-bottom: 1px solid #1e293b;
    margin-bottom: 1rem;
}
.cel-header .cel-logo {
    font-family: 'Syne', sans-serif;
    font-size: 1.25rem;
    font-weight: 700;
    letter-spacing: 0.04em;
    color: #e2e8f0;
}
.cel-header .cel-logo span { color: #38bdf8; }
.cel-datestamp {
    font-family: 'DM Mono', monospace;
    font-size: 0.72rem;
    color: #64748b;
    letter-spacing: 0.05em;
}

/* ── Tab pill styling ── */
div[role="tablist"] {
    gap: 4px !important;
    border-bottom: 1px solid #1e293b !important;
    padding-bottom: 0 !important;
}
button[role="tab"] {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.78rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.04em !important;
    padding: 0.45rem 1.1rem !important;
    border-radius: 6px 6px 0 0 !important;
    border: none !important;
    color: #64748b !important;
    background: transparent !important;
    transition: color 0.18s, background 0.18s !important;
}
button[role="tab"]:hover { color: #e2e8f0 !important; background: #1e293b !important; }
button[role="tab"][aria-selected="true"] {
    color: #38bdf8 !important;
    background: #0f172a !important;
    border-bottom: 2px solid #38bdf8 !important;
}

/* ── Section headings ── */
.cel-section-title {
    font-family: 'Syne', sans-serif;
    font-size: 0.68rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    color: #475569;
    margin: 1.2rem 0 0.5rem 0;
}

/* ── Control card (filter sidebar feel) ── */
.ctrl-card {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 1rem 1rem 0.75rem 1rem;
}
.ctrl-card label {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.7rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #475569 !important;
}

/* ── Signal metric cards ── */
.sig-row { display: flex; gap: 0.5rem; margin: 0.5rem 0 1rem 0; }
.sig-box {
    flex: 1;
    padding: 0.9rem 0.6rem 0.8rem 0.85rem;
    border-radius: 8px;
    border-left: 3px solid transparent;
    background: #0f172a;
    transition: transform 0.15s ease, box-shadow 0.15s ease;
    cursor: default;
    min-width: 0;
}
.sig-box:hover { transform: translateY(-2px); box-shadow: 0 4px 16px rgba(0,0,0,0.35); }
.sig-box .sig-count {
    font-family: 'DM Mono', monospace;
    font-size: 1.8rem;
    font-weight: 500;
    line-height: 1;
    color: #e2e8f0;
}
.sig-box .sig-label {
    font-size: 0.62rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    margin-top: 0.3rem;
    opacity: 0.7;
}
.sig-box .sig-delta {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    margin-top: 0.35rem;
    display: inline-flex;
    align-items: center;
    gap: 3px;
    padding: 1px 6px;
    border-radius: 99px;
    font-weight: 500;
}
.delta-up   { background: rgba(52,211,153,0.15); color: #34d399; }
.delta-down { background: rgba(248,113,113,0.15); color: #f87171; }
.delta-flat { background: rgba(100,116,139,0.12); color: #64748b; }

/* Signal accent colours */
.sig-strong-buy  { border-left-color: #22c55e; } .sig-strong-buy  .sig-label { color: #22c55e; }
.sig-strong-sell { border-left-color: #ef4444; } .sig-strong-sell .sig-label { color: #ef4444; }
.sig-mod-buy     { border-left-color: #4ade80; } .sig-mod-buy     .sig-label { color: #4ade80; }
.sig-mod-sell    { border-left-color: #fb923c; } .sig-mod-sell    .sig-label { color: #fb923c; }
.sig-weak-buy    { border-left-color: #38bdf8; } .sig-weak-buy    .sig-label { color: #38bdf8; }
.sig-weak-sell   { border-left-color: #f59e0b; } .sig-weak-sell   .sig-label { color: #f59e0b; }
.sig-no-action   { border-left-color: #475569; } .sig-no-action   .sig-label { color: #94a3b8; }

/* ── Filter row ── */
.filter-bar {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 10px;
    padding: 0.75rem 1rem;
    margin-bottom: 0.75rem;
    display: flex;
    gap: 0.75rem;
    align-items: flex-end;
}

/* ── Streamlit inputs override ── */
div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {
    background: #0f172a !important;
    border-color: #1e293b !important;
    border-radius: 6px !important;
    font-family: 'Syne', sans-serif !important;
    font-size: 0.82rem !important;
    color: #cbd5e1 !important;
    transition: border-color 0.15s !important;
}
div[data-baseweb="select"] > div:hover,
div[data-baseweb="input"] > div:focus-within {
    border-color: #38bdf8 !important;
}

/* ── Buttons ── */
.stButton > button {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.05em !important;
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    color: #94a3b8 !important;
    border-radius: 6px !important;
    padding: 0.4rem 0.9rem !important;
    transition: all 0.15s !important;
}
.stButton > button:hover {
    border-color: #38bdf8 !important;
    color: #38bdf8 !important;
    background: #0f172a !important;
}

/* ── Download button ── */
.stDownloadButton > button {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.75rem !important;
    font-weight: 600 !important;
    background: rgba(56,189,248,0.1) !important;
    border: 1px solid rgba(56,189,248,0.3) !important;
    color: #38bdf8 !important;
    border-radius: 6px !important;
    padding: 0.4rem 0.9rem !important;
    transition: all 0.15s !important;
}
.stDownloadButton > button:hover {
    background: rgba(56,189,248,0.2) !important;
    border-color: #38bdf8 !important;
}

/* ── Dataframe ── */
.stDataFrame {
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    overflow: hidden;
}

/* ── Metric widget ── */
div[data-testid="metric-container"] {
    background: #0f172a;
    border: 1px solid #1e293b;
    border-radius: 8px;
    padding: 0.8rem 1rem !important;
}
div[data-testid="metric-container"] label {
    font-size: 0.68rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #475569 !important;
    font-weight: 700 !important;
}
div[data-testid="metric-container"] div[data-testid="stMetricValue"] {
    font-family: 'DM Mono', monospace !important;
    color: #e2e8f0 !important;
    font-size: 1.6rem !important;
}

/* ── Chat bubbles ── */
div[data-testid="stChatMessage"] {
    border-radius: 10px !important;
    padding: 0.6rem 0.9rem !important;
    margin-bottom: 0.4rem !important;
    border: 1px solid #1e293b !important;
    background: #0f172a !important;
}
div[data-testid="stChatMessage"][data-testid*="user"] {
    border-left: 3px solid #38bdf8 !important;
}
div[data-testid="stChatMessage"][data-testid*="assistant"] {
    border-left: 3px solid #e8c547 !important;
}
div[data-testid="stChatMessage"] p {
    font-size: 0.88rem !important;
    line-height: 1.6 !important;
    color: #cbd5e1 !important;
}
div[data-testid="stChatInputContainer"] {
    background: #0f172a !important;
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    padding: 0.3rem 0.5rem !important;
    margin-top: 0.5rem !important;
}
div[data-testid="stChatInputContainer"]:focus-within {
    border-color: #38bdf8 !important;
}

/* ── Expander ── */
div[data-testid="stExpander"] {
    border: 1px solid #1e293b !important;
    border-radius: 8px !important;
    background: #0f172a !important;
}
div[data-testid="stExpander"] summary {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.8rem !important;
    font-weight: 600 !important;
    color: #94a3b8 !important;
    letter-spacing: 0.04em !important;
}

/* ── Checkbox & radio ── */
label[data-baseweb="checkbox"] span,
label[data-baseweb="radio"] span {
    font-family: 'Syne', sans-serif !important;
    font-size: 0.82rem !important;
    color: #94a3b8 !important;
}

/* ── Horizontal rule ── */
hr { border-color: #1e293b !important; margin: 0.75rem 0 !important; }

/* ── st.info / warning / error ── */
div[data-testid="stAlert"] {
    border-radius: 8px !important;
    border: 1px solid #1e293b !important;
    font-size: 0.82rem !important;
}

/* ── Subheader & title ── */
h1 {
    font-family: 'Syne', sans-serif !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #e2e8f0 !important;
    margin-bottom: 0.1rem !important;
}
h2, h3 {
    font-family: 'Syne', sans-serif !important;
    font-weight: 600 !important;
    color: #cbd5e1 !important;
}
p, li, span { color: #94a3b8 !important; }

/* ── Caption ── */
small, .stCaption { color: #475569 !important; font-size: 0.72rem !important; }

/* ── Scrollbar ── */
::-webkit-scrollbar { width: 5px; height: 5px; }
::-webkit-scrollbar-track { background: #0f172a; }
::-webkit-scrollbar-thumb { background: #1e293b; border-radius: 3px; }
::-webkit-scrollbar-thumb:hover { background: #334155; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# Branded header
# ─────────────────────────────────────────────
_now = __import__("datetime").datetime.now()
st.markdown(f"""
<div class="cel-header">
  <div class="cel-logo">CELESTIAL <span>BOND ANALYTICS</span></div>
  <div class="cel-datestamp">{_now.strftime("%A, %d %b %Y  ·  %H:%M")} UTC</div>
</div>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# AWS / S3
# ─────────────────────────────────────────────
AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME  = "bonds-celestial-signal"
LOCAL_ZIP    = "ns_curves_20260106.zip"
LOCAL_FOLDER = "ns_curves_0106"


def download_from_s3(file_key: str, local_path: str, force: bool = False):
    if not force and os.path.exists(local_path):
        return local_path
    with st.spinner(f"Downloading {file_key} from S3…"):
        s3 = boto3.client(
            "s3",
            aws_access_key_id=AWS_ACCESS_KEY_ID,
            aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
        )
        s3.download_file(BUCKET_NAME, file_key, local_path)
    return local_path


def file_hash(filepath: str) -> str:
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def unzip_ns_curves(zip_path: str = LOCAL_ZIP, folder: str = LOCAL_FOLDER, force: bool = False):
    zip_path = download_from_s3(file_key="ns_curves_0106.zip", local_path=zip_path, force=force)
    zip_hash = file_hash(zip_path)
    prev_hash = st.session_state.get("ns_zip_hash")
    if force or prev_hash != zip_hash or not os.path.exists(folder):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        with zipfile.ZipFile(zip_path, "r") as z:
            z.extractall(folder)
        st.session_state["ns_zip_hash"] = zip_hash
    return folder, zip_hash


# ─────────────────────────────────────────────
# NS data loaders
# ─────────────────────────────────────────────
@st.cache_data
def load_full_ns_df(country_code: str, zip_hash: str) -> pd.DataFrame:
    # force=False — let hash-based cache invalidation do the work
    folder, _ = unzip_ns_curves(force=False)
    all_files = sorted([
        f for f in os.listdir(folder)
        if f.startswith(country_code) and f.endswith(".parquet")
    ])
    dfs = []
    for f in all_files:
        try:
            df = pd.read_parquet(os.path.join(folder, f))
            if "ISIN"    in df.columns: df["ISIN"]    = df["ISIN"].astype(str).str.strip()
            if "Date"    in df.columns: df["Date"]    = pd.to_datetime(df["Date"], errors="coerce")
            if "Country" not in df.columns: df["Country"] = country_code
            dfs.append(df)
        except Exception as e:
            st.warning(f"Error loading {f}: {e}")
    if not dfs:
        st.warning(f"No parquet files found for '{country_code}'.")
        return pd.DataFrame()
    ns_df = pd.concat(dfs, ignore_index=True)
    if "RESIDUAL" in ns_df.columns and "RESIDUAL_NS" not in ns_df.columns:
        ns_df.rename(columns={"RESIDUAL": "RESIDUAL_NS"}, inplace=True)
    if "RESIDUAL_NS" not in ns_df.columns:
        ns_df["RESIDUAL_NS"] = pd.NA
    ns_df.sort_values("Date", inplace=True)
    return ns_df


def load_ns_curve(country_code: str, date_str: str, zip_hash: str):
    df = load_full_ns_df(country_code, zip_hash=zip_hash)
    if df is not None and not df.empty:
        sub = df[df["Date"].dt.date == pd.to_datetime(date_str).date()]
        if not sub.empty:
            return sub
    return None


# ─────────────────────────────────────────────
# Shared helpers
# ─────────────────────────────────────────────
COUNTRY_OPTIONS = [
    "Italy 🇮🇹", "Spain 🇪🇸", "France 🇫🇷", "Germany 🇩🇪",
    "Finland 🇫🇮", "EU 🇪🇺", "Austria 🇦🇹", "Netherlands 🇳🇱", "Belgium 🇧🇪",
]
COUNTRY_CODE_MAP = {
    "Italy 🇮🇹": "BTPS",   "Spain 🇪🇸": "SPGB",  "France 🇫🇷": "FRTR",
    "Germany 🇩🇪": "BUNDS", "Finland 🇫🇮": "RFGB",  "EU 🇪🇺": "EU",
    "Austria 🇦🇹": "RAGB",  "Netherlands 🇳🇱": "NETHER", "Belgium 🇧🇪": "BGB",
}
LEGEND_SIGNALS = {"strong buy", "moderate buy", "strong sell", "moderate sell"}

import json

def parse_ns_params(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return x
    if isinstance(x, str):
        try:
            return json.loads(x)
        except Exception:
            return None
    return None


# ─────────────────────────────────────────────
# Shared dark Plotly theme
# ─────────────────────────────────────────────
CHART_BG   = "#080f1a"
CHART_GRID = "#1e293b"
CHART_TEXT = "#94a3b8"
CHART_FONT = dict(family="DM Mono, monospace", color=CHART_TEXT, size=11)
NS_LINE_COLOR = "#e8c547"   # gold for NS fit
PRED_COLOR    = "#a78bfa"   # violet for prediction

def dark_layout(fig, title="", height=640, xaxis_title="", yaxis_title=""):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Syne, sans-serif", size=13,
                   color="#cbd5e1"), x=0, xanchor="left", pad=dict(l=0, b=8)),
        height=height,
        paper_bgcolor=CHART_BG,
        plot_bgcolor=CHART_BG,
        font=CHART_FONT,
        margin=dict(l=48, r=20, t=44, b=44),
        xaxis=dict(
            title=dict(text=xaxis_title, font=CHART_FONT),
            gridcolor=CHART_GRID,
            linecolor=CHART_GRID,
            tickfont=CHART_FONT,
            zeroline=False,
        ),
        yaxis=dict(
            title=dict(text=yaxis_title, font=CHART_FONT),
            gridcolor=CHART_GRID,
            linecolor=CHART_GRID,
            tickfont=CHART_FONT,
            zeroline=False,
        ),
        legend=dict(
            bgcolor="rgba(8,15,26,0.85)",
            bordercolor=CHART_GRID,
            borderwidth=1,
            font=dict(family="DM Mono, monospace", size=10, color=CHART_TEXT),
        ),
        hoverlabel=dict(
            bgcolor="#0f172a",
            bordercolor=CHART_GRID,
            font=dict(family="DM Mono, monospace", size=11, color="#e2e8f0"),
        ),
    )
    return fig


# Signal dot colours — vivid against the dark bg
SIGNAL_COLOR_MAP = {
    "strong buy":    "#22c55e",
    "moderate buy":  "#4ade80",
    "weak buy":      "#94a3b8",
    "strong sell":   "#ef4444",
    "moderate sell": "#fb923c",
    "weak sell":     "#94a3b8",
}

def get_country_from_isin(isin):
    country_map = {
        "IT": "🇮🇹 Italy", "ES": "🇪🇸 Spain",  "FR": "🇫🇷 France",
        "DE": "🇩🇪 Germany", "FI": "🇫🇮 Finland", "EU": "🇪🇺 EU",
        "AT": "🇦🇹 Austria", "NL": "🇳🇱 Netherlands", "BE": "🇧🇪 Belgium",
    }
    return country_map.get(isin[:2], "🌍 Unknown")


# ─────────────────────────────────────────────
# Initialise shared state
# ─────────────────────────────────────────────
S3_BUCKET_FILE = "ns_curves_0106.zip"
try:
    zip_path = download_from_s3(file_key=S3_BUCKET_FILE, local_path=LOCAL_ZIP, force=False)
    if not os.path.exists(zip_path):
        raise FileNotFoundError(f"Downloaded file not found: {zip_path}")
    zip_hash = file_hash(zip_path)
except Exception as e:
    st.error(f"Failed to download or hash NS curves zip: {e}")
    zip_path = zip_hash = None


@st.cache_data
def load_trades():
    df = pd.read_pickle("top_trades_agent.pkl")
    for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
        df[col] = df[col].astype(str)
    return df


top_trades_agent = load_trades()


# ═════════════════════════════════════════════
# TABS
# ═════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📈 Nelson-Siegel Curves",
    "📊 Signal Dashboard",
    "🔬 Analysis",
    "🤖 AI Assistant",
])


# ═════════════════════════════════════════════
# TAB 1 — Nelson-Siegel Curves
# ═════════════════════════════════════════════
with tab1:
    sub1, sub2, sub3, sub4, sub5 = st.tabs([
        "Single day curve",
        "Animated curves",
        "Residuals analysis",
        "Compare NS curves",
        "New bond prediction",
    ])

    # ── Single Day ────────────────────────────
    with sub1:
        col_ctrl, col_chart, col_ai = st.columns([1, 3, 2])

        with col_ctrl:
            country_option = st.selectbox("Country", COUNTRY_OPTIONS, key="sd_country")
            selected_country = COUNTRY_CODE_MAP[country_option]
            final_signal_df = pd.read_csv("today_all_signals.csv")
            available_dates = pd.to_datetime(final_signal_df["Date"].unique())
            default_date = available_dates.max()
            date_input = st.date_input("Date", value=default_date, key="sd_date")
            date_str = date_input.strftime("%Y-%m-%d")

        ns_df = load_ns_curve(selected_country, date_str, zip_hash=zip_hash)

        if ns_df is not None and not ns_df.empty:
            col_map = {c.lower(): c for c in ns_df.columns}
            for alias, canonical in [("z_sprd_val", "Z_SPRD"), ("z_sprd", "Z_SPRD"),
                                      ("yearstomaturity", "YTM")]:
                if alias in col_map and canonical not in ns_df.columns:
                    ns_df.rename(columns={col_map[alias]: canonical}, inplace=True)

            ns_df["Maturity"] = pd.to_datetime(ns_df["Maturity"])
            ns_df["YTM"] = (ns_df["Maturity"] - pd.to_datetime(date_input)).dt.days / 365.25
            ns_df = ns_df.merge(final_signal_df[["ISIN", "SIGNAL"]], on="ISIN", how="left")
            ns_df["SIGNAL"] = ns_df["SIGNAL"].str.strip().str.lower()
            ns_df["Signal_Color"] = ns_df["SIGNAL"].map(SIGNAL_COLOR_MAP).fillna("black")

            fig = go.Figure()
            for signal, df_sub in ns_df.groupby("SIGNAL"):
                if df_sub.empty:
                    continue
                color = df_sub["Signal_Color"].iloc[0]
                residuals = df_sub.get("RESIDUAL_NS", pd.Series(np.zeros(len(df_sub)), index=df_sub.index))
                fig.add_trace(go.Scatter(
                    x=df_sub["YTM"],
                    y=df_sub["Z_SPRD"],
                    mode="markers",
                    name=signal.title() if signal in LEGEND_SIGNALS else None,
                    marker=dict(size=7, color=color, symbol="circle"),
                    text=df_sub["SECURITY_NAME"],
                    customdata=np.stack((
                        df_sub["ISIN"],
                        df_sub["Date"].astype(str),
                        residuals,
                    ), axis=-1),
                    hovertemplate=(
                        "Years to maturity: %{x:.2f}<br>"
                        "Z-spread: %{y:.1f} bps<br>"
                        "Residual: %{customdata[2]:.2f} bps<br>"
                        f"Signal: {signal.title()}<br>"
                        "%{text}<extra></extra>"
                    ),
                    showlegend=(signal in LEGEND_SIGNALS),
                ))

            # NS fit line
            if "NS_PARAMS" in ns_df.columns or all(
                c in ns_df.columns for c in ["NS_PARAM_1", "NS_PARAM_2", "NS_PARAM_3", "NS_PARAM_4"]
            ):
                try:
                    ns_params = None
                    if "NS_PARAMS" in ns_df.columns:
                        raw = ns_df["NS_PARAMS"].dropna().iloc[0] if ns_df["NS_PARAMS"].notna().any() else None
                        if raw is not None:
                            if isinstance(raw, (tuple, list, np.ndarray)):
                                ns_params = list(raw)
                            elif isinstance(raw, str):
                                nums = re.findall(r"np\.float64\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)", raw)
                                if len(nums) >= 4:
                                    ns_params = [float(n) for n in nums[:4]]
                                else:
                                    nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", raw)
                                    if len(nums) >= 4:
                                        ns_params = [float(n) for n in nums[:4]]
                    if ns_params is None and all(
                        c in ns_df.columns for c in ["NS_PARAM_1", "NS_PARAM_2", "NS_PARAM_3", "NS_PARAM_4"]
                    ):
                        ns_params = [ns_df[f"NS_PARAM_{i}"].iloc[0] for i in range(1, 5)]
                    if ns_params is not None:
                        mr = np.linspace(ns_df["YTM"].min(), ns_df["YTM"].max(), 100)
                        fig.add_trace(go.Scatter(
                            x=mr, y=nelson_siegel(mr, *ns_params),
                            mode="lines", name="Nelson-Siegel fit",
                            line=dict(color="deepskyblue", width=3),
                        ))
                except Exception as e:
                    st.warning(f"NS curve skipped: {e}")

            dark_layout(fig, title=f"NS curve — {selected_country}  ·  {date_str}",
                        height=620, xaxis_title="Years to maturity",
                        yaxis_title="Z-spread (bps)")
            # override NS fit line colour
            for trace in fig.data:
                if trace.name == "Nelson-Siegel fit":
                    trace.line.color = NS_LINE_COLOR

            with col_chart:
                event = st.plotly_chart(fig, use_container_width=True, on_select="rerun",
                                        key="sd_chart")

            # AI explanation — triggered by chart click or manual select
            with col_ai:
                st.markdown("#### Bond AI explanation")

                # Resolve ISIN from chart click
                clicked_isin = None
                if event and event.selection and event.selection.get("points"):
                    pt = event.selection["points"][0]
                    cd = pt.get("customdata")
                    if cd:
                        clicked_isin = cd[0]

                bond_options = (
                    final_signal_df[["ISIN", "SECURITY_NAME"]]
                    .drop_duplicates()
                    .sort_values("SECURITY_NAME")
                )
                bond_labels = dict(zip(bond_options["ISIN"], bond_options["SECURITY_NAME"]))

                default_idx = 0
                if clicked_isin and clicked_isin in bond_options["ISIN"].values:
                    default_idx = bond_options["ISIN"].tolist().index(clicked_isin)

                selected_isin = st.selectbox(
                    "Select bond (or click chart)",
                    options=bond_options["ISIN"].tolist(),
                    index=default_idx,
                    format_func=lambda i: bond_labels.get(i, i),
                    key="sd_bond_selector",
                )
                st.caption(f"ISIN: `{selected_isin}`")
                if st.button("Explain this bond", key="sd_explain"):
                    hist = final_signal_df[final_signal_df["ISIN"] == selected_isin]
                    diag = format_bond_diagnostics(hist)
                    with st.spinner("Generating explanation…"):
                        explanation = generate_ai_explanation(diag)
                    st.markdown(explanation)
        else:
            with col_chart:
                st.warning("No NS data available for this date.")

    # ── Animated Curves ───────────────────────
    with sub2:
        country_option = st.selectbox("Country", COUNTRY_OPTIONS, key="anim_country")
        selected_country = COUNTRY_CODE_MAP[country_option]

        ns_df = load_full_ns_df(selected_country, zip_hash=zip_hash)
        if ns_df is not None and not ns_df.empty:
            final_signal_df = pd.read_csv("today_all_signals.csv")
            country_isins = ns_df["ISIN"].unique()
            bond_opts = (
                final_signal_df[final_signal_df["ISIN"].isin(country_isins)][["ISIN", "SECURITY_NAME"]]
                .drop_duplicates()
            )
            isin_mat = ns_df.groupby("ISIN")["Maturity"].first().to_dict()
            bond_opts["Maturity"] = pd.to_datetime(bond_opts["ISIN"].map(isin_mat), errors="coerce")
            bond_opts.sort_values("Maturity", inplace=True)
            bond_labels = dict(zip(bond_opts["ISIN"], bond_opts["SECURITY_NAME"]))

            def fmt_anim(isin):
                mat = bond_opts.loc[bond_opts["ISIN"] == isin, "Maturity"].values
                if len(mat) and pd.notnull(mat[0]):
                    return f"{bond_labels.get(isin, isin)} ({pd.to_datetime(mat[0]).strftime('%Y-%m-%d')})"
                return f"{bond_labels.get(isin, isin)} (N/A)"

            show_all = st.checkbox(f"Show all {country_option} bonds", key="anim_all")
            if show_all:
                selected_animation_bonds = bond_opts["ISIN"].tolist()
            else:
                selected_animation_bonds = st.multiselect(
                    "Select bonds for animation",
                    options=bond_opts["ISIN"].tolist(),
                    format_func=fmt_anim,
                    default=[],
                    key="anim_bonds",
                )

            if not selected_animation_bonds:
                st.info("Select at least one bond to display the animation.")
            else:
                ns_filt = ns_df[ns_df["ISIN"].isin(selected_animation_bonds)].copy()
                ns_filt = ns_filt.merge(final_signal_df[["ISIN", "SIGNAL"]], on="ISIN", how="left")
                fig = plot_ns_animation(ns_filt, issuer_label=selected_country,
                                        highlight_isins=selected_animation_bonds)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No NS data available for the selected country.")

    # ── Residuals Analysis ────────────────────
    with sub3:
        country_option = st.selectbox("Country", COUNTRY_OPTIONS, key="res_country")
        selected_country = COUNTRY_CODE_MAP[country_option]

        ns_df = load_full_ns_df(selected_country, zip_hash=zip_hash)
        if ns_df is not None and not ns_df.empty:
            ns_df["Date"] = pd.to_datetime(ns_df["Date"]).dt.normalize()
            ns_df["RESIDUAL_VELOCITY"] = ns_df.groupby("ISIN")["RESIDUAL_NS"].transform(
                lambda x: x.diff()
            )
            isin_mat = ns_df.groupby("ISIN")["Maturity"].first().to_dict()
            bond_opts = ns_df[["ISIN", "SECURITY_NAME"]].drop_duplicates().copy()
            bond_opts["Maturity"] = pd.to_datetime(bond_opts["ISIN"].map(isin_mat), errors="coerce")
            bond_opts.sort_values("Maturity", inplace=True)
            bond_labels = dict(zip(bond_opts["ISIN"], bond_opts["SECURITY_NAME"]))

            def fmt_res(isin):
                mat = bond_opts.loc[bond_opts["ISIN"] == isin, "Maturity"].values
                if len(mat) and pd.notnull(mat[0]):
                    return f"{bond_labels.get(isin, isin)} ({pd.to_datetime(mat[0]).strftime('%Y-%m-%d')})"
                return f"{bond_labels.get(isin, isin)} (N/A)"

            selected_bonds = st.multiselect(
                "Select bonds for residual analysis",
                options=bond_opts["ISIN"].tolist(),
                format_func=fmt_res,
                default=[],
                key="res_bonds",
            )

            if not selected_bonds:
                st.info("Select at least one bond to display residuals.")
            else:
                res_df = ns_df[ns_df["ISIN"].isin(selected_bonds)].copy()
                fig_r = go.Figure()
                fig_v = go.Figure()
                for isin in selected_bonds:
                    bd = res_df[res_df["ISIN"] == isin].sort_values("Date")
                    if bd.empty:
                        continue
                    lbl = bond_labels.get(isin, isin)
                    fig_r.add_trace(go.Scatter(x=bd["Date"], y=bd["RESIDUAL_NS"],
                                               mode="lines+markers", name=lbl))
                    fig_v.add_trace(go.Scatter(x=bd["Date"], y=bd["RESIDUAL_VELOCITY"],
                                               mode="lines+markers", name=lbl))
                dark_layout(fig_r, title="Residuals over time", height=440,
                            xaxis_title="Date", yaxis_title="Residual (bps)")
                dark_layout(fig_v, title="Residual velocity over time", height=440,
                            xaxis_title="Date", yaxis_title="Velocity (bps/day)")
                st.plotly_chart(fig_r, use_container_width=True)
                st.plotly_chart(fig_v, use_container_width=True)
        else:
            st.warning("No NS data available.")

    # ── Compare NS Curves ─────────────────────
    with sub4:
        countries = st.multiselect("Select countries", options=COUNTRY_OPTIONS, key="cmp_countries")
        if countries:
            all_dates = {}
            for c in countries:
                tmp = load_full_ns_df(COUNTRY_CODE_MAP[c], zip_hash=zip_hash)
                if tmp is not None and not tmp.empty:
                    dates = pd.to_datetime(tmp["Date"].unique())
                    all_dates[c] = pd.Series(dates).sort_values(ascending=False)
                else:
                    all_dates[c] = pd.Series(dtype="datetime64[ns]")

            selected_dates = {}
            for c in countries:
                if len(all_dates[c]):
                    fmts = [d.strftime("%Y-%m-%d") for d in all_dates[c]]
                    selected_dates[c] = st.multiselect(
                        f"Dates — {c}", options=fmts, default=[fmts[0]], key=f"cmp_dates_{c}"
                    )

            fig = go.Figure()
            for c in countries:
                for d in selected_dates.get(c, []):
                    curve_df = load_ns_curve(COUNTRY_CODE_MAP[c], d, zip_hash=zip_hash)
                    if curve_df is None or curve_df.empty or "NS_PARAMS" not in curve_df.columns:
                        continue
                    ns_params = parse_ns_params(curve_df["NS_PARAMS"].iloc[0])
                    if ns_params is None:
                        continue
                    if "YTM" not in curve_df.columns or curve_df["YTM"].isna().all():
                        continue
                    max_mat = min(30, curve_df["YTM"].max())
                    mats = np.linspace(0, max_mat, 100)
                    fig.add_trace(go.Scatter(
                        x=mats, y=nelson_siegel(mats, *ns_params),
                        mode="lines", name=f"{c} — {d}",
                    ))
            dark_layout(fig, title="NS curves comparison", height=640,
                        xaxis_title="Years to maturity", yaxis_title="Z-spread (bps)")
            fig.update_xaxes(range=[0, 30])
            st.plotly_chart(fig, use_container_width=True)

    # ── New Bond Prediction ───────────────────
    with sub5:
        country_option = st.selectbox("Country", COUNTRY_OPTIONS, key="pred_country")
        selected_country = COUNTRY_CODE_MAP[country_option]

        col_a, col_b = st.columns(2)
        with col_a:
            new_bond_input = st.text_input("New bond maturity (MM/YY)", value="10/55")
        with col_b:
            auction_concession = st.number_input("Auction concession (bps)", value=0, step=1)

        today_ts = pd.Timestamp.today().normalize()
        start_date = today_ts - pd.Timedelta(days=14)
        ns_df_list = []
        for d in pd.date_range(start_date, today_ts):
            tmp = load_ns_curve(selected_country, d.strftime("%Y-%m-%d"), zip_hash=zip_hash)
            if tmp is not None and not tmp.empty:
                tmp["Date"] = pd.to_datetime(d)
                tmp["YearsToMaturity"] = (pd.to_datetime(tmp["Maturity"]) - today_ts).dt.days / 365.25
                ns_df_list.append(tmp)

        if not ns_df_list:
            st.warning("No NS curve data for the last 2 weeks.")
        else:
            ns_full = pd.concat(ns_df_list, ignore_index=True)
            ns_smooth = ns_full.groupby("YearsToMaturity")["Z_SPRD_VAL"].mean().reset_index()
            ns_std    = ns_full.groupby("YearsToMaturity")["Z_SPRD_VAL"].std().reset_index()

            try:
                month, year = map(int, new_bond_input.split("/"))
                year += 2000 if year < 100 else 0
                new_mat_date = pd.Timestamp(year=year, month=month, day=1)
                new_ytm = (new_mat_date - today_ts).days / 365.25
            except Exception:
                st.error("Invalid format. Use MM/YY.")
                st.stop()

            final_signal_df = pd.read_csv("today_all_signals.csv")
            final_signal_df["Maturity"] = pd.to_datetime(final_signal_df["Maturity"], errors="coerce")
            similar = final_signal_df[
                final_signal_df["Maturity"].notna() &
                (abs((final_signal_df["Maturity"] - new_mat_date).dt.days / 365.25) <= 2)
            ]
            ns_today = ns_full[ns_full["Date"] == ns_full["Date"].max()]
            if not similar.empty:
                similar = similar.merge(
                    ns_today[["ISIN", "Z_SPRD_VAL", "YearsToMaturity"]], on="ISIN", how="left",
                    suffixes=("", "_NS"),
                )

            f_interp = interp1d(ns_smooth["YearsToMaturity"], ns_smooth["Z_SPRD_VAL"],
                                kind="linear", fill_value="extrapolate")
            offsets = []
            for _, row in similar.iterrows():
                if pd.notnull(row.get("Z_SPRD_VAL")):
                    offsets.append(row["Z_SPRD_VAL"] - f_interp(row["YearsToMaturity"]))
            mean_offset = np.nanmean(offsets) if offsets else 0

            predicted_z = f_interp(new_ytm) + mean_offset + auction_concession
            all_mats = ns_smooth["YearsToMaturity"].values
            close_idx = [i for i in np.argsort(np.abs(all_mats - new_ytm))
                         if abs(all_mats[i] - new_ytm) <= 2]
            if close_idx:
                dists = np.abs(all_mats[close_idx] - new_ytm)
                weights = 1 / (dists + 1e-6)
                z_std_use = np.average(ns_std.iloc[close_idx]["Z_SPRD_VAL"], weights=weights)
            else:
                z_std_use = ns_std["Z_SPRD_VAL"].mean()
            z_min, z_max = predicted_z - 1.5 * z_std_use, predicted_z + 1.5 * z_std_use

            st.metric(
                label=f"Predicted Z-spread — {new_bond_input}",
                value=f"{predicted_z:.1f} bps",
                delta=f"Range: {z_min:.1f} – {z_max:.1f} bps",
            )

            ns_today_plot = ns_today.copy()
            ns_today_plot = ns_today_plot.merge(
                final_signal_df[["ISIN", "SIGNAL"]], on="ISIN", how="left"
            )
            ns_today_plot["SIGNAL"] = ns_today_plot["SIGNAL"].str.strip().str.lower()
            ns_today_plot["Signal_Color"] = ns_today_plot["SIGNAL"].map(SIGNAL_COLOR_MAP).fillna("black")

            fig = go.Figure()
            for signal, df_sub in ns_today_plot.groupby("SIGNAL"):
                if df_sub.empty:
                    continue
                color = df_sub["Signal_Color"].iloc[0]
                hover_text = df_sub.get("SECURITY_NAME", df_sub["ISIN"]).fillna("Unknown bond")
                fig.add_trace(go.Scatter(
                    x=df_sub["YearsToMaturity"], y=df_sub["Z_SPRD_VAL"],
                    mode="markers",
                    name=signal.title() if signal in LEGEND_SIGNALS else None,
                    marker=dict(size=6, color=color),
                    text=hover_text,
                    hovertemplate=(
                        "YTM: %{x:.2f}<br>Z-spread: %{y:.1f} bps<br>"
                        f"Signal: {signal.title()}<br>%{{text}}<extra></extra>"
                    ),
                    showlegend=(signal in LEGEND_SIGNALS),
                ))

            if "NS_PARAMS" in ns_today_plot.columns:
                try:
                    ns_params = parse_ns_params(ns_today_plot["NS_PARAMS"].iloc[0])
                    if ns_params is not None:
                        mr = np.linspace(ns_today_plot["YearsToMaturity"].min(),
                                         ns_today_plot["YearsToMaturity"].max(), 100)
                        fig.add_trace(go.Scatter(
                            x=mr, y=nelson_siegel(mr, *ns_params),
                            mode="lines", name="Nelson-Siegel fit",
                            line=dict(color="deepskyblue", width=3),
                        ))
                except Exception as e:
                    st.warning(f"NS curve skipped: {e}")

            fig.add_trace(go.Scatter(
                x=[new_ytm], y=[predicted_z],
                mode="markers+text",
                marker=dict(size=14, color=PRED_COLOR, symbol="star"),
                text=[f"Predicted: {predicted_z:.1f} bps"],
                textposition="top center",
                textfont=dict(color=PRED_COLOR, family="DM Mono, monospace", size=11),
                name="Predicted Z-spread",
            ))
            fig.add_trace(go.Scatter(
                x=[new_ytm - 0.05, new_ytm + 0.05, new_ytm + 0.05, new_ytm - 0.05],
                y=[z_min, z_min, z_max, z_max],
                fill="toself",
                fillcolor="rgba(167,139,250,0.12)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
            ))
            dark_layout(fig, title=f"Predicted Z-spread for new bond {new_bond_input}",
                        height=660, xaxis_title="Years to maturity",
                        yaxis_title="Z-spread (bps)")
            for trace in fig.data:
                if hasattr(trace, "name") and trace.name == "Nelson-Siegel fit":
                    trace.line.color = NS_LINE_COLOR
            st.plotly_chart(fig, use_container_width=True)


# ═════════════════════════════════════════════
# TAB 2 — Signal Dashboard
# ═════════════════════════════════════════════
with tab2:

    @st.cache_data(ttl=300)
    def load_signal_data(force: bool = False) -> pd.DataFrame:
        local_path = "issuer_signals.csv"
        try:
            local_path = download_from_s3(file_key="issuer_signals.csv",
                                           local_path=local_path, force=force)
            return pd.read_csv(local_path)
        except Exception as e:
            st.error(f"Error loading data from S3: {e}")
            if os.path.exists(local_path):
                return pd.read_csv(local_path)
            return pd.DataFrame()

    df = load_signal_data()
    if df.empty:
        st.error("No data available.")
        st.stop()
    df["Country"] = df["ISIN"].apply(get_country_from_isin)

    recent_signals = pd.read_csv("recent_signals.csv")
    recent_signals["Date"] = pd.to_datetime(recent_signals["Date"])
    recent_signals = recent_signals.sort_values("Date")
    today_date = recent_signals["Date"].max()
    yesterday_series = recent_signals[recent_signals["Date"] < today_date]["Date"]
    yesterday_date = yesterday_series.max() if not yesterday_series.empty else today_date
    today_df    = recent_signals[recent_signals["Date"] == today_date]
    yesterday_df = recent_signals[recent_signals["Date"] == yesterday_date]

    SIGNAL_TYPES = [
        "STRONG BUY", "STRONG SELL", "MODERATE BUY", "MODERATE SELL",
        "WEAK BUY", "WEAK SELL", "NO ACTION",
    ]
    signal_counts_today     = today_df["SIGNAL"].value_counts().reindex(SIGNAL_TYPES, fill_value=0)
    signal_counts_yesterday = yesterday_df["SIGNAL"].value_counts().reindex(SIGNAL_TYPES, fill_value=0)
    signal_deltas = signal_counts_today - signal_counts_yesterday

    st.title("Bond Analytics Dashboard")
    st.caption(f"Data as of {today_date.strftime('%d %b %Y')}  ·  Δ vs {yesterday_date.strftime('%d %b %Y')}")

    st.markdown('<div class="cel-section-title">Filters</div>', unsafe_allow_html=True)

    # ── Filters (above the counts so deltas reflect filtered view) ──
    f_col1, f_col2, f_col3, f_col4 = st.columns([2, 2, 3, 1])
    with f_col1:
        selected_countries = st.multiselect(
            "Countries", options=sorted(df["Country"].unique()),
            default=sorted(df["Country"].unique()), key="sig_countries",
        )
    with f_col2:
        FIXED_SIGNALS = [
            "STRONG BUY", "STRONG SELL", "MODERATE BUY",
            "MODERATE SELL", "WEAK BUY", "WEAK SELL", "NO ACTION",
        ]
        selected_signals = st.multiselect(
            "Signals", options=FIXED_SIGNALS,
            default=[s for s in FIXED_SIGNALS if s != "NO ACTION"],
            key="sig_signals",
        )
    with f_col3:
        search_term = st.text_input("Search ISIN or name", placeholder="Type to search…",
                                    key="sig_search")
    with f_col4:
        st.write("")
        st.write("")
        refresh = st.button("🔄 Refresh", key="sig_refresh")
        if refresh:
            st.cache_data.clear()
            st.rerun()

    # Apply filters
    filtered_df = df.copy()
    if selected_countries:
        filtered_df = filtered_df[filtered_df["Country"].isin(selected_countries)]
    if selected_signals:
        filtered_df = filtered_df[filtered_df["SIGNAL"].isin(selected_signals)]
    if search_term:
        mask = (
            filtered_df["ISIN"].str.contains(search_term.upper(), na=False) |
            filtered_df["SECURITY_NAME"].str.contains(search_term.upper(), na=False)
        )
        filtered_df = filtered_df[mask]

    # ── Signal count boxes ──
    filtered_counts = filtered_df["SIGNAL"].value_counts().reindex(SIGNAL_TYPES, fill_value=0)
    st.markdown('<div class="cel-section-title">Signal snapshot — filtered universe</div>', unsafe_allow_html=True)

    BOX_CSS_CLASS = {
        "STRONG BUY":   "sig-strong-buy",
        "STRONG SELL":  "sig-strong-sell",
        "MODERATE BUY": "sig-mod-buy",
        "MODERATE SELL":"sig-mod-sell",
        "WEAK BUY":     "sig-weak-buy",
        "WEAK SELL":    "sig-weak-sell",
        "NO ACTION":    "sig-no-action",
    }

    sig_html = '<div class="sig-row">'
    for sig in SIGNAL_TYPES:
        count  = filtered_counts[sig]
        delta  = signal_deltas[sig]
        d_cls  = "delta-up" if delta > 0 else "delta-down" if delta < 0 else "delta-flat"
        d_sign = "▲ +" if delta > 0 else "▼ " if delta < 0 else "● "
        css_cls = BOX_CSS_CLASS[sig]
        sig_html += f"""
<div class="sig-box {css_cls}">
  <div class="sig-count">{count}</div>
  <div class="sig-label">{sig}</div>
  <div class="sig-delta {d_cls}">{d_sign}{abs(delta)}</div>
</div>"""
    sig_html += "</div>"
    st.markdown(sig_html, unsafe_allow_html=True)

    st.markdown('<div class="cel-section-title">Bond universe</div>', unsafe_allow_html=True)

    if not filtered_df.empty:
        _dl_col, _cnt_col, _spacer = st.columns([1, 2, 6])
        with _dl_col:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "⬇ Export CSV", csv,
                f"bonds_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv", "text/csv",
            )
        with _cnt_col:
            st.markdown(
                f'<div style="padding-top:0.55rem; font-family:DM Mono,monospace; '
                f'font-size:0.72rem; color:#475569;">{len(filtered_df)} bonds matched</div>',
                unsafe_allow_html=True,
            )

        cols_to_display = [
            "SECURITY_NAME", "RESIDUAL_NS", "SIGNAL",
            "Z_Residual_Score", "Volatility_Score", "Market_Stress_Score",
            "Cluster_Score", "Regression_Score", "COMPOSITE_SCORE",
            "Top_Features", "Top_Feature_Effects_Pct",
        ]
        existing_cols = [c for c in cols_to_display if c in filtered_df.columns]
        display_df = filtered_df[existing_cols].copy()
        display_df.rename(columns={
            "RESIDUAL_NS": "Residual",
            "Volatility_Score": "Stability_Score",
            "SIGNAL": "Signal",
        }, inplace=True)

        numeric_cols = [
            "Residual", "Z_Residual_Score", "Stability_Score",
            "Market_Stress_Score", "Cluster_Score", "Regression_Score", "COMPOSITE_SCORE",
        ]
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors="coerce")
        if "Stability_Score" in display_df.columns:
            display_df["Stability_Score"] *= 100

        def extract_maturity_dt(name):
            if isinstance(name, str):
                m = re.search(r"(\d{2}/\d{2}/\d{2,4})$", name)
                if m:
                    for fmt in ("%m/%d/%y", "%m/%d/%Y"):
                        try:
                            return datetime.strptime(m.group(1), fmt)
                        except ValueError:
                            pass
            return pd.NaT

        display_df["Maturity"] = display_df["SECURITY_NAME"].apply(extract_maturity_dt)
        display_df["Maturity"] = display_df["Maturity"].dt.strftime("%Y-%m-%d").fillna("N/A")
        cols_order = ["SECURITY_NAME", "Maturity"] + [
            c for c in display_df.columns if c not in ["SECURITY_NAME", "Maturity"]
        ]
        display_df = display_df[cols_order]

        FEATURE_NAME_MAP = {
            "Cpn": "Coupon", "YAS_RISK": "DV01", "AMT_OUTSTANDING": "Amount Outstanding",
            "Issue_Age": "Issue Age", "REL_SPRD_STD": "Liquidity",
            "GREEN_BOND_LOAN_INDICATOR": "Green Bond",
        }

        if "Top_Features" in display_df.columns and "Top_Feature_Effects_Pct" in display_df.columns:
            def combine_features(feats, pct):
                try:
                    fl = ast.literal_eval(feats) if isinstance(feats, str) else []
                    fl = [FEATURE_NAME_MAP.get(f, f) for f in fl]
                    pl = [int(round(float(v))) for v in
                          pct.replace("[", "").replace("]", "").split()] if isinstance(pct, str) else []
                    return ", ".join(f"{f} ({p}%)" for f, p in zip(fl, pl)) or "N/A"
                except Exception:
                    return "N/A"
            display_df["Top_Features"] = display_df.apply(
                lambda row: combine_features(row["Top_Features"], row["Top_Feature_Effects_Pct"]),
                axis=1,
            )
            display_df.drop(columns=["Top_Feature_Effects_Pct"], inplace=True)

        # Decorate SECURITY_NAME with emoji for upgrades/downgrades
        yesterday_signals_map = yesterday_df.set_index("SECURITY_NAME")["SIGNAL"].to_dict()
        LEVELS = {
            "NO ACTION": 0,
            "WEAK SELL": 1, "WEAK BUY": 1,
            "MODERATE SELL": 2, "MODERATE BUY": 2,
            "STRONG SELL": 3, "STRONG BUY": 3,
        }
        EMOJI_MAP = {
            "STRONG BUY": "🟩", "MODERATE BUY": "💚",
            "STRONG SELL": "🟥", "MODERATE SELL": "💛",
        }

        def decorate_name(row):
            name = row["SECURITY_NAME"]
            today_sig = row["Signal"]
            yest_sig  = yesterday_signals_map.get(name, None)
            tl = LEVELS.get(today_sig, 0)
            yl = LEVELS.get(yest_sig, 0) if yest_sig else 0
            if (tl >= 2 or yl >= 2) and tl != yl:
                emoji = EMOJI_MAP.get(today_sig, "")
                arrow = "↑" if tl > yl else "↓"
                return f"{emoji} {arrow} {name}"
            return name

        display_df["SECURITY_NAME"] = display_df.apply(decorate_name, axis=1)

        HELP_TEXTS = {
            "Residual":           "Residual mispricing (bps off curve)",
            "Z_Residual_Score":   "Z-score of residual. |Z| > 1.5 may indicate opportunities.",
            "Stability_Score":    "Inverse volatility. Higher = more stable. Lower = riskier.",
            "Market_Stress_Score":"Market stress factor. High = more exposed to stress.",
            "Cluster_Score":      "Deviation from peer cluster (bps). |val| > 1.5 likely to mean-revert.",
            "Regression_Score":   "Model-explained mispricing. |val| > 1.5 = strong signal.",
            "COMPOSITE_SCORE":    "Overall mispricing score. |val| > 1.5 = stronger trade signal.",
            "Top_Features":       "Most important drivers of mispricing. % shows relative impact.",
            "Signal":             "Trade signal.",
        }

        column_config = {}
        for col in display_df.columns:
            label = col.replace("_", " ")
            if col in numeric_cols and pd.api.types.is_numeric_dtype(display_df.get(col, pd.Series())):
                if col == "Stability_Score":
                    column_config[col] = st.column_config.NumberColumn(label, format="%.2f%%",
                                                                        help=HELP_TEXTS.get(col))
                else:
                    column_config[col] = st.column_config.NumberColumn(label, format="%.4f",
                                                                        help=HELP_TEXTS.get(col))
            else:
                column_config[col] = st.column_config.TextColumn(label, help=HELP_TEXTS.get(col))

        st.dataframe(display_df, column_config=column_config, use_container_width=True)


# ═════════════════════════════════════════════
# TAB 3 — Analysis  (multi-curve + trades)
# ═════════════════════════════════════════════
with tab3:
    an1, an2 = st.tabs(["Multi-curve comparison", "Top trades"])

    # ── Multi-curve ───────────────────────────
    with an1:
        metric_option = st.radio(
            "Metric", options=["Z-Spread", "Residuals"], horizontal=True, key="an_metric"
        )
        metric_col_map = {"Z-Spread": "Z_SPRD_VAL", "Residuals": "RESIDUAL_NS"}
        selected_metric_col = metric_col_map[metric_option]

        if "curves" not in st.session_state or len(st.session_state.curves) != 2:
            st.session_state.curves = [
                {"id": "curve1", "country": "Italy 🇮🇹", "bond1": None, "bond2": None},
                {"id": "curve2", "country": "Italy 🇮🇹", "bond1": None, "bond2": None},
            ]

        @st.cache_data(ttl=300)
        def load_issuer_signal():
            try:
                return pd.read_csv("issuer_signals.csv")
            except Exception as e:
                st.error(f"Failed to load issuer_signal: {e}")
                return pd.DataFrame()

        issuer_signal = load_issuer_signal()

        # Build global legend labels
        global_legend_labels = {}
        for curve in st.session_state.curves:
            tmp = load_full_ns_df(COUNTRY_CODE_MAP[curve["country"]], zip_hash=zip_hash)
            if tmp is None or tmp.empty:
                continue
            tmp["Date"] = pd.to_datetime(tmp["Date"]).dt.normalize()
            opts = tmp[["ISIN", "SECURITY_NAME", "Maturity"]].drop_duplicates()
            opts = opts.merge(issuer_signal[["ISIN", "SIGNAL"]], on="ISIN", how="left")
            opts["Maturity"] = pd.to_datetime(opts["Maturity"], errors="coerce")
            for _, row in opts.iterrows():
                mat = pd.to_datetime(row["Maturity"]).strftime("%Y-%m-%d") if pd.notnull(row["Maturity"]) else "N/A"
                global_legend_labels[row["ISIN"]] = f"{row['SECURITY_NAME']} ({mat})"

        curve_dfs = []
        for i, curve in enumerate(st.session_state.curves):
            st.subheader(f"Curve {i + 1}")
            cc1, cc2 = st.columns(2)
            with cc1:
                curve["country"] = st.selectbox(
                    f"Country (Curve {i + 1})", COUNTRY_OPTIONS,
                    index=COUNTRY_OPTIONS.index(curve["country"]),
                    key=f"an_country_{curve['id']}",
                )
                ns_df = load_full_ns_df(COUNTRY_CODE_MAP[curve["country"]], zip_hash=zip_hash)
                ns_df["Date"] = pd.to_datetime(ns_df["Date"]).dt.normalize()
                bond_opts = ns_df[["ISIN", "SECURITY_NAME", "Maturity"]].drop_duplicates()
                bond_opts = bond_opts.merge(issuer_signal[["ISIN", "SIGNAL"]], on="ISIN", how="left")
                bond_opts["Maturity"] = pd.to_datetime(bond_opts["Maturity"], errors="coerce")
                bond_opts.sort_values("Maturity", inplace=True)
                bond_labels_c = {}
                for _, row in bond_opts.iterrows():
                    mat = pd.to_datetime(row["Maturity"]).strftime("%Y-%m-%d") if pd.notnull(row["Maturity"]) else "N/A"
                    sig = row["SIGNAL"] if "SIGNAL" in row and pd.notnull(row["SIGNAL"]) else "No signal"
                    bond_labels_c[row["ISIN"]] = f"{row['SECURITY_NAME']} ({mat}) [{sig}]"

            with cc2:
                curve["bond1"] = st.selectbox(
                    f"Bond 1 (Curve {i + 1})", bond_opts["ISIN"],
                    format_func=lambda isin: bond_labels_c.get(isin, isin),
                    key=f"an_bond1_{curve['id']}",
                )
                curve["bond2"] = st.selectbox(
                    f"Bond 2 (Curve {i + 1})", bond_opts["ISIN"],
                    format_func=lambda isin: bond_labels_c.get(isin, isin),
                    key=f"an_bond2_{curve['id']}",
                )

            if curve["bond1"] and curve["bond2"]:
                df1 = ns_df[ns_df["ISIN"] == curve["bond1"]][["Date", selected_metric_col]].rename(
                    columns={selected_metric_col: "B1"})
                df2 = ns_df[ns_df["ISIN"] == curve["bond2"]][["Date", selected_metric_col]].rename(
                    columns={selected_metric_col: "B2"})
                df_c = df1.merge(df2, on="Date", how="outer").sort_values("Date")
                b1_mat = pd.to_datetime(ns_df.loc[ns_df["ISIN"] == curve["bond1"], "Maturity"].iloc[0])
                b2_mat = pd.to_datetime(ns_df.loc[ns_df["ISIN"] == curve["bond2"], "Maturity"].iloc[0])
                df_c["Curve"] = (df_c["B1"] - df_c["B2"]) if b1_mat <= b2_mat else (df_c["B2"] - df_c["B1"])
                df_c["Bond1_ISIN"] = curve["bond1"]
                df_c["Bond2_ISIN"] = curve["bond2"]
                curve_dfs.append(df_c)

        if len(curve_dfs) == 2:
            diff_df = curve_dfs[1][["Date", "Curve"]].merge(
                curve_dfs[0][["Date", "Curve"]], on="Date", suffixes=("_2", "_1")
            )
            diff_df["Curve"] = diff_df["Curve_2"] - diff_df["Curve_1"]
            fig = go.Figure()
            for i, cdf in enumerate(curve_dfs):
                c = st.session_state.curves[i]
                lbl = (f"{global_legend_labels.get(c['bond2'], c['bond2'])} "
                       f"− {global_legend_labels.get(c['bond1'], c['bond1'])}")
                fig.add_trace(go.Scatter(x=cdf["Date"], y=cdf["Curve"], mode="lines", name=lbl))
            fig.add_trace(go.Scatter(
                x=diff_df["Date"], y=diff_df["Curve"],
                mode="lines", name="Differenced curve (Curve 2 − Curve 1)",
                line=dict(color=NS_LINE_COLOR, width=2.5, dash="dot"),
            ))
            dark_layout(fig, title=f"Two-curve {metric_option} comparison", height=640,
                        xaxis_title="Date", yaxis_title=f"{metric_option} difference (bps)")
            st.plotly_chart(fig, use_container_width=True)

    # ── Top Trades ────────────────────────────
    with an2:
        cols_top50 = [
            "A_ISIN", "B_ISIN", "C_ISIN", "D_ISIN",
            "LEG_1", "LEG_2",
            "Trade_ZDiff_30D_Pct", "Diff_of_Diffs_Today",
            "Ranking_Score", "Actionable_Direction",
        ]
        existing_cols_top50 = [c for c in cols_top50 if c in top_trades_agent.columns]

        st.markdown('<div class="cel-section-title">Top 50 trades</div>', unsafe_allow_html=True)
        st.dataframe(top_trades_agent.head(50)[existing_cols_top50], use_container_width=True)

        try:
            st.markdown('<div class="cel-section-title">Trade Z-diff 30D heatmap</div>', unsafe_allow_html=True)
            z_chart = (
                alt.Chart(top_trades_agent.reset_index().head(50))
                .mark_rect()
                .encode(
                    x="Ranking_Score:O",
                    y="index:O",
                    color=alt.Color("Trade_ZDiff_30D_Pct", scale=alt.Scale(scheme="redblue")),
                )
                .properties(height=500, width=800)
            )
            st.altair_chart(z_chart)
        except Exception as e:
            st.warning(f"Heatmap unavailable: {e}")


# ═════════════════════════════════════════════
# TAB 4 — AI Assistant (chat only)
# ═════════════════════════════════════════════
with tab4:
    _chat_hdr, _chat_btn = st.columns([6, 1])
    with _chat_hdr:
        st.markdown('<div class="cel-section-title">Bond AI assistant</div>', unsafe_allow_html=True)
        st.caption("Ask anything about top trades, signals, or specific bonds.")
    with _chat_btn:
        if st.button("Clear chat", key="clear_chat"):
            if "chat_history" in st.session_state:
                del st.session_state["chat_history"]
            st.rerun()

    # Build system prompt once
    if "chat_history" not in st.session_state:
        cols_top3 = [
            "A_ISIN", "B_ISIN", "C_ISIN", "D_ISIN",
            "LEG_1", "LEG_2",
            "Trade_ZDiff_30D_Pct", "Diff_of_Diffs_Today",
            "Ranking_Score", "Actionable_Direction",
        ]
        existing_top3 = [c for c in cols_top3 if c in top_trades_agent.columns]
        top3 = top_trades_agent.nlargest(3, "Ranking_Score")[existing_top3].to_dict(orient="records")
        system_prompt = get_system_prompt(top_trades_agent)
        system_prompt += f"\n\nTop 3 trades summary:\n{top3}"
        st.session_state.chat_history = [{"role": "system", "content": system_prompt}]

    # Render existing messages with native chat UI
    for msg in st.session_state.chat_history[1:]:
        role = "user" if msg["role"] == "user" else "assistant"
        with st.chat_message(role):
            st.markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask about trades, bonds, or signals…")
    if user_input:
        st.session_state.chat_history.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.markdown(user_input)

        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer, *_ = chat_with_trades(user_input, st.session_state.chat_history)
            st.markdown(answer)

        st.session_state.chat_history.append({"role": "assistant", "content": answer})
