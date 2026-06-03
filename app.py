import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import zipfile, shutil, hashlib, os, json, re, ast
from datetime import datetime
import boto3
from scipy.interpolate import interp1d
import altair as alt
from nelson_siegel_fn import plot_ns_animation, nelson_siegel
from ai_explainer_utils import format_bond_diagnostics, generate_ai_explanation
from curve_trade_agent1 import chat_with_trades, get_system_prompt

# ══════════════════════════════════════════════════════════════════
# 1. CONSTANTS  (must be before any st.* call)
# ══════════════════════════════════════════════════════════════════
COUNTRY_OPTIONS = [
    "Italy 🇮🇹","Spain 🇪🇸","France 🇫🇷","Germany 🇩🇪",
    "Finland 🇫🇮","EU 🇪🇺","Austria 🇦🇹","Netherlands 🇳🇱","Belgium 🇧🇪",
]
COUNTRY_CODE_MAP = {
    "Italy 🇮🇹":"BTPS","Spain 🇪🇸":"SPGB","France 🇫🇷":"FRTR",
    "Germany 🇩🇪":"BUNDS","Finland 🇫🇮":"RFGB","EU 🇪🇺":"EU",
    "Austria 🇦🇹":"RAGB","Netherlands 🇳🇱":"NETHER","Belgium 🇧🇪":"BGB",
}
SIGNAL_TYPES = [
    "STRONG BUY","STRONG SELL","MODERATE BUY","MODERATE SELL",
    "WEAK BUY","WEAK SELL","NO ACTION",
]
LEGEND_SIGNALS = {"strong buy","moderate buy","strong sell","moderate sell"}
SIGNAL_COLOR_MAP = {
    "strong buy":"#16a34a","moderate buy":"#4ade80","weak buy":"#94a3b8",
    "strong sell":"#dc2626","moderate sell":"#ea580c","weak sell":"#94a3b8",
}
SIG_CSS = {
    "STRONG BUY":"sig-strong-buy","STRONG SELL":"sig-strong-sell",
    "MODERATE BUY":"sig-mod-buy","MODERATE SELL":"sig-mod-sell",
    "WEAK BUY":"sig-weak-buy","WEAK SELL":"sig-weak-sell","NO ACTION":"sig-no-action",
}

# chart constants
C_PAPER = "#ffffff"
C_PLOT  = "#f9fafb"
C_GRID  = "#e5e9f0"
C_AXIS  = "#9ba3b0"
C_FONT  = dict(family="Space Mono, monospace", color="#1c2536", size=10)
AMBER   = "#e8b84b"
PRED_C  = "#7c3aed"

AWS_ACCESS_KEY_ID     = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
BUCKET_NAME  = "bonds-celestial-signal"
LOCAL_ZIP    = "ns_curves_20260106.zip"
LOCAL_FOLDER = "ns_curves_0106"

# ══════════════════════════════════════════════════════════════════
# 2. PURE HELPERS  (no st.* calls)
# ══════════════════════════════════════════════════════════════════
def get_country_from_isin(isin):
    m = {"IT":"🇮🇹 Italy","ES":"🇪🇸 Spain","FR":"🇫🇷 France","DE":"🇩🇪 Germany",
         "FI":"🇫🇮 Finland","EU":"🇪🇺 EU","AT":"🇦🇹 Austria","NL":"🇳🇱 Netherlands","BE":"🇧🇪 Belgium"}
    return m.get(isin[:2], "🌍 Unknown")

def parse_ns_params(x):
    if isinstance(x, (list, tuple, np.ndarray)): return list(x)
    if isinstance(x, str):
        try: return json.loads(x)
        except Exception: pass
        nums = re.findall(r"np\.float64\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)", x)
        if len(nums) >= 4: return [float(n) for n in nums[:4]]
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x)
        if len(nums) >= 4: return [float(n) for n in nums[:4]]
    return None

def chart(fig, title="", h=600, xt="", yt=""):
    fig.update_layout(
        title=dict(text=title, font=dict(family="Space Mono,monospace",size=11,color="#1c2536"),
                   x=0.01, xanchor="left", pad=dict(b=4)),
        height=h, paper_bgcolor=C_PAPER, plot_bgcolor=C_PLOT,
        font=C_FONT, margin=dict(l=52,r=18,t=40,b=40),
        xaxis=dict(title=dict(text=xt,font=dict(**C_FONT,size=9)),
                   gridcolor=C_GRID,linecolor=C_GRID,zeroline=False,
                   tickfont=dict(family="Space Mono,monospace",color=C_AXIS,size=9)),
        yaxis=dict(title=dict(text=yt,font=dict(**C_FONT,size=9)),
                   gridcolor=C_GRID,linecolor=C_GRID,zeroline=True,
                   zerolinecolor=C_GRID,zerolinewidth=1,
                   tickfont=dict(family="Space Mono,monospace",color=C_AXIS,size=9)),
        legend=dict(bgcolor="rgba(255,255,255,0.9)",bordercolor=C_GRID,borderwidth=1,
                    font=dict(family="Space Mono,monospace",size=9,color="#1c2536")),
        hoverlabel=dict(bgcolor="#1c2333",bordercolor="#263047",
                        font=dict(family="Space Mono,monospace",size=10,color="#dde3ee")),
    )
    return fig

def sh(label):
    st.markdown(f'<div class="sh">{label}</div>', unsafe_allow_html=True)

def fmt_bond(isin, bond_opts, bond_labels):
    mat = bond_opts.loc[bond_opts["ISIN"]==isin,"Maturity"].values
    if len(mat) and pd.notnull(mat[0]):
        return f"{bond_labels.get(isin,isin)}  ·  {pd.to_datetime(mat[0]).strftime('%Y-%m-%d')}"
    return f"{bond_labels.get(isin,isin)}  ·  N/A"

# ══════════════════════════════════════════════════════════════════
# 3. PAGE CONFIG  (first st.* call)
# ══════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="◈ Celestial", page_icon="◈",
    layout="wide", initial_sidebar_state="expanded",
)

# ══════════════════════════════════════════════════════════════════
# 4. CSS
# ══════════════════════════════════════════════════════════════════
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=Inter:wght@300;400;500&display=swap');

:root {
    --bg0:#07090f; --bg1:#0c0f18; --bg2:#111520; --bg3:#161b28;
    --b1:#1a2035;  --b2:#243050;
    --amber:#e8b84b; --sky:#38bdf8; --green:#4ade80; --red:#f87171;
    --muted:#6b7a99; --text:#c8d0e0;
    --mono:'Space Mono',monospace; --sans:'Inter',sans-serif;
}

html,body,[class*="css"]{ font-family:var(--sans)!important; }
.stApp{ background:var(--bg0)!important; }
main .block-container{ padding:0 1.75rem 3rem!important; max-width:100%!important; }
#MainMenu,footer,header{ visibility:hidden; }

/* ── Sidebar ── */
section[data-testid="stSidebar"]{
    background:var(--bg1)!important;
    border-right:1px solid var(--b1)!important;
    width:220px!important; min-width:220px!important;
}
section[data-testid="stSidebar"] > div{ padding:1.5rem 1rem 2rem!important; }

.sb-mark{ font-family:var(--mono); font-size:0.88rem; font-weight:700;
          letter-spacing:0.05em; color:#fff; }
.sb-mark span{ color:var(--amber); }
.sb-sub{ font-family:var(--mono); font-size:0.48rem; letter-spacing:0.2em;
         text-transform:uppercase; color:var(--muted); margin-bottom:1.8rem; margin-top:0.15rem; }

.sb-sep{ font-family:var(--mono); font-size:0.48rem; font-weight:700;
         letter-spacing:0.22em; text-transform:uppercase; color:var(--muted);
         display:flex; align-items:center; gap:0.5rem;
         margin:1.3rem 0 0.65rem; }
.sb-sep::before{ content:''; width:8px; height:1.5px; background:var(--amber); flex-shrink:0; }

.sb-legend{ font-family:var(--mono); font-size:0.6rem; color:var(--muted); line-height:2.1; }

section[data-testid="stSidebar"] label,
section[data-testid="stSidebar"] .stSelectbox label,
section[data-testid="stSidebar"] .stMultiSelect label,
section[data-testid="stSidebar"] .stRadio label,
section[data-testid="stSidebar"] .stDateInput label,
section[data-testid="stSidebar"] .stCheckbox label,
section[data-testid="stSidebar"] .stNumberInput label{
    font-family:var(--mono)!important; font-size:0.52rem!important;
    font-weight:700!important; letter-spacing:0.16em!important;
    text-transform:uppercase!important; color:var(--muted)!important;
}
section[data-testid="stSidebar"] p,
section[data-testid="stSidebar"] span{ color:var(--text)!important; }
section[data-testid="stSidebar"] div[data-baseweb="select"]>div,
section[data-testid="stSidebar"] div[data-baseweb="input"]>div{
    background:var(--bg2)!important; border:1px solid var(--b2)!important;
    border-radius:3px!important; font-family:var(--mono)!important;
    font-size:0.68rem!important; color:var(--text)!important;
}
section[data-testid="stSidebar"] div[data-baseweb="select"]>div:hover,
section[data-testid="stSidebar"] div[data-baseweb="input"]>div:focus-within{
    border-color:var(--amber)!important;
}

/* ── Main inputs ── */
div[data-baseweb="select"]>div, div[data-baseweb="input"]>div{
    background:var(--bg2)!important; border-color:var(--b1)!important;
    border-radius:3px!important; font-family:var(--mono)!important;
    font-size:0.7rem!important; color:var(--text)!important; transition:border-color .15s!important;
}
div[data-baseweb="select"]>div:hover,
div[data-baseweb="input"]>div:focus-within{ border-color:var(--amber)!important; }

/* ── Tabs ── */
.stTabs [data-baseweb="tab-list"]{
    gap:0!important; border-bottom:1px solid var(--b1)!important;
    background:transparent!important; padding:0!important;
}
.stTabs [data-baseweb="tab"]{
    font-family:var(--mono)!important; font-size:0.56rem!important;
    font-weight:700!important; letter-spacing:0.16em!important;
    text-transform:uppercase!important; color:var(--muted)!important;
    background:transparent!important; border:none!important;
    border-bottom:2px solid transparent!important;
    padding:0.7rem 1.4rem!important; transition:color .15s!important;
}
.stTabs [data-baseweb="tab"]:hover{ color:var(--text)!important; }
.stTabs [aria-selected="true"]{
    color:var(--amber)!important;
    border-bottom:2px solid var(--amber)!important;
    background:transparent!important;
}
.stTabs [data-baseweb="tab-panel"]{ padding-top:1.5rem!important; }

/* ── Header ── */
.hdr{ display:flex; align-items:baseline; gap:1rem;
      padding:1rem 0 0.8rem; border-bottom:1px solid var(--b1); margin-bottom:1.5rem; }
.hdr-logo{ font-family:var(--mono); font-size:1rem; font-weight:700;
           letter-spacing:0.05em; color:#fff; }
.hdr-logo span{ color:var(--amber); }
.hdr-ts{ font-family:var(--mono); font-size:0.56rem; letter-spacing:0.1em; color:var(--muted); }

/* ── Section headers ── */
.sh{ font-family:var(--mono); font-size:0.5rem; font-weight:700;
     letter-spacing:0.22em; text-transform:uppercase; color:var(--muted);
     display:flex; align-items:center; gap:0.5rem;
     margin:1.6rem 0 0.7rem; padding-bottom:0.5rem; border-bottom:1px solid var(--b1); }
.sh::before{ content:''; width:8px; height:1.5px; background:var(--amber); flex-shrink:0; }

/* ── Chart cards ── */
.js-plotly-plot{
    background:#fff!important; border:1px solid var(--b1)!important;
    border-radius:4px!important; box-shadow:0 4px 24px rgba(0,0,0,.5)!important;
}

/* ── Signal cards ── */
.sig-row{ display:grid; grid-template-columns:repeat(7,1fr); gap:0.4rem; margin:0.5rem 0 1.2rem; }
.sig-box{
    padding:0.8rem 0.6rem 0.7rem 0.85rem; border-radius:2px;
    border-left:3px solid transparent;
    border-top:1px solid var(--b1); border-right:1px solid var(--b1); border-bottom:1px solid var(--b1);
    background:var(--bg2); transition:transform .14s,box-shadow .14s; cursor:default;
}
.sig-box:hover{ transform:translateY(-2px); box-shadow:0 6px 24px rgba(0,0,0,.55); }
.sig-count{ font-family:var(--mono); font-size:1.6rem; font-weight:700; line-height:1; color:#fff; }
.sig-label{ font-family:var(--mono); font-size:0.5rem; font-weight:700;
             letter-spacing:0.12em; text-transform:uppercase; margin-top:0.28rem; opacity:.6; }
.sig-delta{ font-family:var(--mono); font-size:0.6rem; font-weight:700;
             margin-top:0.3rem; display:inline-flex; align-items:center;
             gap:3px; padding:1px 5px; border-radius:2px; }
.du{ background:rgba(74,222,128,.1); color:var(--green); }
.dd{ background:rgba(248,113,113,.1); color:var(--red); }
.df{ background:rgba(107,122,153,.1); color:var(--muted); }
.sig-strong-buy { border-left-color:#22c55e; } .sig-strong-buy  .sig-label{ color:#22c55e; }
.sig-strong-sell{ border-left-color:#ef4444; } .sig-strong-sell .sig-label{ color:#ef4444; }
.sig-mod-buy    { border-left-color:#4ade80; } .sig-mod-buy     .sig-label{ color:#4ade80; }
.sig-mod-sell   { border-left-color:#fb923c; } .sig-mod-sell    .sig-label{ color:#fb923c; }
.sig-weak-buy   { border-left-color:#38bdf8; } .sig-weak-buy    .sig-label{ color:#38bdf8; }
.sig-weak-sell  { border-left-color:#f59e0b; } .sig-weak-sell   .sig-label{ color:#f59e0b; }
.sig-no-action  { border-left-color:#334155; } .sig-no-action   .sig-label{ color:var(--muted); }

/* ── Buttons ── */
.stButton>button{
    font-family:var(--mono)!important; font-size:0.56rem!important; font-weight:700!important;
    letter-spacing:0.14em!important; text-transform:uppercase!important;
    background:transparent!important; border:1px solid var(--b2)!important;
    color:var(--muted)!important; border-radius:2px!important;
    padding:0.32rem 0.8rem!important; transition:all .15s!important;
}
.stButton>button:hover{ border-color:var(--amber)!important; color:var(--amber)!important; }
.stDownloadButton>button{
    font-family:var(--mono)!important; font-size:0.56rem!important; font-weight:700!important;
    letter-spacing:0.12em!important; text-transform:uppercase!important;
    background:transparent!important; border:1px solid var(--b2)!important;
    color:var(--amber)!important; border-radius:2px!important; transition:all .15s!important;
}
.stDownloadButton>button:hover{ border-color:var(--amber)!important; background:rgba(232,184,75,.08)!important; }

/* ── Metric ── */
div[data-testid="metric-container"]{
    background:var(--bg2); border:1px solid var(--b1);
    border-top:2px solid var(--amber); border-radius:2px; padding:.85rem 1rem!important;
}
div[data-testid="metric-container"] label{
    font-family:var(--mono)!important; font-size:0.54rem!important;
    letter-spacing:.18em!important; text-transform:uppercase!important;
    color:var(--muted)!important; font-weight:700!important;
}
div[data-testid="metric-container"] [data-testid="stMetricValue"]{
    font-family:var(--mono)!important; color:#fff!important;
    font-size:1.45rem!important; font-weight:700!important;
}

/* ── Dataframe ── */
.stDataFrame{ border:1px solid var(--b1)!important; border-radius:3px!important;
              font-family:var(--mono)!important; font-size:0.68rem!important; }

/* ── Chat ── */
div[data-testid="stChatMessage"]{
    border-radius:3px!important; padding:.6rem .9rem!important;
    margin-bottom:.35rem!important; border:1px solid var(--b1)!important; background:var(--bg2)!important;
}
div[data-testid="stChatMessage"] p{
    font-family:var(--sans)!important; font-size:.84rem!important;
    line-height:1.65!important; color:var(--text)!important;
}
div[data-testid="stChatInputContainer"]{
    background:var(--bg2)!important; border:1px solid var(--b2)!important;
    border-radius:3px!important; margin-top:.5rem!important;
}
div[data-testid="stChatInputContainer"]:focus-within{ border-color:var(--amber)!important; }

/* ── Alerts ── */
div[data-testid="stAlert"]{
    border-radius:3px!important; border:1px solid var(--b1)!important;
    background:var(--bg2)!important; font-family:var(--mono)!important;
    font-size:.7rem!important; color:var(--text)!important;
}

/* ── Typography ── */
h1{ font-family:var(--mono)!important; font-size:1.15rem!important;
    font-weight:700!important; color:#fff!important; letter-spacing:-.01em!important; }
h2,h3{ font-family:var(--sans)!important; font-weight:500!important; color:var(--text)!important; }
p,li{ color:var(--muted)!important; }
small,.stCaption{ font-family:var(--mono)!important; color:var(--muted)!important;
                  font-size:.58rem!important; letter-spacing:.06em!important; }
hr{ border-color:var(--b1)!important; margin:.6rem 0!important; }
label{ color:var(--muted)!important; }

/* ── Scrollbar ── */
::-webkit-scrollbar{ width:3px; height:3px; }
::-webkit-scrollbar-track{ background:var(--bg1); }
::-webkit-scrollbar-thumb{ background:var(--b2); border-radius:2px; }
::-webkit-scrollbar-thumb:hover{ background:var(--muted); }
</style>
""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 5. AWS / S3 + DATA LOADERS
# ══════════════════════════════════════════════════════════════════
def download_from_s3(file_key, local_path, force=False):
    if not force and os.path.exists(local_path): return local_path
    with st.spinner(f"Downloading {file_key}…"):
        s3 = boto3.client("s3", aws_access_key_id=AWS_ACCESS_KEY_ID,
                          aws_secret_access_key=AWS_SECRET_ACCESS_KEY)
        s3.download_file(BUCKET_NAME, file_key, local_path)
    return local_path

def file_hash(fp):
    h = hashlib.md5()
    with open(fp,"rb") as f:
        for chunk in iter(lambda: f.read(8192), b""): h.update(chunk)
    return h.hexdigest()

def unzip_ns_curves(zip_path=LOCAL_ZIP, folder=LOCAL_FOLDER, force=False):
    zip_path = download_from_s3("ns_curves_0106.zip", zip_path, force=force)
    zh = file_hash(zip_path)
    if force or st.session_state.get("ns_zip_hash") != zh or not os.path.exists(folder):
        if os.path.exists(folder): shutil.rmtree(folder)
        with zipfile.ZipFile(zip_path,"r") as z: z.extractall(folder)
        st.session_state["ns_zip_hash"] = zh
    return folder, zh

@st.cache_data
def load_full_ns_df(country_code, zip_hash):
    folder, _ = unzip_ns_curves(force=False)
    files = sorted(f for f in os.listdir(folder) if f.startswith(country_code) and f.endswith(".parquet"))
    dfs = []
    for f in files:
        try:
            df = pd.read_parquet(os.path.join(folder, f))
            if "ISIN" in df.columns:    df["ISIN"]    = df["ISIN"].astype(str).str.strip()
            if "Date" in df.columns:    df["Date"]    = pd.to_datetime(df["Date"], errors="coerce")
            if "Country" not in df.columns: df["Country"] = country_code
            dfs.append(df)
        except Exception as e: st.warning(f"Error loading {f}: {e}")
    if not dfs: return pd.DataFrame()
    out = pd.concat(dfs, ignore_index=True)
    if "RESIDUAL" in out.columns and "RESIDUAL_NS" not in out.columns:
        out.rename(columns={"RESIDUAL":"RESIDUAL_NS"}, inplace=True)
    if "RESIDUAL_NS" not in out.columns: out["RESIDUAL_NS"] = pd.NA
    out.sort_values("Date", inplace=True)
    return out

def load_ns_curve(country_code, date_str, zip_hash):
    df = load_full_ns_df(country_code, zip_hash=zip_hash)
    if df is not None and not df.empty:
        sub = df[df["Date"].dt.date == pd.to_datetime(date_str).date()]
        if not sub.empty: return sub
    return None

@st.cache_data
def load_trades():
    df = pd.read_pickle("top_trades_agent.pkl")
    for col in df.select_dtypes(include=["datetime64[ns]"]).columns:
        df[col] = df[col].astype(str)
    return df

@st.cache_data(ttl=300)
def load_signal_data():
    local = "issuer_signals.csv"
    try:
        local = download_from_s3("issuer_signals.csv", local, force=False)
        return pd.read_csv(local)
    except Exception as e:
        st.error(f"S3 load error: {e}")
        return pd.read_csv(local) if os.path.exists(local) else pd.DataFrame()

@st.cache_data(ttl=300)
def load_issuer_signal():
    try: return pd.read_csv("issuer_signals.csv")
    except: return pd.DataFrame()

# ── Bootstrap shared state ──
try:
    zip_path = download_from_s3("ns_curves_0106.zip", LOCAL_ZIP, force=False)
    zip_hash = file_hash(zip_path) if os.path.exists(zip_path) else None
except Exception as e:
    st.error(f"NS curves unavailable: {e}")
    zip_hash = None

top_trades_agent = load_trades()

# ── Available dates for NS date picker ──
_sig_tmp   = pd.read_csv("today_all_signals.csv") if os.path.exists("today_all_signals.csv") else pd.DataFrame(columns=["Date"])
_avail_dt  = pd.to_datetime(_sig_tmp["Date"].unique()) if not _sig_tmp.empty else pd.DatetimeIndex([pd.Timestamp.today()])
_MAX_DATE  = pd.Timestamp(_avail_dt.max()).date()

# ── Signal dashboard country list ──
_sig_raw      = load_signal_data()
_sig_raw["Country"] = _sig_raw["ISIN"].apply(get_country_from_isin) if not _sig_raw.empty else ""
_ALL_SIG_CTRY = sorted(_sig_raw["Country"].unique().tolist()) if not _sig_raw.empty else []

# ══════════════════════════════════════════════════════════════════
# 6. SESSION STATE — track active view for sidebar routing
# ══════════════════════════════════════════════════════════════════
if "view" not in st.session_state:
    st.session_state.view = "single_day"

# ══════════════════════════════════════════════════════════════════
# 7. SIDEBAR — single block, routes by st.session_state.view
# ══════════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown('<div class="sb-mark">◈ CELESTIAL <span>AM</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="sb-sub">Fixed Income · Bond Analytics</div>', unsafe_allow_html=True)

    v = st.session_state.view

    # ── Single day curve ──────────────────────
    if v == "single_day":
        st.markdown('<div class="sb-sep">Single Day Curve</div>', unsafe_allow_html=True)
        sb_sd_country = st.selectbox("Country", COUNTRY_OPTIONS, key="sb_sd_country")
        sb_sd_date    = st.date_input("Date", value=_MAX_DATE, key="sb_sd_date")
        st.markdown('<div class="sb-sep">Legend</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="sb-legend">
  <span style="color:#16a34a">●</span> Strong buy<br>
  <span style="color:#4ade80">●</span> Moderate buy<br>
  <span style="color:#dc2626">●</span> Strong sell<br>
  <span style="color:#ea580c">●</span> Moderate sell<br>
  <span style="color:#94a3b8">●</span> Weak / none<br>
  <span style="color:#e8b84b">—</span> NS fit
</div>""", unsafe_allow_html=True)

    # ── Animated curves ───────────────────────
    elif v == "animated":
        st.markdown('<div class="sb-sep">Animated Curves</div>', unsafe_allow_html=True)
        sb_an_country = st.selectbox("Country", COUNTRY_OPTIONS, key="sb_an_country")
        st.markdown('<div class="sb-sep">Bond Selection</div>', unsafe_allow_html=True)
        sb_an_all = st.checkbox("Show all bonds", key="sb_an_all")

    # ── Residuals ─────────────────────────────
    elif v == "residuals":
        st.markdown('<div class="sb-sep">Residuals Analysis</div>', unsafe_allow_html=True)
        sb_res_country  = st.selectbox("Country", COUNTRY_OPTIONS, key="sb_res_country")
        sb_res_velocity = st.checkbox("Show velocity chart", value=True, key="sb_res_vel")
        st.markdown('<div class="sb-sep">Bond Selection</div>', unsafe_allow_html=True)
        # multiselect rendered here after data is loaded — placeholder for now
        st.caption("Bond selection appears below after data loads.")

    # ── Compare curves ────────────────────────
    elif v == "compare":
        st.markdown('<div class="sb-sep">Compare NS Curves</div>', unsafe_allow_html=True)
        sb_cmp_countries = st.multiselect(
            "Countries", COUNTRY_OPTIONS, default=COUNTRY_OPTIONS[:2], key="sb_cmp_countries"
        )

    # ── New bond prediction ───────────────────
    elif v == "predict":
        st.markdown('<div class="sb-sep">New Bond Prediction</div>', unsafe_allow_html=True)
        sb_pred_country     = st.selectbox("Country", COUNTRY_OPTIONS, key="sb_pred_country")
        sb_pred_maturity    = st.text_input("Maturity (MM/YY)", value="10/55", key="sb_pred_mat")
        sb_pred_concession  = st.number_input("Auction concession (bps)", value=0, step=1, key="sb_pred_conc")

    # ── Signal dashboard ──────────────────────
    elif v == "signals":
        st.markdown('<div class="sb-sep">Signal Dashboard</div>', unsafe_allow_html=True)
        sb_sig_ctry = st.multiselect("Countries", _ALL_SIG_CTRY, default=_ALL_SIG_CTRY, key="sb_sig_ctry")
        sb_sig_sigs = st.multiselect(
            "Signals", SIGNAL_TYPES,
            default=[s for s in SIGNAL_TYPES if s != "NO ACTION"],
            key="sb_sig_sigs",
        )

    # ── Analysis ─────────────────────────────
    elif v == "analysis":
        st.markdown('<div class="sb-sep">Multi-Curve Analysis</div>', unsafe_allow_html=True)
        sb_an_metric   = st.radio("Metric", ["Z-Spread","Residuals"], key="sb_an_metric")
        st.markdown('<div class="sb-sep">Curve 1</div>', unsafe_allow_html=True)
        sb_an_c1       = st.selectbox("Country", COUNTRY_OPTIONS, key="sb_an_c1")
        st.markdown('<div class="sb-sep">Curve 2</div>', unsafe_allow_html=True)
        sb_an_c2       = st.selectbox("Country", COUNTRY_OPTIONS, key="sb_an_c2")

    # ── Top trades ────────────────────────────
    elif v == "trades":
        st.markdown('<div class="sb-sep">Top Trades</div>', unsafe_allow_html=True)
        st.markdown('<div class="sb-legend">Ranked by composite score.<br>Top 50 shown.</div>',
                    unsafe_allow_html=True)

    # ── AI chat ──────────────────────────────
    elif v == "chat":
        st.markdown('<div class="sb-sep">AI Assistant</div>', unsafe_allow_html=True)
        st.markdown("""
<div class="sb-legend">Ask about top trades,<br>bond signals, residuals,<br>or specific ISINs.</div>
""", unsafe_allow_html=True)
        if st.button("Clear conversation", key="sb_clear_chat"):
            st.session_state.pop("chat_history", None)
            st.rerun()

    # ── System (always at bottom) ─────────────
    st.markdown('<div class="sb-sep">System</div>', unsafe_allow_html=True)
    if st.button("↺ Refresh data", key="sb_refresh"):
        st.cache_data.clear()
        st.rerun()
    st.markdown("""
<div style="margin-top:1.5rem;font-family:var(--mono);font-size:0.46rem;
            color:var(--muted);line-height:2.2;letter-spacing:0.06em">
DHARMA ASSET MANAGEMENT<br>CELESTIAL TEAM · INTERNAL
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 8. PAGE HEADER
# ══════════════════════════════════════════════════════════════════
_now = datetime.now()
st.markdown(f"""
<div class="hdr">
  <div class="hdr-logo">◈ CELESTIAL <span>BOND ANALYTICS</span></div>
  <div class="hdr-ts">{_now.strftime("%a %d %b %Y · %H:%M UTC")}</div>
</div>""", unsafe_allow_html=True)

# ══════════════════════════════════════════════════════════════════
# 9. TABS  (on_change updates session_state.view)
# ══════════════════════════════════════════════════════════════════
TAB_LABELS = [
    "📈  NS Curves",
    "📊  Signals",
    "🔬  Analysis",
    "🤖  AI",
]
tab1, tab2, tab3, tab4 = st.tabs(TAB_LABELS)

# ══════════════════════════════════════════════════════════════════
# TAB 1 — Nelson-Siegel Curves
# ══════════════════════════════════════════════════════════════════
with tab1:
    sub1, sub2, sub3, sub4, sub5 = st.tabs([
        "Single day","Animated","Residuals","Compare","Predict",
    ])

    # ── helper: build bond opts dataframe from ns_df ──
    def _bond_opts(ns_df_in, signal_df=None):
        opts = ns_df_in[["ISIN","SECURITY_NAME"]].drop_duplicates().copy()
        mat_map = ns_df_in.groupby("ISIN")["Maturity"].first().to_dict()
        opts["Maturity"] = pd.to_datetime(opts["ISIN"].map(mat_map), errors="coerce")
        opts.sort_values("Maturity", inplace=True)
        return opts

    # ── NS fit line helper ────────────────────
    def _add_ns_fit(fig, ns_df_in, ytm_col="YTM"):
        if "NS_PARAMS" not in ns_df_in.columns and not all(
            c in ns_df_in.columns for c in ["NS_PARAM_1","NS_PARAM_2","NS_PARAM_3","NS_PARAM_4"]
        ): return
        try:
            params = None
            if "NS_PARAMS" in ns_df_in.columns:
                raw = ns_df_in["NS_PARAMS"].dropna()
                if not raw.empty: params = parse_ns_params(raw.iloc[0])
            if params is None and all(f"NS_PARAM_{i}" in ns_df_in.columns for i in range(1,5)):
                params = [ns_df_in[f"NS_PARAM_{i}"].iloc[0] for i in range(1,5)]
            if params:
                mr = np.linspace(ns_df_in[ytm_col].min(), ns_df_in[ytm_col].max(), 120)
                fig.add_trace(go.Scatter(
                    x=mr, y=nelson_siegel(mr, *params),
                    mode="lines", name="NS fit",
                    line=dict(color=AMBER, width=2.5),
                    showlegend=True,
                ))
        except Exception as e:
            st.warning(f"NS fit skipped: {e}")

    # ── scatter by signal ─────────────────────
    def _scatter_signals(fig, df_in, x_col, y_col, label_col="SECURITY_NAME",
                         isin_col="ISIN", date_col="Date", residual_col="RESIDUAL_NS"):
        for signal, grp in df_in.groupby("SIGNAL"):
            if grp.empty: continue
            col = grp["Signal_Color"].iloc[0]
            res = grp[residual_col] if residual_col in grp.columns else pd.Series(np.zeros(len(grp)), index=grp.index)
            fig.add_trace(go.Scatter(
                x=grp[x_col], y=grp[y_col], mode="markers",
                name=signal.title() if signal in LEGEND_SIGNALS else None,
                marker=dict(size=7, color=col, symbol="circle",
                            line=dict(width=0.5, color="rgba(255,255,255,0.3)")),
                text=grp[label_col],
                customdata=np.stack((grp[isin_col], grp[date_col].astype(str), res), axis=-1),
                hovertemplate=(
                    "<b>%{text}</b><br>"
                    "YTM: %{x:.2f}y<br>"
                    "Z-spread: %{y:.1f} bps<br>"
                    "Residual: %{customdata[2]:.2f} bps<br>"
                    f"Signal: {signal.title()}<extra></extra>"
                ),
                showlegend=(signal in LEGEND_SIGNALS),
            ))

    # ═══════════════════════════════════════════
    # SUBTAB 1 — Single day curve
    # ═══════════════════════════════════════════
    with sub1:
        st.session_state.view = "single_day"
        country  = COUNTRY_CODE_MAP[st.session_state.get("sb_sd_country", COUNTRY_OPTIONS[0])]
        date_str = pd.Timestamp(st.session_state.get("sb_sd_date", _MAX_DATE)).strftime("%Y-%m-%d")

        final_signal_df = pd.read_csv("today_all_signals.csv")
        ns_df = load_ns_curve(country, date_str, zip_hash=zip_hash)

        col_plot, col_ai = st.columns([3, 2], gap="medium")

        if ns_df is None or ns_df.empty:
            with col_plot: st.info("No NS data for this date.")
        else:
            # normalise column names
            cm = {c.lower(): c for c in ns_df.columns}
            for alias, canon in [("z_sprd_val","Z_SPRD"),("z_sprd","Z_SPRD"),("yearstomaturity","YTM")]:
                if alias in cm and canon not in ns_df.columns:
                    ns_df.rename(columns={cm[alias]: canon}, inplace=True)
            ns_df["Maturity"]     = pd.to_datetime(ns_df["Maturity"])
            ns_df["YTM"]          = (ns_df["Maturity"] - pd.to_datetime(date_str)).dt.days / 365.25
            ns_df = ns_df.merge(final_signal_df[["ISIN","SIGNAL"]], on="ISIN", how="left")
            ns_df["SIGNAL"]       = ns_df["SIGNAL"].str.strip().str.lower()
            ns_df["Signal_Color"] = ns_df["SIGNAL"].map(SIGNAL_COLOR_MAP).fillna("#94a3b8")

            fig = go.Figure()
            _scatter_signals(fig, ns_df, "YTM", "Z_SPRD")
            _add_ns_fit(fig, ns_df)
            chart(fig, f"NS curve · {country} · {date_str}", h=580,
                  xt="Years to maturity", yt="Z-spread (bps)")
            fig.update_layout(clickmode="event+select")

            with col_plot:
                event = st.plotly_chart(fig, use_container_width=True,
                                        on_select="rerun", key="sd_chart")

            with col_ai:
                sh("Bond AI explanation")
                clicked_isin = None
                if event and event.selection and event.selection.get("points"):
                    cd = event.selection["points"][0].get("customdata")
                    if cd: clicked_isin = cd[0]

                bo = (final_signal_df[["ISIN","SECURITY_NAME"]].drop_duplicates()
                                                                 .sort_values("SECURITY_NAME"))
                bl = dict(zip(bo["ISIN"], bo["SECURITY_NAME"]))
                def_idx = bo["ISIN"].tolist().index(clicked_isin) if clicked_isin and clicked_isin in bo["ISIN"].values else 0
                sel_isin = st.selectbox(
                    "Select bond (or click chart)",
                    bo["ISIN"].tolist(), index=def_idx,
                    format_func=lambda i: bl.get(i, i), key="sd_bond_sel",
                )
                st.caption(f"ISIN: {sel_isin}")
                if st.button("Explain this bond", key="sd_explain"):
                    hist = final_signal_df[final_signal_df["ISIN"]==sel_isin]
                    diag = format_bond_diagnostics(hist)
                    with st.spinner("Generating explanation…"):
                        st.markdown(generate_ai_explanation(diag))

    # ═══════════════════════════════════════════
    # SUBTAB 2 — Animated
    # ═══════════════════════════════════════════
    with sub2:
        st.session_state.view = "animated"
        country = COUNTRY_CODE_MAP[st.session_state.get("sb_an_country", COUNTRY_OPTIONS[0])]
        show_all = st.session_state.get("sb_an_all", False)

        ns_df = load_full_ns_df(country, zip_hash=zip_hash)
        if ns_df is None or ns_df.empty:
            st.warning("No NS data available.")
        else:
            final_signal_df = pd.read_csv("today_all_signals.csv")
            bo   = _bond_opts(ns_df)
            blbl = dict(zip(bo["ISIN"], bo["SECURITY_NAME"]))

            if show_all:
                sel = bo["ISIN"].tolist()
                st.caption(f"Showing all {len(sel)} bonds for {country}")
            else:
                sel = st.multiselect(
                    "Select bonds to animate",
                    bo["ISIN"].tolist(),
                    format_func=lambda i: fmt_bond(i, bo, blbl),
                    default=[], key="anim_sel",
                )

            if not sel:
                st.info("Select bonds above or tick 'Show all bonds' in the sidebar.")
            else:
                ns_filt = ns_df[ns_df["ISIN"].isin(sel)].copy()
                ns_filt = ns_filt.merge(final_signal_df[["ISIN","SIGNAL"]], on="ISIN", how="left")
                fig = plot_ns_animation(ns_filt, issuer_label=country, highlight_isins=sel)
                st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════
    # SUBTAB 3 — Residuals
    # ═══════════════════════════════════════════
    with sub3:
        st.session_state.view = "residuals"
        country      = COUNTRY_CODE_MAP[st.session_state.get("sb_res_country", COUNTRY_OPTIONS[0])]
        show_vel     = st.session_state.get("sb_res_vel", True)

        ns_df = load_full_ns_df(country, zip_hash=zip_hash)
        if ns_df is None or ns_df.empty:
            st.warning("No NS data available.")
        else:
            ns_df["Date"] = pd.to_datetime(ns_df["Date"]).dt.normalize()
            ns_df["RESIDUAL_VELOCITY"] = ns_df.groupby("ISIN")["RESIDUAL_NS"].transform(lambda x: x.diff())
            bo   = _bond_opts(ns_df)
            blbl = dict(zip(bo["ISIN"], bo["SECURITY_NAME"]))

            sel = st.multiselect(
                "Select bonds",
                bo["ISIN"].tolist(),
                format_func=lambda i: fmt_bond(i, bo, blbl),
                default=[], key="res_sel",
            )

            if not sel:
                st.info("Select bonds above to display residuals.")
            else:
                rdf = ns_df[ns_df["ISIN"].isin(sel)].copy()
                fig_r = go.Figure()
                fig_v = go.Figure()
                for isin in sel:
                    bd = rdf[rdf["ISIN"]==isin].sort_values("Date")
                    if bd.empty: continue
                    lbl = blbl.get(isin, isin)
                    fig_r.add_trace(go.Scatter(x=bd["Date"],y=bd["RESIDUAL_NS"],
                                               mode="lines+markers",name=lbl,
                                               marker=dict(size=4)))
                    fig_v.add_trace(go.Scatter(x=bd["Date"],y=bd["RESIDUAL_VELOCITY"],
                                               mode="lines+markers",name=lbl,
                                               marker=dict(size=4)))
                chart(fig_r, "Residuals over time", h=400, xt="Date", yt="Residual (bps)")
                chart(fig_v, "Residual velocity", h=340, xt="Date", yt="bps/day")
                st.plotly_chart(fig_r, use_container_width=True)
                if show_vel:
                    st.plotly_chart(fig_v, use_container_width=True)

    # ═══════════════════════════════════════════
    # SUBTAB 4 — Compare NS curves
    # ═══════════════════════════════════════════
    with sub4:
        st.session_state.view = "compare"
        countries = st.session_state.get("sb_cmp_countries", COUNTRY_OPTIONS[:2])

        if not countries:
            st.info("Select countries in the sidebar.")
        else:
            # per-country date selectors (inline — depend on loaded data)
            all_dates, sel_dates = {}, {}
            for c in countries:
                tmp = load_full_ns_df(COUNTRY_CODE_MAP[c], zip_hash=zip_hash)
                if tmp is not None and not tmp.empty:
                    all_dates[c] = pd.Series(pd.to_datetime(tmp["Date"].unique())).sort_values(ascending=False)

            dcols = st.columns(len(countries))
            for i, c in enumerate(countries):
                if c in all_dates and len(all_dates[c]):
                    fmts = [d.strftime("%Y-%m-%d") for d in all_dates[c]]
                    with dcols[i]:
                        sel_dates[c] = st.multiselect(
                            f"Dates — {c}", fmts, default=[fmts[0]], key=f"cmp_d_{c}"
                        )

            fig = go.Figure()
            for c in countries:
                for d in sel_dates.get(c, []):
                    cd = load_ns_curve(COUNTRY_CODE_MAP[c], d, zip_hash=zip_hash)
                    if cd is None or cd.empty or "NS_PARAMS" not in cd.columns: continue
                    params = parse_ns_params(cd["NS_PARAMS"].iloc[0])
                    if params is None: continue
                    if "YTM" not in cd.columns or cd["YTM"].isna().all(): continue
                    mats = np.linspace(0, min(30, cd["YTM"].max()), 120)
                    fig.add_trace(go.Scatter(x=mats, y=nelson_siegel(mats, *params),
                                             mode="lines", name=f"{c} · {d}"))
            chart(fig, "NS curves comparison", h=600,
                  xt="Years to maturity", yt="Z-spread (bps)")
            fig.update_xaxes(range=[0,30])
            st.plotly_chart(fig, use_container_width=True)

    # ═══════════════════════════════════════════
    # SUBTAB 5 — New bond prediction
    # ═══════════════════════════════════════════
    with sub5:
        st.session_state.view = "predict"
        country    = COUNTRY_CODE_MAP[st.session_state.get("sb_pred_country", COUNTRY_OPTIONS[0])]
        mat_input  = st.session_state.get("sb_pred_mat", "10/55")
        concession = st.session_state.get("sb_pred_conc", 0)

        today_ts = pd.Timestamp.today().normalize()
        ns_list  = []
        for d in pd.date_range(today_ts - pd.Timedelta(days=14), today_ts):
            tmp = load_ns_curve(country, d.strftime("%Y-%m-%d"), zip_hash=zip_hash)
            if tmp is not None and not tmp.empty:
                tmp = tmp.copy()
                tmp["Date"] = pd.to_datetime(d)
                tmp["YearsToMaturity"] = (pd.to_datetime(tmp["Maturity"]) - today_ts).dt.days / 365.25
                ns_list.append(tmp)

        if not ns_list:
            st.warning("No NS data for the last 2 weeks.")
        else:
            ns_full  = pd.concat(ns_list, ignore_index=True)
            ns_smooth = ns_full.groupby("YearsToMaturity")["Z_SPRD_VAL"].mean().reset_index()
            ns_std    = ns_full.groupby("YearsToMaturity")["Z_SPRD_VAL"].std().reset_index()

            try:
                mo, yo = map(int, mat_input.split("/"))
                yo += 2000 if yo < 100 else 0
                new_mat  = pd.Timestamp(year=yo, month=mo, day=1)
                new_ytm  = (new_mat - today_ts).days / 365.25
            except Exception:
                st.error("Invalid format. Use MM/YY.")
                st.stop()

            final_signal_df = pd.read_csv("today_all_signals.csv")
            final_signal_df["Maturity"] = pd.to_datetime(final_signal_df["Maturity"], errors="coerce")
            similar = final_signal_df[
                final_signal_df["Maturity"].notna() &
                (abs((final_signal_df["Maturity"] - new_mat).dt.days / 365.25) <= 2)
            ]
            ns_today = ns_full[ns_full["Date"] == ns_full["Date"].max()]
            if not similar.empty:
                similar = similar.merge(ns_today[["ISIN","Z_SPRD_VAL","YearsToMaturity"]],
                                        on="ISIN", how="left", suffixes=("","_NS"))

            fi = interp1d(ns_smooth["YearsToMaturity"], ns_smooth["Z_SPRD_VAL"],
                          kind="linear", fill_value="extrapolate")
            offsets = [row["Z_SPRD_VAL"] - fi(row["YearsToMaturity"])
                       for _, row in similar.iterrows() if pd.notnull(row.get("Z_SPRD_VAL"))]
            mean_off   = np.nanmean(offsets) if offsets else 0
            pred_z     = fi(new_ytm) + mean_off + concession

            all_mats = ns_smooth["YearsToMaturity"].values
            close_i  = [i for i in np.argsort(np.abs(all_mats - new_ytm)) if abs(all_mats[i] - new_ytm) <= 2]
            if close_i:
                dists    = np.abs(all_mats[close_i] - new_ytm)
                z_std    = np.average(ns_std.iloc[close_i]["Z_SPRD_VAL"], weights=1/(dists+1e-6))
            else:
                z_std    = ns_std["Z_SPRD_VAL"].mean()
            z_min, z_max = pred_z - 1.5*z_std, pred_z + 1.5*z_std

            m1, m2, m3 = st.columns(3)
            m1.metric("Predicted Z-spread", f"{pred_z:.1f} bps")
            m2.metric("Range low",  f"{z_min:.1f} bps")
            m3.metric("Range high", f"{z_max:.1f} bps")

            ns_today_plot = ns_today.copy()
            ns_today_plot = ns_today_plot.merge(final_signal_df[["ISIN","SIGNAL"]], on="ISIN", how="left")
            ns_today_plot["SIGNAL"]       = ns_today_plot["SIGNAL"].str.strip().str.lower()
            ns_today_plot["Signal_Color"] = ns_today_plot["SIGNAL"].map(SIGNAL_COLOR_MAP).fillna("#94a3b8")

            fig = go.Figure()
            for signal, grp in ns_today_plot.groupby("SIGNAL"):
                if grp.empty: continue
                ht = grp.get("SECURITY_NAME", grp["ISIN"]).fillna("Unknown")
                fig.add_trace(go.Scatter(
                    x=grp["YearsToMaturity"], y=grp["Z_SPRD_VAL"],
                    mode="markers", name=signal.title() if signal in LEGEND_SIGNALS else None,
                    marker=dict(size=6, color=grp["Signal_Color"].iloc[0]),
                    text=ht,
                    hovertemplate="YTM: %{x:.2f}y<br>Z: %{y:.1f} bps<br>%{text}<extra></extra>",
                    showlegend=(signal in LEGEND_SIGNALS),
                ))
            _add_ns_fit(fig, ns_today_plot, ytm_col="YearsToMaturity")
            fig.add_trace(go.Scatter(
                x=[new_ytm], y=[pred_z], mode="markers+text",
                marker=dict(size=14, color=PRED_C, symbol="star"),
                text=[f"{pred_z:.1f} bps"], textposition="top center",
                textfont=dict(color=PRED_C, size=10), name="Predicted",
            ))
            fig.add_trace(go.Scatter(
                x=[new_ytm-.06, new_ytm+.06, new_ytm+.06, new_ytm-.06],
                y=[z_min, z_min, z_max, z_max], fill="toself",
                fillcolor="rgba(124,58,237,0.1)", line=dict(color="rgba(0,0,0,0)"),
                showlegend=False,
            ))
            chart(fig, f"Predicted Z-spread · {mat_input}", h=620,
                  xt="Years to maturity", yt="Z-spread (bps)")
            st.plotly_chart(fig, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 2 — Signal Dashboard
# ══════════════════════════════════════════════════════════════════
with tab2:
    st.session_state.view = "signals"

    df = _sig_raw.copy()
    if df.empty:
        st.error("No signal data available.")
        st.stop()

    recent = pd.read_csv("recent_signals.csv")
    recent["Date"] = pd.to_datetime(recent["Date"])
    today_dt   = recent["Date"].max()
    yest_dt    = recent[recent["Date"] < today_dt]["Date"].max() if (recent["Date"] < today_dt).any() else today_dt
    today_df   = recent[recent["Date"] == today_dt]
    yest_df    = recent[recent["Date"] == yest_dt]

    counts_t   = today_df["SIGNAL"].value_counts().reindex(SIGNAL_TYPES, fill_value=0)
    counts_y   = yest_df["SIGNAL"].value_counts().reindex(SIGNAL_TYPES, fill_value=0)
    deltas     = counts_t - counts_y

    # filters from sidebar
    sel_ctry   = st.session_state.get("sb_sig_ctry", _ALL_SIG_CTRY) or _ALL_SIG_CTRY
    sel_sigs   = st.session_state.get("sb_sig_sigs", [s for s in SIGNAL_TYPES if s != "NO ACTION"])
    sel_sigs   = sel_sigs or [s for s in SIGNAL_TYPES if s != "NO ACTION"]

    filtered = df[df["Country"].isin(sel_ctry) & df["SIGNAL"].isin(sel_sigs)].copy()

    # inline search
    sh_col, rf_col = st.columns([6, 1])
    with sh_col:
        search = st.text_input("Search ISIN / name", placeholder="e.g. IT0001234…", key="sig_search")
    with rf_col:
        st.write(""); st.write("")
        if st.button("↺", key="sig_rf"): st.cache_data.clear(); st.rerun()

    if search:
        filtered = filtered[
            filtered["ISIN"].str.contains(search.upper(), na=False) |
            filtered["SECURITY_NAME"].str.contains(search.upper(), na=False)
        ]

    filtered_counts = filtered["SIGNAL"].value_counts().reindex(SIGNAL_TYPES, fill_value=0)

    # ── Signal cards ──
    sh(f"Signal snapshot · {today_dt.strftime('%d %b %Y')}  ·  Δ vs {yest_dt.strftime('%d %b %Y')}")
    html = '<div class="sig-row">'
    for sig in SIGNAL_TYPES:
        cnt = filtered_counts[sig]; dlt = deltas[sig]
        dc  = "du" if dlt>0 else "dd" if dlt<0 else "df"
        ds  = f"▲ +{dlt}" if dlt>0 else f"▼ {dlt}" if dlt<0 else f"● {dlt}"
        html += f"""<div class="sig-box {SIG_CSS[sig]}">
  <div class="sig-count">{cnt}</div>
  <div class="sig-label">{sig}</div>
  <div class="sig-delta {dc}">{ds}</div>
</div>"""
    html += "</div>"
    st.markdown(html, unsafe_allow_html=True)

    # ── Table ──
    sh(f"Bond universe · {len(filtered)} bonds")
    dl_col, _, __ = st.columns([1,2,6])
    with dl_col:
        st.download_button("⬇ CSV", filtered.to_csv(index=False),
                           f"bonds_{datetime.now().strftime('%Y%m%d_%H%M')}.csv","text/csv")

    if not filtered.empty:
        DISP_COLS = ["SECURITY_NAME","RESIDUAL_NS","SIGNAL",
                     "Z_Residual_Score","Volatility_Score","Market_Stress_Score",
                     "Cluster_Score","Regression_Score","COMPOSITE_SCORE",
                     "Top_Features","Top_Feature_Effects_Pct"]
        exist = [c for c in DISP_COLS if c in filtered.columns]
        disp  = filtered[exist].copy()
        disp.rename(columns={"RESIDUAL_NS":"Residual","Volatility_Score":"Stability","SIGNAL":"Signal"}, inplace=True)
        NUM = ["Residual","Z_Residual_Score","Stability","Market_Stress_Score",
               "Cluster_Score","Regression_Score","COMPOSITE_SCORE"]
        for c in NUM:
            if c in disp.columns: disp[c] = pd.to_numeric(disp[c], errors="coerce")
        if "Stability" in disp.columns: disp["Stability"] *= 100

        def _mat(name):
            if isinstance(name, str):
                m = re.search(r"(\d{2}/\d{2}/\d{2,4})$", name)
                if m:
                    for fmt in ("%m/%d/%y","%m/%d/%Y"):
                        try: return datetime.strptime(m.group(1), fmt)
                        except ValueError: pass
            return pd.NaT
        disp["Maturity"] = disp["SECURITY_NAME"].apply(_mat)
        disp["Maturity"] = disp["Maturity"].dt.strftime("%Y-%m-%d").fillna("N/A")
        disp = disp[["SECURITY_NAME","Maturity"] + [c for c in disp.columns if c not in ["SECURITY_NAME","Maturity"]]]

        FEAT_MAP = {"Cpn":"Coupon","YAS_RISK":"DV01","AMT_OUTSTANDING":"Amt Outst",
                    "Issue_Age":"Issue Age","REL_SPRD_STD":"Liquidity","GREEN_BOND_LOAN_INDICATOR":"Green"}
        if "Top_Features" in disp.columns and "Top_Feature_Effects_Pct" in disp.columns:
            def _feats(f, p):
                try:
                    fl = [FEAT_MAP.get(x,x) for x in (ast.literal_eval(f) if isinstance(f,str) else [])]
                    pl = [int(round(float(v))) for v in p.replace("[","").replace("]","").split()] if isinstance(p,str) else []
                    return ", ".join(f"{a} ({b}%)" for a,b in zip(fl,pl)) or "N/A"
                except: return "N/A"
            disp["Top_Features"] = disp.apply(lambda r: _feats(r["Top_Features"],r["Top_Feature_Effects_Pct"]),axis=1)
            disp.drop(columns=["Top_Feature_Effects_Pct"], inplace=True)

        # upgrade/downgrade decoration
        ys_map = yest_df.set_index("SECURITY_NAME")["SIGNAL"].to_dict()
        LVL    = {"NO ACTION":0,"WEAK SELL":1,"WEAK BUY":1,"MODERATE SELL":2,"MODERATE BUY":2,"STRONG SELL":3,"STRONG BUY":3}
        EMO    = {"STRONG BUY":"🟩","MODERATE BUY":"💚","STRONG SELL":"🟥","MODERATE SELL":"💛"}
        def _deco(row):
            n=row["SECURITY_NAME"]; tl=LVL.get(row["Signal"],0)
            yl=LVL.get(ys_map.get(n),0)
            if (tl>=2 or yl>=2) and tl!=yl:
                return f"{EMO.get(row['Signal'],'')} {'↑' if tl>yl else '↓'} {n}"
            return n
        disp["SECURITY_NAME"] = disp.apply(_deco, axis=1)

        HELP = {"Residual":"bps off curve","Z_Residual_Score":"|Z|>1.5 = opportunity",
                "Stability":"Higher = more stable","Market_Stress_Score":"Higher = more exposed",
                "Cluster_Score":"|val|>1.5 likely to mean-revert",
                "Regression_Score":"|val|>1.5 = strong signal",
                "COMPOSITE_SCORE":"|val|>1.5 = stronger signal","Top_Features":"Key drivers"}
        cfg = {}
        for c in disp.columns:
            lbl = c.replace("_"," ")
            if c in NUM and pd.api.types.is_numeric_dtype(disp.get(c, pd.Series())):
                fmt = "%.2f%%" if c=="Stability" else "%.4f"
                cfg[c] = st.column_config.NumberColumn(lbl, format=fmt, help=HELP.get(c))
            else:
                cfg[c] = st.column_config.TextColumn(lbl, help=HELP.get(c))
        st.dataframe(disp, column_config=cfg, use_container_width=True)

# ══════════════════════════════════════════════════════════════════
# TAB 3 — Analysis
# ══════════════════════════════════════════════════════════════════
with tab3:
    an1, an2 = st.tabs(["Multi-curve comparison","Top trades"])

    with an1:
        st.session_state.view = "analysis"
        metric     = st.session_state.get("sb_an_metric","Z-Spread")
        met_col    = "Z_SPRD_VAL" if metric=="Z-Spread" else "RESIDUAL_NS"
        c1_country = COUNTRY_CODE_MAP[st.session_state.get("sb_an_c1", COUNTRY_OPTIONS[0])]
        c2_country = COUNTRY_CODE_MAP[st.session_state.get("sb_an_c2", COUNTRY_OPTIONS[0])]

        if "curves" not in st.session_state or len(st.session_state.curves) != 2:
            st.session_state.curves = [
                {"id":"curve1","country":st.session_state.get("sb_an_c1",COUNTRY_OPTIONS[0]),"bond1":None,"bond2":None},
                {"id":"curve2","country":st.session_state.get("sb_an_c2",COUNTRY_OPTIONS[0]),"bond1":None,"bond2":None},
            ]
        st.session_state.curves[0]["country"] = st.session_state.get("sb_an_c1", COUNTRY_OPTIONS[0])
        st.session_state.curves[1]["country"] = st.session_state.get("sb_an_c2", COUNTRY_OPTIONS[0])

        issuer_sig  = load_issuer_signal()
        curve_dfs   = []
        gl_labels   = {}

        for i, curve in enumerate(st.session_state.curves):
            sh(f"Curve {i+1} · {curve['country']}")
            ns_df = load_full_ns_df(COUNTRY_CODE_MAP[curve["country"]], zip_hash=zip_hash)
            if ns_df is None or ns_df.empty: continue
            ns_df["Date"] = pd.to_datetime(ns_df["Date"]).dt.normalize()

            bo = ns_df[["ISIN","SECURITY_NAME","Maturity"]].drop_duplicates()
            bo = bo.merge(issuer_sig[["ISIN","SIGNAL"]], on="ISIN", how="left") if not issuer_sig.empty else bo
            bo["Maturity"] = pd.to_datetime(bo["Maturity"], errors="coerce")
            bo.sort_values("Maturity", inplace=True)

            blbl = {}
            for _, r in bo.iterrows():
                mat = pd.to_datetime(r["Maturity"]).strftime("%Y-%m-%d") if pd.notnull(r["Maturity"]) else "N/A"
                sig = r.get("SIGNAL","") if pd.notnull(r.get("SIGNAL","")) else "–"
                blbl[r["ISIN"]] = f"{r['SECURITY_NAME']}  ({mat})  [{sig}]"
                gl_labels[r["ISIN"]] = f"{r['SECURITY_NAME']} ({mat})"

            b1_col, b2_col = st.columns(2)
            with b1_col:
                curve["bond1"] = st.selectbox(f"Bond 1", bo["ISIN"], format_func=lambda i:blbl.get(i,i), key=f"an_b1_{curve['id']}")
            with b2_col:
                curve["bond2"] = st.selectbox(f"Bond 2", bo["ISIN"], format_func=lambda i:blbl.get(i,i), key=f"an_b2_{curve['id']}")

            if curve["bond1"] and curve["bond2"]:
                d1 = ns_df[ns_df["ISIN"]==curve["bond1"]][["Date",met_col]].rename(columns={met_col:"B1"})
                d2 = ns_df[ns_df["ISIN"]==curve["bond2"]][["Date",met_col]].rename(columns={met_col:"B2"})
                dc = d1.merge(d2, on="Date", how="outer").sort_values("Date")
                m1 = pd.to_datetime(ns_df.loc[ns_df["ISIN"]==curve["bond1"],"Maturity"].iloc[0])
                m2 = pd.to_datetime(ns_df.loc[ns_df["ISIN"]==curve["bond2"],"Maturity"].iloc[0])
                dc["Curve"] = (dc["B1"]-dc["B2"]) if m1<=m2 else (dc["B2"]-dc["B1"])
                dc["Bond1"] = curve["bond1"]; dc["Bond2"] = curve["bond2"]
                curve_dfs.append(dc)

        if len(curve_dfs) == 2:
            diff = curve_dfs[1][["Date","Curve"]].merge(curve_dfs[0][["Date","Curve"]], on="Date", suffixes=("_2","_1"))
            diff["Curve"] = diff["Curve_2"] - diff["Curve_1"]
            fig = go.Figure()
            for i, cdf in enumerate(curve_dfs):
                c = st.session_state.curves[i]
                lbl = f"{gl_labels.get(c['bond2'],c['bond2'])} − {gl_labels.get(c['bond1'],c['bond1'])}"
                fig.add_trace(go.Scatter(x=cdf["Date"],y=cdf["Curve"],mode="lines",name=lbl))
            fig.add_trace(go.Scatter(x=diff["Date"],y=diff["Curve"],mode="lines",
                                     name="Differenced (C2−C1)",
                                     line=dict(color=AMBER,width=2,dash="dot")))
            chart(fig, f"{metric} two-curve comparison", h=600,
                  xt="Date", yt=f"{metric} diff (bps)")
            st.plotly_chart(fig, use_container_width=True)

    with an2:
        st.session_state.view = "trades"
        TCOLS = ["A_ISIN","B_ISIN","C_ISIN","D_ISIN","LEG_1","LEG_2",
                 "Trade_ZDiff_30D_Pct","Diff_of_Diffs_Today","Ranking_Score","Actionable_Direction"]
        exist = [c for c in TCOLS if c in top_trades_agent.columns]
        sh("Top 50 trades")
        st.dataframe(top_trades_agent.head(50)[exist], use_container_width=True)
        try:
            sh("Z-diff 30D heatmap")
            z = (alt.Chart(top_trades_agent.reset_index().head(50)).mark_rect()
                 .encode(x="Ranking_Score:O", y="index:O",
                         color=alt.Color("Trade_ZDiff_30D_Pct", scale=alt.Scale(scheme="redblue")))
                 .properties(height=480))
            st.altair_chart(z, use_container_width=True)
        except Exception as e:
            st.warning(f"Heatmap unavailable: {e}")

# ══════════════════════════════════════════════════════════════════
# TAB 4 — AI Assistant
# ══════════════════════════════════════════════════════════════════
with tab4:
    st.session_state.view = "chat"

    if "chat_history" not in st.session_state:
        TCOLS3 = ["A_ISIN","B_ISIN","C_ISIN","D_ISIN","LEG_1","LEG_2",
                  "Trade_ZDiff_30D_Pct","Diff_of_Diffs_Today","Ranking_Score","Actionable_Direction"]
        ex3 = [c for c in TCOLS3 if c in top_trades_agent.columns]
        top3 = top_trades_agent.nlargest(3,"Ranking_Score")[ex3].to_dict(orient="records")
        sys_p = get_system_prompt(top_trades_agent) + f"\n\nTop 3 trades:\n{top3}"
        st.session_state.chat_history = [{"role":"system","content":sys_p}]

    for msg in st.session_state.chat_history[1:]:
        with st.chat_message("user" if msg["role"]=="user" else "assistant"):
            st.markdown(msg["content"])

    user_input = st.chat_input("Ask about trades, bonds, or signals…")
    if user_input:
        st.session_state.chat_history.append({"role":"user","content":user_input})
        with st.chat_message("user"): st.markdown(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking…"):
                answer, *_ = chat_with_trades(user_input, st.session_state.chat_history)
            st.markdown(answer)
        st.session_state.chat_history.append({"role":"assistant","content":answer})
