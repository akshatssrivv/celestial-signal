import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
cecwed
import zipfile
import shutil
import hashlib
import os
from nelson_siegel_fn import plot_ns_animation, nelson_siegel
from ai_explainer_utils import format_bond_diagnostics, generate_ai_explanation
import ast
import re
from datetime import datetime
import requests
import boto3
from scipy.interpolate import interp1d
import uuid
from curve_trade_agent1 import chat_with_trades, get_system_prompt
from streamlit_chat import message

# -------------------
# B2 Configuration
# -------------------
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
BUCKET_NAME = "Celestial-Signal"
LOCAL_ZIP = "ns_curves_20251509.zip"
LOCAL_FOLDER = "ns_curves"


# -------------------
# Download from B2
# -------------------
def download_from_b2(file_key: str, local_path: str, force: bool = False):
    """Download a file from B2 bucket."""
    if not force and os.path.exists(local_path):
        return local_path

    with st.spinner(f"Downloading {file_key} from B2..."):
        s3 = boto3.client(
            "s3",
            endpoint_url="https://s3.eu-central-003.backblazeb2.com", 
            aws_access_key_id=B2_KEY_ID,
            aws_secret_access_key=B2_APP_KEY
        )
        s3.download_file(BUCKET_NAME, file_key, local_path)

    return local_path


# -------------------
# Compute MD5 hash
# -------------------
def file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file"""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# -------------------
# Unzip NS curves
# -------------------
def unzip_ns_curves(zip_path: str = LOCAL_ZIP, folder: str = LOCAL_FOLDER, force: bool = False) -> tuple[str, str]:
    """Unzip NS curves from B2 and return (folder, zip_hash)."""
    # Download latest zip from B2
    zip_path = download_from_b2(file_key="ns_curves_1509.zip", local_path=zip_path, force=force)
    zip_hash = file_hash(zip_path)
    prev_hash = st.session_state.get("ns_zip_hash")

    # Only unzip if new or forced
    if force or prev_hash != zip_hash or not os.path.exists(folder):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(folder)
        st.session_state["ns_zip_hash"] = zip_hash

    return folder, zip_hash

# -------------------
# Load full NS DF
# -------------------
@st.cache_data
def load_full_ns_df(country_code: str, zip_hash: str) -> pd.DataFrame:
    """Load all NS curves for a country. Cache invalidates if ZIP changes."""
    folder, _ = unzip_ns_curves(force=True)

    all_files = sorted([
        f for f in os.listdir(folder)
        if f.startswith(country_code) and f.endswith(".parquet")
    ])

    dfs = []
    for f in all_files:
        try:
            df = pd.read_parquet(os.path.join(folder, f))
            dfs.append(df)
        except Exception as e:
            st.warning(f"Error loading file {f}: {e}")
            continue

    if dfs:
        ns_df = pd.concat(dfs, ignore_index=True)

        # Ensure residuals column exists
        if 'RESIDUAL' in ns_df.columns:
            ns_df.rename(columns={'RESIDUAL': 'RESIDUAL_NS'}, inplace=True)
        if 'RESIDUAL_NS' not in ns_df.columns:
            ns_df['RESIDUAL_NS'] = pd.NA

        if "Date" in ns_df.columns:
            ns_df["Date"] = pd.to_datetime(ns_df["Date"])
            ns_df.sort_values("Date", inplace=True)

        if "Country" in ns_df.columns:
            print("Unique country codes in NS DF:", ns_df['Country'].unique())

        return ns_df

    st.warning(f"No parquet files found for country code '{country_code}' in folder '{folder}'.")
    return pd.DataFrame()

# -------------------
# Load NS curve for a specific date
# -------------------
def load_ns_curve(country_code: str, date_str: str, zip_hash: str) -> pd.DataFrame | None:
    """
    Load NS curve for a single day from the full dataset.
    Passing zip_hash ensures cache invalidation when ns_curves.zip changes.
    """
    df = load_full_ns_df(country_code, zip_hash=zip_hash)

    if df is not None and not df.empty:
        df = df[df["Date"] == pd.to_datetime(date_str)]
        if not df.empty:
            return df

    return None


# Make page use full width
st.set_page_config(layout="wide")

# Inject CSS to remove container constraints for the tabs
st.markdown("""
    <style>
    /* Target the main content block */
    main .block-container {
        padding-left: 0rem;
        padding-right: 0rem;
        max-width: 100% !important;
    }
    
    /* Make tabs container full width */
    div[role="tablist"] {
        width: 100% !important;
    }
    </style>
    """, unsafe_allow_html=True)


tab1, tab2, tab3, tab4 = st.tabs(["Nelson-Siegel Curves", "Signal Dashboard", "Analysis", "AI Assisstant"])


with tab2:
            
    st.markdown("""
    <style>
        .metric-box {
            background: white;
            padding: 1.5rem;
            border-radius: 8px;
            border: 1px solid #e1e5e9;
            text-align: center;
        }
        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 0.5rem;
        }
        .metric-label {
            font-size: 0.9rem;
            color: #666;
            font-weight: 500;
        }
        .buy { color: #28a745; }          /* Strong Buy */
        .sell { color: #dc3545; }         /* Strong Sell */
        .mod-buy { color: #20c997; }      /* Moderate Buy */
        .mod-sell { color: #e55353; }     /* Moderate Sell */
        .watch-buy { color: #17a2b8; }    /* Weak Buy */
        .watch-sell { color: #fd7e14; }   /* Weak Sell */
        .no-action { color: #6c757d; }    /* No Action */
    </style>
    """, unsafe_allow_html=True)


    @st.cache_data(ttl=300)
    def load_data(force: bool = False) -> pd.DataFrame:
        """Load issuer_signals.csv from B2 with caching."""
        local_path = "issuer_signals.csv"
        file_key = "issuer_signals.csv"
    
        try:
            local_path = download_from_b2(file_key=file_key, local_path=local_path, force=force)
            df = pd.read_csv(local_path)
            return df
        except Exception as e:
            st.error(f"Error loading data from B2: {e}")
            if os.path.exists(local_path):
                try:
                    df = pd.read_csv(local_path)
                    return df
                except Exception as e2:
                    st.error(f"Error loading local CSV: {e2}")
            return pd.DataFrame()


    def get_country_from_isin(isin):
        """Extract country from ISIN code"""
        country_map = {
            'IT': '🇮🇹 Italy',
            'ES': '🇪🇸 Spain',
            'FR': '🇫🇷 France',
            'DE': '🇩🇪 Germany',
            'FI': '🇫🇮 Finland',
            'EU': '🇪🇺 EU',
            'AT': '🇦🇹 Austria',
            'NL': '🇳🇱 Netherlands',
            'BE': '🇧🇪 Belgium'
        }
        
        return country_map.get(isin[:2], '🌍 Unknown')

    # Load data
    df = load_data()

    if df.empty:
        st.error("No data available")
        st.stop()

    # Add country column
    df['Country'] = df['ISIN'].apply(get_country_from_isin)

    # Title
    st.title("Bond Analytics Dashboard")

    # Get actual signal values from your data (case-sensitive and exact match)
    actual_signals = df['SIGNAL'].unique()
    
    # Assuming you have recent_signals with today and yesterday
    today = pd.Timestamp.today().normalize()
    yesterday = today - pd.Timedelta(days=1)

    recent_signals = pd.read_csv("recent_signals.csv")
    recent_signals['Date'] = pd.to_datetime(recent_signals['Date'])
    
    today_df = recent_signals[recent_signals['Date'] == today]
    yesterday_df = recent_signals[recent_signals['Date'] == yesterday]
    
    signal_types = ['STRONG BUY','STRONG SELL','MODERATE BUY','MODERATE SELL','WEAK BUY','WEAK SELL','NO ACTION']
    
    signal_counts_today = today_df['SIGNAL'].value_counts().reindex(signal_types, fill_value=0)
    signal_counts_yesterday = yesterday_df['SIGNAL'].value_counts().reindex(signal_types, fill_value=0)
    
    signal_deltas = signal_counts_today - signal_counts_yesterday
    
    def format_delta(val):
        if val > 0:
            return f'<span style="color:#28a745">+{val}</span>'
        elif val < 0:
            return f'<span style="color:#dc3545">{val}</span>'
        else:
            return f'<span style="color:#6c757d">{val}</span>'

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    boxes = [
        'STRONG BUY', 'STRONG SELL', 'MODERATE BUY', 
        'MODERATE SELL', 'WEAK BUY', 'WEAK SELL', 'NO ACTION'
    ]
    
    for i, sig in enumerate(boxes):
        count = signal_counts_today[sig]
        delta_html = format_delta(signal_deltas[sig])
        with [col1, col2, col3, col4, col5, col6, col7][i]:
            st.markdown(f"""
            <div class="metric-box">
                <div class="metric-value">{count} <small>{delta_html}</small></div>
                <div class="metric-label">{sig}</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("---")

    # Horizontal filters
    col1, col2, col3 = st.columns([2, 2, 3])

    with col1:
        selected_countries = st.multiselect(
            "Countries",
            options=list(df['Country'].unique()),
            default=list(df['Country'].unique())
        )

    with col2:
        # Get actual signal options from your data
        fixed_signal_options = [
            'STRONG BUY', 'STRONG SELL',
            'MODERATE BUY', 'MODERATE SELL',
            'WEAK BUY', 'WEAK SELL',
            'NO ACTION'
        ]
        default_signals = [sig for sig in fixed_signal_options if sig != 'NO ACTION']
        
        selected_signals = st.multiselect(
            "Signals",
            options=fixed_signal_options,
            default=default_signals
        )

    with col3:
        search_term = st.text_input("Search ISIN or Security Name", placeholder="Type to search...")        

    # Apply filters
    filtered_df = df.copy()

    if selected_countries:
        filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]

    if selected_signals:
        filtered_df = filtered_df[filtered_df['SIGNAL'].isin(selected_signals)]

    if search_term:
        mask = (
            filtered_df['ISIN'].str.contains(search_term.upper(), na=False) |
            filtered_df['SECURITY_NAME'].str.contains(search_term.upper(), na=False)
        )
        filtered_df = filtered_df[mask]

    st.markdown("---")

    st.subheader(f"Bond Data ({len(filtered_df)} bonds)")

    if not filtered_df.empty:
        # Columns to display
        cols_to_display = [
            'SECURITY_NAME', 'RESIDUAL_NS', 'SIGNAL',
            'Z_Residual_Score', 'Volatility_Score', 'Market_Stress_Score',
            'Cluster_Score', 'Regression_Score', 'COMPOSITE_SCORE',
            'Top_Features', 'Top_Feature_Effects_Pct'
        ]
    
        # Keep only existing columns
        existing_cols = [col for col in cols_to_display if col in filtered_df.columns]
        display_df = filtered_df[existing_cols].copy()
    
        # Rename columns
        display_df.rename(columns={
            'RESIDUAL_NS': 'Residual',
            'Volatility_Score': 'Stability_Score',
            'SIGNAL': 'Signal'
        }, inplace=True)
    
        # Convert numeric columns
        numeric_cols = ['Residual', 'Z_Residual_Score', 'Stability_Score',
                        'Market_Stress_Score', 'Cluster_Score', 'Regression_Score', 'COMPOSITE_SCORE']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
    
        if "Stability_Score" in display_df.columns:
            display_df["Stability_Score"] = display_df["Stability_Score"] * 100
    
        # Extract maturity as datetime for sorting
        def extract_maturity_dt(name):
            if isinstance(name, str):
                match = re.search(r'(\d{2}/\d{2}/\d{2,4})$', name)
                if match:
                    date_str = match.group(1)
                    for fmt in ("%m/%d/%y", "%m/%d/%Y"):
                        try:
                            return datetime.strptime(date_str, fmt)
                        except ValueError:
                            continue
            return pd.NaT
    
        display_df['Maturity'] = display_df['SECURITY_NAME'].apply(extract_maturity_dt)
        display_df['Maturity'] = display_df['Maturity'].dt.strftime("%Y-%m-%d")
        display_df['Maturity'] = display_df['Maturity'].fillna("N/A")
    
        # Reorder columns
        cols_order = ['SECURITY_NAME', 'Maturity'] + [c for c in display_df.columns if c not in ['SECURITY_NAME', 'Maturity']]
        display_df = display_df[cols_order]
    
        # Combine Top_Features + Top_Feature_Effects_Pct
        FEATURE_NAME_MAP = {
            "Cpn": "Coupon",
            "YAS_RISK": "DV01",
            "AMT_OUTSTANDING": "Amount Outstanding",
            "Issue_Age": "Issue Age",
            "REL_SPRD_STD": "Liquidity",
            "GREEN_BOND_LOAN_INDICATOR": "Green Bond"
        }
    
        if 'Top_Features' in display_df.columns and 'Top_Feature_Effects_Pct' in display_df.columns:
            def combine_features(feats, pct):
                try:
                    feats_list = ast.literal_eval(feats) if isinstance(feats, str) else []
                    feats_list = [FEATURE_NAME_MAP.get(f, f) for f in feats_list]
                    pct_list = [int(round(float(v))) for v in pct.replace('[','').replace(']','').split()] if isinstance(pct, str) else []
                    combined = [f"{f} ({p}%)" for f, p in zip(feats_list, pct_list)]
                    return ', '.join(combined) if combined else 'N/A'
                except:
                    return 'N/A'
    
            display_df['Top_Features'] = display_df.apply(
                lambda row: combine_features(row['Top_Features'], row['Top_Feature_Effects_Pct']),
                axis=1
            )
            display_df.drop(columns=['Top_Feature_Effects_Pct'], inplace=True)
    
        # --- DEBUG: Check yesterday_df ---
        print(f"Debug: yesterday_df shape: {yesterday_df.shape}")
        print(f"Debug: yesterday_df empty: {yesterday_df.empty}")
        if not yesterday_df.empty:
            print(f"Debug: yesterday_df dates: {yesterday_df['Date'].unique() if 'Date' in yesterday_df.columns else 'No Date column'}")
            print(f"Debug: yesterday_df securities count: {len(yesterday_df['SECURITY_NAME'].unique()) if 'SECURITY_NAME' in yesterday_df.columns else 'No SECURITY_NAME column'}")
    
        # --- UPDATED: Handle empty yesterday_df gracefully ---
        if not yesterday_df.empty and 'SECURITY_NAME' in yesterday_df.columns and 'SIGNAL' in yesterday_df.columns:
            yesterday_signals = yesterday_df.set_index('SECURITY_NAME')['SIGNAL'].to_dict()
            print(f"Debug: yesterday_signals count: {len(yesterday_signals)}")
        else:
            yesterday_signals = {}
            print("Debug: No yesterday signals available - using empty dict")
    
        def decorate_name(row):
            name = row['SECURITY_NAME']
            today_signal = row['Signal']              # raw signal
            yesterday_signal = yesterday_signals.get(name, None)
    
            # DEBUG: Print some examples
            if len(yesterday_signals) > 0 and name in list(yesterday_signals.keys())[:3]:  # First 3 securities
                print(f"Debug: {name} - Today: {today_signal}, Yesterday: {yesterday_signal}")
    
            levels = {
                'NO ACTION': 0,
                'WEAK SELL': 1, 'WEAK BUY': 1,
                'MODERATE SELL': 2, 'MODERATE BUY': 2,
                'STRONG SELL': 3, 'STRONG BUY': 3
            }
    
            today_lvl = levels.get(today_signal, 0)
            yesterday_lvl = levels.get(yesterday_signal, 0) if yesterday_signal else 0
    
            # Only decorate if moving into/out of MODERATE or STRONG
            if (today_lvl >= 2 or yesterday_lvl >= 2) and today_lvl != yesterday_lvl:
                emoji_map = {
                    'STRONG BUY': '🟩',
                    'MODERATE BUY': '💚',
                    'STRONG SELL': '🟥',
                    'MODERATE SELL': '💛'
                }
                emoji = emoji_map.get(today_signal, '')
                if today_lvl > yesterday_lvl:
                    print(f"Debug: UPGRADE - {name}: {yesterday_signal} -> {today_signal}")
                    return f'{emoji} ↑ {name}'  # upgrade
                else:
                    print(f"Debug: DOWNGRADE - {name}: {yesterday_signal} -> {today_signal}")
                    return f'{emoji} ↓ {name}'  # downgrade
            else:
                return name  # unchanged for weak/no action or same level
    
        display_df['SECURITY_NAME'] = display_df.apply(decorate_name, axis=1)
    
        # Column config for tooltips + formatting
        HELP_TEXTS = {
            "Residual": "Residual mispricing (bps off curve)",
            "Z_Residual_Score": "Z-score of residual. |Z| > 1.5 may indicate opportunities.",
            "Stability_Score": "Inverse volatility. Higher = more stable pricing. Lower = riskier.",
            "Market_Stress_Score": "Market stress factor. High = bond more exposed to stress.",
            "Cluster_Score": "Deviation from peer cluster (bps). Absolute > 1.5, likely to mean-revert",
            "Regression_Score": "Model-explained mispricing. Absolute > 1.5, strong signal; likely to mean-revert.",
            "COMPOSITE_SCORE": "Overall mispricing score. Absolute > 1.5 = stronger trade signal.",
            "Top_Features": "Most important drivers of mispricing. % shows relative impact.",
            "Signal": "Trade signal (raw). Arrows/emojis are only shown next to SECURITY_NAME."
        }
    
        column_config = {}
        for col in display_df.columns:
            label = col.replace('_', ' ')
            if col in numeric_cols and pd.api.types.is_numeric_dtype(display_df[col]):
                if col == "Stability_Score":
                    column_config[col] = st.column_config.NumberColumn(
                        label, format="%.2f%%", help=HELP_TEXTS.get(col)
                    )
                else:
                    column_config[col] = st.column_config.NumberColumn(
                        label, format="%.4f", help=HELP_TEXTS.get(col)
                    )
            else:
                column_config[col] = st.column_config.TextColumn(label, help=HELP_TEXTS.get(col))
    
        # Show the table
        st.dataframe(display_df, column_config=column_config)
    
        # Download button
        col1, col2, col3 = st.columns([1, 1, 4])
    
        with col1:
            csv = filtered_df.to_csv(index=False)
            st.download_button(
                "Download CSV",
                csv,
                f"bonds_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv"
            )
    
        with col2:
            if st.button("Refresh Data"):
                st.cache_data.clear()
                st.rerun()

with tab1:
    st.set_page_config(
        page_title="The Curves",
        layout="wide"
    )
    
    # Choose subtab inside Tab 1
    subtab = st.radio(
        "Select View",
        ("Single Day Curve", "Animated Curves", "Residuals Analysis", "Compare NS Curves", "New Bond Prediction")
    )

    B2_BUCKET_FILE = "ns_curves_1509.zip"
    try:
        zip_path = download_from_b2(file_key=B2_BUCKET_FILE, local_path=LOCAL_ZIP, force=False)
        if not os.path.exists(zip_path):
            raise FileNotFoundError(f"Downloaded file not found: {zip_path}")
        zip_hash = file_hash(zip_path)
    except Exception as e:
        st.error(f"Failed to download or hash NS curves zip: {e}")
        zip_path = None
        zip_hash = None
    

    if subtab == "Single Day Curve":

        country_option = st.selectbox(
            "Select Country",
            options=['Italy 🇮🇹', 'Spain 🇪🇸', 'France 🇫🇷', 'Germany 🇩🇪', 'Finland 🇫🇮', 'EU 🇪🇺', 'Austria 🇦🇹', 'Netherlands 🇳🇱', 'Belgium 🇧🇪']
        )

        country_code_map = {
            'Italy 🇮🇹': 'BTPS',
            'Spain 🇪🇸': 'SPGB',
            'France 🇫🇷': 'FRTR',
            'Germany 🇩🇪': 'BUNDS',
            'Finland 🇫🇮': 'RFGB',
            'EU 🇪🇺': 'EU',
            'Austria 🇦🇹': 'RAGB',
            'Netherlands 🇳🇱': 'NETHER',
            'Belgium 🇧🇪': 'BGB'
        }
        
        selected_country = country_code_map[country_option]
        
        final_signal_df = pd.read_csv("today_all_signals.csv")
        available_dates = pd.to_datetime(final_signal_df['Date'].unique())
        default_date = available_dates.max()  # most recent date
        
        date_input = st.date_input("Select Date", value=default_date)
        date_str = date_input.strftime("%Y-%m-%d")
        
        ns_df = load_ns_curve(selected_country, date_str, zip_hash=zip_hash)

        ns_df = load_ns_curve(selected_country, date_str, zip_hash=zip_hash)

        if ns_df is not None and not ns_df.empty:
            # --- Normalize column names for consistency ---
            col_map = {c.lower(): c for c in ns_df.columns}
            if "z_sprd_val" in col_map:
                ns_df.rename(columns={col_map["z_sprd_val"]: "Z_SPRD"}, inplace=True)
            elif "z_sprd" in col_map:
                ns_df.rename(columns={col_map["z_sprd"]: "Z_SPRD"}, inplace=True)
        
            if "yearstomaturity" in col_map:
                ns_df.rename(columns={col_map["yearstomaturity"]: "YTM"}, inplace=True)

        
        if ns_df is not None and not ns_df.empty:
            ns_df['Maturity'] = pd.to_datetime(ns_df['Maturity'])
            curve_date = pd.to_datetime(date_input)
            ns_df['YTM'] = (ns_df['Maturity'] - curve_date).dt.days / 365.25
        
            # Load signals
            final_signal_df = pd.read_csv("today_all_signals.csv")
            ns_df = ns_df.merge(
                final_signal_df[['ISIN', 'SIGNAL']],
                on='ISIN',
                how='left'
            )
        
            # Normalize SIGNAL column
            ns_df['SIGNAL'] = ns_df['SIGNAL'].str.strip().str.lower()
        
            # Map signals to colors
            signal_color_map = {
                'strong buy': 'green',
                'moderate buy': 'lightgreen',
                'weak buy': 'black',
                'strong sell': 'red',
                'moderate sell': 'orange',
                'weak sell': 'black'
            }
            ns_df['Signal_Color'] = ns_df['SIGNAL'].map(signal_color_map).fillna('black')
        
            fig = go.Figure()
            legend_signals = ['strong buy', 'moderate buy', 'strong sell', 'moderate sell']
        
            for signal, df_subset in ns_df.groupby('SIGNAL'):
                if not df_subset.empty:
                    color = df_subset['Signal_Color'].iloc[0]
                    fig.add_trace(go.Scatter(
                        x=df_subset['YTM'],
                        y=df_subset['Z_SPRD'],
                        mode='markers',
                        name=signal.title() if signal in legend_signals else None,
                        marker=dict(size=6, color=color, symbol='circle'),
                        text=df_subset['SECURITY_NAME'],
                        customdata=np.stack((
                            df_subset['ISIN'],
                            df_subset['Date'].astype(str),
                            df_subset.get('RESIDUAL_NS', np.zeros(len(df_subset)))
                        ), axis=-1),
                        hovertemplate=(
                            'Years to Maturity: %{x:.2f}<br>'
                            'Z-Spread: %{y:.1f}bps<br>'
                            'Residual: %{customdata[2]:.2f}bps<br>'
                            'Signal: ' + (signal.title() if signal else "None") + '<br>'
                            '%{text}<extra></extra>'
                        ),
                        showlegend=(signal in legend_signals)
                    ))
        
            # Nelson-Siegel fit
            if 'NS_PARAMS' in ns_df.columns:
                try:
                    ns_params_raw = ns_df['NS_PARAMS'].iloc[0]
                    if isinstance(ns_params_raw, str):
                        import ast
                        ns_params = ast.literal_eval(ns_params_raw)
                    else:
                        ns_params = ns_params_raw
        
                    maturity_range = np.linspace(ns_df['YTM'].min(), ns_df['YTM'].max(), 100)
                    ns_curve = nelson_siegel(maturity_range, *ns_params)
        
                    fig.add_trace(go.Scatter(
                        x=maturity_range,
                        y=ns_curve,
                        mode='lines',
                        name='Nelson-Siegel Fit',
                        line=dict(color='deepskyblue', width=3)
                    ))
                except Exception as e:
                    st.error(f"Error plotting Nelson-Siegel curve: {e}")
        
            fig.update_layout(
                title=f"Nelson-Siegel Curve for {selected_country} on {date_str}",
                xaxis_title="Years to Maturity",
                yaxis_title="Z-Spread (bps)",
                height=700,
                showlegend=True,
                template="plotly_white"
            )
        
            col1, col2 = st.columns([3, 2])
            with col1:
                st.plotly_chart(fig, use_container_width=True)
        
            # AI explanation panel
            with col2:
                bond_options = final_signal_df[['ISIN', 'SECURITY_NAME']].drop_duplicates().sort_values('SECURITY_NAME')
                bond_labels = {row["ISIN"]: row["SECURITY_NAME"] for _, row in bond_options.iterrows()}
        
                selected_isin = st.selectbox(
                    "Select Bond for AI Explanation",
                    options=bond_options['ISIN'].tolist(),
                    format_func=lambda isin: bond_labels.get(isin, isin),
                    key="bond_selector"
                )
        
                selected_name = bond_labels[selected_isin]
                st.write(f"Selected Bond: {selected_name} (ISIN: {selected_isin})")
        
                selected_bond_history = final_signal_df[final_signal_df["ISIN"] == selected_isin]
        
                if st.button("Explain this bond"):
                    diagnostics = format_bond_diagnostics(selected_bond_history)
                    explanation = generate_ai_explanation(diagnostics)
                    st.markdown(f"### AI Explanation for {selected_name}")
                    st.write(explanation)
        
        else:
            st.warning("No Nelson-Siegel data available for this date.")

    
    elif subtab == "New Bond Prediction":

        country_option = st.selectbox(
            "Select Country",
            options=['Italy 🇮🇹', 'Spain 🇪🇸', 'France 🇫🇷', 'Germany 🇩🇪', 
                     'Finland 🇫🇮', 'EU 🇪🇺', 'Austria 🇦🇹', 'Netherlands 🇳🇱', 'Belgium 🇧🇪']
        )
    
        country_code_map = {
            'Italy 🇮🇹': 'BTPS',
            'Spain 🇪🇸': 'SPGB',
            'France 🇫🇷': 'FRTR',
            'Germany 🇩🇪': 'BUNDS',
            'Finland 🇫🇮': 'RFGB',
            'EU 🇪🇺': 'EU',
            'Austria 🇦🇹': 'RAGB',
            'Netherlands 🇳🇱': 'NETHER',
            'Belgium 🇧🇪': 'BGB'
        }
    
        selected_country = country_code_map[country_option]
    
        new_bond_input = st.text_input("Enter New Bond Maturity (MM/YY)", value="10/55")
        auction_concession = st.number_input("Auction concession (bps)", value=0, step=1)
    
        # Load NS curves for last 2 weeks
        today = pd.Timestamp.today().normalize()
        start_date = today - pd.Timedelta(days=14)
        ns_df_list = []
    
        date_range = pd.date_range(start_date, today)
        for d in date_range:
            ns_tmp = load_ns_curve(selected_country, d.strftime("%Y-%m-%d"), zip_hash=zip_hash)
            if ns_tmp is not None and not ns_tmp.empty:
                ns_tmp['Date'] = pd.to_datetime(d)
                ns_tmp['YearsToMaturity'] = (pd.to_datetime(ns_tmp['Maturity']) - today).dt.days / 365.25
                ns_df_list.append(ns_tmp)
    
        if not ns_df_list:
            st.warning("No NS curve data for the last 2 weeks.")
            st.stop()
    
        ns_full = pd.concat(ns_df_list, ignore_index=True)
    
        # Smooth NS: mean Z-spread per YearsToMaturity
        ns_smooth = ns_full.groupby('YearsToMaturity')['Z_SPRD_VAL'].mean().reset_index()
        ns_std = ns_full.groupby('YearsToMaturity')['Z_SPRD_VAL'].std().reset_index()
    
        # Parse new bond maturity
        try:
            month, year = map(int, new_bond_input.split('/'))
            year += 2000 if year < 100 else 0
            new_maturity_date = pd.Timestamp(year=year, month=month, day=1)
            new_years_to_maturity = (new_maturity_date - today).days / 365.25
        except:
            st.error("Invalid bond maturity format. Use MM/YY.")
            st.stop()
    
        # Historical offsets using NS Z-spread
        final_signal_df = pd.read_csv("today_all_signals.csv")
        final_signal_df['Maturity'] = pd.to_datetime(final_signal_df['Maturity'], errors='coerce')
    
        similar_bonds = final_signal_df[
            (final_signal_df['Maturity'].notna()) &
            (abs((final_signal_df['Maturity'] - new_maturity_date).dt.days / 365.25) <= 2)
        ]
    
        # Merge with NS Z-spreads from last 2 weeks (nearest available day)
        if not similar_bonds.empty:
            # Take the NS curve for the bond's date closest to today
            ns_today = ns_full[ns_full['Date'] == ns_full['Date'].max()]
            similar_bonds = similar_bonds.merge(
                ns_today[['ISIN', 'Z_SPRD_VAL', 'YearsToMaturity']],
                on='ISIN',
                how='left',
                suffixes=('', '_NS')
            )
    
        historical_offsets = []
        for _, row in similar_bonds.iterrows():
            if pd.notnull(row['Z_SPRD_VAL']):
                # Interpolate NS at historical bond's YearsToMaturity
                f_interp = interp1d(ns_smooth['YearsToMaturity'], ns_smooth['Z_SPRD_VAL'],
                                    kind='linear', fill_value='extrapolate')
                ns_val = f_interp(row['YearsToMaturity'])
                offset = row['Z_SPRD_VAL'] - ns_val
                historical_offsets.append(offset)
        mean_offset = np.mean(historical_offsets) if historical_offsets else 0
    
        # Interpolate/extrapolate new bond Z-spread
        f_new = interp1d(ns_smooth['YearsToMaturity'], ns_smooth['Z_SPRD_VAL'],
                         kind='linear', fill_value='extrapolate')
        predicted_z = f_new(new_years_to_maturity) + mean_offset + auction_concession
    
        # Confidence band using 4 closest maturities
        all_maturities = ns_smooth['YearsToMaturity'].values
        closest_idx = np.argsort(np.abs(all_maturities - new_years_to_maturity))[:4]
        z_std = ns_std.iloc[closest_idx]['Z_SPRD_VAL'].mean() if not ns_std.empty else 0
        z_min, z_max = predicted_z - 1.5*z_std, predicted_z + 1.5*z_std

        close_idx = np.argsort(np.abs(all_maturities - new_years_to_maturity))
        close_idx = [i for i in close_idx if abs(all_maturities[i] - new_years_to_maturity) <= 2]
        
        if close_idx:
            distances = np.abs(all_maturities[close_idx] - new_years_to_maturity)
            weights = 1 / (distances + 1e-6)
            z_std_use = np.average(ns_std.iloc[close_idx]['Z_SPRD_VAL'], weights=weights)
        else:
            z_std_use = ns_std['Z_SPRD_VAL'].mean()
    
        # --- Plot ---
        signal_color_map = {
            'strong buy': 'green',
            'moderate buy': 'lightgreen',
            'weak buy': 'black',
            'strong sell': 'red',
            'moderate sell': 'orange',
            'weak sell': 'black'
        }
        ns_today_plot = ns_full[ns_full['Date'] == ns_full['Date'].max()]
        ns_today_plot = ns_today_plot.merge(final_signal_df[['ISIN', 'SIGNAL']], on='ISIN', how='left')
        ns_today_plot['SIGNAL'] = ns_today_plot['SIGNAL'].str.strip().str.lower()
        ns_today_plot['Signal_Color'] = ns_today_plot['SIGNAL'].map(signal_color_map).fillna('black')
    
        fig = go.Figure()
        legend_signals = ['strong buy', 'moderate buy', 'strong sell', 'moderate sell']
        for signal, df_subset in ns_today_plot.groupby('SIGNAL'):
            if not df_subset.empty:
                color = df_subset['Signal_Color'].iloc[0]
        
                # Add text and customdata for hover
                hover_text = df_subset.get('SECURITY_NAME', df_subset['ISIN']).fillna("Unknown Bond")
                customdata = np.stack([
                    df_subset['ISIN'],
                    df_subset['Maturity'].dt.strftime('%Y-%m-%d'),
                    df_subset.get('RESIDUAL_NS', pd.Series([np.nan]*len(df_subset)))
                ], axis=-1)
        
                fig.add_trace(go.Scatter(
                    x=df_subset['YearsToMaturity'],
                    y=df_subset['Z_SPRD_VAL'],
                    mode='markers',
                    name=signal.title() if signal in legend_signals else None,
                    marker=dict(size=6, color=color, symbol='circle'),
                    customdata=customdata,
                    text=hover_text,
                    hovertemplate=(
                        'Years to Maturity: %{x:.2f}<br>'
                        'Z-Spread: %{y:.1f}bps<br>'
                        'Residual: %{customdata[2]:.2f}bps<br>'
                        'Signal: ' + (signal.title() if signal else "None") + '<br>'
                        '%{text}<extra></extra>'
                    ),
                    showlegend=(signal in legend_signals)
                ))

    
        # Nelson-Siegel fit
        if 'NS_PARAMS' in ns_today_plot.columns:
            try:
                ns_params_raw = ns_today_plot['NS_PARAMS'].iloc[0]
                if isinstance(ns_params_raw, str):
                    import ast
                    ns_params = ast.literal_eval(ns_params_raw)
                else:
                    ns_params = ns_params_raw
                maturity_range = np.linspace(ns_today_plot['YearsToMaturity'].min(),
                                             ns_today_plot['YearsToMaturity'].max(), 100)
                ns_curve = nelson_siegel(maturity_range, *ns_params)
                fig.add_trace(go.Scatter(
                    x=maturity_range,
                    y=ns_curve,
                    mode='lines',
                    name='Nelson-Siegel Fit',
                    line=dict(color='deepskyblue', width=3)
                ))
            except Exception as e:
                st.error(f"Error plotting NS curve: {e}")
    
        # New bond vertical line
        fig.add_trace(go.Scatter(
            x=[new_years_to_maturity, new_years_to_maturity],
            y=[0, z_min],
            mode='lines',
            line=dict(color='purple', dash='dot', width=1),
            name=f"New Bond {new_bond_input}"
        ))
        
        # Shaded prediction band from z_min to z_max
        fig.add_trace(go.Scatter(
            x=[new_years_to_maturity-0.05, new_years_to_maturity+0.05,
               new_years_to_maturity+0.05, new_years_to_maturity-0.05],  # small width around the bond
            y=[z_min, z_min, z_max, z_max],
            fill='toself',
            fillcolor='rgba(0,0,0,0.9)',  # black with some opacity
            line=dict(color='rgba(0,0,0,0)'),  # no border
            showlegend=False
        ))

        fig.update_layout(
            title=f"Predicted Z-Spread Range for New Bond {new_bond_input}",
            xaxis_title="Years to Maturity",
            yaxis_title="Z-Spread (bps)",
            template="plotly_white",
            height=900,
            showlegend=True
        )
    
        st.plotly_chart(fig, use_container_width=True)

    
    # Animated Curves subtab
    elif subtab == "Animated Curves":
        country_option = st.selectbox(
            "Select Country",
            options=['Italy 🇮🇹', 'Spain 🇪🇸', 'France 🇫🇷', 'Germany 🇩🇪', 'Finland 🇫🇮', 'EU 🇪🇺', 'Austria 🇦🇹', 'Netherlands 🇳🇱', 'Belgium 🇧🇪']
        )
    
        country_code_map = {
            'Italy 🇮🇹': 'BTPS',
            'Spain 🇪🇸': 'SPGB',
            'France 🇫🇷': 'FRTR',
            'Germany 🇩🇪': 'BUNDS',
            'Finland 🇫🇮': 'RFGB',
            'EU 🇪🇺': 'EU',
            'Austria 🇦🇹': 'RAGB',
            'Netherlands 🇳🇱': 'NETHER',
            'Belgium 🇧🇪': 'BGB'
        }

    
        selected_country = country_code_map[country_option]
        
        ns_df = load_full_ns_df(selected_country, zip_hash=zip_hash)
        if ns_df is not None and not ns_df.empty:
            final_signal_df = pd.read_csv("today_all_signals.csv")
            country_isins = ns_df['ISIN'].unique()
            bond_options = final_signal_df[final_signal_df['ISIN'].isin(country_isins)][['ISIN', 'SECURITY_NAME']].drop_duplicates()
    
            isin_maturity_map = ns_df.groupby('ISIN')['Maturity'].first().to_dict()
            bond_options['Maturity'] = bond_options['ISIN'].map(isin_maturity_map)
            bond_options['Maturity'] = pd.to_datetime(bond_options['Maturity'], errors='coerce')
            bond_options.sort_values('Maturity', inplace=True)
    
            bond_labels = {row["ISIN"]: row["SECURITY_NAME"] for _, row in bond_options.iterrows()}
    
            def format_bond_label(isin):
                maturity = bond_options.loc[bond_options['ISIN'] == isin, 'Maturity'].values
                if len(maturity) > 0 and pd.notnull(maturity[0]):
                    maturity_str = pd.to_datetime(maturity[0]).strftime('%Y-%m-%d')
                    return f"{bond_labels.get(isin, isin)} ({maturity_str})"
                else:
                    return f"{bond_labels.get(isin, isin)} (N/A)"

    
            # Multiselect directly with all bonds (no search box)
            selected_animation_bonds = st.multiselect(
                "Select Bonds to Display in Animation",
                options=bond_options['ISIN'].tolist(),
                format_func=format_bond_label,
                default=[],
                key="animation_bond_selector"
            )
    
            if st.button(f"Select All {country_option} Bonds"):
                selected_animation_bonds = bond_options['ISIN'].tolist()
    
            if not selected_animation_bonds:
                st.warning("Select at least one bond to display in the animation.")
            else:
                ns_df_filtered = ns_df[ns_df['ISIN'].isin(selected_animation_bonds)].copy()
                ns_df_filtered = ns_df_filtered.merge(final_signal_df[['ISIN', 'SIGNAL']], on='ISIN', how='left')
    
                fig = plot_ns_animation(
                    ns_df_filtered,
                    issuer_label=selected_country,
                    highlight_isins=selected_animation_bonds
                )
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No Nelson-Siegel data available for the selected country.")


    elif subtab == "Compare NS Curves":

        country_code_map = {
            'Italy 🇮🇹': 'BTPS',
            'Spain 🇪🇸': 'SPGB',
            'France 🇫🇷': 'FRTR',
            'Germany 🇩🇪': 'BUNDS',
            'Finland 🇫🇮': 'RFGB',
            'EU 🇪🇺': 'EU',
            'Austria 🇦🇹': 'RAGB',
            'Netherlands 🇳🇱': 'NETHER',
            'Belgium 🇧🇪': 'BGB'
        }
    
        countries = st.multiselect("Select Countries", options=list(country_code_map.keys()))
    
        if countries:
            all_dates = {}
            for c in countries:
                ns_df_country = load_full_ns_df(country_code_map[c], zip_hash=zip_hash)
                if ns_df_country is not None and not ns_df_country.empty:
                    # Convert to datetime and sort descending
                    dates = pd.to_datetime(ns_df_country['Date'].unique())
                    dates = pd.Series(dates).sort_values(ascending=False)
                    all_dates[c] = dates
                else:
                    all_dates[c] = []
        
            selected_dates = {}
            for c in countries:
                if len(all_dates[c]) > 0:
                    # Format dates to remove time
                    formatted_dates = [d.strftime("%Y-%m-%d") for d in all_dates[c]]
                    # Use multiselect with formatted dates
                    selected_dates[c] = st.multiselect(
                        f"Select Dates for {c}",
                        options=formatted_dates,
                        default=formatted_dates[0]  # latest date as default
                    )


            fig = go.Figure()
            for c in countries:
                for d in selected_dates.get(c, []):
                    ns_df_curve = load_ns_curve(country_code_map[c], d, zip_hash=zip_hash)
                    if ns_df_curve is not None and 'NS_PARAMS' in ns_df_curve.columns:
                        ns_params_raw = ns_df_curve['NS_PARAMS'].iloc[0]
                        if isinstance(ns_params_raw, str):
                            import ast
                            ns_params = ast.literal_eval(ns_params_raw)
                        else:
                            ns_params = ns_params_raw
            
                        max_maturity = min(30, ns_df_curve['YTM'].max())
                        maturities = np.linspace(0, max_maturity, 100)
                        ns_values = nelson_siegel(maturities, *ns_params)
                        fig.add_trace(go.Scatter(
                            x=maturities,
                            y=ns_values,
                            mode='lines',
                            name=f"{c} - {d}"
                        ))
            
                
            fig.update_layout(
                title="Nelson-Siegel Curves Comparison",
                xaxis_title="Years to Maturity",
                yaxis_title="Z-Spread (bps)",
                template="plotly_white",
                height=900,
                width=1200,
                xaxis=dict(range=[0, 30])
            )
            st.plotly_chart(fig, use_container_width=True)
        
        

    elif subtab == "Residuals Analysis":

        country_option = st.selectbox(
            "Select Country",
            options=['Italy 🇮🇹', 'Spain 🇪🇸', 'France 🇫🇷', 'Germany 🇩🇪', 'Finland 🇫🇮', 'EU 🇪🇺', 'Austria 🇦🇹', 'Netherlands 🇳🇱', 'Belgium 🇧🇪']
        )
    
        country_code_map = {
            'Italy 🇮🇹': 'BTPS',
            'Spain 🇪🇸': 'SPGB',
            'France 🇫🇷': 'FRTR',
            'Germany 🇩🇪': 'BUNDS',
            'Finland 🇫🇮': 'RFGB',
            'EU 🇪🇺': 'EU',
            'Austria 🇦🇹': 'RAGB',
            'Netherlands 🇳🇱': 'NETHER',
            'Belgium 🇧🇪': 'BGB'
        }
    
        selected_country = country_code_map[country_option]
        
        # Load full NS dataset
        ns_df = load_full_ns_df(selected_country, zip_hash=zip_hash)

        ns_df['RESIDUAL_VELOCITY'] = ns_df.groupby('ISIN')['RESIDUAL_NS'].transform(lambda x: x.diff())

        if ns_df is not None and not ns_df.empty:
            # Ensure Date is datetime
            ns_df['Date'] = pd.to_datetime(ns_df['Date']).dt.normalize()
            
            # Filter bonds available in NS data
            country_isins = ns_df['ISIN'].unique()
            bond_options = ns_df[ns_df['ISIN'].isin(country_isins)][['ISIN', 'SECURITY_NAME']].drop_duplicates()
            
            # Map maturities from NS data
            isin_maturity_map = ns_df.groupby('ISIN')['Maturity'].first().to_dict()
            bond_options['Maturity'] = bond_options['ISIN'].map(isin_maturity_map)
            bond_options['Maturity'] = pd.to_datetime(bond_options['Maturity'], errors='coerce')
            bond_options.sort_values('Maturity', inplace=True)
            
            # Labels for display
            bond_labels = {row["ISIN"]: row["SECURITY_NAME"] for _, row in bond_options.iterrows()}
            def format_bond_label(isin):
                maturity = bond_options.loc[bond_options['ISIN'] == isin, 'Maturity'].values
                if len(maturity) > 0 and pd.notnull(maturity[0]):
                    maturity_str = pd.to_datetime(maturity[0]).strftime('%Y-%m-%d')
                    return f"{bond_labels.get(isin, isin)} ({maturity_str})"
                else:
                    return f"{bond_labels.get(isin, isin)} (N/A)"
            
            # Multiselect for bonds
            selected_bonds = st.multiselect(
                "Select Bonds for Residual Analysis",
                options=bond_options['ISIN'].tolist(),
                format_func=format_bond_label,
                default=[]
            )
            
            if not selected_bonds:
                st.warning("Select at least one bond to display residuals.")
            else:
                # Filter NS data for selected bonds
                residuals_df = ns_df[ns_df['ISIN'].isin(selected_bonds)].copy()
                
                # Initialize figures
                fig_residuals = go.Figure()
                fig_velocity = go.Figure()
                
                # Plot each bond
                for isin in selected_bonds:
                    bond_data = residuals_df[residuals_df['ISIN'] == isin].sort_values('Date')
                    if not bond_data.empty:
                        # Residuals
                        fig_residuals.add_trace(go.Scatter(
                            x=bond_data['Date'],
                            y=bond_data['RESIDUAL_NS'],
                            mode='lines+markers',
                            name=bond_labels.get(isin, isin)
                        ))
                        # Velocity
                        fig_velocity.add_trace(go.Scatter(
                            x=bond_data['Date'],
                            y=bond_data['RESIDUAL_VELOCITY'],
                            mode='lines+markers',
                            name=bond_labels.get(isin, isin)
                        ))
                
                # Update layouts
                fig_residuals.update_layout(
                    title="Residuals Over Time",
                    xaxis_title="Date",
                    yaxis_title="Residual (bps)",
                    template="plotly_white",
                    height=500
                )
                
                fig_velocity.update_layout(
                    title="Residual Velocity Over Time",
                    xaxis_title="Date",
                    yaxis_title="Velocity (bps/day)",
                    template="plotly_white",
                    height=500
                )
                
                # Display charts
                st.plotly_chart(fig_residuals, use_container_width=True)
                st.plotly_chart(fig_velocity, use_container_width=True)


with tab3:
    st.markdown("## Multi-Curve Comparison")

    # --- Metric Selection ---
    metric_option = st.radio(
        "Select Metric for Curve Calculation",
        options=["Z-Spread", "Residuals"],
        horizontal=True
    )

    # Map selected metric to column name in ns_df
    metric_col_map = {
        "Z-Spread": "Z_SPRD_VAL",
        "Residuals": "RESIDUAL_NS"
    }
    selected_metric_col = metric_col_map[metric_option]

    # Ensure session state has exactly 2 curves
    if "curves" not in st.session_state or len(st.session_state.curves) != 2:
        st.session_state.curves = [
            {"id": "curve1", "country": 'Italy 🇮🇹', "bond1": None, "bond2": None},
            {"id": "curve2", "country": 'Italy 🇮🇹', "bond1": None, "bond2": None}
        ]

    country_options = [
        'Italy 🇮🇹', 'Spain 🇪🇸', 'France 🇫🇷', 'Germany 🇩🇪',
        'Finland 🇫🇮', 'EU 🇪🇺', 'Austria 🇦🇹', 'Netherlands 🇳🇱', 'Belgium 🇧🇪'
    ]
    country_code_map = {
        'Italy 🇮🇹': 'BTPS', 'Spain 🇪🇸': 'SPGB', 'France 🇫🇷': 'FRTR',
        'Germany 🇩🇪': 'BUNDS', 'Finland 🇫🇮': 'RFGB', 'EU 🇪🇺': 'EU',
        'Austria 🇦🇹': 'RAGB', 'Netherlands 🇳🇱': 'NETHER', 'Belgium 🇧🇪': 'BGB'
    }

    # Load issuer_signal data
    @st.cache_data(ttl=300)
    def load_issuer_signal() -> pd.DataFrame:
        try:
            df = pd.read_csv("issuer_signals.csv")
            return df
        except Exception as e:
            st.error(f"Failed to load issuer_signal: {e}")
            return pd.DataFrame()

    issuer_signal = load_issuer_signal()
    curve_dfs = []

    # Build global legend mapping
    global_legend_labels = {}
    for curve in st.session_state.curves:
        selected_country = country_code_map[curve['country']]
        ns_df_tmp = load_full_ns_df(selected_country, zip_hash=zip_hash)
        ns_df_tmp['Date'] = pd.to_datetime(ns_df_tmp['Date']).dt.normalize()
        bond_options_tmp = ns_df_tmp[['ISIN', 'SECURITY_NAME', 'Maturity']].drop_duplicates()
        bond_options_tmp = bond_options_tmp.merge(
            issuer_signal[['ISIN', 'SIGNAL']], on='ISIN', how='left'
        )
        bond_options_tmp['Maturity'] = pd.to_datetime(bond_options_tmp['Maturity'], errors='coerce')
        for _, row in bond_options_tmp.iterrows():
            isin = row['ISIN']
            name = row['SECURITY_NAME']
            maturity = pd.to_datetime(row['Maturity']).strftime('%Y-%m-%d') if pd.notnull(row['Maturity']) else "N/A"
            global_legend_labels[isin] = f"{name} ({maturity})"

    # Process each of the 2 curves
    for i, curve in enumerate(st.session_state.curves):
        st.subheader(f"Curve {i+1}")
        col1, col2 = st.columns(2)
        with col1:
            curve['country'] = st.selectbox(
                f"Select Country (Curve {i+1})",
                country_options,
                index=country_options.index(curve['country']),
                key=f"country_{curve['id']}"
            )
            selected_country = country_code_map[curve['country']]
            ns_df = load_full_ns_df(selected_country, zip_hash=zip_hash)
            ns_df['Date'] = pd.to_datetime(ns_df['Date']).dt.normalize()

            # Bond options with signal
            bond_options = ns_df[['ISIN', 'SECURITY_NAME', 'Maturity']].drop_duplicates()
            bond_options = bond_options.merge(
                issuer_signal[['ISIN', 'SIGNAL']], on='ISIN', how='left'
            )
            bond_options['Maturity'] = pd.to_datetime(bond_options['Maturity'], errors='coerce')
            bond_options.sort_values('Maturity', inplace=True)

            bond_labels = {}
            for _, row in bond_options.iterrows():
                isin = row['ISIN']
                name = row['SECURITY_NAME']
                maturity = pd.to_datetime(row['Maturity']).strftime('%Y-%m-%d') if pd.notnull(row['Maturity']) else "N/A"
                signal = row['SIGNAL'] if 'SIGNAL' in row and pd.notnull(row['SIGNAL']) else "No Signal"
                bond_labels[isin] = f"{name} ({maturity}) [{signal}]"

        with col2:
            curve['bond1'] = st.selectbox(
                f"Select Bond 1 (Curve {i+1})",
                bond_options['ISIN'],
                format_func=lambda isin: bond_labels.get(isin, isin),
                key=f"bond1_{curve['id']}"
            )
            curve['bond2'] = st.selectbox(
                f"Select Bond 2 (Curve {i+1})",
                bond_options['ISIN'],
                format_func=lambda isin: bond_labels.get(isin, isin),
                key=f"bond2_{curve['id']}"
            )

        # Compute curve as bond2 − bond1
        if curve['bond1'] and curve['bond2']:
            df1 = ns_df[ns_df['ISIN'] == curve['bond1']][['Date', selected_metric_col]].rename(columns={selected_metric_col: 'B1'})
            df2 = ns_df[ns_df['ISIN'] == curve['bond2']][['Date', selected_metric_col]].rename(columns={selected_metric_col: 'B2'})
            df_curve = df2.merge(df1, on='Date', how='outer').sort_values('Date')  # bond2 - bond1
            df_curve['Curve'] = df_curve['B2'] - df_curve['B1']
            df_curve['Curve_Name'] = f"Curve {i+1}"
            df_curve['Bond1_ISIN'] = curve['bond1']
            df_curve['Bond2_ISIN'] = curve['bond2']
            curve_dfs.append(df_curve)

    # --- Plot individual curves + combined curve ---
    if len(curve_dfs) == 2:
        combined_curve_df = curve_dfs[1][['Date', 'Curve']].merge(
            curve_dfs[0][['Date', 'Curve']],
            on='Date',
            suffixes=('_2', '_1')
        )
        combined_curve_df['Curve'] = combined_curve_df['Curve_2'] - combined_curve_df['Curve_1']

        fig = go.Figure()

        # Plot individual curves
        for i, curve_df in enumerate(curve_dfs):
            curve = st.session_state.curves[i]
            bond1_label = global_legend_labels.get(curve['bond1'], curve['bond1'])
            bond2_label = global_legend_labels.get(curve['bond2'], curve['bond2'])
            curve_name = f"{bond2_label} − {bond1_label}"  # bond2 - bond1

            fig.add_trace(go.Scatter(
                x=curve_df['Date'],
                y=curve_df['Curve'],
                mode='lines',
                name=curve_name
            ))

        # Plot combined curve
        fig.add_trace(go.Scatter(
            x=combined_curve_df['Date'],
            y=combined_curve_df['Curve'],
            mode='lines',
            name="Combined Curve (Curve2 − Curve1)",
            line=dict(color='black', width=3, dash='dot')
        ))

        fig.update_layout(
            title=f"Two-Curve {metric_option} Comparison",
            xaxis_title="Date",
            yaxis_title=f"{metric_option} Difference (bps)",
            template="plotly_white",
            height=900,
            legend_title="Curves"
        )

        st.plotly_chart(fig, use_container_width=True)



# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": get_system_prompt()}]
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""
if "last_processed_input" not in st.session_state:
    st.session_state.last_processed_input = ""

# --- Tab 4: Chat with bond trading assistant ---
with tab4:
    st.markdown("## Ask anything")
    
    # Display conversation (skip system prompt)
    for i, msg in enumerate(st.session_state.chat_history[1:]):
        is_user = msg["role"] == "user"
        message(msg["content"], is_user=is_user, key=f"msg_{i}")
    
    # Create columns for input and button
    col1, col2 = st.columns([4, 1])
    
    with col1:
        # Input box with on_change callback
        user_input = st.text_input(
            "Your question:",
            key="chat_input_box",
            placeholder="Type your message and press Enter or click Send..."
        )
    
    with col2:
        # Send button
        send_clicked = st.button("Send", key="send_button")
    
    # Check if we should process the input
    should_process = False
    current_input = user_input.strip()
    
    # Process if:
    # 1. Send button was clicked and there's input
    # 2. Input has changed (Enter was pressed) and there's input
    # 3. And we haven't already processed this exact input
    if current_input and current_input != st.session_state.last_processed_input:
        if send_clicked or (current_input != st.session_state.get("previous_input", "")):
            should_process = True
    
    # Store current input for next comparison
    st.session_state.previous_input = current_input
    
    # Process the message if conditions are met
    if should_process:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": current_input})
        
        # Get assistant response
        answer, *_ = chat_with_trades(current_input, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Update last processed input to prevent re-processing
        st.session_state.last_processed_input = current_input
        
        # Clear the input box by deleting and recreating the key
        if "chat_input_box" in st.session_state:
            del st.session_state.chat_input_box
        
        # Rerun to update the display and clear the input
        st.rerun()








