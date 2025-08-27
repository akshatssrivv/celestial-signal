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
import requests
import zipfile


SUPABASE_URL = "https://lpxiwnvxqozkjlgfrbfh.supabase.co/storage/v1/object/public/celestial-signal/ns_curves2708.zip"
LOCAL_ZIP = "ns_curves_20250827.zip"
LOCAL_FOLDER = "ns_curves"

def file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file"""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def download_from_supabase(url: str = SUPABASE_URL, output: str = LOCAL_ZIP, force: bool = False) -> str:
    """Download ns_curves.zip from Supabase, optionally forcing refresh."""
    if force or not os.path.exists(output):
        with st.spinner("Downloading NS curves data from Supabase..."):
            with requests.get(url, stream=True) as r:
                r.raise_for_status()
                with open(output, "wb") as f:
                    for chunk in r.iter_content(chunk_size=8192):
                        f.write(chunk)
    return output


def unzip_ns_curves(zip_path: str = LOCAL_ZIP, folder: str = LOCAL_FOLDER, force: bool = False) -> tuple[str, str]:
    """Unzip NS curves and return (folder, zip_hash)."""
    zip_path = download_from_supabase(SUPABASE_URL, zip_path, force=force)
    zip_hash = file_hash(zip_path)
    prev_hash = st.session_state.get("ns_zip_hash")

    if force or prev_hash != zip_hash or not os.path.exists(folder):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(folder)
        st.session_state["ns_zip_hash"] = zip_hash

    return folder, zip_hash



@st.cache_data
def load_full_ns_df(country_code: str, zip_hash: str) -> pd.DataFrame:
    """Load all NS curves for a country. Cache invalidates if ZIP changes."""
    folder, zip_hash = unzip_ns_curves(zip_path=LOCAL_ZIP, force=True)

    if not os.path.exists(folder):
        st.error(f"Data folder '{folder}' not found after unzip.")
        return pd.DataFrame()

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
            ns_df['RESIDUAL_NS'] = np.nan

        if "Date" in ns_df.columns:
            ns_df["Date"] = pd.to_datetime(ns_df["Date"])
            ns_df.sort_values("Date", inplace=True)

        if "Country" in ns_df.columns:
            print("Unique country codes in NS DF:", ns_df['Country'].unique())

        return ns_df

    st.warning(f"No parquet files found for country code '{country_code}' in folder '{folder}'.")
    return pd.DataFrame()



def load_ns_curve(country_code: str, date_str: str, zip_hash: str) -> pd.DataFrame | None:
    """
    Load NS curve for a single day from the full dataset.
    Passing zip_hash ensures cache invalidation when ns_curves.zip changes.
    """
    df = load_full_ns_df(country_code, zip_hash=zip_hash)

    if df is not None and not df.empty:
        df = df[df["Date"] == date_str]
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


tab1, tab2 = st.tabs(["Nelson-Siegel Curves", "Signal Dashboard"])


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
    def load_data():
        """Load data from CSV with caching"""
        try:
            csv_url = "https://lpxiwnvxqozkjlgfrbfh.supabase.co/storage/v1/object/public/celestial-signal/issuer_signals.csv"
            df = pd.read_csv(csv_url)
            return df
        except Exception as e:
            try:
                df = pd.read_csv('issuer_signals.csv')
                return df
            except Exception as e2:
                st.error(f"Error loading data: {e2}")
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
    
    buy_count = len(df[df['SIGNAL'] == 'STRONG BUY'])
    sell_count = len(df[df['SIGNAL'] == 'STRONG SELL'])
    mod_buy_count = len(df[df['SIGNAL'] == 'MODERATE BUY'])
    mod_sell_count = len(df[df['SIGNAL'] == 'MODERATE SELL'])
    watch_buy_count = len(df[df['SIGNAL'] == 'WEAK BUY'])
    watch_sell_count = len(df[df['SIGNAL'] == 'WEAK SELL'])
    no_action_count = len(df[df['SIGNAL'] == 'NO ACTION'])

    col1, col2, col3, col4, col5, col6, col7 = st.columns(7)
    
    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value buy">{buy_count}</div>
            <div class="metric-label">STRONG BUY</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value sell">{sell_count}</div>
            <div class="metric-label">STRONG SELL</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value mod-buy">{mod_buy_count}</div>
            <div class="metric-label">MODERATE BUY</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value mod-sell">{mod_sell_count}</div>
            <div class="metric-label">MODERATE SELL</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col5:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value watch-buy">{watch_buy_count}</div>
            <div class="metric-label">WEAK BUY</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col6:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value watch-sell">{watch_sell_count}</div>
            <div class="metric-label">WEAK SELL</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col7:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value no-action">{no_action_count}</div>
            <div class="metric-label">NO ACTION</div>
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
            'Volatility_Score': 'Stability_Score'
        }, inplace=True)
    
        # Convert numeric columns
        numeric_cols = ['Residual', 'Z_Residual_Score', 'Stability_Score',
                        'Market_Stress_Score', 'Cluster_Score', 'Regression_Score', 'COMPOSITE_SCORE']
        for col in numeric_cols:
            if col in display_df.columns:
                display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
    
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
    
        # Create display column (optional format, still sortable)
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

        
        # Column config for tooltips + formatting
        HELP_TEXTS = {
            "Residual": "Residual mispricing (bps off curve)",
            "Z_Residual_Score": "Z-score of residual. |Z| > 1.5 may indicate opportunities.",
            "Stability_Score": "Inverse volatility. Higher = more stable pricing. Lower = riskier.",
            "Market_Stress_Score": "Market stress factor. High = bond more exposed to stress.",
            "Cluster_Score": "Deviation from peer cluster (bps). Absolute > 1.5, likely to mean-revert",
            "Regression_Score": "Model-explained mispricing. Absolute > 1.5, strong signal; likely to mean-revert.",
            "COMPOSITE_SCORE": "Overall mispricing score. Absolute > 1.5 = stronger trade signal.",
            "Top_Features": "Most important drivers of mispricing. % shows relative impact."
        }
        
        column_config = {}
        for col in display_df.columns:
            label = col.replace('_', ' ')
            if col in numeric_cols and pd.api.types.is_numeric_dtype(display_df[col]):
                column_config[col] = st.column_config.NumberColumn(label, format="%.2f", help=HELP_TEXTS.get(col))
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
        ("Single Day Curve", "Animated Curves", "Residuals Analysis", "Compare NS Curves")
    )

    # Ensure local zip exists (download from Supabase if missing)
    zip_path = download_from_supabase()
    zip_hash = file_hash(zip_path)

    
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
    
        if ns_df is not None and not ns_df.empty:
            ns_df['Maturity'] = pd.to_datetime(ns_df['Maturity'])
            curve_date = pd.to_datetime(date_input)
            ns_df['YearsToMaturity'] = (ns_df['Maturity'] - curve_date).dt.days / 365.25
    
            # Load signals
            final_signal_df = pd.read_csv("today_all_signals.csv")
            # Merge signals into ns_df
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
    
            # Only include these in legend
            legend_signals = ['strong buy', 'moderate buy', 'strong sell', 'moderate sell']
            
            for signal, df_subset in ns_df.groupby('SIGNAL'):
                if not df_subset.empty:
                    color = df_subset['Signal_Color'].iloc[0]
                    
                    fig.add_trace(go.Scatter(
                        x=df_subset['YearsToMaturity'],
                        y=df_subset['Z_SPRD_VAL'],
                        mode='markers',
                        name=signal.title() if signal in legend_signals else None,
                        marker=dict(size=6,
                                    color=color,
                                    symbol='circle'),
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
    
            # Add Nelson-Siegel fit line if available
            if 'NS_PARAMS' in ns_df.columns:
                try:
                    ns_params_raw = ns_df['NS_PARAMS'].iloc[0]
                    if isinstance(ns_params_raw, str):
                        import ast
                        ns_params = ast.literal_eval(ns_params_raw)
                    else:
                        ns_params = ns_params_raw
    
                    maturity_range = np.linspace(ns_df['YearsToMaturity'].min(), ns_df['YearsToMaturity'].max(), 100)
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
    
                # Single selectbox without search input
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
        countries = st.multiselect(
            "Select Countries",
            options=list(country_code_map.keys()),
            default=["Italy 🇮🇹"]
        )
    
        selected_dates = {}
        for c in countries:
            # Load signals for available dates
            final_signal_df = pd.read_csv("today_all_signals.csv")
            available_dates = pd.to_datetime(final_signal_df['Date'].unique())
            if len(available_dates) == 0:
                continue
            
            default_date = available_dates.max()
            
            # Use st.date_input for each country
            chosen_date = st.date_input(
                f"Select Date for {c}",
                value=default_date,
                min_value=available_dates.min(),
                max_value=available_dates.max()
            )
            
            selected_dates[c] = [chosen_date]  # wrap in list for loop consistency
    
        fig = go.Figure()
        for c in countries:
            for d in selected_dates.get(c, []):
                ns_df_curve = load_ns_curve(country_code_map[c], d.strftime("%Y-%m-%d"), zip_hash=zip_hash)
                if ns_df_curve is not None and 'NS_PARAMS' in ns_df_curve.columns:
                    ns_params_raw = ns_df_curve['NS_PARAMS'].iloc[0]
                    if isinstance(ns_params_raw, str):
                        import ast
                        ns_params = ast.literal_eval(ns_params_raw)
                    else:
                        ns_params = ns_params_raw
    
                    # Cap maturities at 30 years
                    max_maturity = min(30, ns_df_curve['YTM'].max())
                    maturities = np.linspace(0, max_maturity, 100)
    
                    ns_values = nelson_siegel(maturities, *ns_params)
                    fig.add_trace(go.Scatter(
                        x=maturities,
                        y=ns_values,
                        mode='lines',
                        name=f"{c} - {d.strftime('%Y-%m-%d')}"
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













