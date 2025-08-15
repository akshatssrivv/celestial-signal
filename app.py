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


# ------------------------------
# Utility functions
# ------------------------------

def file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file"""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def unzip_ns_curves(zip_path: str = "ns_curves.zip", folder: str = "ns_curves", force: bool = False) -> str:
    """
    Unzip NS curves into folder. Returns folder path.
    Re-extracts only if zip has changed or force=True.
    """
    zip_hash = file_hash(zip_path)
    prev_hash = st.session_state.get("ns_zip_hash")

    if force or prev_hash != zip_hash:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(folder)
        st.session_state["ns_zip_hash"] = zip_hash

    return folder


@st.cache_data
def load_full_ns_df(country_code: str, zip_hash: str) -> pd.DataFrame:
    """
    Load all NS curves for a country.
    Passing zip_hash ensures cache invalidation when zip changes.
    """
    folder = unzip_ns_curves()
    if not os.path.exists(folder):
        st.error(f"Data folder '{folder}' not found.")
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
        if "Date" in ns_df.columns:
            ns_df["Date"] = pd.to_datetime(ns_df["Date"])
            ns_df.sort_values("Date", inplace=True)
        return ns_df

    return pd.DataFrame()


@st.cache_data
def load_ns_curve(country_code: str, date_str: str, zip_hash: str) -> pd.DataFrame | None:
    """
    Load NS curve for a single day.
    Passing zip_hash ensures cache invalidation when zip changes.
    """
    folder = unzip_ns_curves()
    path = os.path.join(folder, f"{country_code}_{date_str}.parquet")
    if os.path.exists(path):
        return pd.read_parquet(path)
    return None

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
        .buy { color: #28a745; }
        .sell { color: #dc3545; }
        .watch-buy { color: #17a2b8; }
        .watch-sell { color: #fd7e14; }
        .no-action { color: #6c757d; }
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
            'DE': '🇩🇪 Germany'
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

    # Signal metrics - horizontal boxes
    col1, col2, col3, col4, col5 = st.columns(5)

    # Get actual signal values from your data (case-sensitive and exact match)
    actual_signals = df['SIGNAL'].unique()
    
    buy_count = len(df[df['SIGNAL'] == 'STRONG BUY'])
    sell_count = len(df[df['SIGNAL'] == 'STRONG SELL'])
    watch_buy_count = len(df[df['SIGNAL'] == 'WEAK BUY'])
    watch_sell_count = len(df[df['SIGNAL'] == 'WEAK SELL'])
    no_action_count = len(df[df['SIGNAL'] == 'NO ACTION'])

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
            <div class="metric-value watch-buy">{watch_buy_count}</div>
            <div class="metric-label">WEAK BUY</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value watch-sell">{watch_sell_count}</div>
            <div class="metric-label">WEAK SELL</div>
        </div>
        """, unsafe_allow_html=True)

    with col5:
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
        fixed_signal_options = ['STRONG BUY', 'STRONG SELL', 'WEAK BUY', 'WEAK SELL', 'NO ACTION']
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

    # Data table with sorting
    st.subheader(f"Bond Data ({len(filtered_df)} bonds)")

    if not filtered_df.empty:
        # Format the dataframe for display
        # Desired columns
        cols_to_display = [
            'SECURITY_NAME', 'RESIDUAL_NS', 'SIGNAL',
            'Z_Residual_Score', 'Volatility_Score', 'Market_Stress_Score',
            'Cluster_Score', 'Regression_Score', 'COMPOSITE_SCORE',
            'Top_Features', 'Top_Feature_Effects_Pct'
        ]
        
        # Keep only columns that exist in the dataframe
        existing_cols = [col for col in cols_to_display if col in filtered_df.columns]
        
        # Create display dataframe safely
        display_df = filtered_df[existing_cols].copy()
        
        # Prepare column config dynamically
        column_config = {}

        # Convert Top_Features strings to lists
        if 'Top_Features' in display_df.columns:
            display_df['Top_Features'] = display_df['Top_Features'].apply(
                lambda x: ', '.join(ast.literal_eval(x)) if isinstance(x, str) else 'N/A'
            )
        
        # Convert Top_Feature_Effects_Pct strings to lists
        if 'Top_Feature_Effects_Pct' in display_df.columns:
            display_df['Top_Feature_Effects_Pct'] = display_df['Top_Feature_Effects_Pct'].apply(
                lambda x: ', '.join([f"{int(round(float(v)))}%" for v in x.replace('[','').replace(']','').split()]) 
                if isinstance(x, str) else 'N/A'
            )

        
        # Prepare column config dynamically
        column_config = {}
        for col in existing_cols:
            if col in ['COMPOSITE_SCORE', 'Z_Residual_Score', 'Cluster_Score', 'Regression_Score',
                       'RESIDUAL_NS', 'Volatility_Score', 'Market_Stress_Score']:
                column_config[col] = st.column_config.NumberColumn(col.replace('_', ' '), format='%.4f')
            elif col in ['ISIN', 'SECURITY_NAME', 'SIGNAL', 'Date', 'Top_Features', 'Top_Feature_Effects_Pct']:
                column_config[col] = col.replace('_', ' ')
        
        # Display table
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

    else:
        st.warning("No bonds match your filters")


with tab1:
    st.set_page_config(
        page_title="The Curves",
        layout="wide"
    )
    
    # Choose subtab inside Tab 1
    subtab = st.radio(
        "Select View",
        ("Animated Curves", "Single Day Curve")
    )

    country_option = st.selectbox(
        "Select Country",
        options=['Italy 🇮🇹', 'Spain 🇪🇸', 'France 🇫🇷', 'Germany 🇩🇪']
    )

    country_code_map = {
        'Italy 🇮🇹': 'BTPS',
        'Spain 🇪🇸': 'SPGB',
        'France 🇫🇷': 'FRTR',
        'Germany 🇩🇪': 'BUNDS'
    }

    selected_country = country_code_map[country_option]

    # Ensure zip exists
    zip_path = "ns_curves.zip"
    if not os.path.exists(zip_path):
        st.error("ns_curves.zip file not found. Upload the file.")
        st.stop()

    # Compute zip hash for cache invalidation
    zip_hash = file_hash(zip_path)

    if subtab == "Animated Curves":
        ns_df = load_full_ns_df(selected_country, zip_hash=zip_hash)
        if ns_df is not None and not ns_df.empty:
            fig = plot_ns_animation(ns_df, issuer_label=selected_country)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No Nelson-Siegel data available for the selected country.")

    elif subtab == "Single Day Curve":
        date_input = st.date_input("Select Date")
        date_str = date_input.strftime("%Y-%m-%d")
    
        ns_df = load_ns_curve(selected_country, date_str, zip_hash=zip_hash)
    
        if ns_df is not None and not ns_df.empty:
            ns_df['Maturity'] = pd.to_datetime(ns_df['Maturity'])
            curve_date = pd.to_datetime(date_input)
            ns_df['YearsToMaturity'] = (ns_df['Maturity'] - curve_date).dt.days / 365.25

            # Separate outliers and regular bonds
            outliers = ns_df.nlargest(7, 'RESIDUAL_NS', keep='all')
            regular = ns_df.drop(outliers.index)

            fig = go.Figure()

            # Add regular bonds (black)
            fig.add_trace(go.Scatter(
                x=regular['YearsToMaturity'],
                y=regular['Z_SPRD_VAL'],
                mode='markers',
                name='Bonds',
                marker=dict(size=6, color='black'),
                text=regular['SECURITY_NAME'],
                customdata=np.stack((regular['ISIN'], regular['Date'].astype(str)), axis=-1),
                hovertemplate='Years to Maturity: %{x:.2f}<br>Z-Spread: %{y:.1f}bps<br>%{text}<extra></extra>'
            ))

            # Add outliers (red diamonds)
            fig.add_trace(go.Scatter(
                x=outliers['YearsToMaturity'],
                y=outliers['Z_SPRD_VAL'],
                mode='markers',
                name='Top 7 Outliers',
                marker=dict(size=8, color='red', symbol='diamond'),
                text=outliers['SECURITY_NAME'],
                customdata=np.stack((outliers['ISIN'], outliers['Date'].astype(str), outliers['RESIDUAL_NS']), axis=-1),
                hovertemplate='Years to Maturity: %{x:.2f}<br>Z-Spread: %{y:.1f}bps<br>Residual: %{customdata[2]:.1f}<br>%{text}<extra></extra>'
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
                def load_final_signal():
                    return pd.read_csv("today_all_signals.csv")
                final_signal_df = load_final_signal()
                
                bond_options = final_signal_df[['ISIN', 'SECURITY_NAME']].drop_duplicates().sort_values('SECURITY_NAME')
                bond_labels = {row["ISIN"]: row["SECURITY_NAME"] for _, row in bond_options.iterrows()}
                
                search_input = st.text_input("Search Bond by Name")
                
                if search_input:
                    filtered_bonds = bond_options[
                        bond_options['SECURITY_NAME'].str.contains(search_input, case=False, na=False)
                    ]
                else:
                    filtered_bonds = bond_options
                
                if not filtered_bonds.empty:
                    selected_isin = st.selectbox(
                        "Select Bond for AI Explanation",
                        options=filtered_bonds['ISIN'].tolist(),
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
                    st.write("No bonds found matching your search.")

        else:
            st.warning("No Nelson-Siegel data available for this date.")











































