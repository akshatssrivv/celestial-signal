import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import plotly.graph_objects as go
import plotly.express as px
from nelson_siegel_fn import plot_ns_animation, nelson_siegel
from ai_explainer_utils import format_bond_diagnostics, generate_ai_explanation
import openai
import os
import shutil
import hashlib


@st.cache_data(ttl=3600)  # Cache AI explanations for 1 hour
def cached_generate_ai_explanation(diagnostics):
    return generate_ai_explanation(diagnostics)
    
@st.cache_resource
def unzip_ns_curves():
    def file_hash(filepath):
        hasher = hashlib.md5()
        with open(filepath, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()
    
    zip_hash = file_hash("ns_curves.zip")

    if os.path.exists("ns_curves"):
        # Load previous hash if stored
        prev_hash = st.session_state.get('ns_zip_hash')
        if prev_hash != zip_hash:
            import shutil
            shutil.rmtree("ns_curves")
            with zipfile.ZipFile("ns_curves.zip", "r") as zip_ref:
                zip_ref.extractall("ns_curves")
            st.session_state['ns_zip_hash'] = zip_hash
    else:
        with zipfile.ZipFile("ns_curves.zip", "r") as zip_ref:
            zip_ref.extractall("ns_curves")
        st.session_state['ns_zip_hash'] = zip_hash



@st.cache_data
def load_ns_curve(country_code, date_str):
    # Ensure data is unzipped first
    unzip_ns_curves()
    path = f"ns_curves/{country_code}_{date_str}.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    else:
        return None

@st.cache_data
def load_full_ns_df(country_code):
    # Ensure data is unzipped first
    unzip_ns_curves()
    
    folder = "ns_curves"
    
    # Check if folder exists after unzipping
    if not os.path.exists(folder):
        st.error(f"Data folder '{folder}' not found. Please ensure ns_curves.zip exists.")
        return pd.DataFrame()
    
    try:
        all_files = [f for f in os.listdir(folder) if f.startswith(country_code) and f.endswith(".parquet")]
    except OSError as e:
        st.error(f"Error accessing folder '{folder}': {e}")
        return pd.DataFrame()

    dfs = []
    for f in sorted(all_files):
        try:
            df = pd.read_parquet(os.path.join(folder, f))
            dfs.append(df)
        except Exception as e:
            st.warning(f"Error loading file {f}: {e}")
            continue

    if dfs:
        return pd.concat(dfs, ignore_index=True)
    else:
        return pd.DataFrame()

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
            csv_url = "https://lpxiwnvxqozkjlgfrbfh.supabase.co/storage/v1/object/public/celestial-signal/today_all_signals.csv"
            df = pd.read_csv(csv_url)
            return df
        except Exception as e:
            try:
                df = pd.read_csv('today_all_signals.csv')
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
    
    buy_count = len(df[df['SIGNAL'] == 'LONG'])
    sell_count = len(df[df['SIGNAL'] == 'SHORT'])
    watch_buy_count = len(df[df['SIGNAL'] == 'WATCHLIST LONG'])
    watch_sell_count = len(df[df['SIGNAL'] == 'WATCHLIST SHORT'])
    no_action_count = len(df[df['SIGNAL'] == 'NO ACTION'])

    with col1:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value buy">{buy_count}</div>
            <div class="metric-label">BUY</div>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value sell">{sell_count}</div>
            <div class="metric-label">SELL</div>
        </div>
        """, unsafe_allow_html=True)

    with col3:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value watch-buy">{watch_buy_count}</div>
            <div class="metric-label">WATCHLIST BUY</div>
        </div>
        """, unsafe_allow_html=True)

    with col4:
        st.markdown(f"""
        <div class="metric-box">
            <div class="metric-value watch-sell">{watch_sell_count}</div>
            <div class="metric-label">WATCHLIST SELL</div>
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
        fixed_signal_options = ['LONG', 'SHORT', 'WATCHLIST LONG', 'WATCHLIST SHORT', 'NO ACTION']
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
        display_df = filtered_df[[
            'ISIN', 'SECURITY_NAME', 'Country', 'SIGNAL', 
            'COMPOSITE_SCORE', 'Z_RESIDUAL_BUCKET', 'Cluster_Deviation_Flipped', 
            'Volatility', 'Regression_Component'
        ]].copy()
    
        # Round numeric columns
        numeric_cols = ['COMPOSITE_SCORE', 'Z_RESIDUAL_BUCKET', 'Cluster_Deviation_Flipped', 'Volatility', 'Regression_Component']
        display_df[numeric_cols] = display_df[numeric_cols].round(4)
    
        # Display table with sorting enabled
        st.dataframe(
            display_df,
            column_config={
                'ISIN': 'ISIN',
                'SECURITY_NAME': 'Security Name',
                'Country': 'Issuer',
                'SIGNAL': 'Signal',
                'COMPOSITE_SCORE': st.column_config.NumberColumn('Composite Score', format='%.4f'),
                'Z_RESIDUAL_BUCKET': st.column_config.NumberColumn('Z-Residual', format='%.4f'),
                'Cluster_Deviation_Flipped': st.column_config.NumberColumn('Cluster Deviation', format='%.4f'),
                'Volatility': st.column_config.NumberColumn('Volatility', format='%.4f'),
                'Regression_Component': st.column_config.NumberColumn('Regression Component', format='%.4f'),
                'Date': 'Date'
            },
            use_container_width=True,
            height=600,
            hide_index=True
        )

        final_signal_df = pd.read_csv("final_signal.csv")
    
        all_bonds = final_signal_df[["ISIN", "SECURITY_NAME"]].drop_duplicates().sort_values("SECURITY_NAME")
        bond_labels = {row["ISIN"]: row["SECURITY_NAME"] for _, row in all_bonds.iterrows()}
            
        selected_isin = st.selectbox(
            "Select bond for detailed AI explanation",
            options=bond_labels.keys(),
            format_func=lambda x: bond_labels[x]
        )
        
        selected_bond_history = final_signal_df[final_signal_df["ISIN"] == selected_isin]
            
        if st.button("Explain this bond"):
            with st.spinner("Generating AI explanation..."):
                diagnostics = format_bond_diagnostics(selected_bond_history)
                explanation = cached_generate_ai_explanation(diagnostics)
                st.markdown("### AI Explanation")
                st.write(explanation)
    
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
    
    # Choose subtab inside Tab 2
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

    if subtab == "Animated Curves":
        # Check if zip file exists before trying to load data
        if not os.path.exists("ns_curves.zip"):
            st.error("ns_curves.zip file not found. Please ensure the file is uploaded to your Streamlit app.")
        else:
            ns_df = load_full_ns_df(selected_country)

            if ns_df is not None and not ns_df.empty:
                fig = plot_ns_animation(ns_df, issuer_label=selected_country)
                st.plotly_chart(fig, use_container_width=True)
            else:
                st.warning("No Nelson-Siegel data available for the selected country.")

    elif subtab == "Single Day Curve":
        if not os.path.exists("ns_curves.zip"):
            st.error("ns_curves.zip file not found. Please ensure the file is uploaded to your Streamlit app.")
        else:
            date_input = st.date_input("Select Date")
            date_str = date_input.strftime("%Y-%m-%d")
            
            ns_df = load_ns_curve(selected_country, date_str)
    
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
                template="plotly_dark"
            )
            
            # Create an explanation placeholder outside columns for stable state
            ai_explanation_placeholder = st.empty()
            
            col1, col2 = st.columns([3, 2])
            
            with col1:
                st.plotly_chart(fig, use_container_width=True)
                from streamlit_plotly_events import plotly_events
                selected_points = plotly_events(fig, click_event=True, hover_event=False)
            
            with col2:
                pass 
            
            # Now outside columns, update explanation box based on selected points:
            if selected_points:
                isin, date_hovered = selected_points[0]['customdata'][:2]
                bond_history = final_signal_df[(final_signal_df['ISIN'] == isin) & (final_signal_df['Date'] == date_hovered)]
                if not bond_history.empty:
                    diagnostics = format_bond_diagnostics(bond_history)
                    explanation = generate_ai_explanation(diagnostics)
                    ai_explanation_placeholder.markdown(f"### AI Explanation for {diagnostics['SECURITY_NAME']} on {diagnostics['Date']}")
                    ai_explanation_placeholder.write(explanation)
                else:
                    ai_explanation_placeholder.markdown("No diagnostics found for selected bond.")
            else:
                ai_explanation_placeholder.markdown("Click a bond on the plot to see AI explanation here.")

    
            else:
                st.warning("No Nelson-Siegel data available for this date.")
