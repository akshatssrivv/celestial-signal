import streamlit as st
import pandas as pd
import numpy as np
import zipfile
import plotly.express as px
from nelson_siegel_fn import plot_ns_animation
import os

@st.cache_resource
def unzip_ns_curves():
    if not os.path.exists("ns_curves"):
        with zipfile.ZipFile("ns_curves.zip", "r") as zip_ref:
            zip_ref.extractall("ns_curves")

@st.cache_data
def load_ns_curve(country_code, date_str):
    path = f"ns_curves/{country_code}_{date_str}.parquet"
    if os.path.exists(path):
        return pd.read_parquet(path)
    else:
        return None

tab1, tab2 = st.tabs(["Signal Dashboard", "Nelson-Siegel Curves"])

with tab1:
    st.set_page_config(
        page_title="Bond Analytics Dashboard",
        page_icon="📊",
        layout="wide"
    )

    # Simple, clean styling
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
    col1, col2, col3, col4 = st.columns(4)

    buy_count = len(df[df['SIGNAL'] == 'LONG'])
    sell_count = len(df[df['SIGNAL'] == 'SHORT'])
    watch_buy_count = len(df[df['SIGNAL'] == 'WATCHLIST LONG'])
    watch_sell_count = len(df[df['SIGNAL'] == 'WATCHLIST SHORT'])

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
        selected_signals = st.multiselect(
            "Signals",
            options=['LONG', 'SHORT', 'WATCHLIST_LONG', 'WATCHLIST_SHORT'],
            default=['LONG', 'SHORT', 'WATCHLIST_LONG', 'WATCHLIST_SHORT']
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
            'Volatility', 'Regression_Component', 'Date'
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
                'Country': 'Country',
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

with tab2:
    st.title("Nelson-Siegel Curve Explorer")

    # Choose subtab inside Tab 2
    subtab = st.radio(
        "Select View",
        ("Animated Yield Curves", "Single Day Yield Curve")
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

    if subtab == "Animated Yield Curves":
        # Load full NS dataframe (all dates) for selected country
        ns_df = load_full_ns_df(selected_country)  # you define this to load whole NS df

        if ns_df is not None and not ns_df.empty:
            # Plot animated NS curve
            fig = plot_ns_animation(ns_df, issuer_label=selected_country)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No Nelson-Siegel data available for the selected country.")

    elif subtab == "Single Day Yield Curve":
        # Select date for single-day plot
        date_input = st.date_input("Select Date")
        date_str = date_input.strftime("%Y-%m-%d")

        # Load filtered NS dataframe for this date
        ns_df = load_ns_curve(selected_country, date_str)

        if ns_df is not None and not ns_df.empty:
            # Plot static single-day curve (similar to before)
            fig = px.line(
                ns_df,
                x='Maturity',
                y='Z_SPRD_VAL',
                title=f"Nelson-Siegel Curve for {selected_country} on {date_str}",
                markers=True
            )
            fig.update_layout(height=700)  # taller plot
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("No Nelson-Siegel data available for this date.")
