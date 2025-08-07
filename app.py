import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page config
st.set_page_config(
    page_title="Bond Analytics Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Minimal, clean CSS that doesn't interfere with dataframes
st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f2937;
        text-align: center;
        margin-bottom: 1rem;
    }
    .subtitle {
        font-size: 1.1rem;
        color: #6b7280;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-container {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1);
        border: 1px solid #e5e7eb;
        text-align: center;
        height: 120px;
        display: flex;
        flex-direction: column;
        justify-content: center;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: 700;
        color: #1f2937;
        margin-bottom: 0.25rem;
    }
    .metric-label {
        font-size: 0.875rem;
        color: #6b7280;
        font-weight: 500;
    }
    .section-header {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1f2937;
        margin: 2rem 0 1rem 0;
        padding-bottom: 0.5rem;
        border-bottom: 2px solid #3b82f6;
    }
    .signal-chip {
        padding: 0.25rem 0.75rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.125rem;
    }
    .signal-sell { background-color: #fee2e2; color: #dc2626; }
    .signal-buy { background-color: #dcfce7; color: #16a34a; }
    .signal-watch-sell { background-color: #fef3c7; color: #d97706; }
    .signal-watch-buy { background-color: #dbeafe; color: #2563eb; }
    
    /* Ensure dataframes are visible */
    .stDataFrame {
        background: white !important;
        border-radius: 8px !important;
        overflow: hidden !important;
        box-shadow: 0 1px 3px 0 rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #f9fafb;
    }
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
        st.error(f"Error loading from URL: {e}")
        try:
            df = pd.read_csv('today_all_signals.csv')
            return df
        except Exception as e2:
            st.error(f"Error loading local file: {e2}")
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

def get_signal_display(signal):
    """Get display format for signals"""
    signal_map = {
        'SHORT': '🔴 SELL',
        'WATCHLIST_SHORT': '🟡 WATCH SELL',
        'WATCHLIST_LONG': '🔵 WATCH BUY',
        'LONG': '🟢 BUY'
    }
    return signal_map.get(signal, signal)

# Load data
with st.spinner('Loading bond data...'):
    df = load_data()

if df.empty:
    st.error("No data available. Please check your data source.")
    st.stop()

# Add derived columns
df['Country'] = df['ISIN'].apply(get_country_from_isin)
df['Risk_Level'] = pd.cut(df['Volatility'], 
                         bins=[-np.inf, 1, 5, np.inf], 
                         labels=['Low', 'Medium', 'High'])
df['Signal_Display'] = df['SIGNAL'].apply(get_signal_display)

# Header
st.markdown('<h1 class="main-title">Bond Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">European Government Bond Mispricing Detection & Trading Signals</p>', unsafe_allow_html=True)

# Sidebar
st.sidebar.header("Filters & Controls")

# Auto-refresh
auto_refresh = st.sidebar.checkbox("Auto-refresh (30s)", value=False)
if auto_refresh:
    st.rerun()

st.sidebar.divider()

# Filters
selected_signals = st.sidebar.multiselect(
    "Trading Signals",
    options=list(df['SIGNAL'].unique()),
    default=list(df['SIGNAL'].unique())
)

selected_countries = st.sidebar.multiselect(
    "Countries",
    options=list(df['Country'].unique()),
    default=list(df['Country'].unique())
)

# Advanced filters in expander
with st.sidebar.expander("Advanced Filters"):
    isin_search = st.text_input("Search ISIN")
    name_search = st.text_input("Search Security Name")
    
    vol_range = st.slider(
        "Volatility Range",
        min_value=float(df['Volatility'].min()),
        max_value=float(df['Volatility'].max()),
        value=(float(df['Volatility'].min()), float(df['Volatility'].max()))
    )
    
    score_range = st.slider(
        "Composite Score Range",
        min_value=float(df['COMPOSITE_SCORE'].min()),
        max_value=float(df['COMPOSITE_SCORE'].max()),
        value=(float(df['COMPOSITE_SCORE'].min()), float(df['COMPOSITE_SCORE'].max()))
    )

# Apply filters
filtered_df = df.copy()

if selected_signals:
    filtered_df = filtered_df[filtered_df['SIGNAL'].isin(selected_signals)]

if selected_countries:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]

if isin_search:
    filtered_df = filtered_df[filtered_df['ISIN'].str.contains(isin_search.upper(), na=False)]

if name_search:
    filtered_df = filtered_df[filtered_df['SECURITY_NAME'].str.contains(name_search.upper(), na=False)]

filtered_df = filtered_df[
    (filtered_df['Volatility'] >= vol_range[0]) & 
    (filtered_df['Volatility'] <= vol_range[1]) &
    (filtered_df['COMPOSITE_SCORE'] >= score_range[0]) & 
    (filtered_df['COMPOSITE_SCORE'] <= score_range[1])
]

# Key Metrics
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value">{len(filtered_df)}</div>
        <div class="metric-label">Total Bonds</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    buy_signals = len(filtered_df[filtered_df['SIGNAL'].isin(['LONG', 'WATCHLIST_LONG'])])
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value" style="color: #16a34a;">{buy_signals}</div>
        <div class="metric-label">Buy Signals</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    sell_signals = len(filtered_df[filtered_df['SIGNAL'].isin(['SHORT', 'WATCHLIST_SHORT'])])
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value" style="color: #dc2626;">{sell_signals}</div>
        <div class="metric-label">Sell Signals</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    avg_score = filtered_df['COMPOSITE_SCORE'].mean() if not filtered_df.empty else 0
    score_color = "#16a34a" if avg_score > 0 else "#dc2626" if avg_score < -1 else "#6b7280"
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value" style="color: {score_color};">{avg_score:.3f}</div>
        <div class="metric-label">Avg Score</div>
    </div>
    """, unsafe_allow_html=True)

with col5:
    avg_vol = filtered_df['Volatility'].mean() if not filtered_df.empty else 0
    vol_color = "#dc2626" if avg_vol > 5 else "#d97706" if avg_vol > 1 else "#16a34a"
    st.markdown(f"""
    <div class="metric-container">
        <div class="metric-value" style="color: {vol_color};">{avg_vol:.2f}</div>
        <div class="metric-label">Avg Volatility</div>
    </div>
    """, unsafe_allow_html=True)

st.markdown("---")

# Charts Section
if not filtered_df.empty:
    # Chart tabs for better organization
    tab1, tab2, tab3 = st.tabs(["📊 Distribution", "🌍 Geographic", "📈 Risk Analysis"])
    
    with tab1:
        col1, col2 = st.columns(2)
        
        with col1:
            # Signal distribution
            signal_counts = filtered_df['SIGNAL'].value_counts()
            colors = {'SHORT': '#dc2626', 'WATCHLIST_SHORT': '#d97706', 
                     'WATCHLIST_LONG': '#2563eb', 'LONG': '#16a34a'}
            
            fig_pie = px.pie(
                values=signal_counts.values,
                names=[get_signal_display(s) for s in signal_counts.index],
                title="Trading Signal Distribution",
                color_discrete_sequence=[colors.get(s, '#6b7280') for s in signal_counts.index]
            )
            fig_pie.update_layout(height=400, showlegend=True)
            st.plotly_chart(fig_pie, use_container_width=True)
        
        with col2:
            # Score distribution
            fig_hist = px.histogram(
                filtered_df,
                x='COMPOSITE_SCORE',
                nbins=30,
                title="Composite Score Distribution",
                color_discrete_sequence=['#3b82f6']
            )
            fig_hist.update_layout(height=400)
            st.plotly_chart(fig_hist, use_container_width=True)
    
    with tab2:
        col1, col2 = st.columns(2)
        
        with col1:
            # Country distribution
            country_data = filtered_df.groupby('Country').agg({
                'ISIN': 'count',
                'COMPOSITE_SCORE': 'mean'
            }).reset_index()
            country_data.columns = ['Country', 'Count', 'Avg_Score']
            
            fig_country = px.bar(
                country_data,
                x='Country',
                y='Count',
                color='Avg_Score',
                title="Bonds by Country",
                color_continuous_scale='RdYlGn',
                text='Count'
            )
            fig_country.update_traces(texttemplate='%{text}', textposition='outside')
            fig_country.update_layout(height=400)
            st.plotly_chart(fig_country, use_container_width=True)
        
        with col2:
            # Risk distribution
            risk_counts = filtered_df['Risk_Level'].value_counts()
            risk_colors = {'Low': '#16a34a', 'Medium': '#d97706', 'High': '#dc2626'}
            
            fig_risk = px.bar(
                x=risk_counts.index,
                y=risk_counts.values,
                title="Risk Level Distribution",
                color=risk_counts.index,
                color_discrete_map=risk_colors
            )
            fig_risk.update_layout(height=400, showlegend=False)
            st.plotly_chart(fig_risk, use_container_width=True)
    
    with tab3:
        # Risk-Return scatter
        fig_scatter = px.scatter(
            filtered_df,
            x='Volatility',
            y='COMPOSITE_SCORE',
            color='Signal_Display',
            size='Volatility',
            hover_data=['ISIN', 'SECURITY_NAME', 'Country'],
            title="Risk vs Return Analysis",
            color_discrete_map={
                '🔴 SELL': '#dc2626',
                '🟡 WATCH SELL': '#d97706',
                '🔵 WATCH BUY': '#2563eb',
                '🟢 BUY': '#16a34a'
            }
        )
        fig_scatter.update_layout(height=500)
        st.plotly_chart(fig_scatter, use_container_width=True)

# Main Data Table - This is the key improvement
st.markdown('<h2 class="section-header">Bond Data Table</h2>', unsafe_allow_html=True)

if not filtered_df.empty:
    # Show record count
    st.caption(f"Showing {len(filtered_df)} of {len(df)} bonds")
    
    # Prepare display dataframe with better formatting
    display_df = filtered_df.copy()
    display_df = display_df.round({
        'COMPOSITE_SCORE': 4,
        'Z_RESIDUAL_BUCKET': 4,
        'Cluster_Deviation_Flipped': 4,
        'Volatility': 3,
        'Regression_Component': 4
    })
    
    # Use native Streamlit dataframe with column configuration
    st.dataframe(
        display_df,
        column_config={
            'ISIN': st.column_config.TextColumn('ISIN', width='medium'),
            'SECURITY_NAME': st.column_config.TextColumn('Security Name', width='large'),
            'Country': st.column_config.TextColumn('Country', width='small'),
            'SIGNAL': st.column_config.SelectboxColumn(
                'Signal',
                options=['SHORT', 'WATCHLIST_SHORT', 'WATCHLIST_LONG', 'LONG'],
                width='medium'
            ),
            'Signal_Display': st.column_config.TextColumn('Signal Display', width='medium'),
            'COMPOSITE_SCORE': st.column_config.NumberColumn(
                'Composite Score',
                format='%.4f',
                help='Overall signal strength'
            ),
            'Z_RESIDUAL_BUCKET': st.column_config.NumberColumn('Z-Residual', format='%.4f'),
            'Cluster_Deviation_Flipped': st.column_config.NumberColumn('Cluster Dev', format='%.4f'),
            'Volatility': st.column_config.NumberColumn('Volatility', format='%.3f'),
            'Regression_Component': st.column_config.NumberColumn('Regression', format='%.4f'),
            'Risk_Level': st.column_config.TextColumn('Risk Level'),
            'Date': st.column_config.DateColumn('Date')
        },
        use_container_width=True,
        height=600,
        hide_index=True
    )
    
    # Download and action buttons
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        csv_data = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download Filtered Data",
            csv_data,
            f"bonds_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col2:
        full_csv = df.to_csv(index=False)
        st.download_button(
            "📊 Download Full Dataset",
            full_csv,
            f"bonds_full_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv",
            use_container_width=True
        )
    
    with col3:
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()
    
    with col4:
        # Export strong signals only
        strong_signals = filtered_df[filtered_df['SIGNAL'].isin(['SHORT', 'LONG'])]
        if not strong_signals.empty:
            strong_csv = strong_signals.to_csv(index=False)
            st.download_button(
                "⚡ Strong Signals Only",
                strong_csv,
                f"strong_signals_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
                "text/csv",
                use_container_width=True
            )

    # Quick Stats
    with st.expander("📊 Quick Statistics"):
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.write("**Signal Distribution:**")
            for signal, count in filtered_df['SIGNAL'].value_counts().items():
                pct = (count / len(filtered_df)) * 100
                st.write(f"• {get_signal_display(signal)}: {count} ({pct:.1f}%)")
        
        with col2:
            st.write("**Score Statistics:**")
            st.write(f"• Mean: {filtered_df['COMPOSITE_SCORE'].mean():.4f}")
            st.write(f"• Median: {filtered_df['COMPOSITE_SCORE'].median():.4f}")
            st.write(f"• Std Dev: {filtered_df['COMPOSITE_SCORE'].std():.4f}")
            st.write(f"• Range: {filtered_df['COMPOSITE_SCORE'].min():.4f} to {filtered_df['COMPOSITE_SCORE'].max():.4f}")
        
        with col3:
            st.write("**Risk Distribution:**")
            for risk, count in filtered_df['Risk_Level'].value_counts().items():
                pct = (count / len(filtered_df)) * 100
                st.write(f"• {risk} Risk: {count} ({pct:.1f}%)")

else:
    st.warning("No bonds match the current filters. Please adjust your criteria.")

# Footer
st.markdown("---")
st.markdown(f"""
<div style='text-align: center; color: #6b7280; padding: 1rem;'>
    <p><strong>European Government Bond Analytics Dashboard</strong></p>
    <p>Last updated: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC')}</p>
</div>
""", unsafe_allow_html=True)
