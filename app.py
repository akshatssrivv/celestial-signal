import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Page config
st.set_page_config(
    page_title="Bond Mispricing Signals Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .signal-buy {
        background-color: #d4edda;
        color: #155724;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .signal-sell {
        background-color: #f8d7da;
        color: #721c24;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
    .signal-hold {
        background-color: #fff3cd;
        color: #856404;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load data from CSV with caching"""
    try:
        csv_url = "https://lpxiwnvxqozkjlgfrbfh.supabase.co/storage/v1/object/public/celestial-signal/today_all_signals.csv"
        df = pd.read_csv(csv_url)
        return df
    except:
        # Fallback to local file
        try:
            df = pd.read_csv('today_all_signals.csv')
            return df
        except:
            st.error("Could not load data from URL or local file")
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

def format_signal_badge(signal):
    """Format signal as colored badge"""
    if signal == 'SHORT':
        return f'<span class="signal-sell">🔴 SELL</span>'
    elif signal == 'WATCHLIST_SHORT':
        return f'<span class="signal-hold">🟡 WATCH SELL</span>'
    elif signal == 'WATCHLIST_LONG':
        return f'<span class="signal-hold">🟡 WATCH BUY</span>'
    elif signal == 'LONG':
        return f'<span class="signal-buy">🟢 BUY</span>'
    else:
        return f'<span class="signal-hold">⚪ {signal}</span>'

# Load data
df = load_data()

if df.empty:
    st.stop()

# Add country column
df['Country'] = df['ISIN'].apply(get_country_from_isin)

# Main title
st.markdown('<h1 class="main-header">📊 Bond Mispricing Signals Dashboard</h1>', unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("🔍 Filters")

# Signal filter
signal_options = ['All'] + list(df['SIGNAL'].unique())
selected_signals = st.sidebar.multiselect(
    'Trading Signals',
    options=signal_options,
    default=['All']
)

# Country filter
country_options = ['All'] + list(df['Country'].unique())
selected_countries = st.sidebar.multiselect(
    'Countries',
    options=country_options,
    default=['All']
)

# ISIN filter
isin_options = ['All'] + list(df['ISIN'].unique())
selected_isins = st.sidebar.multiselect(
    'ISIN Codes',
    options=isin_options,
    default=['All']
)

# Volatility filter
vol_min, vol_max = st.sidebar.slider(
    'Volatility Range',
    min_value=float(df['Volatility'].min()),
    max_value=float(df['Volatility'].max()),
    value=(float(df['Volatility'].min()), float(df['Volatility'].max())),
    step=0.1
)

# Composite score filter
score_min, score_max = st.sidebar.slider(
    'Composite Score Range',
    min_value=float(df['COMPOSITE_SCORE'].min()),
    max_value=float(df['COMPOSITE_SCORE'].max()),
    value=(float(df['COMPOSITE_SCORE'].min()), float(df['COMPOSITE_SCORE'].max())),
    step=0.1
)

# Apply filters
filtered_df = df.copy()

if 'All' not in selected_signals:
    filtered_df = filtered_df[filtered_df['SIGNAL'].isin(selected_signals)]

if 'All' not in selected_countries:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]

if 'All' not in selected_isins:
    filtered_df = filtered_df[filtered_df['ISIN'].isin(selected_isins)]

filtered_df = filtered_df[
    (filtered_df['Volatility'] >= vol_min) & 
    (filtered_df['Volatility'] <= vol_max) &
    (filtered_df['COMPOSITE_SCORE'] >= score_min) & 
    (filtered_df['COMPOSITE_SCORE'] <= score_max)
]

# Key Metrics Row
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_bonds = len(filtered_df)
    st.metric("📈 Total Bonds", total_bonds)

with col2:
    buy_signals = len(filtered_df[filtered_df['SIGNAL'].isin(['LONG', 'WATCHLIST_LONG'])])
    st.metric("🟢 Buy Signals", buy_signals)

with col3:
    sell_signals = len(filtered_df[filtered_df['SIGNAL'].isin(['SHORT', 'WATCHLIST_SHORT'])])
    st.metric("🔴 Sell Signals", sell_signals)

with col4:
    avg_score = filtered_df['COMPOSITE_SCORE'].mean()
    st.metric("⚖️ Avg Score", f"{avg_score:.3f}")

with col5:
    avg_volatility = filtered_df['Volatility'].mean()
    st.metric("📊 Avg Volatility", f"{avg_volatility:.2f}")

# Charts Row
col1, col2 = st.columns(2)

with col1:
    # Signal distribution pie chart
    signal_counts = filtered_df['SIGNAL'].value_counts()
    fig_pie = px.pie(
        values=signal_counts.values,
        names=signal_counts.index,
        title="Signal Distribution",
        color_discrete_map={
            'SHORT': '#ff4444',
            'WATCHLIST_SHORT': '#ffaa44',
            'WATCHLIST_LONG': '#44aa44',
            'LONG': '#44ff44'
        }
    )
    fig_pie.update_layout(height=400)
    st.plotly_chart(fig_pie, use_container_width=True)

with col2:
    # Country distribution bar chart
    country_counts = filtered_df['Country'].value_counts()
    fig_bar = px.bar(
        x=country_counts.index,
        y=country_counts.values,
        title="Bonds by Country",
        color=country_counts.values,
        color_continuous_scale="viridis"
    )
    fig_bar.update_layout(height=400, showlegend=False)
    fig_bar.update_xaxis(title="Country")
    fig_bar.update_yaxis(title="Number of Bonds")
    st.plotly_chart(fig_bar, use_container_width=True)

# Scatter plot
st.subheader("📈 Risk-Return Analysis")
fig_scatter = px.scatter(
    filtered_df,
    x='Volatility',
    y='COMPOSITE_SCORE',
    color='SIGNAL',
    hover_data=['ISIN', 'SECURITY_NAME', 'Country'],
    title="Composite Score vs Volatility",
    color_discrete_map={
        'SHORT': '#ff4444',
        'WATCHLIST_SHORT': '#ffaa44',
        'WATCHLIST_LONG': '#44aa44',
        'LONG': '#44ff44'
    }
)
fig_scatter.update_layout(height=500)
st.plotly_chart(fig_scatter, use_container_width=True)

# Main data table
st.subheader("📋 Bond Analysis Results")
st.write(f"Showing {len(filtered_df)} of {len(df)} bonds")

# Prepare display dataframe
display_df = filtered_df.copy()
display_df = display_df.round({
    'Z_RESIDUAL_BUCKET': 3,
    'Cluster_Deviation_Flipped': 3,
    'Volatility': 2,
    'Regression_Component': 3,
    'COMPOSITE_SCORE': 3
})

# Reorder columns for better display
column_order = [
    'ISIN', 'SECURITY_NAME', 'Country', 'SIGNAL', 'COMPOSITE_SCORE',
    'Z_RESIDUAL_BUCKET', 'Cluster_Deviation_Flipped', 'Volatility', 
    'Regression_Component', 'Date'
]
display_df = display_df[column_order]

# Style the dataframe
def style_dataframe(df):
    def color_signals(val):
        if val == 'SHORT':
            return 'background-color: #ffebee; color: #c62828'
        elif val == 'WATCHLIST_SHORT':
            return 'background-color: #fff3e0; color: #ef6c00'
        elif val == 'WATCHLIST_LONG':
            return 'background-color: #e8f5e8; color: #2e7d32'
        elif val == 'LONG':
            return 'background-color: #e8f5e8; color: #1b5e20'
        return ''
    
    def color_scores(val):
        if val > 0:
            return 'color: #2e7d32; font-weight: bold'
        elif val < -1.5:
            return 'color: #c62828; font-weight: bold'
        return ''
    
    def color_volatility(val):
        if val > 10:
            return 'color: #c62828; font-weight: bold'
        elif val > 5:
            return 'color: #ef6c00'
        elif val < 1:
            return 'color: #2e7d32'
        return ''
    
    return df.style.applymap(color_signals, subset=['SIGNAL']) \
                  .applymap(color_scores, subset=['COMPOSITE_SCORE']) \
                  .applymap(color_volatility, subset=['Volatility']) \
                  .format({
                      'COMPOSITE_SCORE': '{:.3f}',
                      'Z_RESIDUAL_BUCKET': '{:.3f}',
                      'Cluster_Deviation_Flipped': '{:.3f}',
                      'Volatility': '{:.2f}',
                      'Regression_Component': '{:.3f}'
                  })

# Display styled dataframe
styled_df = style_dataframe(display_df)
st.dataframe(styled_df, use_container_width=True, height=600)

# Download button
csv = filtered_df.to_csv(index=False)
st.download_button(
    label="📥 Download Filtered Data as CSV",
    data=csv,
    file_name=f"bond_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
    mime="text/csv"
)

# Summary statistics
st.subheader("📊 Summary Statistics")
col1, col2 = st.columns(2)

with col1:
    st.write("**Signal Summary:**")
    signal_summary = filtered_df['SIGNAL'].value_counts()
    for signal, count in signal_summary.items():
        percentage = (count / len(filtered_df)) * 100
        st.write(f"• {signal}: {count} bonds ({percentage:.1f}%)")

with col2:
    st.write("**Score Statistics:**")
    st.write(f"• Mean Score: {filtered_df['COMPOSITE_SCORE'].mean():.3f}")
    st.write(f"• Median Score: {filtered_df['COMPOSITE_SCORE'].median():.3f}")
    st.write(f"• Std Dev: {filtered_df['COMPOSITE_SCORE'].std():.3f}")
    st.write(f"• Min Score: {filtered_df['COMPOSITE_SCORE'].min():.3f}")
    st.write(f"• Max Score: {filtered_df['COMPOSITE_SCORE'].max():.3f}")

# Footer
st.markdown("---")
st.markdown("*Bond Mispricing Analysis Dashboard - Real-time signals for European government bonds*")
