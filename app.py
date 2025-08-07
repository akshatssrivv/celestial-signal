import streamlit as st
import pandas as pd
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
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
        margin-bottom: 1rem;
    }
    .signal-short {
        background-color: #ffebee;
        color: #c62828;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        display: inline-block;
    }
    .signal-watchlist-short {
        background-color: #fff3e0;
        color: #ef6c00;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        display: inline-block;
    }
    .signal-watchlist-long {
        background-color: #e8f5e8;
        color: #2e7d32;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        display: inline-block;
    }
    .signal-long {
        background-color: #e8f5e8;
        color: #1b5e20;
        padding: 0.25rem 0.5rem;
        border-radius: 0.25rem;
        font-weight: bold;
        display: inline-block;
    }
    .high-score {
        color: #2e7d32;
        font-weight: bold;
    }
    .low-score {
        color: #c62828;
        font-weight: bold;
    }
    .high-volatility {
        color: #c62828;
        font-weight: bold;
    }
    .medium-volatility {
        color: #ef6c00;
    }
    .low-volatility {
        color: #2e7d32;
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

def format_signal_display(signal):
    """Format signal for display"""
    signal_map = {
        'SHORT': '🔴 SELL',
        'WATCHLIST_SHORT': '🟡 WATCH SELL',
        'WATCHLIST_LONG': '🟡 WATCH BUY',
        'LONG': '🟢 BUY'
    }
    return signal_map.get(signal, f'⚪ {signal}')

def style_score(score):
    """Style composite score based on value"""
    if score > 0:
        return f'<span class="high-score">{score:.3f}</span>'
    elif score < -1.5:
        return f'<span class="low-score">{score:.3f}</span>'
    else:
        return f'{score:.3f}'

def style_volatility(vol):
    """Style volatility based on value"""
    if vol > 10:
        return f'<span class="high-volatility">{vol:.2f}</span>'
    elif vol > 5:
        return f'<span class="medium-volatility">{vol:.2f}</span>'
    elif vol < 1:
        return f'<span class="low-volatility">{vol:.2f}</span>'
    else:
        return f'{vol:.2f}'

def style_signal(signal):
    """Style signal based on type"""
    if signal == 'SHORT':
        return f'<span class="signal-short">🔴 SELL</span>'
    elif signal == 'WATCHLIST_SHORT':
        return f'<span class="signal-watchlist-short">🟡 WATCH SELL</span>'
    elif signal == 'WATCHLIST_LONG':
        return f'<span class="signal-watchlist-long">🟡 WATCH BUY</span>'
    elif signal == 'LONG':
        return f'<span class="signal-long">🟢 BUY</span>'
    else:
        return f'⚪ {signal}'

# Load data
df = load_data()

if df.empty:
    st.stop()

# Add country column
df['Country'] = df['ISIN'].apply(get_country_from_isin)

# Main title
st.markdown('<h1 class="main-header">📊 Bond Mispricing Signals Dashboard</h1>', unsafe_allow_html=True)

# Sidebar filters
st.sidebar.header("🔍 Filters & Controls")

# Signal filter
signal_options = list(df['SIGNAL'].unique())
selected_signals = st.sidebar.multiselect(
    '🎯 Trading Signals',
    options=signal_options,
    default=signal_options,
    help="Select which trading signals to display"
)

# Country filter
country_options = list(df['Country'].unique())
selected_countries = st.sidebar.multiselect(
    '🌍 Countries',
    options=country_options,
    default=country_options,
    help="Filter bonds by country"
)

# ISIN filter (searchable)
st.sidebar.write("🔍 **Search ISIN:**")
isin_search = st.sidebar.text_input("Type ISIN to search", "")

# Volatility filter
vol_min, vol_max = st.sidebar.slider(
    '📊 Volatility Range',
    min_value=float(df['Volatility'].min()),
    max_value=float(df['Volatility'].max()),
    value=(float(df['Volatility'].min()), float(df['Volatility'].max())),
    step=0.1,
    help="Filter bonds by volatility level"
)

# Composite score filter
score_min, score_max = st.sidebar.slider(
    '⚖️ Composite Score Range',
    min_value=float(df['COMPOSITE_SCORE'].min()),
    max_value=float(df['COMPOSITE_SCORE'].max()),
    value=(float(df['COMPOSITE_SCORE'].min()), float(df['COMPOSITE_SCORE'].max())),
    step=0.1,
    help="Filter bonds by composite score"
)

# Apply filters
filtered_df = df.copy()

if selected_signals:
    filtered_df = filtered_df[filtered_df['SIGNAL'].isin(selected_signals)]

if selected_countries:
    filtered_df = filtered_df[filtered_df['Country'].isin(selected_countries)]

if isin_search:
    filtered_df = filtered_df[filtered_df['ISIN'].str.contains(isin_search.upper(), na=False)]

filtered_df = filtered_df[
    (filtered_df['Volatility'] >= vol_min) & 
    (filtered_df['Volatility'] <= vol_max) &
    (filtered_df['COMPOSITE_SCORE'] >= score_min) & 
    (filtered_df['COMPOSITE_SCORE'] <= score_max)
]

# Key Metrics Row
st.markdown("### 📈 Key Metrics")
col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_bonds = len(filtered_df)
    st.metric("📊 Total Bonds", total_bonds, delta=f"{total_bonds - len(df)} filtered")

with col2:
    buy_signals = len(filtered_df[filtered_df['SIGNAL'].isin(['LONG', 'WATCHLIST_LONG'])])
    buy_pct = (buy_signals / total_bonds * 100) if total_bonds > 0 else 0
    st.metric("🟢 Buy Signals", buy_signals, delta=f"{buy_pct:.1f}%")

with col3:
    sell_signals = len(filtered_df[filtered_df['SIGNAL'].isin(['SHORT', 'WATCHLIST_SHORT'])])
    sell_pct = (sell_signals / total_bonds * 100) if total_bonds > 0 else 0
    st.metric("🔴 Sell Signals", sell_signals, delta=f"{sell_pct:.1f}%")

with col4:
    avg_score = filtered_df['COMPOSITE_SCORE'].mean() if not filtered_df.empty else 0
    st.metric("⚖️ Avg Score", f"{avg_score:.3f}")

with col5:
    avg_volatility = filtered_df['Volatility'].mean() if not filtered_df.empty else 0
    st.metric("📊 Avg Volatility", f"{avg_volatility:.2f}")

# Signal Distribution
st.markdown("### 📊 Signal Distribution")
col1, col2 = st.columns(2)

with col1:
    if not filtered_df.empty:
        signal_counts = filtered_df['SIGNAL'].value_counts()
        st.write("**Trading Signals:**")
        for signal, count in signal_counts.items():
            percentage = (count / len(filtered_df)) * 100
            st.write(f"• {format_signal_display(signal)}: **{count}** bonds ({percentage:.1f}%)")
    else:
        st.write("No data to display")

with col2:
    if not filtered_df.empty:
        country_counts = filtered_df['Country'].value_counts()
        st.write("**By Country:**")
        for country, count in country_counts.items():
            percentage = (count / len(filtered_df)) * 100
            st.write(f"• {country}: **{count}** bonds ({percentage:.1f}%)")

# Charts using Streamlit native charts
st.markdown("### 📈 Visual Analysis")
col1, col2 = st.columns(2)

with col1:
    if not filtered_df.empty:
        st.write("**Signal Distribution**")
        signal_chart_data = filtered_df['SIGNAL'].value_counts()
        st.bar_chart(signal_chart_data)

with col2:
    if not filtered_df.empty:
        st.write("**Composite Score Distribution**")
        st.histogram(filtered_df['COMPOSITE_SCORE'], bins=20)

# Scatter plot using Streamlit
if not filtered_df.empty:
    st.markdown("### 📈 Risk-Return Analysis")
    st.write("**Composite Score vs Volatility**")
    
    # Create scatter plot data
    scatter_data = filtered_df[['Volatility', 'COMPOSITE_SCORE']].copy()
    st.scatter_chart(scatter_data, x='Volatility', y='COMPOSITE_SCORE', size=20)

# Main data table
st.markdown("### 📋 Bond Analysis Results")
st.write(f"Showing **{len(filtered_df)}** of **{len(df)}** bonds")

if not filtered_df.empty:
    # Prepare display dataframe
    display_df = filtered_df.copy()
    
    # Create formatted columns for display
    display_df['Signal_Display'] = display_df['SIGNAL'].apply(lambda x: style_signal(x))
    display_df['Score_Display'] = display_df['COMPOSITE_SCORE'].apply(lambda x: style_score(x))
    display_df['Volatility_Display'] = display_df['Volatility'].apply(lambda x: style_volatility(x))
    
    # Round numeric columns
    numeric_columns = ['Z_RESIDUAL_BUCKET', 'Cluster_Deviation_Flipped', 'Regression_Component']
    for col in numeric_columns:
        display_df[col] = display_df[col].round(3)
    
    # Reorder columns for better display
    column_config = {
        'ISIN': st.column_config.TextColumn('ISIN', width='medium'),
        'SECURITY_NAME': st.column_config.TextColumn('Security Name', width='large'),
        'Country': st.column_config.TextColumn('Country', width='small'),
        'Signal_Display': st.column_config.TextColumn('Signal', width='medium'),
        'Score_Display': st.column_config.TextColumn('Score', width='small'),
        'Volatility_Display': st.column_config.TextColumn('Volatility', width='small'),
        'Z_RESIDUAL_BUCKET': st.column_config.NumberColumn('Z-Residual', format='%.3f'),
        'Cluster_Deviation_Flipped': st.column_config.NumberColumn('Cluster Dev', format='%.3f'),
        'Regression_Component': st.column_config.NumberColumn('Regression', format='%.3f'),
        'Date': st.column_config.DateColumn('Date')
    }
    
    # Display the main table
    st.dataframe(
        display_df[['ISIN', 'SECURITY_NAME', 'Country', 'Signal_Display', 'Score_Display', 
                   'Volatility_Display', 'Z_RESIDUAL_BUCKET', 'Cluster_Deviation_Flipped', 
                   'Regression_Component', 'Date']],
        column_config=column_config,
        use_container_width=True,
        height=600,
        hide_index=True
    )
    
    # Download button
    csv = filtered_df.to_csv(index=False)
    st.download_button(
        label="📥 Download Filtered Data as CSV",
        data=csv,
        file_name=f"bond_analysis_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
        mime="text/csv",
        use_container_width=True
    )
    
    # Summary statistics
    st.markdown("### 📊 Summary Statistics")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Composite Score Stats:**")
        st.write(f"• Mean: {filtered_df['COMPOSITE_SCORE'].mean():.3f}")
        st.write(f"• Median: {filtered_df['COMPOSITE_SCORE'].median():.3f}")
        st.write(f"• Std Dev: {filtered_df['COMPOSITE_SCORE'].std():.3f}")
    
    with col2:
        st.write("**Volatility Stats:**")
        st.write(f"• Mean: {filtered_df['Volatility'].mean():.2f}")
        st.write(f"• Median: {filtered_df['Volatility'].median():.2f}")
        st.write(f"• Max: {filtered_df['Volatility'].max():.2f}")
    
    with col3:
        st.write("**Risk Categories:**")
        low_vol = len(filtered_df[filtered_df['Volatility'] < 1])
        med_vol = len(filtered_df[(filtered_df['Volatility'] >= 1) & (filtered_df['Volatility'] < 5)])
        high_vol = len(filtered_df[filtered_df['Volatility'] >= 5])
        st.write(f"• Low Risk (<1): {low_vol}")
        st.write(f"• Medium Risk (1-5): {med_vol}")
        st.write(f"• High Risk (≥5): {high_vol}")

else:
    st.warning("No bonds match the current filters. Please adjust your filter criteria.")

# Footer
st.markdown("---")
st.markdown("*Bond Mispricing Analysis Dashboard - Real-time signals for European government bonds*")
st.markdown("*Last updated: " + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S') + "*")
