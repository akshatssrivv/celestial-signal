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

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 2rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    .sub-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 1.5rem 0 1rem 0;
        border-bottom: 2px solid #3498db;
        padding-bottom: 0.5rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1.5rem;
        border-radius: 10px;
        color: white;
        text-align: center;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 1rem;
    }
    .metric-value {
        font-size: 2rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    .metric-label {
        font-size: 0.9rem;
        opacity: 0.9;
    }
    .signal-badge {
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-weight: bold;
        font-size: 0.8rem;
        display: inline-block;
        margin: 0.2rem;
    }
    .signal-sell {
        background-color: #e74c3c;
        color: white;
    }
    .signal-buy {
        background-color: #27ae60;
        color: white;
    }
    .signal-watch-sell {
        background-color: #f39c12;
        color: white;
    }
    .signal-watch-buy {
        background-color: #2ecc71;
        color: white;
    }
    .data-table {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .stDataFrame {
        border: none;
    }
    .filter-section {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        border-left: 4px solid #3498db;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_data(ttl=300)  # Cache for 5 minutes
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

def get_signal_color(signal):
    """Get color for signal type"""
    color_map = {
        'SHORT': '#e74c3c',
        'WATCHLIST_SHORT': '#f39c12',
        'WATCHLIST_LONG': '#2ecc71',
        'LONG': '#27ae60'
    }
    return color_map.get(signal, '#95a5a6')

def format_signal_badge(signal):
    """Format signal as HTML badge"""
    badge_map = {
        'SHORT': '<span class="signal-badge signal-sell">🔴 SELL</span>',
        'WATCHLIST_SHORT': '<span class="signal-badge signal-watch-sell">🟡 WATCH SELL</span>',
        'WATCHLIST_LONG': '<span class="signal-badge signal-watch-buy">🟡 WATCH BUY</span>',
        'LONG': '<span class="signal-badge signal-buy">🟢 BUY</span>'
    }
    return badge_map.get(signal, f'<span class="signal-badge">⚪ {signal}</span>')

def create_metric_card(title, value, delta=None):
    """Create a custom metric card"""
    delta_html = f"<div style='font-size: 0.8rem; margin-top: 0.5rem;'>{delta}</div>" if delta else ""
    return f"""
    <div class="metric-card">
        <div class="metric-value">{value}</div>
        <div class="metric-label">{title}</div>
        {delta_html}
    </div>
    """

# Load data
with st.spinner('🔄 Loading bond data...'):
    df = load_data()

if df.empty:
    st.error("❌ No data available. Please check your data source.")
    st.stop()

# Add derived columns
df['Country'] = df['ISIN'].apply(get_country_from_isin)
df['Risk_Category'] = pd.cut(df['Volatility'], 
                            bins=[-np.inf, 1, 5, np.inf], 
                            labels=['🟢 Low Risk', '🟡 Medium Risk', '🔴 High Risk'])

# Main title with animation
st.markdown('<h1 class="main-header">📊 European Government Bond Analytics Dashboard</h1>', unsafe_allow_html=True)
st.markdown('<p style="text-align: center; font-size: 1.2rem; color: #7f8c8d; margin-bottom: 2rem;">Advanced mispricing detection and trading signals</p>', unsafe_allow_html=True)

# Sidebar filters
st.sidebar.markdown("## 🎛️ Dashboard Controls")
st.sidebar.markdown("---")

# Auto-refresh toggle
auto_refresh = st.sidebar.checkbox("🔄 Auto-refresh (30s)", value=False)
if auto_refresh:
    st.rerun()

st.sidebar.markdown("### 🔍 Data Filters")

# Signal filter with custom styling
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

# Advanced filters in expander
with st.sidebar.expander("🔧 Advanced Filters"):
    # ISIN search
    isin_search = st.text_input("🔍 Search ISIN", "", help="Type part of ISIN to search")
    
    # Security name search
    name_search = st.text_input("📝 Search Security Name", "", help="Type part of security name")
    
    # Volatility filter
    vol_min, vol_max = st.slider(
        '📊 Volatility Range',
        min_value=float(df['Volatility'].min()),
        max_value=float(df['Volatility'].max()),
        value=(float(df['Volatility'].min()), float(df['Volatility'].max())),
        step=0.1,
        help="Filter bonds by volatility level"
    )
    
    # Composite score filter
    score_min, score_max = st.slider(
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

if name_search:
    filtered_df = filtered_df[filtered_df['SECURITY_NAME'].str.contains(name_search.upper(), na=False)]

filtered_df = filtered_df[
    (filtered_df['Volatility'] >= vol_min) & 
    (filtered_df['Volatility'] <= vol_max) &
    (filtered_df['COMPOSITE_SCORE'] >= score_min) & 
    (filtered_df['COMPOSITE_SCORE'] <= score_max)
]

# Key Metrics Dashboard
st.markdown('<h2 class="sub-header">📈 Key Performance Indicators</h2>', unsafe_allow_html=True)

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    total_bonds = len(filtered_df)
    st.markdown(create_metric_card("📊 Total Bonds", total_bonds, f"of {len(df)} total"), unsafe_allow_html=True)

with col2:
    buy_signals = len(filtered_df[filtered_df['SIGNAL'].isin(['LONG', 'WATCHLIST_LONG'])])
    buy_pct = (buy_signals / total_bonds * 100) if total_bonds > 0 else 0
    st.markdown(create_metric_card("🟢 Buy Signals", buy_signals, f"{buy_pct:.1f}%"), unsafe_allow_html=True)

with col3:
    sell_signals = len(filtered_df[filtered_df['SIGNAL'].isin(['SHORT', 'WATCHLIST_SHORT'])])
    sell_pct = (sell_signals / total_bonds * 100) if total_bonds > 0 else 0
    st.markdown(create_metric_card("🔴 Sell Signals", sell_signals, f"{sell_pct:.1f}%"), unsafe_allow_html=True)

with col4:
    avg_score = filtered_df['COMPOSITE_SCORE'].mean() if not filtered_df.empty else 0
    score_trend = "📈" if avg_score > 0 else "📉" if avg_score < -1 else "➡️"
    st.markdown(create_metric_card("⚖️ Avg Score", f"{avg_score:.3f}", score_trend), unsafe_allow_html=True)

with col5:
    avg_volatility = filtered_df['Volatility'].mean() if not filtered_df.empty else 0
    vol_level = "🔴 High" if avg_volatility > 5 else "🟡 Medium" if avg_volatility > 1 else "🟢 Low"
    st.markdown(create_metric_card("📊 Avg Volatility", f"{avg_volatility:.2f}", vol_level), unsafe_allow_html=True)

# Interactive Charts Section
st.markdown('<h2 class="sub-header">📊 Interactive Analytics</h2>', unsafe_allow_html=True)

if not filtered_df.empty:
    # First row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Signal distribution pie chart
        signal_counts = filtered_df['SIGNAL'].value_counts()
        colors = [get_signal_color(signal) for signal in signal_counts.index]
        
        fig_pie = px.pie(
            values=signal_counts.values,
            names=signal_counts.index,
            title="🎯 Trading Signal Distribution",
            color_discrete_sequence=colors,
            hole=0.4
        )
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(
            height=400,
            showlegend=True,
            title_font_size=16,
            font=dict(size=12)
        )
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Country distribution with average scores
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
            title="🌍 Bonds by Country (colored by avg score)",
            color_continuous_scale='RdYlGn',
            text='Count'
        )
        fig_country.update_traces(texttemplate='%{text}', textposition='outside')
        fig_country.update_layout(height=400, title_font_size=16)
        st.plotly_chart(fig_country, use_container_width=True)
    
    # Second row of charts
    col1, col2 = st.columns(2)
    
    with col1:
        # Volatility distribution histogram
        fig_hist = px.histogram(
            filtered_df,
            x='Volatility',
            nbins=20,
            title="📊 Volatility Distribution",
            color_discrete_sequence=['#3498db']
        )
        fig_hist.update_layout(height=400, title_font_size=16)
        fig_hist.update_xaxis(title="Volatility")
        fig_hist.update_yaxis(title="Number of Bonds")
        st.plotly_chart(fig_hist, use_container_width=True)
    
    with col2:
        # Composite score distribution
        fig_score_hist = px.histogram(
            filtered_df,
            x='COMPOSITE_SCORE',
            nbins=20,
            title="⚖️ Composite Score Distribution",
            color_discrete_sequence=['#e74c3c']
        )
        fig_score_hist.update_layout(height=400, title_font_size=16)
        fig_score_hist.update_xaxis(title="Composite Score")
        fig_score_hist.update_yaxis(title="Number of Bonds")
        st.plotly_chart(fig_score_hist, use_container_width=True)
    
    # Risk-Return Scatter Plot (full width)
    st.markdown("### 📈 Risk-Return Analysis")
    
    fig_scatter = px.scatter(
        filtered_df,
        x='Volatility',
        y='COMPOSITE_SCORE',
        color='SIGNAL',
        size='Volatility',
        hover_data=['ISIN', 'SECURITY_NAME', 'Country'],
        title="Risk vs Return: Composite Score vs Volatility",
        color_discrete_map={
            'SHORT': '#e74c3c',
            'WATCHLIST_SHORT': '#f39c12',
            'WATCHLIST_LONG': '#2ecc71',
            'LONG': '#27ae60'
        }
    )
    fig_scatter.update_layout(height=500, title_font_size=18)
    fig_scatter.update_xaxis(title="Volatility (Risk)")
    fig_scatter.update_yaxis(title="Composite Score (Signal Strength)")
    st.plotly_chart(fig_scatter, use_container_width=True)

# Main Data Table
st.markdown('<h2 class="sub-header">📋 Detailed Bond Analysis</h2>', unsafe_allow_html=True)
st.markdown(f"**Showing {len(filtered_df)} of {len(df)} bonds**")

if not filtered_df.empty:
    # Prepare display dataframe
    display_df = filtered_df.copy()
    
    # Format numeric columns
    display_df['COMPOSITE_SCORE'] = display_df['COMPOSITE_SCORE'].round(3)
    display_df['Z_RESIDUAL_BUCKET'] = display_df['Z_RESIDUAL_BUCKET'].round(3)
    display_df['Cluster_Deviation_Flipped'] = display_df['Cluster_Deviation_Flipped'].round(3)
    display_df['Volatility'] = display_df['Volatility'].round(2)
    display_df['Regression_Component'] = display_df['Regression_Component'].round(3)
    
    # Create signal badges for display
    display_df['Signal_Display'] = display_df['SIGNAL'].apply(lambda x: format_signal_badge(x))
    
    # Column configuration for better display
    column_config = {
        'ISIN': st.column_config.TextColumn('ISIN', width='medium', help='International Securities Identification Number'),
        'SECURITY_NAME': st.column_config.TextColumn('Security Name', width='large'),
        'Country': st.column_config.TextColumn('Country', width='small'),
        'SIGNAL': st.column_config.SelectboxColumn('Signal', options=['SHORT', 'WATCHLIST_SHORT', 'WATCHLIST_LONG', 'LONG']),
        'COMPOSITE_SCORE': st.column_config.NumberColumn('Composite Score', format='%.3f', help='Overall mispricing signal strength'),
        'Z_RESIDUAL_BUCKET': st.column_config.NumberColumn('Z-Residual', format='%.3f'),
        'Cluster_Deviation_Flipped': st.column_config.NumberColumn('Cluster Deviation', format='%.3f'),
        'Volatility': st.column_config.NumberColumn('Volatility', format='%.2f', help='Risk measure'),
        'Regression_Component': st.column_config.NumberColumn('Regression Component', format='%.3f'),
        'Risk_Category': st.column_config.TextColumn('Risk Level', width='small'),
        'Date': st.column_config.DateColumn('Analysis Date')
    }
    
    # Display the main table with enhanced styling
    st.dataframe(
        display_df[['ISIN', 'SECURITY_NAME', 'Country', 'SIGNAL', 'COMPOSITE_SCORE', 
                   'Z_RESIDUAL_BUCKET', 'Cluster_Deviation_Flipped', 'Volatility', 
                   'Regression_Component', 'Risk_Category', 'Date']],
        column_config=column_config,
        use_container_width=True,
        height=600,
        hide_index=True
    )
    
    # Action buttons
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # Download filtered data
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Filtered Data (CSV)",
            data=csv,
            file_name=f"bond_analysis_filtered_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Download full data
        full_csv = df.to_csv(index=False)
        st.download_button(
            label="📊 Download Full Dataset (CSV)",
            data=full_csv,
            file_name=f"bond_analysis_full_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col3:
        # Refresh data button
        if st.button("🔄 Refresh Data", use_container_width=True):
            st.cache_data.clear()
            st.rerun()

# Summary Statistics Section
st.markdown('<h2 class="sub-header">📊 Statistical Summary</h2>', unsafe_allow_html=True)

if not filtered_df.empty:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("#### 🎯 Signal Breakdown")
        signal_summary = filtered_df['SIGNAL'].value_counts()
        for signal, count in signal_summary.items():
            percentage = (count / len(filtered_df)) * 100
            st.write(f"• **{signal.replace('_', ' ')}**: {count} bonds ({percentage:.1f}%)")
    
    with col2:
        st.markdown("#### 📈 Score Statistics")
        st.write(f"• **Mean**: {filtered_df['COMPOSITE_SCORE'].mean():.3f}")
        st.write(f"• **Median**: {filtered_df['COMPOSITE_SCORE'].median():.3f}")
        st.write(f"• **Std Dev**: {filtered_df['COMPOSITE_SCORE'].std():.3f}")
        st.write(f"• **Min**: {filtered_df['COMPOSITE_SCORE'].min():.3f}")
        st.write(f"• **Max**: {filtered_df['COMPOSITE_SCORE'].max():.3f}")
    
    with col3:
        st.markdown("#### 🎲 Risk Distribution")
        risk_dist = filtered_df['Risk_Category'].value_counts()
        for risk, count in risk_dist.items():
            percentage = (count / len(filtered_df)) * 100
            st.write(f"• **{risk}**: {count} bonds ({percentage:.1f}%)")

    # Correlation heatmap
    st.markdown("#### 🔥 Correlation Matrix")
    numeric_cols = ['COMPOSITE_SCORE', 'Z_RESIDUAL_BUCKET', 'Cluster_Deviation_Flipped', 'Volatility', 'Regression_Component']
    corr_matrix = filtered_df[numeric_cols].corr()
    
    fig_heatmap = px.imshow(
        corr_matrix,
        text_auto=True,
        aspect="auto",
        title="Correlation Matrix of Key Metrics",
        color_continuous_scale='RdBu'
    )
    fig_heatmap.update_layout(height=400)
    st.plotly_chart(fig_heatmap, use_container_width=True)

else:
    st.warning("⚠️ No bonds match the current filters. Please adjust your filter criteria.")

# Footer
st.markdown("---")
st.markdown("""
<div style='text-align: center; color: #7f8c8d; padding: 2rem;'>
    <h4>🏦 European Government Bond Analytics Dashboard</h4>
    <p>Advanced mispricing detection and trading signals for institutional investors</p>
    <p><em>Last updated: """ + pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S UTC') + """</em></p>
    <p>📊 Data refreshes automatically • 🔒 Secure • ⚡ Real-time analysis</p>
</div>
""", unsafe_allow_html=True)
