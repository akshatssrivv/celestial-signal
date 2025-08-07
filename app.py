import streamlit as st
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Bond Screening Results",
    page_icon="📊",
    layout="wide"
)

# Professional styling matching the image
st.markdown("""
<style>
    .main-title {
        font-size: 1.5rem;
        font-weight: 600;
        color: #1a365d;
        margin-bottom: 1rem;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    .bond-count {
        font-size: 0.9rem;
        color: #718096;
        background: #f7fafc;
        padding: 0.25rem 0.75rem;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        margin-left: auto;
    }
    
    .metric-box {
        background: white;
        padding: 1.5rem;
        border-radius: 8px;
        border: 1px solid #e2e8f0;
        text-align: center;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        margin-bottom: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .buy { color: #38a169; }
    .sell { color: #e53e3e; }
    .watch-buy { color: #3182ce; }
    .watch-sell { color: #dd6b20; }
    
    .filter-container {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
        border: 1px solid #e2e8f0;
    }
    
    /* Custom table styling */
    .stDataFrame {
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
        overflow: hidden !important;
    }
    
    .stDataFrame > div {
        border-radius: 8px !important;
    }
    
    /* Hide default streamlit styling */
    .stDataFrame [data-testid="stMetricValue"] {
        color: inherit !important;
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

def format_signal_display(signal):
    """Format signal for display with color"""
    if signal == 'LONG':
        return '🟢 LONG'
    elif signal == 'SHORT':
        return '🔴 SHORT'
    elif signal == 'WATCHLIST_LONG':
        return '🔵 WATCHLIST LONG'
    elif signal == 'WATCHLIST_SHORT':
        return '🟠 WATCHLIST SHORT'
    return signal

def color_code_value(val, column):
    """Apply color coding to numeric values"""
    if pd.isna(val):
        return val
    
    if column == 'COMPOSITE_SCORE':
        if val > 0:
            return f'<span style="color: #38a169; font-weight: 600;">{val}</span>'
        elif val < 0:
            return f'<span style="color: #e53e3e; font-weight: 600;">{val}</span>'
    elif column in ['Z_RESIDUAL_BUCKET', 'Cluster_Deviation_Flipped', 'Regression_Component']:
        if val > 0:
            return f'<span style="color: #38a169; font-weight: 600;">{val}</span>'
        elif val < 0:
            return f'<span style="color: #e53e3e; font-weight: 600;">{val}</span>'
    elif column == 'Volatility':
        # Format as percentage
        pct_val = f"{val:.1f}%"
        if val > 2:
            return f'<span style="color: #e53e3e; font-weight: 600;">{pct_val}</span>'
        elif val < 0.5:
            return f'<span style="color: #38a169; font-weight: 600;">{pct_val}</span>'
        else:
            return f'<span style="color: #718096; font-weight: 600;">{pct_val}</span>'
    
    return f'<span style="font-weight: 600;">{val}</span>'

# Load data
df = load_data()

if df.empty:
    st.error("No data available")
    st.stop()

# Add country column
df['Country'] = df['ISIN'].apply(get_country_from_isin)

# Header with bond count
header_col1, header_col2 = st.columns([3, 1])
with header_col1:
    st.markdown('<div class="main-title">📊 Bond Screening Results</div>', unsafe_allow_html=True)

# Signal metrics - horizontal boxes
col1, col2, col3, col4 = st.columns(4)

buy_count = len(df[df['SIGNAL'] == 'LONG'])
sell_count = len(df[df['SIGNAL'] == 'SHORT'])
watch_buy_count = len(df[df['SIGNAL'] == 'WATCHLIST_LONG'])
watch_sell_count = len(df[df['SIGNAL'] == 'WATCHLIST_SHORT'])

with col1:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value buy">{buy_count}</div>
        <div class="metric-label">Long</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value sell">{sell_count}</div>
        <div class="metric-label">Short</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value watch-buy">{watch_buy_count}</div>
        <div class="metric-label">Watchlist Long</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-box">
        <div class="metric-value watch-sell">{watch_sell_count}</div>
        <div class="metric-label">Watchlist Short</div>
    </div>
    """, unsafe_allow_html=True)

# Horizontal filters in a styled container
st.markdown('<div class="filter-container">', unsafe_allow_html=True)
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

st.markdown('</div>', unsafe_allow_html=True)

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

# Show filtered count
st.markdown(f'<div class="bond-count">📊 {len(filtered_df)} of {len(df)} bonds</div>', unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# Data table
if not filtered_df.empty:
    # Prepare display dataframe
    display_df = filtered_df[[
        'ISIN', 'SECURITY_NAME', 'SIGNAL', 
        'COMPOSITE_SCORE', 'Volatility', 'Z_RESIDUAL_BUCKET', 
        'Cluster_Deviation_Flipped', 'Regression_Component'
    ]].copy()
    
    # Round numeric columns
    display_df['COMPOSITE_SCORE'] = display_df['COMPOSITE_SCORE'].round(1)
    display_df['Z_RESIDUAL_BUCKET'] = display_df['Z_RESIDUAL_BUCKET'].round(2)
    display_df['Cluster_Deviation_Flipped'] = display_df['Cluster_Deviation_Flipped'].round(2)
    display_df['Volatility'] = display_df['Volatility'].round(1)
    display_df['Regression_Component'] = display_df['Regression_Component'].round(2)
    
    # Display the table with professional styling
    st.dataframe(
        display_df,
        column_config={
            'ISIN': st.column_config.TextColumn('ISIN', width='medium'),
            'SECURITY_NAME': st.column_config.TextColumn('Security Name', width='large'),
            'SIGNAL': st.column_config.SelectboxColumn(
                'Signal',
                options=['LONG', 'SHORT', 'WATCHLIST_LONG', 'WATCHLIST_SHORT'],
                width='medium'
            ),
            'COMPOSITE_SCORE': st.column_config.NumberColumn(
                'Composite Score',
                format='%.1f'
            ),
            'Volatility': st.column_config.NumberColumn(
                'Volatility',
                format='%.1f%%'
            ),
            'Z_RESIDUAL_BUCKET': st.column_config.NumberColumn(
                'Z-Residual',
                format='%.2f'
            ),
            'Cluster_Deviation_Flipped': st.column_config.NumberColumn(
                'Cluster Dev',
                format='%.2f'
            ),
            'Regression_Component': st.column_config.NumberColumn(
                'Regression',
                format='%.2f'
            )
        },
        use_container_width=True,
        height=600,
        hide_index=True
    )
    
    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "📥 Download CSV",
            csv,
            f"bond_screening_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
    
    with col2:
        if st.button("🔄 Refresh"):
            st.cache_data.clear()
            st.rerun()

else:
    st.warning("No bonds match your filters")
