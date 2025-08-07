import streamlit as st
import pandas as pd
import numpy as np

# Page config
st.set_page_config(
    page_title="Bond Screening Results",
    page_icon="📊",
    layout="wide"
)

# Professional styling matching the reference
st.markdown("""
<style>
    /* Hide streamlit default elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Main container */
    .main .block-container {
        padding-top: 2rem;
        padding-left: 3rem;
        padding-right: 3rem;
        max-width: none;
    }
    
    /* Header styling */
    .header-container {
        display: flex;
        align-items: center;
        justify-content: space-between;
        margin-bottom: 2rem;
        padding-bottom: 1rem;
        border-bottom: 1px solid #e2e8f0;
    }
    
    .main-title {
        font-size: 1.75rem;
        font-weight: 600;
        color: #1a202c;
        display: flex;
        align-items: center;
        gap: 0.5rem;
        margin: 0;
    }
    
    .bond-count {
        font-size: 0.875rem;
        color: #718096;
        background: #f7fafc;
        padding: 0.5rem 1rem;
        border-radius: 6px;
        border: 1px solid #e2e8f0;
        display: flex;
        align-items: center;
        gap: 0.5rem;
    }
    
    /* Metrics container */
    .metrics-container {
        display: flex;
        gap: 1.5rem;
        margin-bottom: 2rem;
        justify-content: center;
    }
    
    .metric-box {
        background: white;
        padding: 2rem 1.5rem;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        text-align: center;
        min-width: 180px;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
        transition: box-shadow 0.2s;
    }
    
    .metric-box:hover {
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
    }
    
    .metric-value {
        font-size: 2.75rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 0.5rem;
    }
    
    .metric-label {
        font-size: 0.875rem;
        color: #718096;
        font-weight: 500;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-arrow {
        font-size: 1.5rem;
    }
    
    .buy { color: #38a169; }
    .sell { color: #e53e3e; }
    .watch-buy { color: #3182ce; }
    .watch-sell { color: #ed8936; }
    
    /* Filter container */
    .filter-container {
        background: #f8fafc;
        padding: 1.5rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        border: 1px solid #e2e8f0;
    }
    
    .filter-row {
        display: flex;
        gap: 2rem;
        align-items: end;
    }
    
    /* Table container */
    .table-container {
        background: white;
        border-radius: 12px;
        border: 1px solid #e2e8f0;
        overflow: hidden;
        box-shadow: 0 1px 3px rgba(0, 0, 0, 0.05);
    }
    
    .table-header {
        padding: 1rem 1.5rem;
        border-bottom: 1px solid #e2e8f0;
        background: #f8fafc;
        font-weight: 600;
        color: #1a202c;
        font-size: 0.875rem;
    }
    
    /* Custom streamlit overrides */
    .stSelectbox > div > div {
        background: white;
        border-radius: 8px;
    }
    
    .stTextInput > div > div {
        background: white;
        border-radius: 8px;
    }
    
    .stMultiSelect > div > div {
        background: white;
        border-radius: 8px;
    }
    
    .stDataFrame {
        border: none !important;
    }
    
    .stDataFrame > div {
        border: none !important;
        border-radius: 0 !important;
    }
    
    /* Signal pills */
    .signal-pill {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .signal-long {
        background-color: #c6f6d5;
        color: #22543d;
    }
    
    .signal-short {
        background-color: #fed7d7;
        color: #742a2a;
    }
    
    .signal-watchlist-long {
        background-color: #bee3f8;
        color: #2a4365;
    }
    
    .signal-watchlist-short {
        background-color: #feebc8;
        color: #7b341e;
    }
    
    /* Action buttons */
    .action-buttons {
        display: flex;
        gap: 1rem;
        margin-top: 1.5rem;
        justify-content: flex-start;
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
        'IT': 'Italy',
        'ES': 'Spain', 
        'FR': 'France',
        'DE': 'Germany'
    }
    return country_map.get(isin[:2], 'Unknown')

# Load data
df = load_data()

if df.empty:
    st.error("No data available")
    st.stop()

# Add country column
df['Country'] = df['ISIN'].apply(get_country_from_isin)

# Header with title and count
st.markdown(f"""
<div class="header-container">
    <h1 class="main-title">
        <span style="color: #3182ce;">📊</span>
        Bond Screening Results
    </h1>
    <div class="bond-count">
        <span style="color: #3182ce;">📋</span>
        {len(df)} of {len(df)} bonds
    </div>
</div>
""", unsafe_allow_html=True)

# Signal metrics - centered
buy_count = len(df[df['SIGNAL'] == 'LONG'])
sell_count = len(df[df['SIGNAL'] == 'SHORT'])
watch_buy_count = len(df[df['SIGNAL'] == 'WATCHLIST LONG'])
watch_sell_count = len(df[df['SIGNAL'] == 'WATCHLIST SHORT'])

st.markdown(f"""
<div class="metrics-container">
    <div class="metric-box">
        <div class="metric-value buy">
            <span class="metric-arrow">↗</span>
            {buy_count}
        </div>
        <div class="metric-label">Long</div>
    </div>
    
    <div class="metric-box">
        <div class="metric-value sell">
            <span class="metric-arrow">↘</span>
            {sell_count}
        </div>
        <div class="metric-label">Short</div>
    </div>
    
    <div class="metric-box">
        <div class="metric-value watch-buy">
            <span class="metric-arrow">↗</span>
            {watch_buy_count}
        </div>
        <div class="metric-label">Watchlist Long</div>
    </div>
    
    <div class="metric-box">
        <div class="metric-value watch-sell">
            <span class="metric-arrow">↘</span>
            {watch_sell_count}
        </div>
        <div class="metric-label">Watchlist Short</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Filters
st.markdown('<div class="filter-container">', unsafe_allow_html=True)

col1, col2, col3 = st.columns([2, 2, 3])

with col1:
    selected_countries = st.multiselect(
        "Countries",
        options=list(df['Country'].unique()),
        default=list(df['Country'].unique()),
        key="countries"
    )

with col2:
    selected_signals = st.multiselect(
        "Signals",
        options=['LONG', 'SHORT', 'WATCHLIST_LONG', 'WATCHLIST_SHORT'],
        default=['LONG', 'SHORT', 'WATCHLIST_LONG', 'WATCHLIST_SHORT'],
        key="signals"
    )

with col3:
    search_term = st.text_input(
        "Search ISIN or Security Name", 
        placeholder="Type to search...",
        key="search"
    )

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

# Update bond count after filtering
if len(filtered_df) != len(df):
    st.markdown(f"""
    <div style="text-align: center; margin: 1rem 0;">
        <span class="bond-count">
            Showing {len(filtered_df)} of {len(df)} bonds
        </span>
    </div>
    """, unsafe_allow_html=True)

# Data table
if not filtered_df.empty:
    st.markdown('<div class="table-container">', unsafe_allow_html=True)
    
    # Prepare display dataframe
    display_df = filtered_df[[
        'ISIN', 'SECURITY_NAME', 'SIGNAL', 
        'COMPOSITE_SCORE', 'Volatility', 'Z_RESIDUAL_BUCKET', 
        'Cluster_Deviation_Flipped', 'Regression_Component'
    ]].copy()
    
    # Round and format numeric columns
    display_df['COMPOSITE_SCORE'] = display_df['COMPOSITE_SCORE'].round(1)
    display_df['Z_RESIDUAL_BUCKET'] = display_df['Z_RESIDUAL_BUCKET'].round(2)
    display_df['Cluster_Deviation_Flipped'] = display_df['Cluster_Deviation_Flipped'].round(2)
    display_df['Volatility'] = display_df['Volatility'].round(1)
    display_df['Regression_Component'] = display_df['Regression_Component'].round(2)
    
    # Display the table
    st.dataframe(
        display_df,
        column_config={
            'ISIN': st.column_config.TextColumn('ISIN', width='medium'),
            'SECURITY_NAME': st.column_config.TextColumn('Security Name', width='large'),
            'SIGNAL': st.column_config.SelectboxColumn(
                'Signal',
                options=['LONG', 'SHORT', '', 'WATCHLIST_SHORT'],
                width='medium'
            ),
            'COMPOSITE_SCORE': st.column_config.NumberColumn(
                'Composite Score',
                format='%.1f',
                width='medium'
            ),
            'Volatility': st.column_config.NumberColumn(
                'Volatility',
                format='%.1f%%',
                width='small'
            ),
            'Z_RESIDUAL_BUCKET': st.column_config.NumberColumn(
                'Z-Residual',
                format='%.2f',
                width='medium'
            ),
            'Cluster_Deviation_Flipped': st.column_config.NumberColumn(
                'Cluster Dev',
                format='%.2f',
                width='medium'
            ),
            'Regression_Component': st.column_config.NumberColumn(
                'Regression',
                format='%.2f',
                width='medium'
            )
        },
        use_container_width=True,
        height=600,
        hide_index=True
    )
    
    st.markdown('</div>', unsafe_allow_html=True)
    
    # Action buttons
    st.markdown('<div class="action-buttons">', unsafe_allow_html=True)
    col1, col2, col3 = st.columns([1, 1, 4])
    
    with col1:
        csv = filtered_df.to_csv(index=False)
        st.download_button(
            "Download CSV",
            csv,
            f"bond_screening_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv",
            "text/csv"
        )
    
    with col2:
        if st.button("Refresh Data"):
            st.cache_data.clear()
            st.rerun()
    
    st.markdown('</div>', unsafe_allow_html=True)

else:
    st.warning("No bonds match your filters")
