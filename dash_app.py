import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
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
import boto3
from scipy.interpolate import interp1d
import uuid

# -------------------
# B2 Configuration
# -------------------
B2_KEY_ID = os.getenv("B2_KEY_ID")
B2_APP_KEY = os.getenv("B2_APP_KEY")
BUCKET_NAME = "Celestial-Signal"
LOCAL_ZIP = "ns_curves_20250809.zip"
LOCAL_FOLDER = "ns_curves"

# Global cache for data
cache = {}

# -------------------
# Download from B2
# -------------------
def download_from_b2(file_key: str, local_path: str, force: bool = False):
    """Download a file from B2 bucket."""
    if not force and os.path.exists(local_path):
        return local_path

    s3 = boto3.client(
        "s3",
        endpoint_url="https://s3.eu-central-003.backblazeb2.com", 
        aws_access_key_id=B2_KEY_ID,
        aws_secret_access_key=B2_APP_KEY
    )
    s3.download_file(BUCKET_NAME, file_key, local_path)
    return local_path

# -------------------
# Compute MD5 hash
# -------------------
def file_hash(filepath: str) -> str:
    """Compute MD5 hash of a file"""
    hasher = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            hasher.update(chunk)
    return hasher.hexdigest()

# -------------------
# Unzip NS curves
# -------------------
def unzip_ns_curves(zip_path: str = LOCAL_ZIP, folder: str = LOCAL_FOLDER, force: bool = False) -> tuple[str, str]:
    """Unzip NS curves from B2 and return (folder, zip_hash)."""
    zip_path = download_from_b2(file_key="ns_curves_0809.zip", local_path=zip_path, force=force)
    zip_hash = file_hash(zip_path)
    prev_hash = cache.get("ns_zip_hash")

    if force or prev_hash != zip_hash or not os.path.exists(folder):
        if os.path.exists(folder):
            shutil.rmtree(folder)
        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            zip_ref.extractall(folder)
        cache["ns_zip_hash"] = zip_hash

    return folder, zip_hash

# -------------------
# Load full NS DF
# -------------------
def load_full_ns_df(country_code: str, zip_hash: str) -> pd.DataFrame:
    """Load all NS curves for a country."""
    cache_key = f"ns_df_{country_code}_{zip_hash}"
    if cache_key in cache:
        return cache[cache_key]
    
    folder, _ = unzip_ns_curves(force=True)
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
            print(f"Error loading file {f}: {e}")
            continue

    if dfs:
        ns_df = pd.concat(dfs, ignore_index=True)

        if 'RESIDUAL' in ns_df.columns:
            ns_df.rename(columns={'RESIDUAL': 'RESIDUAL_NS'}, inplace=True)
        if 'RESIDUAL_NS' not in ns_df.columns:
            ns_df['RESIDUAL_NS'] = pd.NA

        if "Date" in ns_df.columns:
            ns_df["Date"] = pd.to_datetime(ns_df["Date"])
            ns_df.sort_values("Date", inplace=True)

        if "Country" in ns_df.columns:
            print("Unique country codes in NS DF:", ns_df['Country'].unique())

        cache[cache_key] = ns_df
        return ns_df

    print(f"No parquet files found for country code '{country_code}' in folder '{folder}'.")
    return pd.DataFrame()

# -------------------
# Load NS curve for a specific date
# -------------------
def load_ns_curve(country_code: str, date_str: str, zip_hash: str) -> pd.DataFrame | None:
    """Load NS curve for a single day from the full dataset."""
    df = load_full_ns_df(country_code, zip_hash=zip_hash)

    if df is not None and not df.empty:
        df = df[df["Date"] == pd.to_datetime(date_str)]
        if not df.empty:
            return df

    return None

# -------------------
# Load data function
# -------------------
def load_data(force: bool = False) -> pd.DataFrame:
    """Load issuer_signals.csv from B2 with caching."""
    cache_key = "signals_data"
    if not force and cache_key in cache:
        return cache[cache_key]
    
    local_path = "issuer_signals.csv"
    file_key = "issuer_signals.csv"

    try:
        local_path = download_from_b2(file_key=file_key, local_path=local_path, force=force)
        df = pd.read_csv(local_path)
        cache[cache_key] = df
        return df
    except Exception as e:
        print(f"Error loading data from B2: {e}")
        if os.path.exists(local_path):
            try:
                df = pd.read_csv(local_path)
                cache[cache_key] = df
                return df
            except Exception as e2:
                print(f"Error loading local CSV: {e2}")
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

# Initialize Dash app
app = dash.Dash(__name__)
app.title = "Bond Analytics Dashboard"

# Custom CSS
app.index_string = '''
<!DOCTYPE html>
<html>
    <head>
        {%metas%}
        <title>{%title%}</title>
        {%favicon%}
        {%css%}
        <style>
            .metric-box {
                background: white;
                padding: 1.5rem;
                border-radius: 8px;
                border: 1px solid #e1e5e9;
                text-align: center;
                margin: 0.5rem;
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
            .mod-buy { color: #20c997; }
            .mod-sell { color: #e55353; }
            .watch-buy { color: #17a2b8; }
            .watch-sell { color: #fd7e14; }
            .no-action { color: #6c757d; }
            .tab-content {
                padding: 20px;
            }
            .filters-row {
                display: flex;
                gap: 20px;
                margin-bottom: 20px;
                align-items: end;
            }
            .filter-item {
                flex: 1;
            }
        </style>
    </head>
    <body>
        {%app_entry%}
        <footer>
            {%config%}
            {%scripts%}
            {%renderer%}
        </footer>
    </body>
</html>
'''

# App layout
app.layout = html.Div([
    dcc.Tabs(id="main-tabs", value='tab-1', children=[
        dcc.Tab(label='Nelson-Siegel Curves', value='tab-1'),
        dcc.Tab(label='Signal Dashboard', value='tab-2'),
        dcc.Tab(label='Analysis', value='tab-3'),
    ]),
    html.Div(id='tabs-content')
])

# Callback for tab content
@app.callback(Output('tabs-content', 'children'),
              Input('main-tabs', 'value'))
def render_content(active_tab):
    if active_tab == 'tab-1':
        return render_nelson_siegel_tab()
    elif active_tab == 'tab-2':
        return render_signal_dashboard_tab()
    elif active_tab == 'tab-3':
        return render_analysis_tab()

def render_signal_dashboard_tab():
    # Load data
    df = load_data()
    if df.empty:
        return html.Div("No data available", style={'color': 'red', 'fontSize': '24px'})
    
    # Add country column
    df['Country'] = df['ISIN'].apply(get_country_from_isin)
    
    # Count signals
    buy_count = len(df[df['SIGNAL'] == 'STRONG BUY'])
    sell_count = len(df[df['SIGNAL'] == 'STRONG SELL'])
    mod_buy_count = len(df[df['SIGNAL'] == 'MODERATE BUY'])
    mod_sell_count = len(df[df['SIGNAL'] == 'MODERATE SELL'])
    watch_buy_count = len(df[df['SIGNAL'] == 'WEAK BUY'])
    watch_sell_count = len(df[df['SIGNAL'] == 'WEAK SELL'])
    no_action_count = len(df[df['SIGNAL'] == 'NO ACTION'])
    
    return html.Div([
        html.H1("Bond Analytics Dashboard", style={'textAlign': 'center', 'marginBottom': '30px'}),
        
        # Metrics row
        html.Div([
            html.Div([
                html.Div(str(buy_count), className="metric-value buy"),
                html.Div("STRONG BUY", className="metric-label")
            ], className="metric-box", style={'width': '14%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div(str(sell_count), className="metric-value sell"),
                html.Div("STRONG SELL", className="metric-label")
            ], className="metric-box", style={'width': '14%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div(str(mod_buy_count), className="metric-value mod-buy"),
                html.Div("MODERATE BUY", className="metric-label")
            ], className="metric-box", style={'width': '14%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div(str(mod_sell_count), className="metric-value mod-sell"),
                html.Div("MODERATE SELL", className="metric-label")
            ], className="metric-box", style={'width': '14%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div(str(watch_buy_count), className="metric-value watch-buy"),
                html.Div("WEAK BUY", className="metric-label")
            ], className="metric-box", style={'width': '14%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div(str(watch_sell_count), className="metric-value watch-sell"),
                html.Div("WEAK SELL", className="metric-label")
            ], className="metric-box", style={'width': '14%', 'display': 'inline-block'}),
            
            html.Div([
                html.Div(str(no_action_count), className="metric-value no-action"),
                html.Div("NO ACTION", className="metric-label")
            ], className="metric-box", style={'width': '14%', 'display': 'inline-block'}),
        ], style={'display': 'flex', 'justifyContent': 'space-around', 'marginBottom': '30px'}),
        
        html.Hr(),
        
        # Filters
        html.Div([
            html.Div([
                html.Label("Countries"),
                dcc.Dropdown(
                    id='country-filter',
                    options=[{'label': c, 'value': c} for c in df['Country'].unique()],
                    value=list(df['Country'].unique()),
                    multi=True
                )
            ], className="filter-item"),
            
            html.Div([
                html.Label("Signals"),
                dcc.Dropdown(
                    id='signal-filter',
                    options=[
                        {'label': 'STRONG BUY', 'value': 'STRONG BUY'},
                        {'label': 'STRONG SELL', 'value': 'STRONG SELL'},
                        {'label': 'MODERATE BUY', 'value': 'MODERATE BUY'},
                        {'label': 'MODERATE SELL', 'value': 'MODERATE SELL'},
                        {'label': 'WEAK BUY', 'value': 'WEAK BUY'},
                        {'label': 'WEAK SELL', 'value': 'WEAK SELL'},
                        {'label': 'NO ACTION', 'value': 'NO ACTION'}
                    ],
                    value=['STRONG BUY', 'STRONG SELL', 'MODERATE BUY', 'MODERATE SELL', 'WEAK BUY', 'WEAK SELL'],
                    multi=True
                )
            ], className="filter-item"),
            
            html.Div([
                html.Label("Search ISIN or Security Name"),
                dcc.Input(
                    id='search-input',
                    type='text',
                    placeholder='Type to search...',
                    style={'width': '100%'}
                )
            ], className="filter-item"),
        ], className="filters-row"),
        
        html.Hr(),
        
        # Data table
        html.Div([
            html.H3(id="table-title"),
            html.Div(id="bond-table"),
            
            html.Br(),
            html.Div([
                html.Button("Download CSV", id="download-btn", n_clicks=0),
                html.Button("Refresh Data", id="refresh-btn", n_clicks=0, style={'marginLeft': '10px'}),
                dcc.Download(id="download-csv")
            ])
        ])
    ])

def render_nelson_siegel_tab():
    return html.Div([
        dcc.RadioItems(
            id='ns-subtab',
            options=[
                {'label': 'Single Day Curve', 'value': 'single'},
                {'label': 'Animated Curves', 'value': 'animated'},
                {'label': 'Residuals Analysis', 'value': 'residuals'},
                {'label': 'Compare NS Curves', 'value': 'compare'},
                {'label': 'New Bond Prediction', 'value': 'prediction'}
            ],
            value='single',
            labelStyle={'display': 'inline-block', 'marginRight': '20px'}
        ),
        html.Div(id='ns-subtab-content')
    ])

def render_analysis_tab():
    return html.Div([
        html.H2("Bond Pair Residual Curve Comparison"),
        
        html.Div([
            html.Label("Select Country"),
            dcc.Dropdown(
                id='analysis-country-dropdown',
                options=[
                    {'label': 'Italy 🇮🇹', 'value': 'BTPS'},
                    {'label': 'Spain 🇪🇸', 'value': 'SPGB'},
                    {'label': 'France 🇫🇷', 'value': 'FRTR'},
                    {'label': 'Germany 🇩🇪', 'value': 'BUNDS'},
                    {'label': 'Finland 🇫🇮', 'value': 'RFGB'},
                    {'label': 'EU 🇪🇺', 'value': 'EU'},
                    {'label': 'Austria 🇦🇹', 'value': 'RAGB'},
                    {'label': 'Netherlands 🇳🇱', 'value': 'NETHER'},
                    {'label': 'Belgium 🇧🇪', 'value': 'BGB'}
                ],
                value='BTPS'
            )
        ], style={'width': '30%', 'marginBottom': '20px'}),
        
        html.Div([
            html.H4("Curve A (Issuer 1)"),
            html.Div([
                html.Div([
                    html.Label("Select Bond A1"),
                    dcc.Dropdown(id='bond-a1-dropdown')
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Select Bond A2"),
                    dcc.Dropdown(id='bond-a2-dropdown')
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
            ])
        ], style={'marginBottom': '20px'}),
        
        html.Div([
            html.H4("Curve B (Issuer 2)"),
            html.Div([
                html.Div([
                    html.Label("Select Bond B1"),
                    dcc.Dropdown(id='bond-b1-dropdown')
                ], style={'width': '48%', 'display': 'inline-block'}),
                html.Div([
                    html.Label("Select Bond B2"),
                    dcc.Dropdown(id='bond-b2-dropdown')
                ], style={'width': '48%', 'display': 'inline-block', 'marginLeft': '4%'})
            ])
        ], style={'marginBottom': '20px'}),
        
        dcc.Checklist(
            id='show-diff-checkbox',
            options=[{'label': 'Show difference between curves', 'value': 'show_diff'}],
            value=[]
        ),
        
        dcc.Graph(id='pair-comparison-chart')
    ])

# Callbacks for Signal Dashboard
@app.callback(
    [Output('bond-table', 'children'),
     Output('table-title', 'children')],
    [Input('country-filter', 'value'),
     Input('signal-filter', 'value'),
     Input('search-input', 'value'),
     Input('refresh-btn', 'n_clicks')]
)
def update_bond_table(selected_countries, selected_signals, search_term, refresh_clicks):
    # Load data (refresh if button clicked)
    force_refresh = refresh_clicks > 0 if refresh_clicks else False
    df = load_data(force=force_refresh)
    
    if df.empty:
        return html.Div("No data available"), "Bond Data (0 bonds)"
    
    # Add country column
    df['Country'] = df['ISIN'].apply(get_country_from_isin)
    
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
    
    # Prepare display data
    cols_to_display = [
        'SECURITY_NAME', 'RESIDUAL_NS', 'SIGNAL',
        'Z_Residual_Score', 'Volatility_Score', 'Market_Stress_Score',
        'Cluster_Score', 'Regression_Score', 'COMPOSITE_SCORE',
        'Top_Features', 'Top_Feature_Effects_Pct'
    ]
    
    existing_cols = [col for col in cols_to_display if col in filtered_df.columns]
    display_df = filtered_df[existing_cols].copy()
    
    # Rename columns
    display_df.rename(columns={
        'RESIDUAL_NS': 'Residual',
        'Volatility_Score': 'Stability_Score',
        'SIGNAL': 'Signal'
    }, inplace=True)
    
    # Convert numeric columns
    numeric_cols = ['Residual', 'Z_Residual_Score', 'Stability_Score',
                    'Market_Stress_Score', 'Cluster_Score', 'Regression_Score', 'COMPOSITE_SCORE']
    for col in numeric_cols:
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors='coerce')
    
    if "Stability_Score" in display_df.columns:
        display_df["Stability_Score"] = display_df["Stability_Score"] * 100
    
    # Extract maturity
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
    display_df['Maturity'] = display_df['Maturity'].dt.strftime("%Y-%m-%d")
    display_df['Maturity'] = display_df['Maturity'].fillna("N/A")
    
    # Process features
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
    
    # Create table
    columns = [{"name": col, "id": col, "type": "numeric" if col in numeric_cols else "text"} 
               for col in display_df.columns]
    
    # Format numeric columns
    for col in columns:
        if col["id"] in numeric_cols:
            if col["id"] == "Stability_Score":
                col["format"] = {"specifier": ".2%"}
            else:
                col["format"] = {"specifier": ".4f"}
    
    table = dash_table.DataTable(
        data=display_df.to_dict('records'),
        columns=columns,
        page_size=20,
        sort_action="native",
        filter_action="native",
        style_table={'overflowX': 'auto'},
        style_cell={
            'textAlign': 'left',
            'padding': '10px',
            'fontFamily': 'Arial'
        },
        style_header={
            'backgroundColor': 'rgb(230, 230, 230)',
            'fontWeight': 'bold'
        }
    )
    
    title = f"Bond Data ({len(filtered_df)} bonds)"
    
    return table, title

@app.callback(
    Output("download-csv", "data"),
    Input("download-btn", "n_clicks"),
    [State('country-filter', 'value'),
     State('signal-filter', 'value'),
     State('search-input', 'value')],
    prevent_initial_call=True
)
def download_csv(n_clicks, selected_countries, selected_signals, search_term):
    if n_clicks > 0:
        df = load_data()
        df['Country'] = df['ISIN'].apply(get_country_from_isin)
        
        # Apply same filters
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
        
        return dcc.send_data_frame(filtered_df.to_csv, f"bonds_{pd.Timestamp.now().strftime('%Y%m%d_%H%M')}.csv")

# Callback for NS subtabs
@app.callback(Output('ns-subtab-content', 'children'),
              Input('ns-subtab', 'value'))
def render_ns_subtab(selected_subtab):
    if selected_subtab == 'single':
        return render_single_day_curve()
    elif selected_subtab == 'animated':
        return render_animated_curves()
    elif selected_subtab == 'residuals':
        return render_residuals_analysis()
    elif selected_subtab == 'compare':
        return render_compare_curves()
    elif selected_subtab == 'prediction':
        return render_new_bond_prediction()

def render_single_day_curve():
    # Get available dates
    final_signal_df = pd.read_csv("today_all_signals.csv")
    available_dates = pd.to_datetime(final_signal_df['Date'].unique())
    default_date = available_dates.max().strftime('%Y-%m-%d')
    
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Select Country"),
                dcc.Dropdown(
                    id='single-country-dropdown',
                    options=[
                        {'label': 'Italy 🇮🇹', 'value': 'BTPS'},
                        {'label': 'Spain 🇪🇸', 'value': 'SPGB'},
                        {'label': 'France 🇫🇷', 'value': 'FRTR'},
                        {'label': 'Germany 🇩🇪', 'value': 'BUNDS'},
                        {'label': 'Finland 🇫🇮', 'value': 'RFGB'},
                        {'label': 'EU 🇪🇺', 'value': 'EU'},
                        {'label': 'Austria 🇦🇹', 'value': 'RAGB'},
                        {'label': 'Netherlands 🇳🇱', 'value': 'NETHER'},
                        {'label': 'Belgium 🇧🇪', 'value': 'BGB'}
                    ],
                    value='BTPS'
                )
            ], style={'width': '30%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label("Select Date"),
                dcc.DatePickerSingle(
                    id='single-date-picker',
                    date=default_date,
                    display_format='YYYY-MM-DD'
                )
            ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'})
        ]),
        
        html.Div([
            html.Div([
                dcc.Graph(id='single-day-chart')
            ], style={'width': '70%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label("Select Bond for AI Explanation"),
                dcc.Dropdown(id='bond-selector'),
                html.Div(id='selected-bond-info'),
                html.Button("Explain this bond", id='explain-btn', n_clicks=0),
                html.Div(id='ai-explanation')
            ], style={'width': '28%', 'display': 'inline-block', 'marginLeft': '2%', 'verticalAlign': 'top'})
        ])
    ])

def render_animated_curves():
    return html.Div([
        html.Div([
            html.Label("Select Country"),
            dcc.Dropdown(
                id='animated-country-dropdown',
                options=[
                    {'label': 'Italy 🇮🇹', 'value': 'BTPS'},
                    {'label': 'Spain 🇪🇸', 'value': 'SPGB'},
                    {'label': 'France 🇫🇷', 'value': 'FRTR'},
                    {'label': 'Germany 🇩🇪', 'value': 'BUNDS'},
                    {'label': 'Finland 🇫🇮', 'value': 'RFGB'},
                    {'label': 'EU 🇪🇺', 'value': 'EU'},
                    {'label': 'Austria 🇦🇹', 'value': 'RAGB'},
                    {'label': 'Netherlands 🇳🇱', 'value': 'NETHER'},
                    {'label': 'Belgium 🇧🇪', 'value': 'BGB'}
                ],
                value='BTPS'
            )
        ], style={'width': '30%', 'marginBottom': '20px'}),
        
        html.Div([
            html.Label("Select Bonds to Display in Animation"),
            dcc.Dropdown(
                id='animated-bonds-dropdown',
                multi=True,
                placeholder="Select bonds..."
            )
        ], style={'marginBottom': '20px'}),
        
        html.Button("Select All Bonds", id='select-all-animated-btn', n_clicks=0),
        
        dcc.Graph(id='animated-curves-chart')
    ])

def render_residuals_analysis():
    return html.Div([
        html.Div([
            html.Label("Select Country"),
            dcc.Dropdown(
                id='residuals-country-dropdown',
                options=[
                    {'label': 'Italy 🇮🇹', 'value': 'BTPS'},
                    {'label': 'Spain 🇪🇸', 'value': 'SPGB'},
                    {'label': 'France 🇫🇷', 'value': 'FRTR'},
                    {'label': 'Germany 🇩🇪', 'value': 'BUNDS'},
                    {'label': 'Finland 🇫🇮', 'value': 'RFGB'},
                    {'label': 'EU 🇪🇺', 'value': 'EU'},
                    {'label': 'Austria 🇦🇹', 'value': 'RAGB'},
                    {'label': 'Netherlands 🇳🇱', 'value': 'NETHER'},
                    {'label': 'Belgium 🇧🇪', 'value': 'BGB'}
                ],
                value='BTPS'
            )
        ], style={'width': '30%', 'marginBottom': '20px'}),
        
        html.Div([
            html.Label("Select Bonds for Residual Analysis"),
            dcc.Dropdown(
                id='residuals-bonds-dropdown',
                multi=True,
                placeholder="Select bonds..."
            )
        ], style={'marginBottom': '20px'}),
        
        dcc.Graph(id='residuals-chart'),
        dcc.Graph(id='velocity-chart')
    ])

def render_compare_curves():
    return html.Div([
        html.Div([
            html.Label("Select Countries"),
            dcc.Dropdown(
                id='compare-countries-dropdown',
                options=[
                    {'label': 'Italy 🇮🇹', 'value': 'BTPS'},
                    {'label': 'Spain 🇪🇸', 'value': 'SPGB'},
                    {'label': 'France 🇫🇷', 'value': 'FRTR'},
                    {'label': 'Germany 🇩🇪', 'value': 'BUNDS'},
                    {'label': 'Finland 🇫🇮', 'value': 'RFGB'},
                    {'label': 'EU 🇪🇺', 'value': 'EU'},
                    {'label': 'Austria 🇦🇹', 'value': 'RAGB'},
                    {'label': 'Netherlands 🇳🇱', 'value': 'NETHER'},
                    {'label': 'Belgium 🇧🇪', 'value': 'BGB'}
                ],
                multi=True,
                value=['BTPS']
            )
        ], style={'width': '50%', 'marginBottom': '20px'}),
        
        html.Div(id='compare-date-selectors'),
        
        dcc.Graph(id='compare-curves-chart')
    ])

def render_new_bond_prediction():
    return html.Div([
        html.Div([
            html.Div([
                html.Label("Select Country"),
                dcc.Dropdown(
                    id='prediction-country-dropdown',
                    options=[
                        {'label': 'Italy 🇮🇹', 'value': 'BTPS'},
                        {'label': 'Spain 🇪🇸', 'value': 'SPGB'},
                        {'label': 'France 🇫🇷', 'value': 'FRTR'},
                        {'label': 'Germany 🇩🇪', 'value': 'BUNDS'},
                        {'label': 'Finland 🇫🇮', 'value': 'RFGB'},
                        {'label': 'EU 🇪🇺', 'value': 'EU'},
                        {'label': 'Austria 🇦🇹', 'value': 'RAGB'},
                        {'label': 'Netherlands 🇳🇱', 'value': 'NETHER'},
                        {'label': 'Belgium 🇧🇪', 'value': 'BGB'}
                    ],
                    value='BTPS'
                )
            ], style={'width': '30%', 'display': 'inline-block'}),
            
            html.Div([
                html.Label("Enter New Bond Maturity (MM/YY)"),
                dcc.Input(
                    id='new-bond-maturity-input',
                    type='text',
                    value='10/55',
                    placeholder='MM/YY'
                )
            ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'}),
            
            html.Div([
                html.Label("Auction concession (bps)"),
                dcc.Input(
                    id='auction-concession-input',
                    type='number',
                    value=0,
                    step=1
                )
            ], style={'width': '30%', 'display': 'inline-block', 'marginLeft': '5%'})
        ], style={'marginBottom': '20px'}),
        
        dcc.Graph(id='prediction-chart')
    ])

# Callbacks for Nelson-Siegel subtabs

@app.callback(
    [Output('single-day-chart', 'figure'),
     Output('bond-selector', 'options'),
     Output('bond-selector', 'value')],
    [Input('single-country-dropdown', 'value'),
     Input('single-date-picker', 'date')]
)
def update_single_day_chart(country_code, date_str):
    try:
        zip_path = download_from_b2(file_key="ns_curves_0809.zip", local_path=LOCAL_ZIP, force=False)
        zip_hash = file_hash(zip_path)
        
        ns_df = load_ns_curve(country_code, date_str, zip_hash=zip_hash)
        
        if ns_df is None or ns_df.empty:
            return go.Figure().add_annotation(text="No data available", showarrow=False), [], None
        
        # Process data
        col_map = {c.lower(): c for c in ns_df.columns}
        if "z_sprd_val" in col_map:
            ns_df.rename(columns={col_map["z_sprd_val"]: "Z_SPRD"}, inplace=True)
        elif "z_sprd" in col_map:
            ns_df.rename(columns={col_map["z_sprd"]: "Z_SPRD"}, inplace=True)
        
        if "yearstomaturity" in col_map:
            ns_df.rename(columns={col_map["yearstomaturity"]: "YTM"}, inplace=True)
        
        ns_df['Maturity'] = pd.to_datetime(ns_df['Maturity'])
        curve_date = pd.to_datetime(date_str)
        ns_df['YTM'] = (ns_df['Maturity'] - curve_date).dt.days / 365.25
        
        # Load signals
        final_signal_df = pd.read_csv("today_all_signals.csv")
        ns_df = ns_df.merge(final_signal_df[['ISIN', 'SIGNAL']], on='ISIN', how='left')
        
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
        legend_signals = ['strong buy', 'moderate buy', 'strong sell', 'moderate sell']
        
        for signal, df_subset in ns_df.groupby('SIGNAL'):
            if not df_subset.empty:
                color = df_subset['Signal_Color'].iloc[0]
                fig.add_trace(go.Scatter(
                    x=df_subset['YTM'],
                    y=df_subset['Z_SPRD'],
                    mode='markers',
                    name=signal.title() if signal in legend_signals else None,
                    marker=dict(size=6, color=color, symbol='circle'),
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
        
        # Nelson-Siegel fit
        if 'NS_PARAMS' in ns_df.columns:
            try:
                ns_params_raw = ns_df['NS_PARAMS'].iloc[0]
                if isinstance(ns_params_raw, str):
                    ns_params = ast.literal_eval(ns_params_raw)
                else:
                    ns_params = ns_params_raw
                
                maturity_range = np.linspace(ns_df['YTM'].min(), ns_df['YTM'].max(), 100)
                ns_curve = nelson_siegel(maturity_range, *ns_params)
                
                fig.add_trace(go.Scatter(
                    x=maturity_range,
                    y=ns_curve,
                    mode='lines',
                    name='Nelson-Siegel Fit',
                    line=dict(color='deepskyblue', width=3)
                ))
            except Exception as e:
                print(f"Error plotting Nelson-Siegel curve: {e}")
        
        fig.update_layout(
            title=f"Nelson-Siegel Curve for {country_code} on {date_str}",
            xaxis_title="Years to Maturity",
            yaxis_title="Z-Spread (bps)",
            height=700,
            showlegend=True,
            template="plotly_white"
        )
        
        # Bond options for selector
        bond_options = [{'label': f"{row['SECURITY_NAME']} ({row['ISIN']})", 'value': row['ISIN']} 
                       for _, row in final_signal_df[['ISIN', 'SECURITY_NAME']].drop_duplicates().iterrows()]
        default_bond = bond_options[0]['value'] if bond_options else None
        
        return fig, bond_options, default_bond
        
    except Exception as e:
        print(f"Error in single day chart: {e}")
        return go.Figure().add_annotation(text="Error loading data", showarrow=False), [], None

@app.callback(
    [Output('selected-bond-info', 'children'),
     Output('ai-explanation', 'children')],
    [Input('bond-selector', 'value'),
     Input('explain-btn', 'n_clicks')],
    prevent_initial_call=True
)
def update_bond_explanation(selected_isin, n_clicks):
    if not selected_isin:
        return "", ""
    
    try:
        final_signal_df = pd.read_csv("today_all_signals.csv")
        bond_info = final_signal_df[final_signal_df['ISIN'] == selected_isin]
        
        if bond_info.empty:
            return "Bond not found", ""
        
        selected_name = bond_info['SECURITY_NAME'].iloc[0]
        info_text = f"Selected Bond: {selected_name} (ISIN: {selected_isin})"
        
        explanation = ""
        if n_clicks and n_clicks > 0:
            try:
                diagnostics = format_bond_diagnostics(bond_info)
                explanation = html.Div([
                    html.H4(f"AI Explanation for {selected_name}"),
                    html.P(generate_ai_explanation(diagnostics))
                ])
            except Exception as e:
                explanation = f"Error generating explanation: {e}"
        
        return info_text, explanation
        
    except Exception as e:
        return f"Error: {e}", ""

@app.callback(
    [Output('animated-bonds-dropdown', 'options'),
     Output('animated-bonds-dropdown', 'value')],
    [Input('animated-country-dropdown', 'value'),
     Input('select-all-animated-btn', 'n_clicks')],
    [State('animated-bonds-dropdown', 'value')]
)
def update_animated_bonds_options(country_code, select_all_clicks, current_selection):
    try:
        zip_path = download_from_b2(file_key="ns_curves_0809.zip", local_path=LOCAL_ZIP, force=False)
        zip_hash = file_hash(zip_path)
        
        ns_df = load_full_ns_df(country_code, zip_hash=zip_hash)
        if ns_df is None or ns_df.empty:
            return [], []
        
        final_signal_df = pd.read_csv("today_all_signals.csv")
        country_isins = ns_df['ISIN'].unique()
        bond_options = final_signal_df[final_signal_df['ISIN'].isin(country_isins)][['ISIN', 'SECURITY_NAME']].drop_duplicates()
        
        isin_maturity_map = ns_df.groupby('ISIN')['Maturity'].first().to_dict()
        bond_options['Maturity'] = bond_options['ISIN'].map(isin_maturity_map)
        bond_options['Maturity'] = pd.to_datetime(bond_options['Maturity'], errors='coerce')
        bond_options.sort_values('Maturity', inplace=True)
        
        options = []
        for _, row in bond_options.iterrows():
            maturity = row['Maturity']
            if pd.notnull(maturity):
                maturity_str = pd.to_datetime(maturity).strftime('%Y-%m-%d')
                label = f"{row['SECURITY_NAME']} ({maturity_str})"
            else:
                label = f"{row['SECURITY_NAME']} (N/A)"
            options.append({'label': label, 'value': row['ISIN']})
        
        # Handle select all button
        ctx = callback_context
        if ctx.triggered and ctx.triggered[0]['prop_id'] == 'select-all-animated-btn.n_clicks':
            return options, [opt['value'] for opt in options]
        
        return options, current_selection or []
        
    except Exception as e:
        print(f"Error updating animated bonds options: {e}")
        return [], []

@app.callback(
    Output('animated-curves-chart', 'figure'),
    [Input('animated-country-dropdown', 'value'),
     Input('animated-bonds-dropdown', 'value')]
)
def update_animated_curves_chart(country_code, selected_bonds):
    if not selected_bonds:
        return go.Figure().add_annotation(text="Select bonds to display", showarrow=False)
    
    try:
        zip_path = download_from_b2(file_key="ns_curves_0809.zip", local_path=LOCAL_ZIP, force=False)
        zip_hash = file_hash(zip_path)
        
        ns_df = load_full_ns_df(country_code, zip_hash=zip_hash)
        if ns_df is None or ns_df.empty:
            return go.Figure().add_annotation(text="No data available", showarrow=False)
        
        ns_df_filtered = ns_df[ns_df['ISIN'].isin(selected_bonds)].copy()
        final_signal_df = pd.read_csv("today_all_signals.csv")
        ns_df_filtered = ns_df_filtered.merge(final_signal_df[['ISIN', 'SIGNAL']], on='ISIN', how='left')
        
        fig = plot_ns_animation(ns_df_filtered, issuer_label=country_code, highlight_isins=selected_bonds)
        return fig
        
    except Exception as e:
        print(f"Error in animated curves: {e}")
        return go.Figure().add_annotation(text="Error loading data", showarrow=False)

@app.callback(
    [Output('residuals-bonds-dropdown', 'options'),
     Output('residuals-bonds-dropdown', 'value')],
    Input('residuals-country-dropdown', 'value')
)
def update_residuals_bonds_options(country_code):
    try:
        zip_path = download_from_b2(file_key="ns_curves_0809.zip", local_path=LOCAL_ZIP, force=False)
        zip_hash = file_hash(zip_path)
        
        ns_df = load_full_ns_df(country_code, zip_hash=zip_hash)
        if ns_df is None or ns_df.empty:
            return [], []
        
        country_isins = ns_df['ISIN'].unique()
        bond_options = ns_df[ns_df['ISIN'].isin(country_isins)][['ISIN', 'SECURITY_NAME']].drop_duplicates()
        
        isin_maturity_map = ns_df.groupby('ISIN')['Maturity'].first().to_dict()
        bond_options['Maturity'] = bond_options['ISIN'].map(isin_maturity_map)
        bond_options['Maturity'] = pd.to_datetime(bond_options['Maturity'], errors='coerce')
        bond_options.sort_values('Maturity', inplace=True)
        
        options = []
        for _, row in bond_options.iterrows():
            maturity = row['Maturity']
            if pd.notnull(maturity):
                maturity_str = pd.to_datetime(maturity).strftime('%Y-%m-%d')
                label = f"{row['SECURITY_NAME']} ({maturity_str})"
            else:
                label = f"{row['SECURITY_NAME']} (N/A)"
            options.append({'label': label, 'value': row['ISIN']})
        
        return options, []
        
    except Exception as e:
        print(f"Error updating residuals bonds options: {e}")
        return [], []

@app.callback(
    [Output('residuals-chart', 'figure'),
     Output('velocity-chart', 'figure')],
    [Input('residuals-country-dropdown', 'value'),
     Input('residuals-bonds-dropdown', 'value')]
)
def update_residuals_charts(country_code, selected_bonds):
    if not selected_bonds:
        empty_fig = go.Figure().add_annotation(text="Select bonds to display", showarrow=False)
        return empty_fig, empty_fig
    
    try:
        zip_path = download_from_b2(file_key="ns_curves_0809.zip", local_path=LOCAL_ZIP, force=False)
        zip_hash = file_hash(zip_path)
        
        ns_df = load_full_ns_df(country_code, zip_hash=zip_hash)
        if ns_df is None or ns_df.empty:
            empty_fig = go.Figure().add_annotation(text="No data available", showarrow=False)
            return empty_fig, empty_fig
        
        ns_df['Date'] = pd.to_datetime(ns_df['Date']).dt.normalize()
        ns_df['RESIDUAL_VELOCITY'] = ns_df.groupby('ISIN')['RESIDUAL_NS'].transform(lambda x: x.diff())
        
        residuals_df = ns_df[ns_df['ISIN'].isin(selected_bonds)].copy()
        
        bond_labels = dict(zip(residuals_df['ISIN'], residuals_df['SECURITY_NAME']))
        
        fig_residuals = go.Figure()
        fig_velocity = go.Figure()
        
        for isin in selected_bonds:
            bond_data = residuals_df[residuals_df['ISIN'] == isin].sort_values('Date')
            if not bond_data.empty:
                label = bond_labels.get(isin, isin)
                
                fig_residuals.add_trace(go.Scatter(
                    x=bond_data['Date'],
                    y=bond_data['RESIDUAL_NS'],
                    mode='lines+markers',
                    name=label
                ))
                
                fig_velocity.add_trace(go.Scatter(
                    x=bond_data['Date'],
                    y=bond_data['RESIDUAL_VELOCITY'],
                    mode='lines+markers',
                    name=label
                ))
        
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
        
        return fig_residuals, fig_velocity
        
    except Exception as e:
        print(f"Error in residuals charts: {e}")
        empty_fig = go.Figure().add_annotation(text="Error loading data", showarrow=False)
        return empty_fig, empty_fig

@app.callback(
    Output('compare-date-selectors', 'children'),
    Input('compare-countries-dropdown', 'value')
)
def update_compare_date_selectors(selected_countries):
    if not selected_countries:
        return html.Div()
    
    try:
        zip_path = download_from_b2(file_key="ns_curves_0809.zip", local_path=LOCAL_ZIP, force=False)
        zip_hash = file_hash(zip_path)
        
        selectors = []
        for country in selected_countries:
            ns_df_country = load_full_ns_df(country, zip_hash=zip_hash)
            if ns_df_country is not None and not ns_df_country.empty:
                dates = pd.to_datetime(ns_df_country['Date'].unique())
                dates = pd.Series(dates).sort_values(ascending=False)
                formatted_dates = [d.strftime("%Y-%m-%d") for d in dates]
                
                selectors.append(
                    html.Div([
                        html.Label(f"Select Dates for {country}"),
                        dcc.Dropdown(
                            id=f'compare-dates-{country}',
                            options=[{'label': d, 'value': d} for d in formatted_dates],
                            value=[formatted_dates[0]] if formatted_dates else [],
                            multi=True
                        )
                    ], style={'marginBottom': '15px'})
                )
        
        return html.Div(selectors)
        
    except Exception as e:
        print(f"Error updating compare date selectors: {e}")
        return html.Div("Error loading date options")

@app.callback(
    Output('compare-curves-chart', 'figure'),
    [Input('compare-countries-dropdown', 'value')] + 
    [Input(f'compare-dates-{country}', 'value') for country in ['BTPS', 'SPGB', 'FRTR', 'BUNDS', 'RFGB', 'EU', 'RAGB', 'NETHER', 'BGB']]
)
def update_compare_curves_chart(selected_countries, *date_values):
    if not selected_countries:
        return go.Figure().add_annotation(text="Select countries to compare", showarrow=False)
    
    try:
        zip_path = download_from_b2(file_key="ns_curves_0809.zip", local_path=LOCAL_ZIP, force=False)
        zip_hash = file_hash(zip_path)
        
        country_list = ['BTPS', 'SPGB', 'FRTR', 'BUNDS', 'RFGB', 'EU', 'RAGB', 'NETHER', 'BGB']
        date_dict = {country: dates for country, dates in zip(country_list, date_values) if dates}
        
        fig = go.Figure()
        
        for country in selected_countries:
            selected_dates = date_dict.get(country, [])
            for date_str in selected_dates:
                ns_df_curve = load_ns_curve(country, date_str, zip_hash=zip_hash)
                if ns_df_curve is not None and not ns_df_curve.empty and 'NS_PARAMS' in ns_df_curve.columns:
                    try:
                        ns_params_raw = ns_df_curve['NS_PARAMS'].iloc[0]
                        if isinstance(ns_params_raw, str):
                            ns_params = ast.literal_eval(ns_params_raw)
                        else:
                            ns_params = ns_params_raw
                        
                        max_maturity = min(30, ns_df_curve['YTM'].max() if 'YTM' in ns_df_curve.columns else 30)
                        maturities = np.linspace(0, max_maturity, 100)
                        ns_values = nelson_siegel(maturities, *ns_params)
                        
                        fig.add_trace(go.Scatter(
                            x=maturities,
                            y=ns_values,
                            mode='lines',
                            name=f"{country} - {date_str}"
                        ))
                    except Exception as e:
                        print(f"Error processing curve for {country} on {date_str}: {e}")
        
        fig.update_layout(
            title="Nelson-Siegel Curves Comparison",
            xaxis_title="Years to Maturity",
            yaxis_title="Z-Spread (bps)",
            template="plotly_white",
            height=900,
            xaxis=dict(range=[0, 30])
        )
        
        return fig
        
    except Exception as e:
        print(f"Error in compare curves: {e}")
        return go.Figure().add_annotation(text="Error loading data", showarrow=False)

@app.callback(
    Output('prediction-chart', 'figure'),
    [Input('prediction-country-dropdown', 'value'),
     Input('new-bond-maturity-input', 'value'),
     Input('auction-concession-input', 'value')]
)
def update_prediction_chart(country_code, new_bond_input, auction_concession):
    try:
        zip_path = download_from_b2(file_key="ns_curves_0809.zip", local_path=LOCAL_ZIP, force=False)
        zip_hash = file_hash(zip_path)
        
        # Load NS curves for last 2 weeks
        today = pd.Timestamp.today().normalize()
        start_date = today - pd.Timedelta(days=14)
        ns_df_list = []
        
        date_range = pd.date_range(start_date, today)
        for d in date_range:
            ns_tmp = load_ns_curve(country_code, d.strftime("%Y-%m-%d"), zip_hash=zip_hash)
            if ns_tmp is not None and not ns_tmp.empty:
                ns_tmp['Date'] = pd.to_datetime(d)
                ns_tmp['YearsToMaturity'] = (pd.to_datetime(ns_tmp['Maturity']) - today).dt.days / 365.25
                ns_df_list.append(ns_tmp)
        
        if not ns_df_list:
            return go.Figure().add_annotation(text="No NS curve data for the last 2 weeks", showarrow=False)
        
        ns_full = pd.concat(ns_df_list, ignore_index=True)
        
        # Smooth NS: mean Z-spread per YearsToMaturity
        ns_smooth = ns_full.groupby('YearsToMaturity')['Z_SPRD_VAL'].mean().reset_index()
        ns_std = ns_full.groupby('YearsToMaturity')['Z_SPRD_VAL'].std().reset_index()
        
        # Parse new bond maturity
        try:
            month, year = map(int, new_bond_input.split('/'))
            year += 2000 if year < 100 else 0
            new_maturity_date = pd.Timestamp(year=year, month=month, day=1)
            new_years_to_maturity = (new_maturity_
