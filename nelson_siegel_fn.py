import numpy as np
import pandas as pd
import plotly.graph_objects as go

def nelson_siegel(t, beta0, beta1, beta2, tau):
    t = np.array(t)
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = (1 - np.exp(-t / tau)) / (t / tau)
        term2 = term1 - np.exp(-t / tau)
        return beta0 + beta1 * term1 + beta2 * term2

def plot_ns_animation(ns_df, issuer_label, highlight_isins=None):
    # Define color + size mapping for signals
    signal_style_map = {
        'strong buy':   {"color": "darkgreen", "size": 10},
        'moderate buy': {"color": "lightgreen", "size": 9},
        'strong sell':  {"color": "darkred", "size": 10},
        'moderate sell':{"color": "orange", "size": 9},
        'weak buy':     {"color": "black", "size": 6},
        'weak sell':    {"color": "black", "size": 6},
        'no action':    {"color": "black", "size": 6}
    }

    # Apply mapping (fallback: black, small)
    ns_df['Signal_Color'] = ns_df['SIGNAL'].map(
        lambda s: signal_style_map.get(str(s).lower(), {"color": "black", "size": 6})["color"]
    )
    ns_df['Signal_Size'] = ns_df['SIGNAL'].map(
        lambda s: signal_style_map.get(str(s).lower(), {"color": "black", "size": 6})["size"]
    )

    fig = go.Figure()

    # Scatter plot with animation frames (by Date)
    fig.add_trace(go.Scatter(
        x=ns_df['YTM'],
        y=ns_df['Z_SPRD_VAL'],
        mode='markers',
        marker=dict(
            size=ns_df['Signal_Size'],
            color=ns_df['Signal_Color'],
            opacity=0.8,
            line=dict(width=0.5, color='white')
        ),
        text=ns_df['SECURITY_NAME'],
        customdata=np.stack((
            ns_df['ISIN'],
            ns_df['Date'].astype(str),
            ns_df.get('RESIDUAL_NS', np.zeros(len(ns_df)))
        ), axis=-1),
        hovertemplate=(
            'Years to Maturity: %{x:.2f}<br>'
            'Z-Spread: %{y:.1f}bps<br>'
            'Residual: %{customdata[2]:.2f}bps<br>'
            'Signal: %{marker.color}<br>'
            '%{text}<extra></extra>'
        ),
        showlegend=False,
        animation_frame=ns_df['Date'],
    ))

    fig.update_layout(
        title=f"Nelson-Siegel Curve Animation — {issuer_label}",
        xaxis_title="Years to Maturity",
        yaxis_title="Z-Spread (bps)",
        legend_title="Signal",
        template="plotly_white"
    )

    return fig
