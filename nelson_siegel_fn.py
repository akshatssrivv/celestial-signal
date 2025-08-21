import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

def nelson_siegel(t, beta0, beta1, beta2, tau):
    t = np.array(t)
    with np.errstate(divide='ignore', invalid='ignore'):
        term1 = (1 - np.exp(-t / tau)) / (t / tau)
        term2 = term1 - np.exp(-t / tau)
        return beta0 + beta1 * term1 + beta2 * term2

def plot_ns_animation(
    ns_df,
    issuer_label="Issuer",
    resid_threshold=20,
    ytm_range=np.linspace(0.1, 50, 200),
    template="plotly_dark",
    highlight_isins=None  # NEW: list of bonds to highlight
):
    highlight_isins = highlight_isins or []

    # Map SIGNAL to colors
    signal_color_map = {
        'STRONG BUY': 'green',
        'MODERATE BUY': 'lightgreen',
        'STRONG SELL': 'red',
        'MODERATE SELL': 'salmon',
        None: 'black'  # default
    }

    dates = sorted(ns_df['Date'].unique())
    if not dates:
        print(f"[{issuer_label}] No available dates to animate.")
        return

    fig = go.Figure()

    # Helper function for marker color/size
    def get_marker_style(row):
        isin = row['ISIN']
        signal = row.get('SIGNAL', None)
        color = signal_color_map.get(signal, 'black')
        size = 10 if isin in highlight_isins else 6
        symbol = 'diamond' if row['RESIDUAL_NS'] >= resid_threshold else 'circle'
        return color, size, symbol

    # INITIAL frame for first date
    first_daily = ns_df[ns_df['Date'] == dates[0]]
    colors, sizes, symbols = zip(*first_daily.apply(get_marker_style, axis=1))

    fig.add_trace(go.Scatter(
        x=first_daily['YTM'],
        y=first_daily['Z_SPRD_VAL'],
        mode='markers',
        name='Bonds',
        marker=dict(color=colors, size=sizes, symbol=symbols),
        text=first_daily['SECURITY_NAME'],
        customdata=first_daily['RESIDUAL_NS'],
        hovertemplate='YTM: %{x:.2f}<br>Z: %{y:.1f}bps<br>Residual: %{customdata:.1f}<br>%{text}<extra></extra>'
    ))

    # Add Nelson-Siegel fit
    try:
        ns_params = first_daily['NS_PARAMS'].iloc[0]
        fig.add_trace(go.Scatter(
            x=ytm_range,
            y=nelson_siegel(ytm_range, *ns_params),
            mode='lines',
            name='Nelson-Siegel Fit',
            line=dict(color='deepskyblue', width=3)
        ))
    except Exception as e:
        print(f"[ERROR] Could not plot NS curve: {e}")

    # Create animation frames
    frames = []
    for d in dates:
        daily = ns_df[ns_df['Date'] == d]
        colors, sizes, symbols = zip(*daily.apply(get_marker_style, axis=1))
        try:
            fit_curve = nelson_siegel(ytm_range, *daily['NS_PARAMS'].iloc[0])
            frames.append(go.Frame(
                name=str(d.date()),
                data=[
                    go.Scatter(
                        x=daily['YTM'],
                        y=daily['Z_SPRD_VAL'],
                        mode='markers',
                        marker=dict(color=colors, size=sizes, symbol=symbols),
                        text=daily['SECURITY_NAME'],
                        customdata=daily['RESIDUAL_NS'],
                        hovertemplate='YTM: %{x:.2f}<br>Z: %{y:.1f}bps<br>Residual: %{customdata:.1f}<br>%{text}<extra></extra>',
                        name='Bonds'
                    ),
                    go.Scatter(
                        x=ytm_range,
                        y=fit_curve,
                        mode='lines',
                        line=dict(color='deepskyblue', width=3),
                        name='Nelson-Siegel Fit'
                    )
                ]
            ))
        except Exception as e:
            print(f"[ERROR] Error creating frame for {d}: {e}")

    fig.frames = frames

    # Animation buttons & layout remain the same
    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "x": 0.05, "y": 1.1,
            "direction": "right",
            "buttons": [
                {"label": "▶️ Play", "method": "animate", "args": [None, {"frame": {"duration": 500, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]},
                {"label": "⏸️ Pause", "method": "animate", "args": [[None], {"frame": {"duration": 0, "redraw": False}, "mode": "immediate"}]}
            ],
            "showactive": False
        }],
        sliders=[{
            "steps": [dict(
                method="animate",
                args=[[str(d.date())], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
                label=str(d.date())
            ) for d in dates],
            "transition": {"duration": 0},
            "x": 0.05,
            "len": 0.9
        }],
        title=f"{issuer_label} Z-Spread Curve Animation with Nelson-Siegel Fit",
        xaxis_title="Years to Maturity",
        yaxis_title="Z-Spread (bps)",
        template=template,
        height=800,
        width=1200,
        showlegend=True
    )

    return fig
