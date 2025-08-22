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
    highlight_isins=None
):
    highlight_isins = highlight_isins or []

    # Map SIGNAL to colors
    signal_color_map = {
        'STRONG BUY': 'darkgreen',
        'MODERATE BUY': 'green',
        'STRONG SELL': 'darkred',
        'MODERATE SELL': 'orange'
    }

    # Ensure dates exist
    dates = sorted(ns_df['Date'].unique())
    if not dates:
        print(f"[{issuer_label}] No available dates to animate.")
        return

    # Determine fixed axis ranges
    x_min, x_max = ns_df['YTM'].min(), ns_df['YTM'].max()
    y_min, y_max = ns_df['Z_SPRD_VAL'].min(), ns_df['Z_SPRD_VAL'].max()
    
    # Add a small buffer to avoid points being on the edge
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05
    
    fig = go.Figure()

    # Helper for marker style
    def get_marker_style(row):
        signal = row.get('SIGNAL', None)
        isin = row['ISIN']
        if signal in signal_color_map:
            color = signal_color_map[signal]
            size = 8 if isin in highlight_isins else 7
        else:
            color = 'black'
            size = 6
        symbol = 'circle' if abs(row['RESIDUAL_NS']) >= resid_threshold else 'circle'
        return color, size, symbol

    # INITIAL frame
    first_daily = ns_df[ns_df['Date'] == dates[0]]
    colors, sizes, symbols = zip(*first_daily.apply(get_marker_style, axis=1))

    fig.add_trace(go.Scatter(
        x=first_daily['YTM'],
        y=first_daily['Z_SPRD_VAL'],
        mode='markers',
        marker=dict(color=colors, size=sizes, symbol=symbols, line=dict(width=1, color="black")),
        text=first_daily['SECURITY_NAME'],
        customdata=np.stack([first_daily['RESIDUAL_NS'], first_daily['SIGNAL']], axis=-1),
        hovertemplate=(
            "YTM: %{x:.2f} yrs<br>"
            "Z-Spread: %{y:.1f} bps<br>"
            "Residual: %{customdata[0]:.1f} bps<br>"
            "Signal: %{customdata[1]}<br>"
            "%{text}<extra></extra>"
        ),
        name="Bonds"
    ))

    # NS curve
    try:
        ns_params = first_daily['NS_PARAMS'].iloc[0]
        fig.add_trace(go.Scatter(
            x=ytm_range,
            y=nelson_siegel(ytm_range, *ns_params),
            mode='lines',
            line=dict(color='deepskyblue', width=3),
            name='Nelson-Siegel Fit'
        ))
    except Exception as e:
        print(f"[ERROR] Could not plot NS curve: {e}")

    # Frames
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
                        marker=dict(color=colors, size=sizes, symbol=symbols, line=dict(width=1, color="black")),
                        text=daily['SECURITY_NAME'],
                        customdata=np.stack([daily['RESIDUAL_NS'], daily['SIGNAL']], axis=-1),
                        hovertemplate=(
                            "YTM: %{x:.2f} yrs<br>"
                            "Z-Spread: %{y:.1f} bps<br>"
                            "Residual: %{customdata[0]:.1f} bps<br>"
                            "Signal: %{customdata[1]}<br>"
                            "%{text}<extra></extra>"
                        ),
                        name="Bonds"
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

    fig.update_layout(
        xaxis=dict(title="Years to Maturity", range=[x_min - x_pad, x_max + x_pad]),
        yaxis=dict(title="Z-Spread (bps)", range=[y_min - y_pad, y_max + y_pad]),
        updatemenus=[{
            "type": "buttons",
            "x": 0.05, "y": 1.1,
            "direction": "right",
            "buttons": [
                {"label": "▶️ Play", "method": "animate", "args": [None, {"frame": {"duration": 150, "redraw": True}, "fromcurrent": True, "mode": "immediate"}]},
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
        template=template,
        height=800,
        width=1200,
        showlegend=True
    )

    return fig

