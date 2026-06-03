import numpy as np
import pandas as pd
import plotly.graph_objects as go
import json
import re


def parse_ns_params(x):
    if isinstance(x, (list, tuple, np.ndarray)):
        return list(x)
    if isinstance(x, str):
        # Handle "np.float64(...)" format
        nums = re.findall(r"np\.float64\(([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)\)", x)
        if len(nums) >= 4:
            return [float(n) for n in nums[:4]]
        # Handle plain JSON array or comma-separated numbers
        try:
            parsed = json.loads(x)
            if isinstance(parsed, list) and len(parsed) >= 4:
                return [float(v) for v in parsed[:4]]
        except Exception:
            pass
        # Fallback: extract any numbers
        nums = re.findall(r"[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?", x)
        if len(nums) >= 4:
            return [float(n) for n in nums[:4]]
    return None


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

    signal_color_map = {
        'STRONG BUY':    'darkgreen',
        'MODERATE BUY':  'green',
        'STRONG SELL':   'darkred',
        'MODERATE SELL': 'orange',
    }

    dates = sorted(ns_df['Date'].unique())
    if not dates:
        print(f"[{issuer_label}] No available dates to animate.")
        return go.Figure()

    # Fixed axis ranges with padding
    x_min, x_max = ns_df['YTM'].min(), ns_df['YTM'].max()
    y_min, y_max = ns_df['Z_SPRD_VAL'].min(), ns_df['Z_SPRD_VAL'].max()
    x_pad = (x_max - x_min) * 0.05
    y_pad = (y_max - y_min) * 0.05

    # ── Helper ────────────────────────────────
    def get_marker_style(row):
        signal = row.get('SIGNAL', None)
        isin   = row['ISIN']
        if signal in signal_color_map:
            color = signal_color_map[signal]
            size  = 8 if isin in highlight_isins else 7
        else:
            color = 'grey'
            size  = 6
        return color, size, 'circle'

    # ── Initial trace (first date) ────────────
    first_daily = ns_df[ns_df['Date'] == dates[0]]
    colors, sizes, symbols = zip(*first_daily.apply(get_marker_style, axis=1))

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=first_daily['YTM'],
        y=first_daily['Z_SPRD_VAL'],
        mode='markers',
        marker=dict(
            color=list(colors),
            size=list(sizes),
            symbol=list(symbols),
            line=dict(width=1, color='black'),
        ),
        text=first_daily['SECURITY_NAME'],
        customdata=np.stack([first_daily['RESIDUAL_NS'], first_daily['SIGNAL']], axis=-1),
        hovertemplate=(
            "YTM: %{x:.2f} yrs<br>"
            "Z-Spread: %{y:.1f} bps<br>"
            "Residual: %{customdata[0]:.1f} bps<br>"
            "Signal: %{customdata[1]}<br>"
            "%{text}<extra></extra>"
        ),
        name='Bonds',
    ))

    # Initial NS fit line
    ns_params_init = parse_ns_params(first_daily['NS_PARAMS'].iloc[0])
    if ns_params_init is not None:
        fig.add_trace(go.Scatter(
            x=ytm_range,
            y=nelson_siegel(ytm_range, *ns_params_init),
            mode='lines',
            line=dict(color='deepskyblue', width=3),
            name='Nelson-Siegel Fit',
        ))
    else:
        # Placeholder so trace index stays consistent with frames
        fig.add_trace(go.Scatter(x=[], y=[], mode='lines',
                                 line=dict(color='deepskyblue', width=3),
                                 name='Nelson-Siegel Fit'))

    # ── Frames ────────────────────────────────
    frames = []
    for d in dates:
        daily = ns_df[ns_df['Date'] == d]
        if daily.empty:
            continue

        colors, sizes, symbols = zip(*daily.apply(get_marker_style, axis=1))

        ns_params = parse_ns_params(daily['NS_PARAMS'].iloc[0])
        fit_y     = nelson_siegel(ytm_range, *ns_params) if ns_params is not None else []

        frames.append(go.Frame(
            name=str(d.date()) if hasattr(d, 'date') else str(d),
            data=[
                go.Scatter(
                    x=daily['YTM'],
                    y=daily['Z_SPRD_VAL'],
                    mode='markers',
                    marker=dict(
                        color=list(colors),
                        size=list(sizes),
                        symbol=list(symbols),
                        line=dict(width=1, color='black'),
                    ),
                    text=daily['SECURITY_NAME'],
                    customdata=np.stack([daily['RESIDUAL_NS'], daily['SIGNAL']], axis=-1),
                    hovertemplate=(
                        "YTM: %{x:.2f} yrs<br>"
                        "Z-Spread: %{y:.1f} bps<br>"
                        "Residual: %{customdata[0]:.1f} bps<br>"
                        "Signal: %{customdata[1]}<br>"
                        "%{text}<extra></extra>"
                    ),
                    name='Bonds',
                ),
                go.Scatter(
                    x=ytm_range if ns_params is not None else [],
                    y=fit_y,
                    mode='lines',
                    line=dict(color='deepskyblue', width=3),
                    name='Nelson-Siegel Fit',
                ),
            ],
        ))

    fig.frames = frames

    # ── Layout ────────────────────────────────
    date_labels = [
        str(d.date()) if hasattr(d, 'date') else str(d)
        for d in dates
    ]

    fig.update_layout(
        title=f"{issuer_label} — Z-Spread Curve Animation with Nelson-Siegel Fit",
        template=template,
        height=800,
        width=1200,
        showlegend=True,
        xaxis=dict(title='Years to Maturity', range=[x_min - x_pad, x_max + x_pad]),
        yaxis=dict(title='Z-Spread (bps)',    range=[y_min - y_pad, y_max + y_pad]),
        updatemenus=[{
            'type':      'buttons',
            'x':         0.05,
            'y':         1.08,
            'direction': 'right',
            'showactive': False,
            'buttons': [
                {
                    'label':  '▶ Play',
                    'method': 'animate',
                    'args': [
                        None,
                        {
                            'frame':       {'duration': 150, 'redraw': True},
                            'fromcurrent': True,
                            'mode':        'immediate',
                            'transition':  {'duration': 50},
                        },
                    ],
                },
                {
                    'label':  '⏸ Pause',
                    'method': 'animate',
                    'args': [
                        [None],
                        {
                            'frame':      {'duration': 0, 'redraw': False},
                            'mode':       'immediate',
                            'transition': {'duration': 0},
                        },
                    ],
                },
            ],
        }],
        sliders=[{
            'currentvalue': {
                'prefix':  'Date: ',
                'visible': True,
                'xanchor': 'center',
                'font':    {'size': 14},
            },
            'transition': {'duration': 50},
            'x':   0.05,
            'len': 0.9,
            'pad': {'t': 50},
            'steps': [
                {
                    'method': 'animate',
                    'label':  lbl,
                    'args': [
                        [lbl],
                        {
                            'frame':      {'duration': 0, 'redraw': True},
                            'mode':       'immediate',
                            'transition': {'duration': 0},
                        },
                    ],
                }
                for lbl in date_labels
            ],
        }],
    )

    return fig
