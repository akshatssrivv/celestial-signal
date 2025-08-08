import numpy as np
import pandas as pd
import plotly.graph_objects as go

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
    template="plotly_dark"
):

    dates = sorted(ns_df['Date'].unique())
    if not dates:
        print(f"[{issuer_label}] No available dates to animate.")
        return

    first_date = dates[0]
    first_daily = ns_df[ns_df['Date'] == first_date]

    first_outliers = first_daily[np.abs(first_daily['RESIDUAL_NS']) > resid_threshold]
    first_inliers = first_daily[np.abs(first_daily['RESIDUAL_NS']) <= resid_threshold]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=first_inliers['YTM'],
        y=first_inliers['Z_SPRD_VAL'],
        mode='markers',
        name='Inliers',
        marker=dict(size=6, color='white'),
        text=first_inliers['ISIN'],
        hovertemplate='YTM: %{x:.2f}<br>Z: %{y:.1f}bps<br>%{text}<extra></extra>'
    ))

    fig.add_trace(go.Scatter(
        x=first_outliers['YTM'],
        y=first_outliers['Z_SPRD_VAL'],
        mode='markers',
        name='Outliers',
        marker=dict(size=8, color='red', symbol='diamond'),
        text=first_outliers['ISIN'],
        hovertemplate='YTM: %{x:.2f}<br>Z: %{y:.1f}bps<br>%{text}<extra></extra>'
    ))

    ns_params = first_daily['NS_PARAMS'].iloc[0]
    fig.add_trace(go.Scatter(
        x=ytm_range,
        y=nelson_siegel(ytm_range, *ns_params),
        mode='lines',
        name='Nelson-Siegel Fit',
        line=dict(color='deepskyblue', width=2)
    ))

    frames = []
    for d in dates:
        daily = ns_df[ns_df['Date'] == d]

        outliers = daily.reindex(np.abs(daily['RESIDUAL_NS']).nlargest(7).index)
        inliers = daily.drop(outliers.index)

        fit_curve = nelson_siegel(ytm_range, *daily['NS_PARAMS'].iloc[0])

        frames.append(go.Frame(
            name=str(d.date()),
            data=[
                go.Scatter(
                    x=inliers['YTM'],
                    y=inliers['Z_SPRD_VAL'],
                    mode='markers',
                    marker=dict(size=6, color='white'),
                    text=inliers['ISIN'],
                    hovertemplate='YTM: %{x:.2f}<br>Z: %{y:.1f}bps<br>%{text}<extra></extra>'
                ),
                go.Scatter(
                    x=outliers['YTM'],
                    y=outliers['Z_SPRD_VAL'],
                    mode='markers',
                    marker=dict(size=8, color='red', symbol='diamond'),
                    text=outliers['ISIN'],
                    hovertemplate='YTM: %{x:.2f}<br>Z: %{y:.1f}bps<br>%{text}<extra></extra>'
                ),
                go.Scatter(
                    x=ytm_range,
                    y=fit_curve,
                    mode='lines',
                    line=dict(color='deepskyblue', width=2),
                    name='Nelson-Siegel Fit'
                )
            ]
        ))

    fig.frames = frames

    fig.update_layout(
        updatemenus=[{
            "type": "buttons",
            "x": 0.05, "y": 1.1,
            "direction": "right",
            "buttons": [
                {
                    "label": "▶️ Play",
                    "method": "animate",
                    "args": [None, {
                        "frame": {"duration": 500, "redraw": True},
                        "fromcurrent": True,
                        "mode": "immediate"
                    }]
                },
                {
                    "label": "⏸️ Pause",
                    "method": "animate",
                    "args": [[None], {
                        "frame": {"duration": 0, "redraw": False},
                        "mode": "immediate"
                    }]
                }
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
        height=600,
        width=900,
        showlegend=True
    )

    return fig
