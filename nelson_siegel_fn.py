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

def plot_ns_animation(ns_df, issuer_label, highlight_isins=None):
    
    # safety: ensure Date is string (frames require discrete names)
    ns_df = ns_df.copy()
    ns_df["Date"] = ns_df["Date"].astype(str)

    # color already mapped to ns_df["Signal_Color"] outside
    frames = []
    for date, df_subset in ns_df.groupby("Date"):
        frames.append(go.Frame(
            data=[
                go.Scatter(
                    x=df_subset["YTM"],
                    y=df_subset["Z_SPRD_VAL"],
                    mode="markers",
                    marker=dict(
                        size=10,
                        color=df_subset["Signal_Color"],
                        line=dict(width=1, color="black")
                    ),
                    text=df_subset["SECURITY_NAME"],
                    customdata=np.stack((
                        df_subset["ISIN"],
                        df_subset["RESIDUAL_NS"]
                    ), axis=-1),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "ISIN: %{customdata[0]}<br>"
                        "Years to Maturity: %{x:.2f}<br>"
                        "Z-Spread: %{y:.1f}bps<br>"
                        "Residual_NS: %{customdata[1]:.2f}<extra></extra>"
                    )
                )
            ],
            name=date
        ))

    # initial frame = first date
    fig = go.Figure(
        data=frames[0].data,
        frames=frames,
        layout=go.Layout(
            title=f"{issuer_label} Bonds — Z-Spread vs YearsToMaturity",
            xaxis=dict(title="Years to Maturity"),
            yaxis=dict(title="Z-Spread (bps)"),
            updatemenus=[{
                "type": "buttons",
                "showactive": False,
                "buttons": [
                    {"label": "Play", "method": "animate", "args": [None]},
                    {"label": "Pause", "method": "animate", "args": [[None], {"mode": "immediate"}]}
                ]
            }],
            sliders=[{
                "steps": [
                    {"args": [[f.name], {"mode": "immediate"}], "label": f.name, "method": "animate"}
                    for f in frames
                ],
                "x": 0.1, "y": -0.1, "len": 0.9
            }]
        )
    )

    return fig
