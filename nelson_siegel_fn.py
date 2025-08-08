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

    print(f"[DEBUG] DataFrame columns: {list(ns_df.columns)}")
    print(f"[DEBUG] DataFrame shape: {ns_df.shape}")
    
    dates = sorted(ns_df['Date'].unique())
    if not dates:
        print(f"[{issuer_label}] No available dates to animate.")
        return

    first_date = dates[0]
    first_daily = ns_df[ns_df['Date'] == first_date]
    print(f"[DEBUG] First date: {first_date}")
    print(f"[DEBUG] First daily shape: {first_daily.shape}")
    
    # Check if we have the required columns
    required_cols = ['YTM', 'Z_SPRD_VAL', 'RESIDUAL_NS', 'SECURITY_NAME', 'NS_PARAMS']
    missing_cols = [col for col in required_cols if col not in ns_df.columns]
    if missing_cols:
        print(f"[ERROR] Missing columns: {missing_cols}")
        print(f"[DEBUG] Available columns: {list(ns_df.columns)}")
        return None

    fig = go.Figure()

    # Get top 7 outliers for first date
    first_outliers = first_daily.nlargest(7, 'RESIDUAL_NS', keep='all')
    first_regular = first_daily.drop(first_outliers.index)

    # Add regular bonds (black)
    fig.add_trace(go.Scatter(
        x=first_regular['YTM'],
        y=first_regular['Z_SPRD_VAL'],
        mode='markers',
        name='Bonds',
        marker=dict(size=6, color='black'),
        text=first_regular['SECURITY_NAME'],
        hovertemplate='YTM: %{x:.2f}<br>Z: %{y:.1f}bps<br>%{text}<extra></extra>'
    ))

    # Add top 7 outliers (red)
    fig.add_trace(go.Scatter(
        x=first_outliers['YTM'],
        y=first_outliers['Z_SPRD_VAL'],
        mode='markers',
        name='Top 7 Outliers',
        marker=dict(size=8, color='red', symbol='diamond'),
        text=first_outliers['SECURITY_NAME'],
        hovertemplate='YTM: %{x:.2f}<br>Z: %{y:.1f}bps<br>Residual: %{customdata:.1f}<br>%{text}<extra></extra>',
        customdata=first_outliers['RESIDUAL_NS']
    ))

    # Add Nelson-Siegel curve
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

    # Create frames for animation
    frames = []
    for d in dates:
        daily = ns_df[ns_df['Date'] == d]
        print(f"[DEBUG] Date {d}: {len(daily)} bonds")

        try:
            # Generate Nelson-Siegel curve for this date
            fit_curve = nelson_siegel(ytm_range, *daily['NS_PARAMS'].iloc[0])

            # Get top 7 outliers for this date
            daily_outliers = daily.nlargest(7, 'RESIDUAL_NS', keep='all')
            daily_regular = daily.drop(daily_outliers.index)

            frames.append(go.Frame(
                name=str(d.date()),
                data=[
                    # Regular bonds (black)
                    go.Scatter(
                        x=daily_regular['YTM'],
                        y=daily_regular['Z_SPRD_VAL'],
                        mode='markers',
                        marker=dict(size=6, color='black'),
                        text=daily_regular['SECURITY_NAME'],
                        hovertemplate='YTM: %{x:.2f}<br>Z: %{y:.1f}bps<br>%{text}<extra></extra>',
                        name='Bonds'
                    ),
                    # Top 7 outliers (red)
                    go.Scatter(
                        x=daily_outliers['YTM'],
                        y=daily_outliers['Z_SPRD_VAL'],
                        mode='markers',
                        marker=dict(size=8, color='red', symbol='diamond'),
                        text=daily_outliers['SECURITY_NAME'],
                        hovertemplate='YTM: %{x:.2f}<br>Z: %{y:.1f}bps<br>Residual: %{customdata:.1f}<br>%{text}<extra></extra>',
                        customdata=daily_outliers['RESIDUAL_NS'],
                        name='Top 7 Outliers'
                    ),
                    # Nelson-Siegel curve
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
        height=800,
        width=1200,
        showlegend=True
    )
    
    return fig
