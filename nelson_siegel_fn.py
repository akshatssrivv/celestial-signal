import numpy as np
import pandas as pd
import plotly.graph_objects as go

def nelson_siegel(maturities, beta0, beta1, beta2, tau):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = maturities / tau
        term1 = (1 - np.exp(-x)) / x
        term2 = term1 - np.exp(-x)
        return beta0 + beta1 * term1 + beta2 * term2

def plot_ns_animation(
    ns_df,
    issuer_label="Issuer",
    resid_threshold=20,
    ytm_range=np.linspace(0.1, 50, 200),
    template="plotly_dark"
):
    # Required columns
    required_cols = ['Date', 'YTM', 'Z_SPRD_VAL', 'ISIN', 'RESIDUAL_NS', 'NS_PARAMS']
    if not all(col in ns_df.columns for col in required_cols):
        print(f"[{issuer_label}] ❌ Missing required columns in input data.")
        print("Available columns:", ns_df.columns.tolist())
        return

    # Drop rows with NaNs in required columns
    ns_df = ns_df.dropna(subset=required_cols).copy()

    # Ensure 'Date' is datetime
    ns_df['Date'] = pd.to_datetime(ns_df['Date'])

    # Sort by date
    ns_df.sort_values(by='Date', inplace=True)

    # Get unique dates
    dates = ns_df['Date'].dropna().unique()
    if len(dates) == 0:
        print(f"[{issuer_label}] ❌ No valid dates found in dataset.")
        return

    # Get first day for initial plot
    first_date = dates[0]
    first_daily = ns_df[ns_df['Date'] == first_date]

    # Separate inliers and outliers
    first_inliers = first_daily[np.abs(first_daily['RESIDUAL_NS']) <= resid_threshold]
    first_outliers = first_daily[np.abs(first_daily['RESIDUAL_NS']) > resid_threshold]

    # Fit curve
    try:
        ns_params = first_daily['NS_PARAMS'].iloc[0]
        if not isinstance(ns_params, (list, tuple)) or len(ns_params) != 4:
            print(f"[{issuer_label}] ❌ Invalid NS_PARAMS on first day: {ns_params}")
            return
        fitted = nelson_siegel(ytm_range, *ns_params)
    except Exception as e:
        print(f"[{issuer_label}] ❌ Curve fitting failed:", e)
        return

    # Create initial figure
    fig = go.Figure(
        data=[
            go.Scatter(x=ytm_range, y=fitted, mode='lines', name='NS Curve', line=dict(color='white')),
            go.Scatter(x=first_inliers['YTM'], y=first_inliers['Z_SPRD_VAL'],
                       mode='markers', name='Inliers', marker=dict(color='cyan', size=8)),
            go.Scatter(x=first_outliers['YTM'], y=first_outliers['Z_SPRD_VAL'],
                       mode='markers', name='Outliers', marker=dict(color='red', size=8)),
        ],
        layout=go.Layout(
            title=f"{issuer_label} - {first_date.strftime('%Y-%m-%d')}",
            xaxis_title='YTM (Years)',
            yaxis_title='Z-SPREAD',
            template=template,
            updatemenus=[dict(
                type='buttons',
                showactive=False,
                buttons=[dict(label='Play', method='animate', args=[None])]
            )]
        ),
        frames=[]
    )

    # Animation frames
    for d in dates:
        daily = ns_df[ns_df['Date'] == d].dropna(subset=required_cols)
        if daily.empty:
            continue

        try:
            params = daily['NS_PARAMS'].iloc[0]
            if not isinstance(params, (list, tuple)) or len(params) != 4:
                print(f"[{issuer_label}] ⚠️ Skipping date {d} due to bad NS_PARAMS:", params)
                continue
            fit_curve = nelson_siegel(ytm_range, *params)
        except Exception as e:
            print(f"[{issuer_label}] ⚠️ Curve fitting error on {d}:", e)
            continue

        inliers = daily[np.abs(daily['RESIDUAL_NS']) <= resid_threshold]
        outliers = daily[np.abs(daily['RESIDUAL_NS']) > resid_threshold]

        frame = go.Frame(
            data=[
                go.Scatter(x=ytm_range, y=fit_curve, mode='lines'),
                go.Scatter(x=inliers['YTM'], y=inliers['Z_SPRD_VAL'], mode='markers'),
                go.Scatter(x=outliers['YTM'], y=outliers['Z_SPRD_VAL'], mode='markers')
            ],
            name=d.strftime('%Y-%m-%d'),
            layout=go.Layout(title_text=f"{issuer_label} - {d.strftime('%Y-%m-%d')}")
        )
        fig.frames.append(frame)

    fig.show()
