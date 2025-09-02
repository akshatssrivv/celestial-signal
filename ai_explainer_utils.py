from openai import OpenAI
import pandas as pd

client = OpenAI()

def generate_ai_explanation(diagnostics):
    prompt = prompt = f"""
    You are an expert fixed income trader and risk analyst.
    
    Analyze the bond {diagnostics['SECURITY_NAME']} ({diagnostics['ISIN']}) as of {diagnostics['Date']}, which currently has a trading signal of '{diagnostics['SIGNAL']}' and a composite mispricing score of {diagnostics['COMPOSITE_SCORE'] if diagnostics['COMPOSITE_SCORE'] is not None else 'N/A'} ({diagnostics['COMPOSITE_Strength_Category']}, {diagnostics['COMPOSITE_Market_Category']} in market, {diagnostics['COMPOSITE_Issuer_Category']} among issuer peers).
    
    If any of the following metrics are missing, skip that part of the analysis rather than making assumptions.
    
    Structure your explanation to give the most intuitive, actionable view possible, covering:
    
    1. **Own-History Momentum**
       {f"- Compare today’s composite score to 1 week ago ({diagnostics['COMPOSITE_SCORE_1W_AGO']}, change {diagnostics['Composite_1W_Change']} / {diagnostics['Composite_1W_Change_Pct']}%)" if diagnostics['COMPOSITE_SCORE_1W_AGO'] is not None else '- No 1-week ago data available.'}
       {f"- Compare to 1 month ago ({diagnostics['COMPOSITE_SCORE_1M_AGO']}, change {diagnostics['Composite_1M_Change']} / {diagnostics['Composite_1M_Change_Pct']}%)" if diagnostics['COMPOSITE_SCORE_1M_AGO'] is not None else '- No 1-month ago data available.'}
    
    2. **Cross-Sectional Positioning**
       {f"- Residual Z-score: {diagnostics['Z_RESIDUAL_BUCKET']} (issuer percentile: {diagnostics['Z_Residual_Score_Issuer_Percentile']}, market percentile: {diagnostics['Z_Residual_Score_Market_Percentile']})" if diagnostics['Z_RESIDUAL_BUCKET'] is not None else '- Residual Z-score data unavailable.'}
       {f"- Cluster deviation score: {diagnostics['Cluster_Score']} (issuer percentile: {diagnostics['Cluster_Score_Issuer_Percentile']}, market percentile: {diagnostics['Cluster_Score_Market_Percentile']})" if diagnostics['Cluster_Score'] is not None else '- Cluster deviation data unavailable.'}
    
    3. **Risk & Stability**
       {f"- Current volatility: {diagnostics['Volatility']} (issuer percentile: {diagnostics['Volatility_Issuer_Percentile']}, market percentile: {diagnostics['Volatility_Market_Percentile']}), trend: {diagnostics['Volatility_Trend']}" if diagnostics['Volatility'] is not None else '- Volatility data unavailable.'}
    
    4. **Model Alignment**
       {f"- Regression component: {diagnostics['Regression_Component']} (issuer percentile: {diagnostics['Regression_Score_Issuer_Percentile']})" if diagnostics['Regression_Component'] is not None else '- Regression data unavailable.'}
    
    5. **Relative Strength**
       {f"- Compared to market average: {diagnostics['COMPOSITE_SCORE_Relative_Strength']}×, issuer percentile: {diagnostics['COMPOSITE_SCORE_Issuer_Percentile']}" if diagnostics.get('COMPOSITE_SCORE_Relative_Strength') is not None else '- Relative strength data unavailable.'}
    
    6. **Recommendation**
       - Provide a decisive trading stance (buy, sell, hold, watchlist, avoid) that weighs signal strength, trend direction, volatility, and peer positioning.
       - Highlight the *why* in plain English — e.g., “High mispricing score, strengthening momentum, low volatility — attractive buy” or “Signal weakening, volatility rising — risk outweighs opportunity”.
    
    Guidelines:
    - Write like a trader speaking to another trader — concise, numbers-backed, practical.
    - Explain what each key number *means* in terms of opportunity and risk, not just the raw value.
    - Use directional language (strengthening, fading, stable, volatile) to make the situation clear in one read.
    - Focus on what to do **now**, given the context.
    
    Example:
    "The composite score has risen by 0.45 (+28%) vs last month, putting it in the top quartile of the market. Residual Z-score is 1.8, well above issuer average, pointing to relative undervaluation. Volatility is in the bottom quartile and stable, indicating low risk of price whipsaws. Regression alignment is positive at +0.65. Overall — strengthening undervaluation signal with low risk — a strong buy candidate."
    """
    



    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert fixed income trader and risk analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )
    answer = response.choices[0].message.content
    return answer

def format_bond_diagnostics(history_df):
    import pandas as pd

    def safe_get(series, key):
        val = series.get(key, None)
        if val is None or pd.isna(val):
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    def safe_round(val, digits=2):
        if val is None:
            return None
        try:
            return round(val, digits)
        except Exception:
            return None

    latest_row = history_df.sort_values("Date").iloc[-1]

    return {
        "ISIN": latest_row.get("ISIN", None),
        "SECURITY_NAME": latest_row.get("SECURITY_NAME", "Unknown"),
        "Date": str(latest_row.get("Date", None)),
        "SIGNAL": latest_row.get("SIGNAL", None),

        # Composite
        "COMPOSITE_SCORE": safe_round(safe_get(latest_row, "COMPOSITE_SCORE")),
        "COMPOSITE_Strength_Category": latest_row.get("COMPOSITE_Strength_Category") or "Unknown",
        "COMPOSITE_Market_Category": latest_row.get("COMPOSITE_Market_Category") or "Unknown",
        "COMPOSITE_Issuer_Category": latest_row.get("COMPOSITE_Issuer_Category") or "Unknown",
        "COMPOSITE_SCORE_1W_AGO": safe_round(safe_get(latest_row, "COMPOSITE_SCORE_1W_AGO")),
        "Composite_1W_Change": safe_round(safe_get(latest_row, "COMPOSITE_SCORE_1W_Change")),
        "Composite_1W_Change_Pct": safe_round(safe_get(latest_row, "COMPOSITE_SCORE_1W_Change_Pct"), 1),
        "COMPOSITE_SCORE_1M_AGO": safe_round(safe_get(latest_row, "COMPOSITE_SCORE_1M_AGO")),
        "Composite_1M_Change": safe_round(safe_get(latest_row, "COMPOSITE_SCORE_1M_Change")),
        "Composite_1M_Change_Pct": safe_round(safe_get(latest_row, "COMPOSITE_SCORE_1M_Change_Pct"), 1),

        # Residual Z-score
        "Z_RESIDUAL_BUCKET": safe_round(safe_get(latest_row, "Z_Residual_Score")),
        "Z_Residual_Score_1W_Change": safe_round(safe_get(latest_row, "Z_Residual_Score_1W_Change")),
        "Z_Residual_Score_1W_Change_Pct": safe_round(safe_get(latest_row, "Z_Residual_Score_1W_Change_Pct"), 1),
        "Z_Residual_Score_1M_Change": safe_round(safe_get(latest_row, "Z_Residual_Score_1M_Change")),
        "Z_Residual_Score_1M_Change_Pct": safe_round(safe_get(latest_row, "Z_Residual_Score_1M_Change_Pct"), 1),
        "Z_Residual_Score_Market_Percentile": safe_round(safe_get(latest_row, "Z_Residual_Score_Market_Percentile"), 0) or 50,
        "Z_Residual_Score_Issuer_Percentile": safe_round(safe_get(latest_row, "Z_Residual_Score_Issuer_Percentile"), 0) or 50,
        "Z_Residual_Score_Relative_Strength": safe_round(safe_get(latest_row, "Z_Residual_Score_Relative_Strength")) or 1.0,

        # Cluster score
        "Cluster_Score": safe_round(safe_get(latest_row, "Cluster_Score")),
        "Cluster_Score_1W_Change": safe_round(safe_get(latest_row, "Cluster_Score_1W_Change")),
        "Cluster_Score_1W_Change_Pct": safe_round(safe_get(latest_row, "Cluster_Score_1W_Change_Pct"), 1),
        "Cluster_Score_1M_Change": safe_round(safe_get(latest_row, "Cluster_Score_1M_Change")),
        "Cluster_Score_1M_Change_Pct": safe_round(safe_get(latest_row, "Cluster_Score_1M_Change_Pct"), 1),
        "Cluster_Score_Market_Percentile": safe_round(safe_get(latest_row, "Cluster_Score_Market_Percentile"), 0) or 50,
        "Cluster_Score_Issuer_Percentile": safe_round(safe_get(latest_row, "Cluster_Score_Issuer_Percentile"), 0) or 50,
        "Cluster_Score_Relative_Strength": safe_round(safe_get(latest_row, "Cluster_Score_Relative_Strength")) or 1.0,

        # Regression
        "Regression_Component": safe_round(safe_get(latest_row, "Regression_Score")),
        "Regression_Score_1W_Change": safe_round(safe_get(latest_row, "Regression_Score_1W_Change")),
        "Regression_Score_1W_Change_Pct": safe_round(safe_get(latest_row, "Regression_Score_1W_Change_Pct"), 1),
        "Regression_Score_1M_Change": safe_round(safe_get(latest_row, "Regression_Score_1M_Change")),
        "Regression_Score_1M_Change_Pct": safe_round(safe_get(latest_row, "Regression_Score_1M_Change_Pct"), 1),
        "Regression_Score_Market_Percentile": safe_round(safe_get(latest_row, "Regression_Score_Market_Percentile"), 0) or 50,
        "Regression_Score_Issuer_Percentile": safe_round(safe_get(latest_row, "Regression_Score_Issuer_Percentile"), 0) or 50,
        "Regression_Score_Relative_Strength": safe_round(safe_get(latest_row, "Regression_Score_Relative_Strength")) or 1.0,

        # Volatility
        "Volatility": safe_round(safe_get(latest_row, "Volatility_Score")),
        "VOLATILITY_1M_AGO": safe_round(safe_get(latest_row, "VOLATILITY_1M_AGO")),
        "Volatility_1M_Change": safe_round(safe_get(latest_row, "Volatility_1M_Change")),
        "Volatility_Trend": latest_row.get("Volatility_Trend") or "stable",
        "Volatility_Market_Percentile": safe_round(safe_get(latest_row, "Volatility_Market_Percentile"), 0) or 50,
        "Volatility_Issuer_Percentile": safe_round(safe_get(latest_row, "Volatility_Issuer_Percentile"), 0) or 50,
    }
