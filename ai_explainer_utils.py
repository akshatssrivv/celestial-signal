from openai import OpenAI
import pandas as pd

client = OpenAI()

def generate_ai_explanation(diagnostics):
    prompt = f"""
    You are an expert fixed income trader and risk analyst.
    
    Analyze the bond {diagnostics['SECURITY_NAME']} ({diagnostics['ISIN']}) as of {diagnostics['Date']}, which currently has a trading signal of '{diagnostics['SIGNAL']}' and a composite mispricing score of {diagnostics['COMPOSITE_SCORE']:.2f} ({diagnostics['COMPOSITE_Strength_Category']}, {diagnostics['COMPOSITE_Market_Category']} in market, {diagnostics['COMPOSITE_Issuer_Category']} among issuer peers).
    
    Structure your explanation to give the most intuitive, actionable view possible, covering:
    
    1. **Own-History Momentum**  
       - Compare today’s composite score to 1 week ago ({diagnostics['COMPOSITE_SCORE_1W_AGO']:.2f}, change {diagnostics['COMPOSITE_SCORE_1W_Change']:+.2f} / {diagnostics['COMPOSITE_SCORE_1W_Change_Pct']:+.1f}%) and 1 month ago ({diagnostics['COMPOSITE_SCORE_1M_AGO']:.2f}, change {diagnostics['COMPOSITE_SCORE_1M_Change']:+.2f} / {diagnostics['COMPOSITE_SCORE_1M_Change_Pct']:+.1f}%).  
       - Describe whether momentum is strengthening, weakening, or reversing.
    
    2. **Cross-Sectional Positioning**  
       - Residual Z-score: {diagnostics['Z_RESIDUAL_BUCKET']:.2f} ({diagnostics['Z_Residual_Score_Issuer_Category']} vs issuer, {diagnostics['Z_Residual_Score_Market_Category']} vs market).  
         Explain whether this suggests overpricing or underpricing relative to peers.  
       - Cluster deviation score: {diagnostics['Cluster_Score']:.2f} ({diagnostics['Cluster_Score_Issuer_Category']} vs issuer, {diagnostics['Cluster_Score_Market_Category']} vs market).  
         Explain whether deviation is unusually high or low.
    
    3. **Risk & Stability**  
       - Current volatility: {diagnostics['Volatility']:.2f} ({diagnostics['Volatility_Issuer_Percentile']:.0f}th percentile vs issuer, {diagnostics['Volatility_Market_Percentile']:.0f}th percentile vs market), trend over last month: {diagnostics['Volatility_Trend']}.  
         Interpret what this implies for price stability, stop-loss risk, and position sizing.
    
    4. **Model Alignment**  
       - Regression component: {diagnostics['Regression_Component']:.2f} ({diagnostics['Regression_Score_Issuer_Category']} vs issuer).  
         Explain whether the model sees this bond’s pricing as aligned or divergent from macro yield curve expectations.
    
    5. **Relative Strength**  
       - How today’s signal strength compares to the average bond in the market ({diagnostics['COMPOSITE_SCORE_Relative_Strength']:.2f}× the average) and to the issuer average ({diagnostics['COMPOSITE_SCORE_Issuer_Percentile']:.0f}th percentile).
    
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
    def safe_get(series, key, default=0):
        return series[key] if key in series and pd.notna(series[key]) else default
        
    def safe_round(val):
        try:
            return round(val, 2)
        except:
            return 0

    latest_row = history_df.sort_values("Date").iloc[-1]

    return {
    "ISIN": latest_row["ISIN"],
    "SECURITY_NAME": latest_row["SECURITY_NAME"],
    "Date": str(latest_row["Date"]),
    "SIGNAL": latest_row["SIGNAL"],
    
    # Composite scores & trends
    "COMPOSITE_SCORE": round(safe_get(latest_row, "COMPOSITE_SCORE"), 2),
    "COMPOSITE_Strength_Category": latest_row.get("COMPOSITE_Strength_Category", "Unknown"),
    "COMPOSITE_Market_Category": latest_row.get("COMPOSITE_Market_Category", "Unknown"),
    "COMPOSITE_Issuer_Category": latest_row.get("COMPOSITE_Issuer_Category", "Unknown"),
    "COMPOSITE_SCORE_1W_AGO": round(safe_get(latest_row, "COMPOSITE_SCORE_1W_AGO"), 2),
    "Composite_1W_Change": round(safe_get(latest_row, "COMPOSITE_SCORE_1W_Change"), 2),
    "Composite_1W_Change_Pct": round(safe_get(latest_row, "COMPOSITE_SCORE_1W_Change_Pct"), 1),
    "COMPOSITE_SCORE_1M_AGO": round(safe_get(latest_row, "COMPOSITE_SCORE_1M_AGO"), 2),
    "Composite_1M_Change": round(safe_get(latest_row, "COMPOSITE_SCORE_1M_Change"), 2),
    "Composite_1M_Change_Pct": round(safe_get(latest_row, "COMPOSITE_SCORE_1M_Change_Pct"), 1),
    
    # Residual Z-score
    "Z_RESIDUAL_BUCKET": round(safe_get(latest_row, "Z_Residual_Score"), 2),
    "Z_Residual_Score_1W_Change": round(safe_get(latest_row, "Z_Residual_Score_1W_Change"), 2),
    "Z_Residual_Score_1W_Change_Pct": round(safe_get(latest_row, "Z_Residual_Score_1W_Change_Pct"), 1),
    "Z_Residual_Score_1M_Change": round(safe_get(latest_row, "Z_Residual_Score_1M_Change"), 2),
    "Z_Residual_Score_1M_Change_Pct": round(safe_get(latest_row, "Z_Residual_Score_1M_Change_Pct"), 1),
    "Z_Residual_Score_Market_Percentile": round(latest_row.get("Z_Residual_Score_Market_Percentile", 50), 0),
    "Z_Residual_Score_Issuer_Percentile": round(latest_row.get("Z_Residual_Score_Issuer_Percentile", 50), 0),
    "Z_Residual_Score_Relative_Strength": round(latest_row.get("Z_Residual_Score_Relative_Strength", 1.0), 2),
    
    # Cluster score
    "Cluster_Score": round(safe_get(latest_row, "Cluster_Score"), 2),
    "Cluster_Score_1W_Change": round(safe_get(latest_row, "Cluster_Score_1W_Change"), 2),
    "Cluster_Score_1W_Change_Pct": round(safe_get(latest_row, "Cluster_Score_1W_Change_Pct"), 1),
    "Cluster_Score_1M_Change": round(safe_get(latest_row, "Cluster_Score_1M_Change"), 2),
    "Cluster_Score_1M_Change_Pct": round(safe_get(latest_row, "Cluster_Score_1M_Change_Pct"), 1),
    "Cluster_Score_Market_Percentile": round(latest_row.get("Cluster_Score_Market_Percentile", 50), 0),
    "Cluster_Score_Issuer_Percentile": round(latest_row.get("Cluster_Score_Issuer_Percentile", 50), 0),
    "Cluster_Score_Relative_Strength": round(latest_row.get("Cluster_Score_Relative_Strength", 1.0), 2),
    
    # Regression component
    "Regression_Component": round(safe_get(latest_row, "Regression_Score"), 2),
    "Regression_Score_1W_Change": round(safe_get(latest_row, "Regression_Score_1W_Change"), 2),
    "Regression_Score_1W_Change_Pct": round(safe_get(latest_row, "Regression_Score_1W_Change_Pct"), 1),
    "Regression_Score_1M_Change": round(safe_get(latest_row, "Regression_Score_1M_Change"), 2),
    "Regression_Score_1M_Change_Pct": round(safe_get(latest_row, "Regression_Score_1M_Change_Pct"), 1),
    "Regression_Score_Market_Percentile": round(latest_row.get("Regression_Score_Market_Percentile", 50), 0),
    "Regression_Score_Issuer_Percentile": round(latest_row.get("Regression_Score_Issuer_Percentile", 50), 0),
    "Regression_Score_Relative_Strength": round(latest_row.get("Regression_Score_Relative_Strength", 1.0), 2),
    
    # Volatility
    "Volatility": round(safe_get(latest_row, "Volatility_Score"), 2),
    "VOLATILITY_1M_AGO": round(safe_get(latest_row, "VOLATILITY_1M_AGO"), 2),
    "Volatility_1M_Change": round(safe_get(latest_row, "Volatility_1M_Change"), 2),
    "Volatility_Trend": latest_row.get("Volatility_Trend", "stable"),
    "Volatility_Market_Percentile": round(latest_row.get("Volatility_Market_Percentile", 50), 0),
    "Volatility_Issuer_Percentile": round(latest_row.get("Volatility_Issuer_Percentile", 50), 0),
    
    # Optional / helper
    "Issuer_Proxy": latest_row.get("Issuer_Proxy", ""),
}
