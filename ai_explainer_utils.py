from openai import OpenAI
import pandas as pd

client = OpenAI()

def generate_ai_explanation(diagnostics):
    prompt = f"""
    You are an expert fixed income trader and risk analyst.
    
    Explain the bond {diagnostics['SECURITY_NAME']} ({diagnostics['ISIN']}) as of {diagnostics['Date']}, which currently has a signal '{diagnostics['SIGNAL']}' and a composite score of {diagnostics['COMPOSITE_SCORE']:.2f}.
    
    Please provide a concise, actionable explanation including:
    
    1. How this composite score compares to the bond’s own score 1 week ago ({diagnostics['COMPOSITE_SCORE_1W_AGO']:.2f}), and 1 month ago ({diagnostics['COMPOSITE_SCORE_1M_AGO']:.2f}), including percent changes ({diagnostics['Composite_1W_Change']:+.2f} and {diagnostics['Composite_1M_Change']:+.2f}).
    2. How the residual Z-score and cluster deviation compare to the average or typical values for similar bonds. For example, say if the bond's residual Z-score ({diagnostics['Z_RESIDUAL_BUCKET']:.2f}) is above or below average for bonds with similar maturity or issuer.
    3. The current volatility ({diagnostics['Volatility']:.2f}) and its trend over the last month ({diagnostics['Volatility_Trend']}), explaining what that means for price stability or risk.
    4. How the regression component ({diagnostics['Regression_Component']:.2f}) fits into expected market behavior.
    5. Based on these factors, give a clear trader recommendation (e.g., hold, buy, sell, watchlist), emphasizing the magnitude and direction of changes, not just labels.
    
    Use simple language focused on what this means practically for trading decisions. Avoid jargon but use the numbers to explain relative strength or weakness and changes over time.
    
    Example:
    "The bond’s composite score has decreased by 0.10 (-15%) compared to last month, indicating weakening mispricing. Its residual Z-score is 0.5 standard deviations below the peer average, suggesting slight overpricing relative to issuer peers. Volatility has risen by 12% this month, signaling increasing risk. Given these factors, it is advisable to hold or avoid initiating new positions until signals improve."
    
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
        "COMPOSITE_SCORE": round(safe_get("COMPOSITE_SCORE"), 2),
        "Z_RESIDUAL_BUCKET": round(safe_get("Z_RESIDUAL_BUCKET"), 2),
        "Cluster_Deviation_Flipped": round(safe_get("Cluster_Deviation_Flipped"), 2),
        "Volatility": round(safe_get("Volatility"), 2),
        "Regression_Component": round(safe_get("Regression_Component"), 2),
        "Composite_1W_Change": round(safe_get("Composite_1W_Change"), 2),
        "COMPOSITE_SCORE_1W_AGO": round(safe_get("COMPOSITE_SCORE_1W_AGO"), 2),
        "Composite_1M_Change": round(safe_get("Composite_1M_Change"), 2),
        "COMPOSITE_SCORE_1M_AGO": round(safe_get("COMPOSITE_SCORE_1M_AGO"), 2),
        "Volatility_Trend": latest_row.get("Volatility_Trend", "stable"),
    }
