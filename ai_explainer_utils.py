from openai import OpenAI
import pandas as pd

client = OpenAI()

def generate_ai_explanation(diagnostics):
    prompt = f"""
    You are an expert fixed income trader and risk analyst.

    Explain the behavior of the bond {diagnostics['SECURITY_NAME']} ({diagnostics['ISIN']}) on {diagnostics['Date']}.

    It currently has a signal: {diagnostics['SIGNAL']} with a composite score of {diagnostics['COMPOSITE_SCORE']}.

    Provide a clear, concise explanation that covers:

    - How the bond is performing today compared to its issuer peers and bonds with similar maturity.
    - Whether its mispricing signal is improving or weakening compared to 1 week and 1 month ago.
    - The meaning of each component below in plain language, focusing on what the values imply for trading or risk:
      * Residual Z-Score: {diagnostics['Z_RESIDUAL_BUCKET']}
      * Cluster Deviation: {diagnostics['Cluster_Deviation_Flipped']}
      * Volatility: {diagnostics['Volatility']} (trend: {diagnostics.get('Volatility_Trend', 'stable')})
      * Regression Component: {diagnostics['Regression_Component']}

    Include insights about investor sentiment if applicable, and provide a clear recommendation or action item for traders based on these factors.

    Use no jargon, and keep the explanation actionable and easy to understand.
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


def format_bond_diagnostics(row):
    def safe_get(series, key, default=0):
        return series[key] if key in series and pd.notna(series[key]) else default

    def safe_round(val):
        try:
            return round(val, 2)
        except:
            return 0

    return {
        "ISIN": safe_get(row, "ISIN", ""),
        "SECURITY_NAME": safe_get(row, "SECURITY_NAME", ""),
        "Date": str(safe_get(row, "Date", "")),
        "SIGNAL": safe_get(row, "SIGNAL", ""),
        "COMPOSITE_SCORE": safe_round(safe_get(row, "COMPOSITE_SCORE")),
        "Z_RESIDUAL_BUCKET": safe_round(safe_get(row, "Z_RESIDUAL_BUCKET")),
        "Cluster_Deviation_Flipped": safe_round(safe_get(row, "Cluster_Deviation_Flipped")),
        "Volatility": safe_round(safe_get(row, "Volatility")),
        "Volatility_Trend": safe_get(row, "Volatility_Trend", "stable"),
        "Regression_Component": safe_round(safe_get(row, "Regression_Component")),
        "Composite_1W_Change": safe_round(safe_get(row, "Composite_1W_Change")),
        "Composite_1M_Change": safe_round(safe_get(row, "Composite_1M_Change")),
        # Uncomment these if you have percentile columns
        # "Residual_Z_Percentile": safe_round(safe_get(row, "Residual_Z_Percentile", 50)),
        # "Cluster_Deviation_Percentile": safe_round(safe_get(row, "Cluster_Deviation_Percentile", 50)),
        # "Regression_Component_Percentile": safe_round(safe_get(row, "Regression_Component_Percentile", 50)),
    }
