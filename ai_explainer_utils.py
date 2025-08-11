from openai import OpenAI

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
      * Residual Z-Score: {diagnostics['Z_RESIDUAL_BUCKET']} (percentile among issuer peers: {diagnostics['Residual_Z_Percentile']}%)
      * Cluster Deviation: {diagnostics['Cluster_Deviation_Flipped']} (percentile within maturity group: {diagnostics['Cluster_Deviation_Percentile']}%)
      * Volatility: {diagnostics['Volatility']} (trend: {diagnostics.get('Volatility_Trend', 'stable')})
      * Regression Component: {diagnostics['Regression_Component']} (percentile vs issuer peers: {diagnostics['Regression_Component_Percentile']}%)
    
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


def format_bond_diagnostics(row, peer_stats, historical_stats):
    """
    peer_stats: dict with percentiles for metrics compared to issuer peers or maturity groups
    historical_stats: dict with time-based comparisons like 1w ago composite, volatility trends, etc.
    """

    def safe_round(val):
        try:
            return round(val, 2)
        except:
            return 0

    return {
        "ISIN": row["ISIN"],
        "SECURITY_NAME": row["SECURITY_NAME"],
        "Date": str(row["Date"]),
        "SIGNAL": row["SIGNAL"],
        "COMPOSITE_SCORE": safe_round(row["COMPOSITE_SCORE"]),
        "Z_RESIDUAL_BUCKET": safe_round(row.get("Z_RESIDUAL_BUCKET", 0)),
        # Uncomment if you calculate these percentile columns in your df:
        # "Residual_Z_Percentile": safe_round(row.get("Residual_Z_Percentile", 50)),
        "Cluster_Deviation_Flipped": safe_round(row.get("Cluster_Deviation_Flipped", 0)),
        # "Cluster_Deviation_Percentile": safe_round(row.get("Cluster_Deviation_Percentile", 50)),
        "Volatility": safe_round(row.get("Volatility", 0)),
        "Volatility_Trend": row.get("Volatility_Trend", "stable"),
        "Regression_Component": safe_round(row.get("Regression_Component", 0)),
        # "Regression_Component_Percentile": safe_round(row.get("Regression_Component_Percentile", 50)),
        "Composite_1W_Change": safe_round(row.get("Composite_1W_Change", 0)),
        "Composite_1M_Change": safe_round(row.get("Composite_1M_Change", 0)),
    }
