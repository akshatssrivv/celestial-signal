import openai

def generate_ai_explanation(diagnostics):
    prompt = f"""
    You are a bond trading analyst. 
    Explain why the bond {diagnostics['SECURITY_NAME']} ({diagnostics['ISIN']}) 
    has a signal of {diagnostics['Signal']} with composite score {diagnostics['Composite_Score']} on {diagnostics['Date']}.

    Context:
    Residual Z: {diagnostics['Residual_Z']}
    Cluster Deviation: {diagnostics['Cluster_Deviation']}
    Volatility: {diagnostics['Volatility']}
    Anomaly Flag: {diagnostics['Anomaly_Flag']}
    Regression Component: {diagnostics['Regression_Component']}
    Confidence: {diagnostics['Confidence']}

    Give a short, clear explanation from a trader's perspective.
    """

    response = openai.ChatCompletion.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are an expert fixed income trader and risk analyst."},
            {"role": "user", "content": prompt}
        ],
        temperature=0.3
    )

    return response.choices[0].message["content"]
