from openai import OpenAI

client = OpenAI()

def generate_ai_explanation(diagnostics):
    prompt = f"""
    You are a bond trading analyst. 
    Explain why the bond {diagnostics['SECURITY_NAME']} ({diagnostics['ISIN']}) 
    has a signal of {diagnostics['SIGNAL']} with composite score {diagnostics['COMPOSITE_SCORE']} on {diagnostics['Date']}.

    Context:
    Residual Z-Score: {diagnostics['Z_RESIDUAL_BUCKET']}
    Cluster Deviation: {diagnostics['Cluster_Deviation_Flipped']}
    Volatility: {diagnostics['Volatility']}
    Regression Component: {diagnostics['Regression_Component']}

    Give a short, clear explanation from a trader's perspective.
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
    return {
        "ISIN": row["ISIN"],
        "SECURITY_NAME": row["SECURITY_NAME"],
        "Date": str(row["Date"]),  # keep string if needed
        "SIGNAL": row["SIGNAL"],  # exact column name
        "COMPOSITE_SCORE": round(row["COMPOSITE_SCORE"], 2),
        "Z_RESIDUAL_BUCKET": round(row.get("Z_RESIDUAL_BUCKET", 0), 2),
        "Cluster_Deviation_Flipped": round(row.get("Cluster_Deviation_Flipped", 0), 2),
        "Volatility": round(row.get("Volatility", 0), 2),
        "Regression_Component": round(row.get("Regression_Component", 0), 2),
    }
