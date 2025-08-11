def format_bond_diagnostics(row):
    return {
        "ISIN": row["ISIN"],
        "SECURITY_NAME": row["SECURITY_NAME"],
        "Date": str(row["Date"]),  # convert datetime to string if needed
        "Signal": row["SIGNAL"],
        "Composite_Score": round(row["COMPOSITE_SCORE"], 2),
        "Residual_Z": round(row.get("Z_Residual_Score", 0), 2),
        "Cluster_Deviation": round(row.get("Cluster_Deviation", 0), 2),
        "Volatility": round(row.get("Volatility", 0), 2),
        "Anomaly_Flag": bool(row.get("Anomaly_Flag", False)),
        "Regression_Component": round(row.get("Regression_Component", 0), 2),
        "Confidence": round(row.get("Confidence", 0), 2),
        # Add any other relevant columns here
    }

