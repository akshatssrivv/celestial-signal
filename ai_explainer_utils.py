def format_bond_diagnostics(row):
    return {
        "ISIN": row["ISIN"],
        "SECURITY_NAME": row["SECURITY_NAME"],
        "Date": str(row["Date"]),  # convert datetime to string if needed
        "Signal": row["SIGNAL"],
        "Composite Score": round(row["COMPOSITE_SCORE"], 2),
        "Residual Z-Score": round(row.get("Z_RESIDUAL_BUCKET", 0), 2),
        "Cluster Deviation": round(row.get("Cluster_Deviation_Flipped", 0), 2),
        "Volatility": round(row.get("Volatility", 0), 2),
        "Regression Component": round(row.get("Regression_Component", 0), 2),
    }

