import streamlit as st
import pandas as pd
import numpy as np
import altair as alt
from openai import OpenAI

# -------------------------
# Load trades
# -------------------------
@st.cache_data
def load_trades():
    df = pd.read_pickle("top_trades_agent.pkl")
    return df

top_trades_agent = load_trades()

# -------------------------
# Compute Leg Direction
# -------------------------
def compute_leg_direction(row):
    ad_buy = row['A_Signal'].endswith('BUY') and row['D_Signal'].endswith('BUY')
    bc_sell = row['B_Signal'].endswith('SELL') and row['C_Signal'].endswith('SELL')
    if ad_buy and bc_sell:
        return 'Leg1: Steepener, Leg2: Flattener'
    elif not ad_buy and not bc_sell:
        return 'Leg1: Flattener, Leg2: Steepener'
    else:
        return 'Mixed / Complex'

top_trades_agent['Leg_Direction'] = top_trades_agent.apply(compute_leg_direction, axis=1)

# -------------------------
# Actionability / Confidence
# -------------------------
def compute_confidence(row):
    # You can adjust thresholds based on Ranking_Score and ZDiff
    if row['Ranking_Score'] >= 90 and abs(row['Trade_ZDiff_30D_Pct']) > 0.8:
        return 'High'
    elif row['Ranking_Score'] >= 70:
        return 'Medium'
    else:
        return 'Low'

top_trades_agent['Confidence'] = top_trades_agent.apply(compute_confidence, axis=1)

# -------------------------
# OpenAI client
# -------------------------
client = OpenAI()

# -------------------------
# GPT system prompt
# -------------------------
def get_system_prompt(top_trades):
    trade_summaries = []
    for i, row in top_trades.head(50).iterrows():
        summary = f"""
Trade {i+1} | Rank: {row['Ranking_Score']} | Confidence: {row['Confidence']} | Actionable: {row['Actionable_Direction']}
Leg1 ({row['LEG_1']}):
  - A: {row['A_ISIN']}, Name: {row['A_Name']}, Maturity: {row['A_Maturity']}, Signal: {row['A_Signal']}, DV01: {row['Target_DV01_A']}, Notional: {row['Notional_A']}, Z: {row['Z_A']}, YAS_RISK_1M: {row['A_YAS_RISK_1M']}
  - B: {row['B_ISIN']}, Name: {row['B_Name']}, Maturity: {row['B_Maturity']}, Signal: {row['B_Signal']}, DV01: {row['Target_DV01_B']}, Notional: {row['Notional_B']}, Z: {row['Z_B']}, YAS_RISK_1M: {row['B_YAS_RISK_1M']}
Leg2 ({row['LEG_2']}):
  - C: {row['C_ISIN']}, Name: {row['C_Name']}, Maturity: {row['C_Maturity']}, Signal: {row['C_Signal']}, DV01: {row['Target_DV01_C']}, Notional: {row['Notional_C']}, Z: {row['Z_C']}, YAS_RISK_1M: {row['C_YAS_RISK_1M']}
  - D: {row['D_ISIN']}, Name: {row['D_Name']}, Maturity: {row['D_Maturity']}, Signal: {row['D_Signal']}, DV01: {row['Target_DV01_D']}, Notional: {row['Notional_D']}, Z: {row['Z_D']}, YAS_RISK_1M: {row['D_YAS_RISK_1M']}
Leg_Direction: {row['Leg_Direction']}
Trade_ZDiff_30D_Pct: {row['Trade_ZDiff_30D_Pct']}, Diff_of_Diffs_Today: {row['Diff_of_Diffs_Today']}
"""
        trade_summaries.append(summary)
    trade_text = "\n".join(trade_summaries)

    system_prompt = f"""
You are a bond trading assistant. Explain trades concisely, actionably, and in human-readable form. Always:
- Divide trades into Leg1 and Leg2
- Show per-bond signals, DV01, Notional, Maturity, YAS_RISK_1M, Z-values
- Indicate Leg Direction: Steepener / Flattener
- Prioritize top trades by Ranking_Score
- Include confidence levels (High/Medium/Low)
- Never hallucinate numbers

Top trades:

{trade_text}
"""
    return system_prompt

# -------------------------
# GPT chat function
# -------------------------
def chat_with_trades(user_input, history):
    if not history:
        history = [{"role": "system", "content": get_system_prompt(top_trades_agent)}]

    messages = history + [{"role": "user", "content": user_input}]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )

    answer = response.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})
    return answer, messages
