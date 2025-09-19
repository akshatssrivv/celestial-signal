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
    """
    Determine curve positioning:
    - Steepener: Long long-end, Short short-end
    - Flattener: Long short-end, Short long-end
    - Otherwise: Complex / Mixed
    """
    short_signals = [row['A_Signal'], row['C_Signal']]
    long_signals = [row['B_Signal'], row['D_Signal']]
    if any(s.endswith("BUY") for s in long_signals) and any(s.endswith("SELL") for s in short_signals):
        return "Steepener"
    elif any(s.endswith("SELL") for s in long_signals) and any(s.endswith("BUY") for s in short_signals):
        return "Flattener"
    else:
        return "Complex / Mixed"

top_trades_agent['Leg_Direction'] = top_trades_agent.apply(compute_leg_direction, axis=1)

# -------------------------
# Actionability / Confidence
# -------------------------
def compute_confidence(row):
    """
    Confidence is based on multiple diagnostic dimensions:
    - Ranking_Score
    - Trade_ZDiff_30D_Pct
    - Absolute deviation
    - Diff of diffs
    """
    positive_signals = sum([
        row['Trade_ZDiff_30D_Pct'] > 0.8,
        row['Ranking_Score'] >= 85,
        abs(row['Abs_Deviation_30D_bps']) > 10,
        abs(row['Diff_of_Diffs_Today']) > abs(row['Rolling_Std_DoD'])
    ])
    if positive_signals >= 3:
        return 'High'
    elif positive_signals == 2:
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
  - A: {row['A_ISIN']}, {row['A_Name']}, Mat: {row['A_Maturity']}, Signal: {row['A_Signal']}
    DV01: {row['Target_DV01_A']}, Notional: {row['Notional_A']}, Z: {row['Z_A']}, YAS_RISK_1M: {row['A_YAS_RISK_1M']}
  - B: {row['B_ISIN']}, {row['B_Name']}, Mat: {row['B_Maturity']}, Signal: {row['B_Signal']}
    DV01: {row['Target_DV01_B']}, Notional: {row['Notional_B']}, Z: {row['Z_B']}, YAS_RISK_1M: {row['B_YAS_RISK_1M']}

Leg2 ({row['LEG_2']}):
  - C: {row['C_ISIN']}, {row['C_Name']}, Mat: {row['C_Maturity']}, Signal: {row['C_Signal']}
    DV01: {row['Target_DV01_C']}, Notional: {row['Notional_C']}, Z: {row['Z_C']}, YAS_RISK_1M: {row['C_YAS_RISK_1M']}
  - D: {row['D_ISIN']}, {row['D_Name']}, Mat: {row['D_Maturity']}, Signal: {row['D_Signal']}
    DV01: {row['Target_DV01_D']}, Notional: {row['Notional_D']}, Z: {row['Z_D']}, YAS_RISK_1M: {row['D_YAS_RISK_1M']}

Leg_Direction: {row['Leg_Direction']}
Diagnostics:
  Trade_ZDiff_30D_Pct: {row['Trade_ZDiff_30D_Pct']}
  Diff_of_Diffs_Today: {row['Diff_of_Diffs_Today']}
  Abs_Deviation_30D_bps: {row['Abs_Deviation_30D_bps']}
  Rolling_Mean_DoD: {row['Rolling_Mean_DoD']}
  Rolling_Std_DoD: {row['Rolling_Std_DoD']}
"""
        trade_summaries.append(summary)
    trade_text = "\n".join(trade_summaries)

    system_prompt = f"""
You are a bond trading assistant. When explaining trades:
- Always justify each bond’s signal using diagnostics (Z-scores, deviations, risk metrics).
- Divide into Leg1 and Leg2, and give Analyst Notes for each bond.
- Explain curve logic: Steepener / Flattener / Complex.
- Summarize with 2–3 actionable bullets.
- Include confidence (High/Medium/Low).
- Never hallucinate numbers. Only use provided fields.

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
        temperature=0.2,  # small variation for natural tone
        max_tokens=1200   # ensure enough space for reasoning
    )

    answer = response.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})
    return answer, messages
