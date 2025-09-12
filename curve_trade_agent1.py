import streamlit as st
from openai import OpenAI
import pandas as pd

# -------------------------
# Load the top trades (pickle format)
# -------------------------
@st.cache_data
def load_trades():
    df = pd.read_pickle("top_trades_agent.pkl")
    # Convert any Timestamp columns to string
    for col in df.select_dtypes(include=['datetime64[ns]']).columns:
        df[col] = df[col].astype(str)
    return df

top_trades_agent = load_trades()

# -------------------------
# Initialize OpenAI client
# -------------------------
client = OpenAI(api_key=st.secrets["OPENAI_API_KEY"])

# -------------------------
# Build system prompt
# -------------------------
def get_system_prompt():
    # Convert top_trades_agent to JSON string for GPT (limited to top 50 trades for token efficiency)
    trade_json_str = top_trades_agent.head(50).to_dict(orient="records")
    
    system_prompt = f"""
You are a bond trading assistant for a hedge fund. You have access to the top trades below:

{trade_json_str}

Each trade has metrics including:
- ZSpreadDiffs (Leg1 and Leg2)
- Diff_of_Diffs
- CompositeDiffs
- ResidualDiffs
- ClusterDiffs
- VolatilityAvgs
- Direction
- RankScore
- LongEndFlag
- SignalPriorities (B1-B4)

Answer any questions from the user about these trades. You can:
- Explain why a trade is attractive
- Compare trades
- Rank or filter trades by metrics
- Highlight top trades

Only use the data provided; do not hallucinate numbers.
Be concise, human-readable, and actionable.
"""
    return system_prompt

# -------------------------
# Chat function
# -------------------------
def chat_with_trades(user_input, history):
    # Ensure system prompt is first in conversation
    if not history:
        history = [{"role": "system", "content": get_system_prompt()}]

    messages = history + [{"role": "user", "content": user_input}]
    
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=messages,
        temperature=0
    )
    
    answer = response.choices[0].message.content
    messages.append({"role": "assistant", "content": answer})
    return answer, messages
