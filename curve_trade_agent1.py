import streamlit as st
import pandas as pd
from openai import OpenAI

# -------------------------
# Load top trades
# -------------------------
@st.cache_data
def load_trades():
    df = pd.read_pickle("top_trades_agent.pkl")
    # Convert datetime to string
    for col in df.select_dtypes(include=['datetime64[ns]']).columns:
        df[col] = df[col].astype(str)
    return df

top_trades_agent = load_trades()

# -------------------------
# Merge Z-Spreads for each bond (today & 30D ago)
# -------------------------
# Assuming you have per-bond signals/history with ZSpread columns
# today_all_signals['ISIN', 'ZSpread'], signals_history_df['ISIN', 'ZSpread_30D_Ago']

def merge_zspreads(trades_df, today_df, history_df):
    for leg in ['Bond1','Bond2','Bond3','Bond4']:
        trades_df[f'{leg}_ZSpread'] = trades_df[leg+'_ISIN'].map(
            today_df.set_index('ISIN')['ZSpread']
        )
        trades_df[f'{leg}_ZSpread_30D'] = trades_df[leg+'_ISIN'].map(
            history_df.set_index('ISIN')['ZSpread_30D_Ago']
        )
    # Compute leg-level
    trades_df['Leg1_ZSpread'] = trades_df['Bond2_ZSpread'] - trades_df['Bond1_ZSpread']
    trades_df['Leg2_ZSpread'] = trades_df['Bond4_ZSpread'] - trades_df['Bond3_ZSpread']
    trades_df['Trade_ZSpread'] = (trades_df['Bond4_ZSpread'] - trades_df['Bond3_ZSpread'] -
                                  trades_df['Bond2_ZSpread'] + trades_df['Bond1_ZSpread'])
    # Compute 30D changes
    trades_df['Leg1_ZSpread_Change'] = trades_df['Leg1_ZSpread'] - (trades_df['Bond2_ZSpread_30D'] - trades_df['Bond1_ZSpread_30D'])
    trades_df['Leg2_ZSpread_Change'] = trades_df['Leg2_ZSpread'] - (trades_df['Bond4_ZSpread_30D'] - trades_df['Bond3_ZSpread_30D'])
    trades_df['Trade_ZSpread_Change'] = trades_df['Trade_ZSpread'] - ((trades_df['Bond4_ZSpread_30D'] - trades_df['Bond3_ZSpread_30D']) - (trades_df['Bond2_ZSpread_30D'] - trades_df['Bond1_ZSpread_30D']))
    return trades_df

# -------------------------
# Compute Leg Steepener/Flattener
# -------------------------
def compute_leg_direction(row):
    # Buy AD / Sell BC → Leg1 Steepener, Leg2 Flattener
    bond1_sig = row['Bond1_SIGNAL']
    bond2_sig = row['Bond2_SIGNAL']
    bond3_sig = row['Bond3_SIGNAL']
    bond4_sig = row['Bond4_SIGNAL']

    # Determine if AD is buy / BC is sell
    ad_buy = bond1_sig.endswith('BUY') and bond4_sig.endswith('BUY')
    bc_sell = bond2_sig.endswith('SELL') and bond3_sig.endswith('SELL')

    if ad_buy and bc_sell:
        return 'Leg1: Steepener, Leg2: Flattener'
    elif not ad_buy and not bc_sell:
        return 'Leg1: Flattener, Leg2: Steepener'
    else:
        return 'Mixed / Complex'

top_trades_agent['Leg_Direction'] = top_trades_agent.apply(compute_leg_direction, axis=1)

# -------------------------
# Initialize OpenAI
# -------------------------
client = OpenAI()

# -------------------------
# Build system prompt
# -------------------------
def get_system_prompt(top_trades):
    # Only top 50 trades for efficiency
    trade_json_str = top_trades.head(50).to_dict(orient="records")
    
    system_prompt = f"""
You are a bond trading assistant for a hedge fund. You always wear three hats:
1. Trader: explains trade legs, steepeners/flatteners, and actionable signals
2. Quant: gives z-spreads, leg-level and trade-level, 30-day changes, and metrics
3. Storyteller: explains trades in concise, human-readable terms

Here are the top trades (max 50 for token efficiency):

{trade_json_str}

Whenever asked, you will:
- Explain the top N trades
- Divide into Leg1 (AB) and Leg2 (CD)
- Show z-spreads per bond, per leg, trade-level, and 30-day change
- Show signals per bond
- Indicate steepener/flattener for each leg
- Include rank and Rank_Score
- Be concise, actionable, and human-readable
- Never hallucinate numbers; only use data provided
"""
    return system_prompt

# -------------------------
# Chat function
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

# -------------------------
# Streamlit UI
# -------------------------
st.title("Bond AI Trade Assistant")
st.write("Ask about top trades, legs, Z-spreads, signals, and steepener/flattener analysis.")

user_input = st.text_area("Your question to the agent:")

if st.button("Ask"):
    answer, _ = chat_with_trades(user_input, history=[])
    st.markdown(answer)
