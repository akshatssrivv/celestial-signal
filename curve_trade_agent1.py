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
# Merge Z-Spreads (today & 30D ago)
# -------------------------
def merge_zspreads(trades_df, today_df, history_df):
    for leg in ['Bond1','Bond2','Bond3','Bond4']:
        trades_df[f'{leg}_ZSpread'] = trades_df[leg+'_ISIN'].map(today_df.set_index('ISIN')['ZSpread'])
        trades_df[f'{leg}_ZSpread_30D'] = trades_df[leg+'_ISIN'].map(history_df.set_index('ISIN')['ZSpread_30D_Ago'])
    
    # Leg-level and trade-level
    trades_df['Leg1_ZSpread'] = trades_df['Bond2_ZSpread'] - trades_df['Bond1_ZSpread']
    trades_df['Leg2_ZSpread'] = trades_df['Bond4_ZSpread'] - trades_df['Bond3_ZSpread']
    trades_df['Trade_ZSpread'] = (trades_df['Bond4_ZSpread'] - trades_df['Bond3_ZSpread']) - (trades_df['Bond2_ZSpread'] - trades_df['Bond1_ZSpread'])
    
    # 30D changes
    trades_df['Leg1_ZSpread_Change'] = trades_df['Leg1_ZSpread'] - (trades_df['Bond2_ZSpread_30D'] - trades_df['Bond1_ZSpread_30D'])
    trades_df['Leg2_ZSpread_Change'] = trades_df['Leg2_ZSpread'] - (trades_df['Bond4_ZSpread_30D'] - trades_df['Bond3_ZSpread_30D'])
    trades_df['Trade_ZSpread_Change'] = trades_df['Trade_ZSpread'] - ((trades_df['Bond4_ZSpread_30D'] - trades_df['Bond3_ZSpread_30D']) - (trades_df['Bond2_ZSpread_30D'] - trades_df['Bond1_ZSpread_30D']))
    
    return trades_df

# -------------------------
# Compute Leg Steepener/Flattener
# -------------------------
def compute_leg_direction(row):
    bond1_sig, bond2_sig, bond3_sig, bond4_sig = row['Bond1_SIGNAL'], row['Bond2_SIGNAL'], row['Bond3_SIGNAL'], row['Bond4_SIGNAL']
    
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
# OpenAI client
# -------------------------
client = OpenAI()

# -------------------------
# System prompt
# -------------------------
def get_system_prompt(top_trades):
    trade_json_str = top_trades.head(50).to_dict(orient="records")
    
    system_prompt = f"""
You are a bond trading assistant for a hedge fund. You wear three hats at all times:
1. Trader: explains trade legs, steepeners/flatteners, actionable signals
2. Quant: provides Z-Spreads, leg-level and trade-level, 30-day changes, metrics
3. Storyteller: explains trades concisely in human-readable form

Here are the top trades (max 50):

{trade_json_str}

Whenever asked, you will:
- Explain the top N trades
- Divide into Leg1 (AB) and Leg2 (CD)
- Show Z-spreads per bond, per leg, trade-level, and 30-day change
- Show signals per bond
- Indicate steepener/flattener for each leg
- Include rank and Rank_Score
- Be concise, actionable, human-readable
- Never hallucinate numbers; only use provided data
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

# --- Initialize session state ---
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [{"role": "system", "content": get_system_prompt(top_trades_agent)}]
if "chat_input" not in st.session_state:
    st.session_state.chat_input = ""
if "last_processed_input" not in st.session_state:
    st.session_state.last_processed_input = ""

# --- Chat input ---
user_input = st.text_input("Your question:", key="chat_input_box", placeholder="Type your question here...")

if st.button("Send"):
    current_input = user_input.strip()
    if current_input and current_input != st.session_state.last_processed_input:
        # Append user message
        st.session_state.chat_history.append({"role": "user", "content": current_input})
        
        # Get assistant response
        answer, _ = chat_with_trades(current_input, st.session_state.chat_history)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})
        
        # Update last processed input
        st.session_state.last_processed_input = current_input
        
        # Clear input box
        st.session_state.chat_input_box = ""
        
        # Rerun to display updated chat
        st.experimental_rerun()

# --- Display conversation ---
for i, msg in enumerate(st.session_state.chat_history[1:]):
    is_user = msg["role"] == "user"
    st.markdown(f"**{'You' if is_user else 'Assistant'}:** {msg['content']}")
