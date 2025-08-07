import streamlit as st
import pandas as pd

csv_url = "https://lpxiwnvxqozkjlgfrbfh.supabase.co/storage/v1/object/public/celestial-signal/today_all_signals.csv"
df = pd.read_csv(csv_url)

df = pd.read_csv('today_all_signals.csv')

st.title("Bond Mispricing Signals Dashboard")

st.dataframe(df)

isin_filter = st.multiselect('Filter by ISIN', options=df['ISIN'].unique())
if isin_filter:
    df = df[df['ISIN'].isin(isin_filter)]
    st.dataframe(df)
