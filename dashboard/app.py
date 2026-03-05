import pandas as pd
import streamlit as st
from pathlib import Path
import plotly.express as px

st.set_page_config(page_title="Claude Code Analytics", layout="wide")

DATA_DIR = Path("data/processed")

@st.cache_data
def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / name)

st.title("Claude Code Usage Analytics")

events = load_csv("events_processed.csv")
sessions = load_csv("sessions_summary.csv")

c1, c2, c3 = st.columns(3)
with c1:
    st.metric("Events", f"{len(events):,}")
with c2:
    st.metric("Sessions", f"{len(sessions):,}")
with c3:
    st.metric("Total Tokens", f"{events['tokens'].sum():,}")

st.divider()

left, right = st.columns(2)

tokens_by_hour = load_csv("tokens_by_hour.csv")
left.plotly_chart(px.line(tokens_by_hour, x="hour", y="tokens", title="Tokens by Hour (Peak Usage)"), use_container_width=True)

tokens_by_day = load_csv("tokens_by_day.csv")
right.plotly_chart(px.line(tokens_by_day, x="date", y="tokens", title="Tokens by Day"), use_container_width=True)

st.divider()
st.subheader("Sessions Summary (Top 20 by Tokens)")
st.dataframe(sessions.head(20), use_container_width=True)