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

events["ts"] = pd.to_datetime(events["ts"], errors="coerce")

# Sidebar filters
with st.sidebar:
    st.header("Filters")

    min_dt = events["ts"].min()
    max_dt = events["ts"].max()

    date_range = st.date_input(
        "Date range",
        value=(min_dt.date(), max_dt.date()),
        min_value=min_dt.date(),
        max_value=max_dt.date(),
    )

    def pick_multiselect(col, label):
        if col in events.columns:
            opts = sorted([x for x in events[col].dropna().unique().tolist()])
            return st.multiselect(label, options=opts, default=[])
        return []

    practices = pick_multiselect("practice", "Practice")
    levels = pick_multiselect("level", "Level")
    locations = pick_multiselect("location", "Location")

# Apply filters
f = events.copy()
start, end = date_range
f = f[(f["ts"].dt.date >= start) & (f["ts"].dt.date <= end)]

if practices:
    f = f[f["practice"].isin(practices)]
if levels:
    f = f[f["level"].isin(levels)]
if locations:
    f = f[f["location"].isin(locations)]

# KPI row
c1, c2, c3, c4 = st.columns(4)
c1.metric("Events", f"{len(f):,}")
c2.metric("Total Tokens", f"{int(f['tokens'].sum()):,}")
c3.metric("Unique Users", f"{f['email'].nunique():,}")
c4.metric("Unique Sessions", f"{f['session_id'].nunique():,}")

st.divider()

# Charts row 1
left, right = st.columns(2)

by_level = f.groupby("level", dropna=False)["tokens"].sum().reset_index().sort_values("tokens", ascending=False)
left.plotly_chart(px.bar(by_level, x="level", y="tokens", title="Tokens by Level"), use_container_width=True)

by_practice = f.groupby("practice", dropna=False)["tokens"].sum().reset_index().sort_values("tokens", ascending=False)
right.plotly_chart(px.bar(by_practice, x="practice", y="tokens", title="Tokens by Practice"), use_container_width=True)

st.divider()

# Charts row 2
left2, right2 = st.columns(2)

by_hour = f.groupby("hour", dropna=False)["tokens"].sum().reset_index().sort_values("hour")
left2.plotly_chart(px.line(by_hour, x="hour", y="tokens", title="Tokens by Hour (Peak Usage)"), use_container_width=True)

by_day = f.groupby("date", dropna=False)["tokens"].sum().reset_index().sort_values("date")
right2.plotly_chart(px.line(by_day, x="date", y="tokens", title="Tokens by Day"), use_container_width=True)

st.divider()

# Tables
st.subheader("Top Users by Tokens (Filtered)")
top_users = f.groupby("email")["tokens"].sum().reset_index().sort_values("tokens", ascending=False).head(25)
st.dataframe(top_users, use_container_width=True)

st.subheader("Top Sessions by Tokens (Filtered)")
top_sessions = (
    f.groupby("session_id", dropna=False)
    .agg(tokens=("tokens", "sum"), events=("message", "count"), users=("email", "nunique"))
    .reset_index()
    .sort_values("tokens", ascending=False)
    .head(25)
)
top_sessions["tokens_per_event"] = (top_sessions["tokens"] / top_sessions["events"]).round(2)
st.dataframe(top_sessions, use_container_width=True)

st.subheader("Sessions Summary (Global, top 20)")
st.dataframe(sessions.head(20), use_container_width=True)