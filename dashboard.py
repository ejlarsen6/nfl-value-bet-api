"""
dashboard.py -- Streamlit dashboard for the NFL value-bet model.

This is a pure CLIENT of the deployed API -- it doesn't load the model or
features itself. It calls /games to discover this week's matchups, then
/predict for each one, and presents the results.

Run locally:
    streamlit run dashboard.py
"""

import streamlit as st
import pandas as pd
import requests

# Point this at your live Render URL once deployed.
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://127.0.0.1:8000")

st.set_page_config(page_title="NFL Value-Bet Dashboard", layout="wide")

st.title("🏈 NFL Model vs. Market Dashboard")

st.warning(
    "**Educational / analytical project only.** This tool compares a statistical "
    "model against betting market prices for research purposes. It is not "
    "betting advice, and past patterns here do not predict future results.",
    icon="⚠️",
)


@st.cache_data(ttl=300)
def fetch_games():
    resp = requests.get(f"{API_BASE_URL}/games", timeout=10)
    resp.raise_for_status()
    return resp.json()


@st.cache_data(ttl=300)
def fetch_prediction(season, week, home_team, away_team):
    resp = requests.post(
        f"{API_BASE_URL}/predict",
        json={"season": season, "week": week, "home_team": home_team, "away_team": away_team},
        timeout=20,
    )
    resp.raise_for_status()
    return resp.json()


try:
    games = fetch_games()
except requests.exceptions.RequestException as e:
    st.error(f"Couldn't reach the API at {API_BASE_URL}. Is it running/deployed? ({e})")
    st.stop()

if not games:
    st.info("No upcoming games found in the current feature table.")
    st.stop()

rows = []
for g in games:
    print(g, type(g))
    try:
        pred = fetch_prediction(g["season"], g["week"], g["home_team"], g["away_team"])
        print(pred)
        rows.append({
            "Matchup": f"{g['away_team']} @ {g['home_team']}",
            "Week": g["week"],
            "Model Prob (Home)": pred["model_home_win_prob"],
            "Market Prob (Home)": pred["market_home_win_prob"],
            "Edge": pred["edge"],
            "Value Flag": "🔥 Value" if pred["value_flag"] else "",
        })
    except requests.exceptions.RequestException as e:
        st.error(f"Request Error: {e}")
        continue  # skip games the API couldn't score, rather than crashing the whole dashboard

df = pd.DataFrame(rows)
print(len(df), df.columns)
df["Abs Edge"] = df["Edge"].abs()
df = df.sort_values("Abs Edge", ascending=False).drop(columns="Abs Edge")

st.subheader(f"Week {games[0]['week']} — {len(df)} games")


def highlight_value(row):
    color = "background-color: #fff3cd" if row["Value Flag"] else ""
    return [color] * len(row)


st.dataframe(
    df.style.apply(highlight_value, axis=1).format({
        "Model Prob (Home)": "{:.1%}",
        "Market Prob (Home)": "{:.1%}",
        "Edge": "{:+.1%}",
    }),
    use_container_width=True,
    hide_index=True,
)

st.caption(
    "Sorted by absolute edge (largest model/market disagreement first). "
    "Flagged rows exceed the model's configured value threshold."
)