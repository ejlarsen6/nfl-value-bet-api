import os
import polars as pl
import pandas as pd
import numpy as np
import requests
import nflreadpy as nfl
from dotenv import load_dotenv
load_dotenv()
print(os.getenv("ODDS_API_KEY"))

CURRENT_SEASON = 2026
ODDS_API_KEY = os.environ.get("ODDS_API_KEY")  # never hardcode this -- set as an env var
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/americanfootball_nfl/odds"

ROLLING_COLS = ["passing_epa_per_play", "rushing_epa_per_play", "passing_cpoe", "def_interceptions", "fumble_recovery_opp", "passing_interceptions", "fumbles_lost_total", \
                "team_point_diff", "def_sacks", "sacks_suffered", "passing_first_downs", "rushing_first_downs", "penalties", "penalty_yards", "def_qb_hits", "fg_pct"]
WINDOWS = (3, 8)

ODDS_API_TO_NFLVERSE = {
    "Arizona Cardinals": "ARI",
    "Atlanta Falcons": "ATL",
    "Baltimore Ravens": "BAL",
    "Buffalo Bills": "BUF",
    "Carolina Panthers": "CAR",
    "Chicago Bears": "CHI",
    "Cincinnati Bengals": "CIN",
    "Cleveland Browns": "CLE",
    "Dallas Cowboys": "DAL",
    "Denver Broncos": "DEN",
    "Detroit Lions": "DET",
    "Green Bay Packers": "GB",
    "Houston Texans": "HOU",
    "Indianapolis Colts": "IND",
    "Jacksonville Jaguars": "JAX",
    "Kansas City Chiefs": "KC",
    "Los Angeles Rams": "LA",
    "Los Angeles Chargers": "LAC",
    "Las Vegas Raiders": "LV",
    "Miami Dolphins": "MIA",
    "Minnesota Vikings": "MIN",
    "New England Patriots": "NE",
    "New Orleans Saints": "NO",
    "New York Giants": "NYG",
    "New York Jets": "NYJ",
    "Philadelphia Eagles": "PHI",
    "Pittsburgh Steelers": "PIT",
    "Seattle Seahawks": "SEA",
    "San Francisco 49ers": "SF",
    "Tampa Bay Buccaneers": "TB",
    "Tennessee Titans": "TEN",
    "Washington Commanders": "WAS",
}


# Get team form
def get_current_team_form(seasons: list[int]) -> pl.DataFrame:
    ts = nfl.load_team_stats(seasons=seasons)
    sch = nfl.load_schedules(seasons=seasons).filter(pl.col("home_score").is_not_null()).select(["game_id", "home_team", "away_team", "home_score", "away_score"])
    ts = ts.join(sch, on = "game_id", how = "inner")

    ts = ts.with_columns(
        is_home=pl.when(pl.col("team") == pl.col("home_team"))
        .then(1)
        .otherwise(0),
        team_score=pl.when(pl.col("team") == pl.col("home_team"))
        .then(pl.col("home_score"))
        .otherwise(pl.col("away_score")),
        opp_score=pl.when(pl.col("team") == pl.col("home_team"))
        .then(pl.col("away_score"))
        .otherwise(pl.col("home_score")),
    ).with_columns(
        team_point_diff=pl.col("team_score") - pl.col("opp_score")
    )

    ts = ts.with_columns([
        (pl.col("passing_epa") / pl.col("attempts")).alias("passing_epa_per_play"),
        (pl.col("rushing_epa") / pl.col("carries")).alias("rushing_epa_per_play"),
    ])
    ts = ts.sort(["team", "season", "week"])

    exprs = [
        pl.col(c).rolling_mean(window_size=w, min_samples=1).over("team").alias(f"{c}_roll{w}")
        for c in ROLLING_COLS for w in WINDOWS
    ]
    asof = ts.with_columns(exprs)
    form = asof.group_by("team", maintain_order=True).tail(1).select(
        ["team"] + [f"{c}_roll{w}" for c in ROLLING_COLS for w in WINDOWS]
    )
    return form

def get_upcoming_games(current_season: int) -> pl.DataFrame:
    sched = nfl.load_schedules(seasons=[current_season])
    upcoming = sched.filter(pl.col("home_score").is_null())
    return upcoming.select(["game_id", "season", "week", "home_team", "away_team"])

def attach_team_form(upcoming: pl.DataFrame, form: pl.DataFrame) -> pl.DataFrame:
    home_form = form.rename({c: f"home_{c}" for c in form.columns if c != "team"}).rename({"team": "home_team"})
    away_form = form.rename({c: f"away_{c}" for c in form.columns if c != "team"}).rename({"team": "away_team"})
    return upcoming.join(home_form, on="home_team", how="left").join(away_form, on="away_team", how="left")


def fetch_current_odds() -> pd.DataFrame:
    """
    Pulls current moneylines for upcoming NFL games from The Odds API.
    Returns one row per game with home/away team names and moneylines.
    """
    if ODDS_API_KEY is None:
        raise RuntimeError("Set the ODDS_API_KEY environment variable before running this.")

    params = {
        "apiKey": ODDS_API_KEY,
        "regions": "us",
        "markets": "h2h",       # moneyline market
        "oddsFormat": "american",
    }
    resp = requests.get(ODDS_API_URL, params=params, timeout=15)
    resp.raise_for_status()
    data = resp.json()

    rows = []
    for game in data:
        home_team = game["home_team"]
        away_team = game["away_team"]
        # Eventually average across bookmakers
        if not game.get("bookmakers"):
            continue
        outcomes = game["bookmakers"][0]["markets"][0]["outcomes"]
        home_ml = next((o["price"] for o in outcomes if o["name"] == home_team), None)
        away_ml = next((o["price"] for o in outcomes if o["name"] == away_team), None)
        rows.append({"home_team_full": home_team, "away_team_full": away_team,
                     "home_moneyline": home_ml, "away_moneyline": away_ml})

    return pd.DataFrame(rows)

def american_odds_to_prob(odds: float) -> float:
    """
    Converts an american odds floating point number into an implied probability. 
    """
    odds = int(odds)
    if odds < 0:
        prob = odds / (odds - 100)
    else:
        prob = 100 / (100 + odds)
    return prob

def prob_to_american(probability: float) -> int:
    """
    Converts an implied probability (0.0 to 1.0) into American odds.
    Returns a string representation with the correct '+' or '-' sign.
    """
    # Handle edge cases for invalid inputs
    if not (0.0 < probability < 1.0):
        raise ValueError("Probability must be strictly between 0 and 1.")

    if probability >= 0.5:
        odds = round((-100 * probability) / (1 - probability))
        return odds
    else:
        odds = round((100 * (1 - probability)) / probability)
        return odds

def devig(home_odds, away_odds):
    """
    Converts a pair of American moneylines into vig-free (fair) implied
    probabilities.
    """
    home_true_probs = []
    away_true_probs = []
    for i in range(len(home_odds)):
        home_odds_i = home_odds.iloc[i]
        away_odds_i = away_odds.iloc[i]

        if pd.isna(home_odds_i) or pd.isna(away_odds_i):
            home_true_probs.append(None)
            away_true_probs.append(None)
            continue

        home_prob = american_odds_to_prob(home_odds_i)
        away_prob = american_odds_to_prob(away_odds_i)
        market_width = home_prob + away_prob

        home_true_probs.append(home_prob / market_width)
        away_true_probs.append(away_prob / market_width)

    return (home_true_probs, away_true_probs)


def main(current_season: int, historical_seasons: list[int]):
    print("Computing current team form...")
    try:
        form = get_current_team_form(historical_seasons + [current_season])
    except ConnectionError:
        print(f"  No team stats available yet for {current_season} (season hasn't started) -- using historical seasons only.")
        form = get_current_team_form(historical_seasons)

    print("Pulling upcoming games...")
    upcoming = get_upcoming_games(current_season)

    print("Joining team form to upcoming games...")
    features = attach_team_form(upcoming, form)

    print("Fetching live odds...")
    odds = fetch_current_odds()
    odds["home_devigged_prob"], odds["away_devigged_prob"] = devig(
        odds["home_moneyline"], odds["away_moneyline"]
    )
    odds["home_team_abbr"] = odds["home_team_full"].map(ODDS_API_TO_NFLVERSE)
    odds["away_team_abbr"] = odds["away_team_full"].map(ODDS_API_TO_NFLVERSE)

    # TODO: join `features` (polars, nflverse abbreviations) to `odds`
    # (pandas, full team names) via a team-name mapping table -- this is
    # the piece that needs the 32-team lookup mentioned above.

    features_pd = features.to_pandas()

    final = features_pd.merge(odds, left_on=["home_team", "away_team"], right_on=["home_team_abbr", "away_team_abbr"], how="left").drop(columns = ["home_team_abbr", "away_team_abbr"])

    output_path = "data/current_features.parquet"
    final.to_parquet(output_path)
    print(f"Saved to {output_path}")


if __name__ == "__main__":
    main(current_season=2026, historical_seasons=list(range(2006, 2026)))