from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
import xgboost as xgb
import numpy as np
import pandas as pd
import nflreadpy as nfl

app = FastAPI(title="NFL Pred API", version="0.1.0")

VALUE_THRESHOLD = 0.05
MODEL_PATH = 'model_v1.json'
FEATURES_PATH = '../data/current_features.parquet'
MODEL_FEATURE_COLS = [
'home_passing_epa_per_play_roll3',
 'home_passing_epa_per_play_roll8',
 'home_rushing_epa_per_play_roll3',
 'home_rushing_epa_per_play_roll8',
 'home_passing_cpoe_roll3',
 'home_passing_cpoe_roll8',
 'home_def_interceptions_roll3',
 'home_def_interceptions_roll8',
 'home_fumble_recovery_opp_roll3',
 'home_fumble_recovery_opp_roll8',
 'home_passing_interceptions_roll3',
 'home_passing_interceptions_roll8',
 'home_fumbles_lost_total_roll3',
 'home_fumbles_lost_total_roll8',
 'home_team_point_diff_roll3',
 'home_team_point_diff_roll8',
 'home_def_sacks_roll3',
 'home_def_sacks_roll8',
 'home_sacks_suffered_roll3',
 'home_sacks_suffered_roll8',
 'home_passing_first_downs_roll3',
 'home_passing_first_downs_roll8',
 'home_rushing_first_downs_roll3',
 'home_rushing_first_downs_roll8',
 'home_penalties_roll3',
 'home_penalties_roll8',
 'home_penalty_yards_roll3',
 'home_penalty_yards_roll8',
 'home_def_qb_hits_roll3',
 'home_def_qb_hits_roll8',
 'home_fg_pct_roll3',
 'home_fg_pct_roll8',
 'away_passing_epa_per_play_roll3',
 'away_passing_epa_per_play_roll8',
 'away_rushing_epa_per_play_roll3',
 'away_rushing_epa_per_play_roll8',
 'away_passing_cpoe_roll3',
 'away_passing_cpoe_roll8',
 'away_def_interceptions_roll3',
 'away_def_interceptions_roll8',
 'away_fumble_recovery_opp_roll3',
 'away_fumble_recovery_opp_roll8',
 'away_passing_interceptions_roll3',
 'away_passing_interceptions_roll8',
 'away_fumbles_lost_total_roll3',
 'away_fumbles_lost_total_roll8',
 'away_team_point_diff_roll3',
 'away_team_point_diff_roll8',
 'away_def_sacks_roll3',
 'away_def_sacks_roll8',
 'away_sacks_suffered_roll3',
 'away_sacks_suffered_roll8',
 'away_passing_first_downs_roll3',
 'away_passing_first_downs_roll8',
 'away_rushing_first_downs_roll3',
 'away_rushing_first_downs_roll8',
 'away_penalties_roll3',
 'away_penalties_roll8',
 'away_penalty_yards_roll3',
 'away_penalty_yards_roll8',
 'away_def_qb_hits_roll3',
 'away_def_qb_hits_roll8',
 'away_fg_pct_roll3',
 'away_fg_pct_roll8'
]

model = xgb.XGBClassifier()
try:
    model.load_model(MODEL_PATH)
    MODEL_LOADED = True
except Exception as e:
    MODEL_LOADED = False
    MODEL_LOAD_ERROR = str(e)

try:
    features_df = pd.read_parquet(FEATURES_PATH)
    features_df = features_df.set_index(["season", "week", "home_team", "away_team"])
    FEATURES_LOADED = True
except Exception as e:
    FEATURES_LOADED = False
    FEATURES_LOAD_ERROR = str(e)

class Matchup(BaseModel):
    season: int
    week: int
    home_team: str
    away_team: str

class PredictionResponse(BaseModel):
    model_home_win_prob: float
    market_home_win_prob: float
    edge: float
    value_flag: bool

class GameResponse(BaseModel):
    season: int
    week: int
    away_team: str
    home_team: str
    gamedate: str
    gametime: str


@app.get("/health")
def health():
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail=f"Model failed to load: {MODEL_LOAD_ERROR}")
    return {"status": "ok", "model_loaded": True}

@app.get("/games")
async def games() -> list[GameResponse]:
    sched = nfl.load_schedules([nfl.get_current_season()]).to_pandas()
    current_week = nfl.get_current_week()
    return [
        GameResponse(season =game.season, week = game.week, away_team=game.away_team, home_team = game.home_team, gamedate = game.gameday, gametime = game.gametime) for game in sched.itertuples() if game.week >= current_week
    ]

@app.get("/debug/lookup")
def debug_lookup(season: int, week: int, home_team: str, away_team: str):
    """TEMPORARY diagnostic endpoint -- remove once the lookup bug is found.
    Reproduces the exact lookup /predict does, but returns full diagnostic
    detail instead of a bare 404, so this can be tested without shell access."""
    if not FEATURES_LOADED:
        raise HTTPException(status_code=503, detail=f"Feature table not loaded: {FEATURES_LOAD_ERROR}")
 
    key = (season, week, home_team, away_team)
    result = {
        "requested_key": {"season": season, "week": week, "home_team": home_team, "away_team": away_team},
        "requested_key_types": [str(type(k)) for k in key],
        "index_level_dtypes": {str(name): str(dtype) for name, dtype in zip(features_df.index.names, features_df.index.dtypes)},
        "index_has_duplicates": bool(features_df.index.duplicated().any()),
    }
 
    close_matches = [
        {"season": idx[0], "week": idx[1], "home_team": idx[2], "away_team": idx[3],
         "types": [str(type(v)) for v in idx]}
        for idx in features_df.index if idx[0] == season and idx[1] == week
    ]
    result["rows_with_matching_season_week"] = close_matches
 
    try:
        row = features_df.loc[key]
        result["lookup_succeeded"] = True
    except KeyError as e:
        result["lookup_succeeded"] = False
        result["key_error"] = str(e)
 
    return result

@app.post("/predict", response_model=PredictionResponse)
def predict(matchup: Matchup):
    if not MODEL_LOADED:
        raise HTTPException(status_code=503, detail="Model not loaded")
 
    key = (matchup.season, matchup.week, matchup.home_team, matchup.away_team)
    try:
        row = features_df.loc[key]
    except (KeyError, NameError):
        raise HTTPException(
            status_code=404,
            detail=f"No features found for {matchup.away_team} @ {matchup.home_team}, "
                   f"season {matchup.season} week {matchup.week}. "
                   f"Check team abbreviations and that this week's features have been computed."
        )
 
    X = np.array([row[MODEL_FEATURE_COLS].values.astype(float)])
    model_prob = float(model.predict_proba(X)[:, 1][0])
    market_prob = float(row["home_devigged_prob"])
 
    edge = model_prob - market_prob
    value_flag = abs(edge) > VALUE_THRESHOLD
 
    return PredictionResponse(
        model_home_win_prob=round(model_prob, 4),
        market_home_win_prob=round(market_prob, 4),
        edge=round(edge, 4),
        value_flag=value_flag,
    )