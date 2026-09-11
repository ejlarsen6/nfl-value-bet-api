"""
pull_data.py

Pulls raw NFL data needed for the outcome-prediction + value-bet model:
  1. Schedules — game results PLUS historical closing spread/moneyline/total
     (nflverse bakes real historical odds into this table, so you don't need
     a paid odds-API historical tier just to get started)
  2. Team-level weekly stats — for building "recent form" features

Data source: nflverse (https://github.com/nflverse/nflverse-data), accessed
via the nflreadpy package. Free, no API key, no rate limits.

Run:
    pip install nflreadpy --break-system-packages   # or just `pip install nflreadpy`
    python pull_data.py
"""

import nflreadpy as nfl
import polars as pl
from pathlib import Path

DATA_DIR = Path(__file__).parent / "data"
DATA_DIR.mkdir(exist_ok=True)

# ---------------------------------------------------------------------------
# Config: how far back to pull. Moneylines are populated from 2006 onward;
# spreads go back to 1999. 2006+ gives ~20 seasons — plenty for a first model,
# and keeps the "era" of the league reasonably consistent (modern passing rules).
# ---------------------------------------------------------------------------
START_SEASON = 2006
END_SEASON = 2025          # last fully completed season; current season pulled separately below
CURRENT_SEASON = 2026


def pull_schedules() -> pl.DataFrame:
    """Game-level results + closing odds. This is your label + market-baseline source."""
    seasons = list(range(START_SEASON, END_SEASON + 1))
    sched = nfl.load_schedules(seasons=seasons)

    # Keep regular season + playoffs, drop preseason noise for now
    sched = sched.filter(pl.col("game_type").is_in(["REG", "WC", "DIV", "CON", "SB"]))

    keep_cols = [
        "game_id", "season", "game_type", "week", "gameday",
        "away_team", "home_team", "away_score", "home_score", "result",
        "home_moneyline", "away_moneyline",
        "spread_line", "home_spread_odds", "away_spread_odds",
        "total_line", "over_odds", "under_odds",
        "away_rest", "home_rest", "div_game", "roof", "surface",
        "temp", "wind",
    ]
    sched = sched.select([c for c in keep_cols if c in sched.columns])
    return sched


def pull_team_stats() -> pl.DataFrame:
    """Weekly team-level box score stats, used to build 'recent form' features."""
    seasons = list(range(START_SEASON, END_SEASON + 1))
    stats = nfl.load_team_stats(seasons=seasons)
    return stats


def pull_current_season() -> pl.DataFrame:
    """This season's schedule (upcoming games have no result yet) — used at
    inference time, and appended to training data as games complete."""
    sched = nfl.load_schedules(seasons=[CURRENT_SEASON])
    return sched


def main():
    print(f"Pulling schedules {START_SEASON}-{END_SEASON} (includes historical odds)...")
    schedules = pull_schedules()
    schedules.write_parquet(DATA_DIR / "schedules.parquet")
    print(f"  -> {schedules.shape[0]} games saved to data/schedules.parquet")

    # Sanity check: how much odds coverage do we actually have?
    has_moneyline = schedules.filter(pl.col("home_moneyline").is_not_null()).shape[0]
    print(f"  -> {has_moneyline}/{schedules.shape[0]} games have moneyline odds "
          f"({has_moneyline / schedules.shape[0]:.0%})")

    print(f"\nPulling team stats {START_SEASON}-{END_SEASON}...")
    team_stats = pull_team_stats()
    team_stats.write_parquet(DATA_DIR / "team_stats.parquet")
    print(f"  -> {team_stats.shape[0]} team-game rows saved to data/team_stats.parquet")

    print(f"\nPulling current season ({CURRENT_SEASON}) schedule for inference...")
    current = pull_current_season()
    current.write_parquet(DATA_DIR / "current_season.parquet")
    print(f"  -> {current.shape[0]} games saved to data/current_season.parquet")

    print("\nDone. Files in data/:")
    for f in sorted(DATA_DIR.glob("*.parquet")):
        print(f"  {f.name}")


if __name__ == "__main__":
    main()
