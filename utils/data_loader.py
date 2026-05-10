"""
utils/data_loader.py

Handles all data ingestion:
  - NHL API (play-by-play, rosters, shifts)
  - MoneyPuck CSV (shots with pre-computed features)
  - Local cache to avoid redundant API calls
"""

import json
import time
import logging
from pathlib import Path
from typing import Optional

import pandas as pd
import requests
from tqdm import tqdm

from config import NHL_API_BASE, MONEYPUCK_BASE, DATA_RAW, SEASONS

logger = logging.getLogger(__name__)


# ── NHL API ────────────────────────────────────────────────────────────────────

def _get(url: str, retries: int = 3, backoff: float = 1.5) -> dict:
    """GET with retry/backoff. Returns parsed JSON."""
    for attempt in range(retries):
        try:
            resp = requests.get(url, timeout=10)
            resp.raise_for_status()
            return resp.json()
        except requests.RequestException as exc:
            if attempt == retries - 1:
                raise
            wait = backoff ** attempt
            logger.warning(f"Request failed ({exc}), retrying in {wait:.1f}s...")
            time.sleep(wait)


def fetch_schedule(season: int) -> list[dict]:
    """
    Return a list of game dicts for a full regular season.
    season=2023 → 2023-24 season.
    """
    cache = DATA_RAW / f"schedule_{season}.json"
    if cache.exists():
        return json.loads(cache.read_text())

    season_str = f"{season}{season+1}"
    url = f"{NHL_API_BASE}/schedule/game-type/2/season/{season_str}"
    data = _get(url)
    games = [
        {"game_id": g["id"], "date": g["gameDate"],
         "home": g["homeTeam"]["abbrev"], "away": g["awayTeam"]["abbrev"]}
        for week in data.get("gameWeek", [])
        for g in week.get("games", [])
    ]
    cache.write_text(json.dumps(games))
    logger.info(f"Fetched {len(games)} games for {season_str}")
    return games


def fetch_play_by_play(game_id: int) -> pd.DataFrame:
    """
    Fetch and flatten play-by-play events for a single game.
    Returns one row per event with coordinates, event type, team, player.
    """
    cache = DATA_RAW / f"pbp_{game_id}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    url = f"{NHL_API_BASE}/gamecenter/{game_id}/play-by-play"
    data = _get(url)

    rows = []
    for play in data.get("plays", []):
        details = play.get("details", {})
        coords = details.get("xCoord"), details.get("yCoord")
        rows.append({
            "game_id": game_id,
            "event_id": play.get("eventId"),
            "period": play.get("periodDescriptor", {}).get("number"),
            "time_in_period": play.get("timeInPeriod"),
            "time_remaining": play.get("timeRemaining"),
            "event_type": play.get("typeDescKey"),
            "x_coord": coords[0],
            "y_coord": coords[1],
            "zone": details.get("zoneCode"),
            "shot_type": details.get("shotType"),
            "scoring_player_id": details.get("scoringPlayerId"),
            "assist1_player_id": details.get("assist1PlayerId"),
            "assist2_player_id": details.get("assist2PlayerId"),
            "goalie_id": details.get("goalieInNetId"),
            "blocking_player_id": details.get("blockingPlayerId"),
            "home_score": play.get("homeScore"),
            "away_score": play.get("awayScore"),
            "situation_code": play.get("situationCode"),  # encodes strength state
        })

    df = pd.DataFrame(rows)
    if not df.empty:
        df.to_parquet(cache, index=False)
    return df


def fetch_season_pbp(season: int, max_games: Optional[int] = None) -> pd.DataFrame:
    """
    Pull all play-by-play for a season. Caches each game individually.
    max_games: limit for dev/testing.
    """
    games = fetch_schedule(season)
    if max_games:
        games = games[:max_games]

    dfs = []
    for game in tqdm(games, desc=f"PBP {season}"):
        try:
            df = fetch_play_by_play(game["game_id"])
            df["home_team"] = game["home"]
            df["away_team"] = game["away"]
            df["game_date"] = game["date"]
            dfs.append(df)
        except Exception as exc:
            logger.warning(f"Skipping game {game['game_id']}: {exc}")

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Season {season}: {len(combined):,} events from {len(dfs)} games")
    return combined


# ── MoneyPuck ──────────────────────────────────────────────────────────────────

MONEYPUCK_SHOT_COLUMNS = [
    "shotID", "season", "name", "id", "team", "homeTeamCode", "awayTeamCode",
    "isPlayoffGame", "game_id", "time", "period", "team", "location",
    "event", "homeTeamGoals", "awayTeamGoals", "homeSkatersOnIce",
    "awaySkatersOnIce", "shooterPlayerId", "shooterName", "goalieIdForShot",
    "goalieNameForShot", "xCordAdjusted", "yCordAdjusted", "shotAngleAdjusted",
    "shotDistance", "shotType", "shotRebound", "shotRush", "shotAnglePlusRebound",
    "shotGoalProbability", "goal", "xGoal", "xGoalModel",
]


def fetch_moneypuck_shots(season: int) -> pd.DataFrame:
    cache = DATA_RAW / f"moneypuck_shots_{season}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    season_str = f"{season}{season + 1}"

    url = (
        "https://moneypuck.com/moneypuck/playerData/careers/"
        f"gameByGame/shots_{season_str}.csv"
    )

    logger.info(f"Downloading MoneyPuck shots: {url}")

    headers = {
        "User-Agent": "Mozilla/5.0"
    }

    response = requests.get(url, headers=headers, timeout=30, allow_redirects=True)
    response.raise_for_status()

    if "data_license" in response.url or "text/html" in response.headers.get("Content-Type", ""):
        raise ValueError(
            f"MoneyPuck blocked direct CSV access. Final URL: {response.url}"
        )

    from io import StringIO
    df = pd.read_csv(StringIO(response.text))

    df.to_parquet(cache, index=False)
    logger.info(f"MoneyPuck {season}: {len(df):,} shots")

    return df


def fetch_moneypuck_multi(seasons: list[int]) -> pd.DataFrame:
    """Pull and concatenate MoneyPuck data across multiple seasons."""
    dfs = [fetch_moneypuck_shots(s) for s in seasons]
    return pd.concat(dfs, ignore_index=True)


# ── Rosters ────────────────────────────────────────────────────────────────────

def fetch_roster(team: str, season: int) -> pd.DataFrame:
    """Return roster for a team-season with player positions."""
    cache = DATA_RAW / f"roster_{team}_{season}.parquet"
    if cache.exists():
        return pd.read_parquet(cache)

    season_str = f"{season}{season+1}"
    url = f"{NHL_API_BASE}/roster/{team}/{season_str}"
    data = _get(url)

    rows = []
    for position_group in ["forwards", "defensemen", "goalies"]:
        for p in data.get(position_group, []):
            rows.append({
                "player_id": p["id"],
                "first_name": p["firstName"]["default"],
                "last_name": p["lastName"]["default"],
                "position": p.get("positionCode"),
                "shoots_catches": p.get("shootsCatches"),
                "jersey_number": p.get("sweaterNumber"),
                "birth_date": p.get("birthDate"),
                "height_inches": p.get("heightInInches"),
                "weight_pounds": p.get("weightInPounds"),
            })

    df = pd.DataFrame(rows)
    df["team"] = team
    df["season"] = season
    df.to_parquet(cache, index=False)
    return df
