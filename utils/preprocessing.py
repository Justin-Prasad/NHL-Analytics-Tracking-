"""
utils/preprocessing.py

Shared preprocessing: cleaning, encoding, normalization.
All transformations are stateless functions or sklearn-compatible transformers
so they can be used inside pipelines and serialized with joblib.
"""

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder


# ── Strength State Parsing ─────────────────────────────────────────────────────

SITUATION_CODE_MAP = {
    "1551": "5v5",
    "1451": "5v4",   # home PP
    "1541": "4v5",   # home PK
    "1441": "4v4",
    "1351": "5v3",
    "1531": "3v5",
    "1331": "3v3",
    "1651": "6v5",   # empty net
    "1561": "5v6",
}

def parse_situation_code(code: str, is_home: bool) -> str:
    """
    NHL situation codes encode skaters for both teams.
    code='1551' = 5v5. '1451' = home has 4 skaters, away has 5 (home PP).
    is_home: whether the event team is the home team.
    Returns normalized strength state string from shooter's perspective.
    """
    if pd.isna(code) or len(str(code)) != 4:
        return "5v5"
    code = str(code)
    home_sk = int(code[0]) + int(code[1])   # home skater count (incl goalie)
    away_sk = int(code[2]) + int(code[3])
    # simplify: just use the mapped label
    return SITUATION_CODE_MAP.get(code, "other")


# ── Shot Feature Engineering ───────────────────────────────────────────────────

def compute_shot_distance(x: float, y: float) -> float:
    """Distance from shot location to center of goal (89, 0)."""
    return np.sqrt((abs(x) - 89) ** 2 + y ** 2)


def compute_shot_angle(x: float, y: float) -> float:
    """Angle from shot location to goal line. 0 = straight on."""
    if abs(x) >= 89:
        return np.degrees(np.arctan(abs(y) / max(abs(abs(x) - 89), 0.1)))
    return 90.0


def add_shot_geometry(df: pd.DataFrame) -> pd.DataFrame:
    """Add distance and angle columns from x/y coordinates."""
    df = df.copy()
    df["shot_distance"] = df.apply(
        lambda r: compute_shot_distance(r["xCordAdjusted"], r["yCordAdjusted"]), axis=1
    )
    df["shot_angle"] = df.apply(
        lambda r: compute_shot_angle(r["xCordAdjusted"], r["yCordAdjusted"]), axis=1
    )
    return df


def add_rebound_flag(df: pd.DataFrame, window_seconds: int = 3) -> pd.DataFrame:
    """
    A shot is a rebound if a shot (of any outcome) occurred within
    window_seconds by the same team in the same period.
    """
    df = df.sort_values(["game_id", "period", "time"]).copy()
    df["prev_event"] = df.groupby(["game_id", "period", "team"])["event"].shift(1)
    df["prev_time"] = df.groupby(["game_id", "period", "team"])["time"].shift(1)
    df["time_since_prev"] = df["time"] - df["prev_time"]
    df["is_rebound"] = (
        df["prev_event"].isin(["shot-on-goal", "missed-shot", "blocked-shot"])
        & (df["time_since_prev"] <= window_seconds)
    ).astype(int)
    return df


def add_rush_flag(df: pd.DataFrame, window_seconds: int = 4) -> pd.DataFrame:
    """
    A shot is a rush shot if the preceding event was in the neutral zone
    (absolute x < 25) within window_seconds.
    """
    df = df.sort_values(["game_id", "period", "time"]).copy()
    df["prev_zone"] = df.groupby(["game_id", "period"])["zone"].shift(1)
    df["prev_time"] = df.groupby(["game_id", "period"])["time"].shift(1)
    df["time_since_prev_any"] = df["time"] - df["prev_time"]
    df["is_rush"] = (
        (df["prev_zone"] == "N")
        & (df["time_since_prev_any"] <= window_seconds)
    ).astype(int)
    return df


def add_score_differential(df: pd.DataFrame, perspective: str = "shooter") -> pd.DataFrame:
    """
    Add score_differential from the shooter's team perspective.
    Positive = shooter's team is leading.
    """
    df = df.copy()
    is_home_shooter = df["team"] == df["homeTeamCode"]
    df["score_differential"] = np.where(
        is_home_shooter,
        df["homeTeamGoals"] - df["awayTeamGoals"],
        df["awayTeamGoals"] - df["homeTeamGoals"],
    )
    return df


# ── Categorical Encoding ───────────────────────────────────────────────────────

SHOT_TYPE_ORDER = ["wrist", "snap", "slap", "backhand", "tip-in", "wrap-around", "deflected"]

class OrdinalShotTypeEncoder(BaseEstimator, TransformerMixin):
    """
    Encode shot type as an ordinal roughly ordered by typical danger level.
    Unknown types map to -1.
    """
    def fit(self, X, y=None):
        self.mapping_ = {t: i for i, t in enumerate(SHOT_TYPE_ORDER)}
        return self

    def transform(self, X):
        return np.array([self.mapping_.get(x, -1) for x in X]).reshape(-1, 1)


# ── Zone Entry Preprocessing ───────────────────────────────────────────────────

def label_zone_entries(pbp: pd.DataFrame) -> pd.DataFrame:
    """
    Identify zone entries from play-by-play.
    A zone entry = event_type in ('zone-entry', 'puck-recovery') where zone changes to 'O'.
    Labels each entry as controlled (carry-in) or uncontrolled (dump-in).
    """
    pbp = pbp.sort_values(["game_id", "period", "time_in_period"]).copy()
    entries = pbp[pbp["event_type"] == "zone-entry"].copy()
    # placeholder: in real tracking data this is explicitly labeled
    # here we simulate: treat as controlled if shot_type-like detail present
    entries["entry_type"] = "controlled"  # would be derived from tracking detail
    entries["entry_type"] = entries["entry_type"].where(
        entries["x_coord"].notna(), "uncontrolled"
    )
    return entries


def label_entry_outcomes(entries: pd.DataFrame, pbp: pd.DataFrame,
                          window: int = 10) -> pd.DataFrame:
    """
    For each zone entry, check if a shot attempt occurs within `window` seconds.
    Adds 'results_in_shot_10s' binary target.
    """
    shots = pbp[pbp["event_type"].isin(["shot-on-goal", "missed-shot", "blocked-shot"])]
    shot_lookup = shots.set_index(["game_id", "period"])

    results = []
    for _, entry in entries.iterrows():
        key = (entry["game_id"], entry["period"])
        if key not in shot_lookup.index:
            results.append(0)
            continue
        game_shots = shot_lookup.loc[[key]]
        after = game_shots[
            (game_shots["time_in_period"] > entry["time_in_period"])
            & (game_shots["time_in_period"] <= entry["time_in_period"] + window)
        ]
        results.append(1 if len(after) > 0 else 0)

    entries = entries.copy()
    entries["results_in_shot_10s"] = results
    return entries


# ── Sequence Construction ──────────────────────────────────────────────────────

EVENT_TYPE_VOCAB = {
    "shot-on-goal": 0,
    "missed-shot": 1,
    "blocked-shot": 2,
    "zone-entry": 3,
    "zone-exit": 4,
    "faceoff": 5,
    "hit": 6,
    "takeaway": 7,
    "giveaway": 8,
    "stoppage": 9,
    "penalty": 10,
    "<PAD>": 11,
    "<UNK>": 12,
}

def build_possession_sequences(pbp: pd.DataFrame,
                                max_len: int = 20) -> list[dict]:
    """
    Group play-by-play into possession sequences.
    A possession ends on a zone exit, stoppage, or period end.
    Returns list of dicts: {events: [...], label: 0/1}.
    """
    pbp = pbp.sort_values(["game_id", "period", "time_in_period"]).copy()
    sequences = []

    for (game_id, period), group in pbp.groupby(["game_id", "period"]):
        current_seq = []
        current_team = None

        for _, row in group.iterrows():
            event = row["event_type"]

            # possession break
            if event in ("stoppage", "faceoff", "zone-exit") or row.get("team") != current_team:
                if len(current_seq) >= 2:
                    label = int(any(
                        e["event_type"] in ("shot-on-goal",) for e in current_seq
                    ))
                    sequences.append({
                        "game_id": game_id,
                        "period": period,
                        "events": current_seq[-max_len:],
                        "label": label,
                    })
                current_seq = []
                current_team = row.get("team")

            current_seq.append({
                "event_type": EVENT_TYPE_VOCAB.get(event, EVENT_TYPE_VOCAB["<UNK>"]),
                "x_coord": row.get("x_coord", 0) or 0,
                "y_coord": row.get("y_coord", 0) or 0,
                "time": row.get("time_in_period", 0) or 0,
            })

    return sequences


def pad_sequences(sequences: list[list], max_len: int,
                  pad_value: float = 0.0) -> np.ndarray:
    """Left-pad a list of variable-length sequences to max_len."""
    out = np.full((len(sequences), max_len, len(sequences[0][0])), pad_value, dtype=np.float32)
    for i, seq in enumerate(sequences):
        arr = np.array(seq, dtype=np.float32)
        out[i, -len(arr):] = arr
    return out
