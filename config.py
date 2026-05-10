"""
config.py — Central configuration for NHL Analytics Platform.
All paths, hyperparameters, and constants live here.
"""

from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────

ROOT = Path(__file__).parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models" / "saved"
REPORTS_DIR = ROOT / "reports" / "output"

for d in [DATA_RAW, DATA_PROCESSED, MODELS_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ── Data Sources ───────────────────────────────────────────────────────────────

NHL_API_BASE = "https://api-web.nhle.com/v1"
MONEYPUCK_BASE = "https://moneypuck.com/moneypuck/playerData/careers/gameByGame"

SEASONS = [2021, 2022, 2023]   # seasons to pull (2021 = 2021-22)
CURRENT_SEASON = 2023

# ── Shot / xG Model ────────────────────────────────────────────────────────────

XG_FEATURES = [
    "shot_distance",
    "shot_angle",
    "shot_type",           # wrist, slap, snap, backhand, tip, wrap
    "is_rebound",          # shot within 3s of prior shot
    "is_rush",             # event preceded by neutral zone carry
    "prior_event_type",
    "prior_event_distance",
    "score_differential",
    "period",
    "strength_state",      # 5v5, 5v4, 4v5, etc.
    "shooter_hand",
    "x_coord",
    "y_coord",
]

XG_TARGET = "is_goal"

XG_XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 5,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "use_label_encoder": False,
    "eval_metric": "logloss",
    "random_state": 42,
}

# ── Zone Entry Model ───────────────────────────────────────────────────────────

ZONE_ENTRY_FEATURES = [
    "entry_type",          # controlled (carry) vs uncontrolled (dump)
    "entry_x",
    "entry_y",
    "score_differential",
    "strength_state",
    "period",
    "seconds_remaining",
    "prior_zone_time",     # seconds in neutral zone before entry
    "attacking_team_id",
    "defending_team_id",
]

ZONE_ENTRY_TARGET = "results_in_shot_10s"
ZONE_ENTRY_WINDOW_SECONDS = 10

# ── Sequence / LSTM Model ──────────────────────────────────────────────────────

SEQUENCE_MAX_LEN = 20        # max events per sequence
SEQUENCE_EVENT_TYPES = [
    "shot", "pass", "zone_entry", "zone_exit",
    "faceoff", "hit", "takeaway", "giveaway", "block",
]

LSTM_PARAMS = {
    "input_size": 16,        # feature vector per event
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.3,
    "bidirectional": True,
    "batch_size": 256,
    "lr": 1e-3,
    "epochs": 50,
    "patience": 7,           # early stopping patience
}

SCORING_CHANCE_WINDOW_SECONDS = 10

# ── Evaluation ─────────────────────────────────────────────────────────────────

EVAL_CV_FOLDS = 5
EVAL_TEST_SEASONS = [2023]   # hold-out seasons for final eval

# ── Reporting ──────────────────────────────────────────────────────────────────

REPORT_TOP_N_PLAYERS = 20
REPORT_TEAM_COLORS = {
    "TOR": ("#003E7E", "#FFFFFF"),
    "EDM": ("#041E42", "#FF4C00"),
    "BOS": ("#FFB81C", "#000000"),
    "NYR": ("#0038A8", "#CE1126"),
    "VGK": ("#B4975A", "#333F42"),
    # add more as needed
}
