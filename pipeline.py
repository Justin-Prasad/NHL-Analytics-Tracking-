"""
pipeline.py

End-to-end orchestration script. Runs stages in sequence or individually.

Usage:
    python pipeline.py --stage ingest --seasons 2021 2022 2023
    python pipeline.py --stage features
    python pipeline.py --stage train
    python pipeline.py --stage evaluate --season 2023
    python pipeline.py --stage report --team TOR --season 2023
    python pipeline.py --all
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log"),
    ],
)
logger = logging.getLogger("pipeline")

from config import (
    DATA_PROCESSED, DATA_RAW, MODELS_DIR, SEASONS, XG_FEATURES, XG_TARGET
)


# ── Stage: Ingest ──────────────────────────────────────────────────────────────

def stage_ingest(seasons: list[int]) -> None:
    from utils.data_loader import fetch_moneypuck_shots, fetch_season_pbp

    for season in seasons:
        logger.info(f"Ingesting MoneyPuck shots: {season}")
        shots = fetch_moneypuck_shots(season)
        logger.info(f"  {len(shots):,} shots")

        logger.info(f"Ingesting NHL play-by-play: {season}")
        pbp = fetch_season_pbp(season)
        pbp.to_parquet(DATA_RAW / f"pbp_{season}.parquet", index=False)
        logger.info(f"  {len(pbp):,} events")

    logger.info("Ingest complete")


# ── Stage: Features ────────────────────────────────────────────────────────────

def stage_features(seasons: list[int]) -> None:
    from utils.preprocessing import (
        add_rebound_flag, add_rush_flag, add_score_differential,
        add_shot_geometry, build_possession_sequences, label_entry_outcomes,
        label_zone_entries,
    )

    all_shots = []
    all_sequences = []

    for season in seasons:
        logger.info(f"Feature engineering: {season}")

        # MoneyPuck shots → xG features
        shots = pd.read_parquet(DATA_RAW / f"moneypuck_shots_{season}.parquet")
        shots = add_shot_geometry(shots)
        shots = add_rebound_flag(shots)
        shots = add_rush_flag(shots)
        shots = add_score_differential(shots)
        shots["season"] = season
        all_shots.append(shots)

        # PBP → zone entries + sequences
        pbp_path = DATA_RAW / f"pbp_{season}.parquet"
        if pbp_path.exists():
            pbp = pd.read_parquet(pbp_path)
            entries = label_zone_entries(pbp)
            entries = label_entry_outcomes(entries, pbp)
            entries.to_parquet(DATA_PROCESSED / f"zone_entries_{season}.parquet", index=False)

            seqs = build_possession_sequences(pbp)
            import json
            (DATA_PROCESSED / f"sequences_{season}.jsonl").write_text(
                "\n".join(json.dumps(s) for s in seqs)
            )
            logger.info(f"  Zone entries: {len(entries):,} | Sequences: {len(seqs):,}")

    combined_shots = pd.concat(all_shots, ignore_index=True)
    combined_shots.to_parquet(DATA_PROCESSED / "shots_all.parquet", index=False)
    logger.info(f"Total shots saved: {len(combined_shots):,}")


# ── Stage: Train ───────────────────────────────────────────────────────────────

def stage_train(train_seasons: list[int], test_seasons: list[int]) -> None:
    from sklearn.model_selection import train_test_split

    from models.sequence_model import SequenceLSTM, SequenceTrainer
    from models.xg_model import XGModel
    from models.zone_entry import ZoneEntryModel

    shots = pd.read_parquet(DATA_PROCESSED / "shots_all.parquet")

    # Train/test split by season (temporal)
    train_shots = shots[shots["season"].isin(train_seasons)]
    test_shots = shots[shots["season"].isin(test_seasons)]
    logger.info(f"xG Train: {len(train_shots):,} | Test: {len(test_shots):,}")

    # ── xG Model ──────────────────────────────────────────────────────────────
    xg_model = XGModel(calibrate=True)

    # CV on train set
    cv_results = xg_model.cross_validate(train_shots, train_shots[XG_TARGET])
    logger.info(f"xG CV results: {cv_results}")

    # Final fit + eval
    xg_model.fit(train_shots, train_shots[XG_TARGET])
    eval_results = xg_model.evaluate(test_shots, test_shots[XG_TARGET])
    logger.info(f"xG Test eval: {eval_results}")

    xg_model.save(MODELS_DIR / "xg_model.pkl")

    # Add xG predictions to shots
    shots["xg"] = xg_model.predict_proba(shots)
    shots.to_parquet(DATA_PROCESSED / "shots_with_xg.parquet", index=False)

    # ── Zone Entry Model ───────────────────────────────────────────────────────
    ze_files = list(DATA_PROCESSED.glob("zone_entries_*.parquet"))
    if ze_files:
        entries = pd.concat([pd.read_parquet(f) for f in ze_files], ignore_index=True)
        from config import ZONE_ENTRY_FEATURES, ZONE_ENTRY_TARGET

        train_entries = entries[entries["season"].isin(train_seasons)]
        test_entries = entries[entries["season"].isin(test_seasons)]

        ze_model = ZoneEntryModel()
        ze_model.fit(train_entries, train_entries[ZONE_ENTRY_TARGET])
        ze_eval = ze_model.evaluate(test_entries, test_entries[ZONE_ENTRY_TARGET])
        logger.info(f"Zone entry eval: {ze_eval}")
        ze_model.save(MODELS_DIR / "zone_entry_model.pkl")

    # ── LSTM Sequence Model ────────────────────────────────────────────────────
    import json
    seq_files_train = [DATA_PROCESSED / f"sequences_{s}.jsonl" for s in train_seasons]
    seq_files_test = [DATA_PROCESSED / f"sequences_{s}.jsonl" for s in test_seasons]

    def load_seqs(files):
        seqs = []
        for f in files:
            if f.exists():
                seqs += [json.loads(l) for l in f.read_text().strip().splitlines()]
        return seqs

    train_seqs = load_seqs(seq_files_train)
    test_seqs = load_seqs(seq_files_test)

    if train_seqs:
        logger.info(f"LSTM Train seqs: {len(train_seqs):,} | Test: {len(test_seqs):,}")
        lstm = SequenceLSTM()
        trainer = SequenceTrainer(lstm)
        trainer.fit(train_seqs, test_seqs[:5000] if test_seqs else train_seqs[-2000:])
        trainer.save(MODELS_DIR / "sequence_lstm.pt")

    logger.info("Training complete")


# ── Stage: Report ──────────────────────────────────────────────────────────────

def stage_report(team: str, season: int) -> None:
    from utils.evaluation import player_xg_summary

    shots = pd.read_parquet(DATA_PROCESSED / "shots_with_xg.parquet")
    team_shots = shots[(shots["team"] == team) & (shots["season"] == season)]

    player_summary = player_xg_summary(team_shots, xg_col="xg")

    # Simulate game logs and zone entry data for report
    # (in prod: load from processed outputs)
    import numpy as np
    dates = pd.date_range("2023-10-10", periods=82, freq="4D")
    game_logs = pd.DataFrame({
        "game_date": dates,
        "xg_for": np.random.normal(2.5, 0.6, 82).clip(0.5, 5.0),
        "xg_against": np.random.normal(2.45, 0.6, 82).clip(0.5, 5.0),
    })

    zone_data = pd.DataFrame({
        "line_id": ["L1", "L1", "L2", "L2", "L3", "L3", "L4", "L4"],
        "entry_type": ["controlled", "uncontrolled"] * 4,
        "n": [85, 42, 72, 61, 65, 70, 30, 58],
        "shot_rate": [0.62, 0.31, 0.58, 0.29, 0.55, 0.28, 0.50, 0.27],
    })

    from reports.generate_report import generate_report
    output = f"reports/output/{team}_{season}.html"
    generate_report(team, season, output, team_shots, player_summary, zone_data, game_logs)


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="NHL Analytics Pipeline")
    parser.add_argument("--stage", choices=["ingest", "features", "train", "report", "all"])
    parser.add_argument("--seasons", nargs="+", type=int, default=SEASONS)
    parser.add_argument("--train_seasons", nargs="+", type=int, default=[2021, 2022])
    parser.add_argument("--test_seasons", nargs="+", type=int, default=[2023])
    parser.add_argument("--team", default="TOR")
    parser.add_argument("--season", type=int, default=2023)
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    run_all = args.all or args.stage == "all"

    if run_all or args.stage == "ingest":
        logger.info("=== STAGE: INGEST ===")
        stage_ingest(args.seasons)

    if run_all or args.stage == "features":
        logger.info("=== STAGE: FEATURES ===")
        stage_features(args.seasons)

    if run_all or args.stage == "train":
        logger.info("=== STAGE: TRAIN ===")
        stage_train(args.train_seasons, args.test_seasons)

    if run_all or args.stage == "report":
        logger.info("=== STAGE: REPORT ===")
        stage_report(args.team, args.season)


if __name__ == "__main__":
    main()
