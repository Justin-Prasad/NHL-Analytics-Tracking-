"""
tests/test_features.py

Unit tests for feature engineering functions.
Run with: pytest tests/ -v --cov=utils
"""

import numpy as np
import pandas as pd
import pytest

from utils.preprocessing import (
    add_rebound_flag,
    add_rush_flag,
    add_score_differential,
    add_shot_geometry,
    build_possession_sequences,
    compute_shot_angle,
    compute_shot_distance,
    pad_sequences,
)


# ── Shot Geometry ──────────────────────────────────────────────────────────────

class TestShotGeometry:
    def test_distance_straight_on(self):
        # Shot from top of crease, on center: (83, 0) → 6 feet out
        dist = compute_shot_distance(83, 0)
        assert abs(dist - 6.0) < 0.1

    def test_distance_behind_net(self):
        # Behind net: x=93, y=0 → 4 feet behind
        dist = compute_shot_distance(93, 0)
        assert dist > 0

    def test_angle_straight_on(self):
        # Dead center: angle should be ~0
        angle = compute_shot_angle(80, 0)
        assert angle < 5

    def test_angle_extreme(self):
        # Sharp angle from boards: angle >> 0
        angle = compute_shot_angle(89, 40)
        assert angle > 60

    def test_add_shot_geometry_creates_cols(self):
        df = pd.DataFrame({
            "xCordAdjusted": [80, 65, 89],
            "yCordAdjusted": [0, 10, 30],
        })
        result = add_shot_geometry(df)
        assert "shot_distance" in result.columns
        assert "shot_angle" in result.columns
        assert result["shot_distance"].notna().all()


# ── Rebound / Rush Flags ───────────────────────────────────────────────────────

class TestReboundFlag:
    def _make_shots_df(self):
        return pd.DataFrame({
            "game_id": [1, 1, 1, 1],
            "period": [1, 1, 1, 1],
            "team": ["TOR", "TOR", "TOR", "MTL"],
            "event": ["shot-on-goal", "shot-on-goal", "shot-on-goal", "shot-on-goal"],
            "time": [100, 102, 110, 115],
        })

    def test_rebound_within_window(self):
        df = self._make_shots_df()
        result = add_rebound_flag(df, window_seconds=3)
        # Second shot (t=102) is 2s after first → rebound
        assert result.iloc[1]["is_rebound"] == 1

    def test_no_rebound_outside_window(self):
        df = self._make_shots_df()
        result = add_rebound_flag(df, window_seconds=3)
        # Third shot (t=110) is 8s after second → not rebound
        assert result.iloc[2]["is_rebound"] == 0

    def test_cross_team_no_rebound(self):
        df = self._make_shots_df()
        result = add_rebound_flag(df, window_seconds=30)
        # MTL shot (t=115) follows TOR shot but different team → not a rebound
        assert result[result["team"] == "MTL"]["is_rebound"].iloc[0] == 0


# ── Score Differential ─────────────────────────────────────────────────────────

class TestScoreDifferential:
    def test_home_leading(self):
        df = pd.DataFrame({
            "team": ["TOR"],
            "homeTeamCode": ["TOR"],
            "homeTeamGoals": [3],
            "awayTeamGoals": [1],
        })
        result = add_score_differential(df)
        assert result["score_differential"].iloc[0] == 2

    def test_away_trailing(self):
        df = pd.DataFrame({
            "team": ["MTL"],
            "homeTeamCode": ["TOR"],
            "homeTeamGoals": [3],
            "awayTeamGoals": [1],
        })
        result = add_score_differential(df)
        assert result["score_differential"].iloc[0] == -2


# ── Sequence Building ──────────────────────────────────────────────────────────

class TestSequences:
    def _make_pbp(self):
        return pd.DataFrame({
            "game_id": [1] * 6,
            "period": [1] * 6,
            "team": ["TOR", "TOR", "TOR", "stoppage", "TOR", "TOR"],
            "event_type": [
                "zone-entry", "shot-on-goal", "missed-shot",
                "stoppage", "faceoff", "shot-on-goal"
            ],
            "x_coord": [50, 75, 70, 0, 0, 80],
            "y_coord": [5, 2, -3, 0, 0, 10],
            "time_in_period": [120, 122, 125, 130, 135, 140],
        })

    def test_sequences_produced(self):
        pbp = self._make_pbp()
        seqs = build_possession_sequences(pbp, max_len=20)
        assert len(seqs) > 0

    def test_sequence_label_present(self):
        pbp = self._make_pbp()
        seqs = build_possession_sequences(pbp, max_len=20)
        for s in seqs:
            assert "label" in s
            assert s["label"] in (0, 1)

    def test_sequence_max_len_enforced(self):
        pbp = self._make_pbp()
        seqs = build_possession_sequences(pbp, max_len=3)
        for s in seqs:
            assert len(s["events"]) <= 3


# ── Padding ────────────────────────────────────────────────────────────────────

class TestPadding:
    def test_output_shape(self):
        sequences = [
            [[1, 2, 3], [4, 5, 6]],
            [[7, 8, 9]],
        ]
        padded = pad_sequences(sequences, max_len=5)
        assert padded.shape == (2, 5, 3)

    def test_padding_value(self):
        sequences = [[[1.0, 2.0]]]
        padded = pad_sequences(sequences, max_len=4, pad_value=-99.0)
        # First 3 rows should be padded
        assert padded[0, 0, 0] == -99.0
        assert padded[0, 3, 0] == 1.0
