"""
models/xg_model.py

Expected Goals (xG) model.
Primary: XGBoost classifier
Baseline: Logistic Regression (for calibration comparison)

Usage:
    from models.xg_model import XGModel
    model = XGModel()
    model.fit(X_train, y_train)
    preds = model.predict_proba(X_test)
    model.save("models/saved/xg_v1.pkl")
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

from config import XG_FEATURES, XG_TARGET, XG_XGBOOST_PARAMS, EVAL_CV_FOLDS

logger = logging.getLogger(__name__)


class XGModel:
    """
    Expected Goals classifier. Wraps XGBoost with optional isotonic calibration.

    Attributes
    ----------
    model : fitted XGBClassifier
    baseline : fitted LogisticRegression (for comparison)
    feature_names : list of feature columns used in training
    """

    def __init__(self, calibrate: bool = True):
        self.calibrate = calibrate
        self.feature_names = XG_FEATURES
        self.model = None
        self.baseline = None

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "XGModel":
        """
        Fit XGBoost + logistic baseline on training data.
        Applies isotonic calibration if self.calibrate=True.
        """
        logger.info(f"Training xG model on {len(X):,} shots ({y.mean():.3%} goal rate)")

        X_arr = self._prepare(X)

        # XGBoost
        xgb = XGBClassifier(**XG_XGBOOST_PARAMS)
        if self.calibrate:
            self.model = CalibratedClassifierCV(xgb, method="isotonic", cv=3)
        else:
            self.model = xgb
        self.model.fit(X_arr, y)

        # Logistic baseline
        self.baseline = Pipeline([
            ("scaler", StandardScaler()),
            ("lr", LogisticRegression(max_iter=1000, C=1.0)),
        ])
        self.baseline.fit(X_arr, y)

        logger.info("xG model training complete")
        return self

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """
        Season-stratified cross-validation. Returns dict of mean metrics.
        Should be run before final fit on full training data.
        """
        X_arr = self._prepare(X)
        cv = StratifiedKFold(n_splits=EVAL_CV_FOLDS, shuffle=True, random_state=42)

        metrics = {"auc": [], "log_loss": [], "brier": []}
        for fold, (tr, va) in enumerate(cv.split(X_arr, y)):
            xgb = XGBClassifier(**XG_XGBOOST_PARAMS)
            xgb.fit(X_arr[tr], y.iloc[tr])
            probs = xgb.predict_proba(X_arr[va])[:, 1]
            metrics["auc"].append(roc_auc_score(y.iloc[va], probs))
            metrics["log_loss"].append(log_loss(y.iloc[va], probs))
            metrics["brier"].append(brier_score_loss(y.iloc[va], probs))
            logger.info(f"  Fold {fold+1}: AUC={metrics['auc'][-1]:.4f} "
                        f"LogLoss={metrics['log_loss'][-1]:.4f} "
                        f"Brier={metrics['brier'][-1]:.4f}")

        summary = {k: np.mean(v) for k, v in metrics.items()}
        logger.info(f"CV summary: {summary}")
        return summary

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Return goal probability for each shot."""
        self._check_fitted()
        return self.model.predict_proba(self._prepare(X))[:, 1]

    def predict_baseline(self, X: pd.DataFrame) -> np.ndarray:
        """Return logistic regression baseline probabilities."""
        self._check_fitted()
        return self.baseline.predict_proba(self._prepare(X))[:, 1]

    # ── Evaluation ─────────────────────────────────────────────────────────────

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        """Full evaluation report on a held-out set."""
        self._check_fitted()
        probs = self.predict_proba(X)
        base_probs = self.predict_baseline(X)

        results = {
            "n_shots": len(y),
            "goal_rate": float(y.mean()),
            "xgb_auc": roc_auc_score(y, probs),
            "xgb_log_loss": log_loss(y, probs),
            "xgb_brier": brier_score_loss(y, probs),
            "baseline_auc": roc_auc_score(y, base_probs),
            "baseline_log_loss": log_loss(y, base_probs),
            "baseline_brier": brier_score_loss(y, base_probs),
        }

        # calibration
        frac_pos, mean_pred = calibration_curve(y, probs, n_bins=10)
        results["calibration"] = {
            "mean_predicted": mean_pred.tolist(),
            "fraction_positive": frac_pos.tolist(),
        }

        logger.info(
            f"Eval: AUC={results['xgb_auc']:.4f} "
            f"LogLoss={results['xgb_log_loss']:.4f} "
            f"Brier={results['xgb_brier']:.4f}"
        )
        return results

    def feature_importance(self) -> pd.DataFrame:
        """Return feature importances (gain) from the XGBoost model."""
        self._check_fitted()
        # Unwrap calibrated classifier
        estimator = (
            self.model.calibrated_classifiers_[0].estimator
            if self.calibrate else self.model
        )
        scores = estimator.feature_importances_
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": scores,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved xG model to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "XGModel":
        obj = joblib.load(path)
        logger.info(f"Loaded xG model from {path}")
        return obj

    # ── Internals ──────────────────────────────────────────────────────────────

    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        """Select features and encode categoricals."""
        X = X.copy()

        # encode shot_type if present as string
        if "shot_type" in X.columns and X["shot_type"].dtype == object:
            shot_type_map = {
                "wrist": 0, "snap": 1, "slap": 2,
                "backhand": 3, "tip-in": 4, "wrap-around": 5,
                "deflected": 6,
            }
            X["shot_type"] = X["shot_type"].map(shot_type_map).fillna(-1)

        if "strength_state" in X.columns and X["strength_state"].dtype == object:
            strength_map = {"5v5": 0, "5v4": 1, "4v5": 2, "4v4": 3, "other": 4}
            X["strength_state"] = X["strength_state"].map(strength_map).fillna(0)

        if "prior_event_type" in X.columns and X["prior_event_type"].dtype == object:
            event_map = {
                "shot-on-goal": 0, "missed-shot": 1, "blocked-shot": 2,
                "faceoff": 3, "hit": 4, "takeaway": 5, "giveaway": 6,
                "zone-entry": 7, "stoppage": 8,
            }
            X["prior_event_type"] = X["prior_event_type"].map(event_map).fillna(-1)

        if "shooter_hand" in X.columns and X["shooter_hand"].dtype == object:
            X["shooter_hand"] = (X["shooter_hand"] == "R").astype(int)

        # select only features used in training (fill any missing with 0)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        return X[self.feature_names].fillna(0).to_numpy(dtype=np.float32)

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("Model not fitted. Call .fit() first.")
