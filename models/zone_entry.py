"""
models/zone_entry.py

Zone Entry Success Classifier.
Predicts whether a zone entry (controlled or dump-in) results in a shot
attempt within 10 seconds — the key metric separating effective from
ineffective zone entries.

Usage:
    from models.zone_entry import ZoneEntryModel
    model = ZoneEntryModel()
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)
"""

import logging
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    classification_report,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from config import (
    EVAL_CV_FOLDS,
    ZONE_ENTRY_FEATURES,
    ZONE_ENTRY_TARGET,
)

logger = logging.getLogger(__name__)


class ZoneEntryModel:
    """
    Binary classifier: does this zone entry lead to a shot within 10s?

    Key insight from hockey analytics research: controlled zone entries
    generate shots at roughly 2x the rate of dump-ins. This model captures
    finer-grained predictors beyond simple entry type.
    """

    ENTRY_TYPE_MAP = {"controlled": 1, "uncontrolled": 0}
    STRENGTH_MAP = {"5v5": 0, "5v4": 1, "4v5": 2, "4v4": 3, "other": 4}

    def __init__(self):
        self.feature_names = ZONE_ENTRY_FEATURES
        self.model = None

    # ── Training ───────────────────────────────────────────────────────────────

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "ZoneEntryModel":
        logger.info(
            f"Training zone entry model on {len(X):,} entries "
            f"({y.mean():.1%} result in shot)"
        )
        X_arr = self._prepare(X)
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=(1 - y.mean()) / y.mean(),  # handle class imbalance
            eval_metric="logloss",
            random_state=42,
        )
        self.model.fit(X_arr, y)
        logger.info("Zone entry model training complete")
        return self

    def cross_validate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        X_arr = self._prepare(X)
        cv = StratifiedKFold(n_splits=EVAL_CV_FOLDS, shuffle=True, random_state=42)
        aucs, aps = [], []

        for fold, (tr, va) in enumerate(cv.split(X_arr, y)):
            m = XGBClassifier(
                n_estimators=300, max_depth=4, learning_rate=0.05,
                eval_metric="logloss", random_state=42,
            )
            m.fit(X_arr[tr], y.iloc[tr])
            probs = m.predict_proba(X_arr[va])[:, 1]
            aucs.append(roc_auc_score(y.iloc[va], probs))
            aps.append(average_precision_score(y.iloc[va], probs))
            logger.info(f"  Fold {fold+1}: AUC={aucs[-1]:.4f} AP={aps[-1]:.4f}")

        return {"auc": np.mean(aucs), "average_precision": np.mean(aps)}

    # ── Inference ──────────────────────────────────────────────────────────────

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        self._check_fitted()
        return self.model.predict_proba(self._prepare(X))[:, 1]

    # ── Analytics ──────────────────────────────────────────────────────────────

    def evaluate(self, X: pd.DataFrame, y: pd.Series) -> dict:
        self._check_fitted()
        probs = self.predict_proba(X)
        preds = (probs >= 0.5).astype(int)
        return {
            "auc": roc_auc_score(y, probs),
            "average_precision": average_precision_score(y, probs),
            "classification_report": classification_report(y, preds, output_dict=True),
        }

    def entry_type_summary(self, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
        """
        Compare controlled vs uncontrolled entry success rates — both actual and predicted.
        Key stakeholder-facing insight.
        """
        df = X.copy()
        df["actual"] = y.values
        df["predicted"] = self.predict_proba(X)

        summary = df.groupby("entry_type").agg(
            n_entries=("actual", "count"),
            actual_shot_rate=("actual", "mean"),
            predicted_shot_rate=("predicted", "mean"),
        ).round(3)

        summary["lift_vs_dump"] = (
            summary["actual_shot_rate"] / summary["actual_shot_rate"].min()
        )
        return summary

    def line_combo_breakdown(self, X: pd.DataFrame, y: pd.Series,
                              line_col: str = "line_id") -> pd.DataFrame:
        """
        Breakdown of zone entry success by line combination.
        Surface to coaching staff: which lines are winning the zone more efficiently?
        """
        df = X.copy()
        df["actual"] = y.values
        df["predicted"] = self.predict_proba(X)

        summary = (
            df.groupby([line_col, "entry_type"])
            .agg(
                n=("actual", "count"),
                shot_rate=("actual", "mean"),
                xShotRate=("predicted", "mean"),
            )
            .reset_index()
            .sort_values("shot_rate", ascending=False)
        )
        return summary

    def feature_importance(self) -> pd.DataFrame:
        self._check_fitted()
        return pd.DataFrame({
            "feature": self.feature_names,
            "importance": self.model.feature_importances_,
        }).sort_values("importance", ascending=False).reset_index(drop=True)

    # ── Persistence ────────────────────────────────────────────────────────────

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)
        logger.info(f"Saved zone entry model to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "ZoneEntryModel":
        return joblib.load(path)

    # ── Internals ──────────────────────────────────────────────────────────────

    def _prepare(self, X: pd.DataFrame) -> np.ndarray:
        X = X.copy()
        if "entry_type" in X.columns and X["entry_type"].dtype == object:
            X["entry_type"] = X["entry_type"].map(self.ENTRY_TYPE_MAP).fillna(0)
        if "strength_state" in X.columns and X["strength_state"].dtype == object:
            X["strength_state"] = X["strength_state"].map(self.STRENGTH_MAP).fillna(0)
        for col in self.feature_names:
            if col not in X.columns:
                X[col] = 0
        return X[self.feature_names].fillna(0).to_numpy(dtype=np.float32)

    def _check_fitted(self):
        if self.model is None:
            raise RuntimeError("Model not fitted.")
