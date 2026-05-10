"""
utils/evaluation.py

Shared evaluation utilities for all models:
  - ROC / PR curves
  - Calibration plots
  - Player/team xG summaries
  - Standardized logging format
"""

import logging
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    log_loss,
    precision_recall_curve,
    roc_auc_score,
    roc_curve,
)
from sklearn.calibration import calibration_curve

logger = logging.getLogger(__name__)


# ── Core Metrics ───────────────────────────────────────────────────────────────

def full_metrics(y_true: np.ndarray, y_prob: np.ndarray,
                 label: str = "model") -> dict:
    """
    Compute standard binary classification metrics.
    Returns a flat dict suitable for logging or DataFrame row.
    """
    return {
        "label": label,
        "n": len(y_true),
        "base_rate": float(y_true.mean()),
        "auc_roc": roc_auc_score(y_true, y_prob),
        "auc_pr": average_precision_score(y_true, y_prob),
        "log_loss": log_loss(y_true, y_prob),
        "brier": brier_score_loss(y_true, y_prob),
    }


def print_metrics(metrics: dict) -> None:
    label = metrics.get("label", "model")
    logger.info(
        f"[{label}] n={metrics['n']:,} | base_rate={metrics['base_rate']:.3%} | "
        f"AUC-ROC={metrics['auc_roc']:.4f} | AUC-PR={metrics['auc_pr']:.4f} | "
        f"LogLoss={metrics['log_loss']:.4f} | Brier={metrics['brier']:.4f}"
    )


# ── Plotting ───────────────────────────────────────────────────────────────────

def plot_roc_pr(y_true: np.ndarray, probas: dict[str, np.ndarray],
                title: str = "Model Comparison",
                save_path: Optional[str] = None) -> plt.Figure:
    """
    ROC and PR curves for one or more models.
    probas: {"model_name": y_prob_array}
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    fig.suptitle(title, fontsize=14)

    for name, y_prob in probas.items():
        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc = roc_auc_score(y_true, y_prob)
        axes[0].plot(fpr, tpr, label=f"{name} (AUC={auc:.3f})", lw=1.8)

        prec, rec, _ = precision_recall_curve(y_true, y_prob)
        ap = average_precision_score(y_true, y_prob)
        axes[1].plot(rec, prec, label=f"{name} (AP={ap:.3f})", lw=1.8)

    # ROC
    axes[0].plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5)
    axes[0].set_xlabel("False Positive Rate")
    axes[0].set_ylabel("True Positive Rate")
    axes[0].set_title("ROC Curve")
    axes[0].legend()
    axes[0].set_aspect("equal")

    # PR
    axes[1].axhline(y_true.mean(), color="k", linestyle="--", lw=0.8, alpha=0.5,
                    label=f"Baseline ({y_true.mean():.3%})")
    axes[1].set_xlabel("Recall")
    axes[1].set_ylabel("Precision")
    axes[1].set_title("Precision–Recall Curve")
    axes[1].legend()

    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_calibration(y_true: np.ndarray, probas: dict[str, np.ndarray],
                     n_bins: int = 10, save_path: Optional[str] = None) -> plt.Figure:
    """
    Calibration curves. Ideally points cluster along the diagonal.
    Over-confident model → above diagonal. Under-confident → below.
    """
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot([0, 1], [0, 1], "k--", lw=0.8, alpha=0.5, label="Perfect calibration")

    for name, y_prob in probas.items():
        frac_pos, mean_pred = calibration_curve(y_true, y_prob, n_bins=n_bins)
        ax.plot(mean_pred, frac_pos, "o-", label=name, lw=1.8, ms=5)

    ax.set_xlabel("Mean Predicted Probability")
    ax.set_ylabel("Fraction of Positives")
    ax.set_title("Calibration Curve")
    ax.legend()
    ax.set_aspect("equal")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def plot_shot_heatmap(shots: pd.DataFrame, xg_col: str = "xg",
                      title: str = "Shot xG Heatmap",
                      save_path: Optional[str] = None) -> plt.Figure:
    """
    2D hexbin heatmap of shot locations weighted by xG.
    Useful stakeholder visual: shows where dangerous shots originate.
    """
    fig, ax = plt.subplots(figsize=(10, 8))

    # Draw rink outline (simplified)
    _draw_rink_half(ax)

    hb = ax.hexbin(
        shots["x_coord"].abs(), shots["y_coord"],
        C=shots[xg_col],
        reduce_C_function=np.mean,
        gridsize=25,
        cmap="RdYlGn_r",
        vmin=0, vmax=0.25,
        extent=[25, 89, -42.5, 42.5],
    )
    cb = plt.colorbar(hb, ax=ax)
    cb.set_label("Mean xG")
    ax.set_title(title)
    ax.set_xlabel("Feet from center ice")
    ax.set_ylabel("Feet from center line")
    plt.tight_layout()
    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
    return fig


def _draw_rink_half(ax: plt.Axes) -> None:
    """Draw simplified offensive half-rink outline."""
    from matplotlib.patches import Arc, Circle, FancyArrowPatch

    # boards
    ax.set_xlim(25, 100)
    ax.set_ylim(-42.5, 42.5)

    # goal crease
    crease = plt.Circle((89, 0), 6, color="#6baed6", alpha=0.3, zorder=1)
    ax.add_patch(crease)

    # goal line
    ax.axvline(89, color="red", lw=0.8, alpha=0.5)

    # blue line
    ax.axvline(25, color="blue", lw=1.5, alpha=0.4)

    ax.set_facecolor("#f7f7f7")


# ── Player / Team Summaries ────────────────────────────────────────────────────

def player_xg_summary(shots: pd.DataFrame,
                       xg_col: str = "xg",
                       player_col: str = "shooterName",
                       goal_col: str = "goal",
                       min_shots: int = 30) -> pd.DataFrame:
    """
    Per-player xG summary. Key for evaluating shooting talent vs luck.

    xG - Goals = "finishing" differential.
    Positive → underperforming (shooting below expected).
    Negative → overperforming (could regress toward expectation).
    """
    summary = (
        shots.groupby(player_col)
        .agg(
            shots=(goal_col, "count"),
            goals=(goal_col, "sum"),
            xG=(xg_col, "sum"),
        )
        .query(f"shots >= {min_shots}")
        .reset_index()
    )
    summary["goals_minus_xg"] = summary["goals"] - summary["xG"]
    summary["shooting_pct"] = summary["goals"] / summary["shots"]
    summary["xg_per_shot"] = summary["xG"] / summary["shots"]
    return summary.sort_values("xG", ascending=False).reset_index(drop=True)


def team_xg_over_time(shots: pd.DataFrame,
                       xg_col: str = "xg",
                       date_col: str = "game_date",
                       team_col: str = "team") -> pd.DataFrame:
    """
    Rolling team xG for/against by date. Good for stakeholder trend charts.
    """
    shots[date_col] = pd.to_datetime(shots[date_col])
    shots = shots.sort_values(date_col)

    records = []
    for team in shots[team_col].unique():
        team_shots = shots[shots[team_col] == team].copy()
        team_shots["cumulative_xg_for"] = team_shots[xg_col].cumsum()
        team_shots["team"] = team
        records.append(team_shots[[date_col, "team", "cumulative_xg_for"]])

    return pd.concat(records, ignore_index=True)
