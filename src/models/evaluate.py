"""Model evaluation utilities and lead-ranking metrics reporting.

Altered for capacity-constrained sales outreach (5,000 call limit):
- Primary metrics: PR-AUC (Average Precision), ROC-AUC, Conversions@k, Precision@k, Recall@k, Lift@k.
- Secondary diagnostics: Threshold-based F1, accuracy, classification report, and confusion matrix.
"""
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix,
    classification_report,
)

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def compute_ranking_metrics_at_k(
    y_true: pd.Series,
    y_proba: np.ndarray,
    k: int = 5000,
    pos_label: str = "yes"
) -> Dict[str, Any]:
    """Compute capacity-constrained lead ranking metrics.

    Ranks prospects descending by predicted probability of positive conversion.
    Evaluates cumulative conversions, precision, recall, and lift within top k calls.

    Args:
        y_true: True binary target labels.
        y_proba: Continuous predicted probabilities for the positive class.
        k: Call outreach limit (e.g. 5,000 calls).
        pos_label: Positive class label.

    Returns:
        Dict[str, Any]: Ranking metrics including lift and conversions at k.
    """
    n_samples = len(y_true)
    y_binary = (pd.Series(y_true).reset_index(drop=True) == pos_label).astype(int)
    total_positives = int(y_binary.sum())
    base_rate = total_positives / n_samples if n_samples > 0 else 0.0

    k_eval = min(k, n_samples)

    # Rank descending
    ranked_indices = np.argsort(-y_proba)
    top_k_indices = ranked_indices[:k_eval]
    conversions_at_k = int(y_binary.iloc[top_k_indices].sum())

    precision_at_k = conversions_at_k / k_eval if k_eval > 0 else 0.0
    recall_at_k = conversions_at_k / total_positives if total_positives > 0 else 0.0
    lift_at_k = precision_at_k / base_rate if base_rate > 0 else 0.0

    # Discrimination metrics
    roc_auc = float(roc_auc_score(y_binary, y_proba)) if total_positives > 0 else None
    pr_auc = float(average_precision_score(y_binary, y_proba)) if total_positives > 0 else None

    return {
        "k_evaluated": k_eval,
        "total_samples": n_samples,
        "base_rate": base_rate,
        "conversions_at_k": conversions_at_k,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "lift_at_k": lift_at_k,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    pos_label: str = "yes",
    k: int = 5000
) -> Dict[str, Any]:
    """Evaluate a trained model or pipeline with ranking metrics and classification diagnostics.

    Args:
        model: Trained estimator or Pipeline.
        X_test: Test features.
        y_test: True labels.
        pos_label: Positive class label (default: 'yes').
        k: Outreach capacity constraint (default: 5000).

    Returns:
        Dict with primary ranking metrics and secondary diagnostic reports.
    """
    y_pred = model.predict(X_test)
    y_test_series = pd.Series(y_test).reset_index(drop=True)
    y_test_binary = (y_test_series == pos_label).astype(int)

    # Compute predicted probabilities
    y_proba = None
    ranking_metrics = {}
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_test)
            classes = list(model.classes_)
            if pos_label in classes:
                pos_idx = classes.index(pos_label)
                y_proba = probas[:, pos_idx]
                ranking_metrics = compute_ranking_metrics_at_k(
                    y_true=y_test_series,
                    y_proba=y_proba,
                    k=k,
                    pos_label=pos_label
                )
        except Exception as e:
            logger.warning(f"Could not compute probability ranking metrics: {e}")

    # Fallback to decision function if predict_proba is unavailable
    elif hasattr(model, "decision_function"):
        try:
            scores = model.decision_function(X_test)
            y_proba = (scores - scores.min()) / (scores.max() - scores.min() + 1e-9)
            ranking_metrics = compute_ranking_metrics_at_k(
                y_true=y_test_series,
                y_proba=y_proba,
                k=k,
                pos_label=pos_label
            )
        except Exception as e:
            logger.warning(f"Could not compute decision ranking metrics: {e}")

    # Secondary diagnostic metrics
    diagnostics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    # Consolidated output
    metrics = {
        **ranking_metrics,
        "diagnostics": diagnostics,
        # Preserve top-level keys for backward compatibility
        "pr_auc": ranking_metrics.get("pr_auc"),
        "roc_auc": ranking_metrics.get("roc_auc"),
        "precision_at_k": ranking_metrics.get("precision_at_k"),
        "recall_at_k": ranking_metrics.get("recall_at_k"),
        "lift_at_k": ranking_metrics.get("lift_at_k"),
        "conversions_at_k": ranking_metrics.get("conversions_at_k"),
        "f1": diagnostics["f1"],
        "accuracy": diagnostics["accuracy"],
    }

    logger.info(
        f"Evaluation -> PR-AUC: {metrics.get('pr_auc', 0.0):.4f}, "
        f"ROC-AUC: {metrics.get('roc_auc', 0.0):.4f}, "
        f"Lift@k: {metrics.get('lift_at_k', 0.0):.2f}x, "
        f"Conversions@k: {metrics.get('conversions_at_k', 0)}"
    )
    return metrics
