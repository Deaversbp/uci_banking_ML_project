"""Model evaluation utilities and metrics reporting."""
from typing import Dict, Any
import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

def evaluate_model(
    model: Any,
    X_test,
    y_test,
    pos_label: str = "yes"
) -> Dict[str, Any]:
    """Evaluate a trained model or pipeline on a test set.

    Args:
        model: Trained estimator or Pipeline.
        X_test: Test features.
        y_test: True labels.
        pos_label: The positive class label (default: 'yes').

    Returns:
        Dict with metrics and reports.
    """
    y_pred = model.predict(X_test)

    # Compute predicted probabilities if supported
    y_proba = None
    roc_auc = None
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(X_test)
            # Find index of pos_label
            classes = list(model.classes_)
            if pos_label in classes:
                pos_idx = classes.index(pos_label)
                y_proba = probas[:, pos_idx]
                # Convert y_test to binary 0/1 for ROC-AUC
                y_test_binary = (y_test == pos_label).astype(int)
                roc_auc = float(roc_auc_score(y_test_binary, y_proba))
        except Exception as e:
            logger.warning(f"Could not compute ROC-AUC: {e}")

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred, pos_label=pos_label, zero_division=0)),
        "recall": float(recall_score(y_test, y_pred, pos_label=pos_label, zero_division=0)),
        "f1": float(f1_score(y_test, y_pred, pos_label=pos_label, zero_division=0)),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(y_test, y_pred, zero_division=0),
    }

    logger.info(f"Evaluation results -> Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']}")
    return metrics
