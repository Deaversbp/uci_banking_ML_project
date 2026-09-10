"""Non-ML baselines for pre-call outreach prioritization.

Provides:
1. Random Selection Baseline: Theoretical benchmark under random dialing.
2. Business-Rule Heuristic Baseline: Non-ML prioritization using pre-call domain rules
   (prior successful contacts, absence of debt burden, and balance tie-breaking).
"""
from typing import Dict, Any
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


def evaluate_random_baseline(
    y_true: pd.Series,
    k: int = 5000,
    pos_label: str = "yes"
) -> Dict[str, Any]:
    """Calculate theoretical expected performance under random outreach.

    Args:
        y_true: Ground truth target series.
        k: Call capacity constraint.
        pos_label: Positive class label.

    Returns:
        Dict with baseline ranking metrics.
    """
    n_total = len(y_true)
    n_pos = int((y_true == pos_label).sum())
    base_rate = n_pos / n_total
    k_eval = min(k, n_total)

    expected_conversions = k_eval * base_rate
    recall = expected_conversions / n_pos if n_pos > 0 else 0.0

    return {
        "baseline_name": "Random Selection Baseline",
        "capacity_k": k_eval,
        "total_population": n_total,
        "base_rate": base_rate,
        "conversions_at_k": float(expected_conversions),
        "precision_at_k": base_rate,
        "recall_at_k": recall,
        "lift_at_k": 1.0,
        "pr_auc": base_rate,
        "roc_auc": 0.5,
    }


def business_rule_baseline_score(X: pd.DataFrame) -> pd.Series:
    """Compute a deterministic heuristic score using legitimate pre-call signals.

    Heuristic Logic:
    - Tier 1 (+1000 pts): Prior campaign success (poutcome == 'success').
      Historically converts at ~64.7% (highest-propensity cohort).
    - Tier 2 (+500 pts): Previously contacted prospects (pdays != -1) without personal loans.
    - Tier 3 (+200 pts): Debt-free clients with neither housing nor personal loans.
    - Tie-breaking: Client average yearly balance (higher liquidity preferred).

    Args:
        X: Feature dataframe.

    Returns:
        pd.Series: Continuous heuristic ranking score.
    """
    scores = pd.Series(0.0, index=X.index)

    # 1. Prior success bonus
    if "poutcome" in X.columns:
        scores += np.where(X["poutcome"] == "success", 1000.0, 0.0)

    # 2. Prior contact without personal loan bonus
    if "pdays" in X.columns and "loan" in X.columns:
        prior_contact_clean = (X["pdays"] != -1) & (X["loan"] == "no")
        scores += np.where(prior_contact_clean, 500.0, 0.0)

    # 3. Debt-free bonus
    if "housing" in X.columns and "loan" in X.columns:
        debt_free = (X["housing"] == "no") & (X["loan"] == "no")
        scores += np.where(debt_free, 200.0, 0.0)

    # 4. Deterministic tie-breaker: normalized balance
    if "balance" in X.columns:
        bal = X["balance"].fillna(0)
        # Scaled balance contribution (clipped to [-10, 50] to break ties without overwhelming tiers)
        scores += np.clip(bal / 1000.0, -10.0, 50.0)

    return scores


def evaluate_business_rule_baseline(
    X: pd.DataFrame,
    y_true: pd.Series,
    k: int = 5000,
    pos_label: str = "yes"
) -> Dict[str, Any]:
    """Evaluate the heuristic business-rule baseline at outreach capacity k.

    Args:
        X: Feature dataframe.
        y_true: Ground truth target series.
        k: Call capacity constraint.
        pos_label: Positive class label.

    Returns:
        Dict with ranking metrics for the business-rule benchmark.
    """
    scores = business_rule_baseline_score(X)
    n_total = len(y_true)
    n_pos = int((y_true == pos_label).sum())
    base_rate = n_pos / n_total if n_total > 0 else 0.0
    k_eval = min(k, n_total)

    y_binary = (y_true == pos_label).astype(int)

    # Rank descending
    ranked_indices = scores.sort_values(ascending=False).index[:k_eval]
    top_k_targets = y_true.loc[ranked_indices]
    conversions = int((top_k_targets == pos_label).sum())

    precision_at_k = conversions / k_eval if k_eval > 0 else 0.0
    recall_at_k = conversions / n_pos if n_pos > 0 else 0.0
    lift_at_k = precision_at_k / base_rate if base_rate > 0 else 0.0

    try:
        roc_auc = float(roc_auc_score(y_binary, scores))
        pr_auc = float(average_precision_score(y_binary, scores))
    except Exception as e:
        logger.warning(f"Could not compute AUC for business rule: {e}")
        roc_auc, pr_auc = None, None

    logger.info(
        f"Business-Rule Baseline @ {k_eval:,} calls: {conversions:,} conversions, "
        f"Precision: {precision_at_k*100:.2f}%, Lift: {lift_at_k:.2f}x"
    )

    return {
        "baseline_name": "Business-Rule Heuristic Baseline",
        "capacity_k": k_eval,
        "total_population": n_total,
        "base_rate": base_rate,
        "conversions_at_k": conversions,
        "precision_at_k": precision_at_k,
        "recall_at_k": recall_at_k,
        "lift_at_k": lift_at_k,
        "pr_auc": pr_auc,
        "roc_auc": roc_auc,
    }
