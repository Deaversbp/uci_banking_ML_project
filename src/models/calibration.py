"""Probability Calibration and Reliability Diagnostics for Supervised Models.

Methodological Guarantees:
- Strict Pre-Campaign Contract: Enforces canonical 11 raw pre-campaign features.
  Purges post-decision execution variables (duration, contact, month, day, campaign, y).
- Nested Cross-Validation: Evaluates calibration strictly on out-of-fold outer-validation splits.
  Calibrator and base classifier are fitted solely on outer-training splits using internal CV
  (via CalibratedClassifierCV with ensemble=False). Outer-validation data is never used for fitting.
- Partition Isolation: Operates strictly on the 80% development partition; holdout test partition is never accessed.
- Business Alignment: Quantifies impact on capacity-constrained lead ranking (5,000 / 45,211 capacity fraction).
  Evaluates fold-by-fold ranking guardrails at fold-specific capacity k_fold = 800.
"""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit
from scipy.stats import spearmanr
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.features.build_features import (
    create_pre_call_pipeline,
)
from src.features.feature_contract import (
    select_canonical_pre_campaign_features,
    validate_pre_campaign_feature_contract,
    CANONICAL_PRE_CAMPAIGN_RAW_FEATURES,
    CANONICAL_FORBIDDEN_FEATURES,
)
from src.models.evaluate import compute_ranking_metrics_at_k
from src.models.train import compute_capacity_k
from src.utils.logger import setup_logger, get_project_root

logger = setup_logger(__name__)


def compute_calibration_slope_and_intercept(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    clip_eps: float = 1e-6,
) -> Tuple[float, float]:
    """Compute calibration intercept and slope using logistic regression on log-odds.

    Model: logit(P(Y=1)) = alpha + beta * logit(p)
    - Perfect calibration corresponds to alpha = 0 (intercept) and beta = 1 (slope).
    - Slope < 1 indicates overconfident predictions (too extreme).
    - Slope > 1 indicates underconfident predictions (too central).
    - Intercept > 0 indicates systematic underprediction of overall probability.
    - Intercept < 0 indicates systematic overprediction of overall probability.

    Args:
        y_true: Binary ground truth labels (0 or 1).
        y_prob: Predicted continuous probabilities in [0, 1].
        clip_eps: Epsilon to clip probabilities away from 0 and 1 before logit transform.

    Returns:
        Tuple[float, float]: (calibration_intercept, calibration_slope)
    """
    y_arr = np.asarray(y_true).astype(int)
    p_clipped = np.clip(np.asarray(y_prob, dtype=float), clip_eps, 1.0 - clip_eps)
    logit_p = logit(p_clipped).reshape(-1, 1)

    # Use large C to perform unregularized logistic regression without deprecation warnings
    lr = LogisticRegression(C=1e9, solver="lbfgs")
    lr.fit(logit_p, y_arr)

    intercept = float(lr.intercept_[0])
    slope = float(lr.coef_[0][0])
    return intercept, slope


def compute_calibration_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    k: Optional[int] = None,
    pos_label: int = 1,
) -> Dict[str, Any]:
    """Compute comprehensive probability quality metrics and optional ranking metrics.

    Args:
        y_true: Ground truth binary labels (0 or 1).
        y_prob: Predicted continuous probabilities.
        k: Optional capacity cutoff for ranking metrics.
        pos_label: Positive target label.

    Returns:
        Dict[str, Any]: Dictionary containing Brier score, log-loss, slope, intercept, AUCs, and ranking metrics.
    """
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(y_prob, dtype=float)

    brier = float(brier_score_loss(y_arr, p_arr))
    # Clip probabilities slightly to avoid log_loss infinite penalties on extreme values
    p_loss = np.clip(p_arr, 1e-15, 1.0 - 1e-15)
    loss = float(log_loss(y_arr, p_loss))

    intercept, slope = compute_calibration_slope_and_intercept(y_arr, p_arr)

    roc = float(roc_auc_score(y_arr, p_arr)) if len(np.unique(y_arr)) > 1 else np.nan
    pr = float(average_precision_score(y_arr, p_arr)) if len(np.unique(y_arr)) > 1 else np.nan

    metrics: Dict[str, Any] = {
        "brier_score": brier,
        "log_loss": loss,
        "calibration_intercept": intercept,
        "calibration_slope": slope,
        "roc_auc": roc,
        "pr_auc": pr,
    }

    if k is not None:
        ranking = compute_ranking_metrics_at_k(pd.Series(y_arr), p_arr, k=k, pos_label=pos_label)
        metrics.update({
            "k_evaluated": ranking["k_evaluated"],
            "conversions_at_k": ranking["conversions_at_k"],
            "precision_at_k": ranking["precision_at_k"],
            "recall_at_k": ranking["recall_at_k"],
            "lift_at_k": ranking["lift_at_k"],
        })

    return metrics


def generate_oof_calibrated_predictions(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    n_outer_splits: int = 5,
    n_inner_splits: int = 3,
    random_state: int = 42,
    pos_label: str = "yes",
) -> pd.DataFrame:
    """Generate nested out-of-fold calibrated predictions across the development set.

    Evaluation Design:
    - Enforces canonical pre-campaign feature contract; purges all forbidden execution variables.
    - 5-fold outer StratifiedKFold cross-validation partitions the development data.
    - For each outer fold:
      1. Pipeline (feature engineer + preprocessor + RF) is trained on outer training data.
      2. CalibratedClassifierCV(..., ensemble=False, cv=inner_cv) trains the calibrator
         using out-of-fold predictions strictly within the outer training data.
      3. The outer validation fold is scored using:
         - Uncalibrated base pipeline (raw ranking score)
         - Sigmoid-calibrated classifier
         - Isotonic-calibrated classifier
      4. Outer validation labels are NEVER accessed during estimator or calibrator training.
    - Every development observation receives exactly one OOF prediction per method.

    Args:
        X_dev: Raw development feature DataFrame.
        y_dev: Development target Series ('yes' / 'no').
        n_outer_splits: Number of outer validation splits.
        n_inner_splits: Number of inner validation splits for calibration fitting.
        random_state: Random state for reproducible splitting.
        pos_label: Positive class label string.

    Returns:
        pd.DataFrame: Out-of-fold predictions with columns:
            ['orig_idx', 'actual_str', 'actual', 'raw_score', 'sigmoid_prob', 'isotonic_prob', 'fold']
    """
    X_clean = select_canonical_pre_campaign_features(X_dev)
    validate_pre_campaign_feature_contract(X_clean, raise_on_violation=True)

    skf_outer = StratifiedKFold(n_splits=n_outer_splits, shuffle=True, random_state=random_state)

    n_dev = len(X_clean)
    raw_scores = np.zeros(n_dev, dtype=float)
    sigmoid_probs = np.zeros(n_dev, dtype=float)
    isotonic_probs = np.zeros(n_dev, dtype=float)
    oof_folds = np.zeros(n_dev, dtype=int)

    rf_params = {
        "n_estimators": 100,
        "max_depth": 12,
        "class_weight": None,
        "random_state": random_state,
        "n_jobs": -1,
    }

    logger.info(
        f"Starting compliant nested probability calibration across {n_outer_splits} outer folds "
        f"and {n_inner_splits} inner folds on {n_dev:,} development records..."
    )

    for fold_idx, (train_idx, val_idx) in enumerate(skf_outer.split(X_clean, y_dev)):
        X_fold_train = X_clean.iloc[train_idx]
        y_fold_train = y_dev.iloc[train_idx]
        X_fold_val = X_clean.iloc[val_idx]
        y_fold_val = y_dev.iloc[val_idx]

        validate_pre_campaign_feature_contract(X_fold_train, raise_on_violation=True)
        validate_pre_campaign_feature_contract(X_fold_val, raise_on_violation=True)

        # Construct leak-free pipeline under canonical contract
        rf = RandomForestClassifier(**rf_params)
        base_pipeline = create_pre_call_pipeline(classifier=rf, raw_feature_df=X_fold_train)

        # Internal CV for calibrator training strictly within outer-training data
        inner_cv = StratifiedKFold(n_splits=n_inner_splits, shuffle=True, random_state=random_state)

        # 1. Sigmoid / Platt calibration
        cal_sigmoid = CalibratedClassifierCV(
            estimator=base_pipeline,
            method="sigmoid",
            cv=inner_cv,
            ensemble=False,
        )
        cal_sigmoid.fit(X_fold_train, y_fold_train)

        # 2. Isotonic calibration
        cal_isotonic = CalibratedClassifierCV(
            estimator=base_pipeline,
            method="isotonic",
            cv=inner_cv,
            ensemble=False,
        )
        cal_isotonic.fit(X_fold_train, y_fold_train)

        # Extract base fitted pipeline to obtain exact uncalibrated raw scores
        fitted_base_pipeline = cal_sigmoid.calibrated_classifiers_[0].estimator
        classes = list(fitted_base_pipeline.classes_)
        if pos_label not in classes:
            raise ValueError(f"Positive label '{pos_label}' not in classes: {classes}")
        pos_idx = classes.index(pos_label)

        # Generate outer validation predictions
        raw_scores[val_idx] = fitted_base_pipeline.predict_proba(X_fold_val)[:, pos_idx]
        sigmoid_probs[val_idx] = cal_sigmoid.predict_proba(X_fold_val)[:, pos_idx]
        isotonic_probs[val_idx] = cal_isotonic.predict_proba(X_fold_val)[:, pos_idx]
        oof_folds[val_idx] = fold_idx

        logger.info(
            f"Calibration Fold {fold_idx + 1}/{n_outer_splits} finished "
            f"({len(val_idx):,} val rows scored)."
        )

    y_binary = (pd.Series(y_dev).reset_index(drop=True) == pos_label).astype(int)

    df_oof = pd.DataFrame({
        "orig_idx": X_dev.index,
        "actual_str": y_dev.values,
        "actual": y_binary.values,
        "raw_score": raw_scores,
        "sigmoid_prob": sigmoid_probs,
        "isotonic_prob": isotonic_probs,
        "fold": oof_folds,
    }, index=X_dev.index)

    # Verification assertions
    assert len(df_oof) == n_dev, f"Row count mismatch: {len(df_oof)} vs {n_dev}"
    assert not df_oof[["raw_score", "sigmoid_prob", "isotonic_prob"]].isna().any().any(), "NaN scores detected"
    for forbidden in CANONICAL_FORBIDDEN_FEATURES:
        assert forbidden not in df_oof.columns, f"Forbidden field '{forbidden}' found in output"

    return df_oof


def build_reliability_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "quantile",
    min_count: int = 30,
) -> pd.DataFrame:
    """Build a detailed reliability / calibration table across score bins.

    Supports both equal-width ('uniform') and equal-frequency ('quantile') binning.
    Equal-frequency binning ensures robust sample counts in low-prevalence regimes (~11.7%).

    Args:
        y_true: Ground truth binary targets (0 or 1).
        y_prob: Predicted probabilities.
        n_bins: Number of probability bins (default: 10).
        strategy: Binning strategy: 'quantile' (equal frequency) or 'uniform' (equal width).
        min_count: Minimum sample threshold to flag unstable bin estimates (default: 30).

    Returns:
        pd.DataFrame: Table containing bin intervals, counts, mean predicted prob,
            observed rate, calibration gap, and small-sample flags.
    """
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(y_prob, dtype=float)
    n_samples = len(y_arr)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(p_arr, bin_edges, right=True)
        bin_indices = np.clip(bin_indices, 1, n_bins)
        ranges = [f"[{bin_edges[b - 1]:.2f}, {bin_edges[b]:.2f}]" for b in range(1, n_bins + 1)]
    elif strategy == "quantile":
        ranks = pd.Series(p_arr).rank(method="first")
        bin_indices = (pd.qcut(ranks, q=n_bins, labels=False) + 1).values
        ranges = []
        for b in range(1, n_bins + 1):
            mask = (bin_indices == b)
            if mask.sum() > 0:
                p_min, p_max = p_arr[mask].min(), p_arr[mask].max()
                ranges.append(f"[{p_min:.4f}, {p_max:.4f}]")
            else:
                ranges.append("[N/A]")
    else:
        raise ValueError(f"Unsupported strategy '{strategy}'. Use 'uniform' or 'quantile'.")

    rows: List[Dict[str, Any]] = []
    for b in range(1, n_bins + 1):
        mask = (bin_indices == b)
        count = int(mask.sum())
        bin_label = ranges[b - 1]

        if count == 0:
            rows.append({
                "bin_idx": b,
                "bin_range": bin_label,
                "count": 0,
                "pct_of_total": 0.0,
                "positives": 0,
                "mean_predicted_prob": np.nan,
                "observed_rate": np.nan,
                "abs_calibration_gap": np.nan,
                "small_sample_flag": True,
            })
            continue

        positives = int(y_arr[mask].sum())
        mean_pred = float(p_arr[mask].mean())
        obs_rate = float(positives / count)
        gap = float(abs(mean_pred - obs_rate))

        rows.append({
            "bin_idx": b,
            "bin_range": bin_label,
            "count": count,
            "pct_of_total": float(count / n_samples * 100),
            "positives": positives,
            "mean_predicted_prob": mean_pred,
            "observed_rate": obs_rate,
            "abs_calibration_gap": gap,
            "small_sample_flag": bool(count < min_count),
        })

    df_table = pd.DataFrame(rows)
    assert df_table["count"].sum() == n_samples, f"Bin count sum {df_table['count'].sum()} != {n_samples}"
    assert df_table["positives"].sum() == y_arr.sum(), f"Positives sum {df_table['positives'].sum()} != {y_arr.sum()}"
    return df_table


def evaluate_fold_level_ranking(
    df_oof_cal: pd.DataFrame,
    full_capacity: int = 5000,
    total_population: int = 45211,
) -> Dict[str, Any]:
    """Evaluate fold-by-fold ranking performance and ranking preservation.

    Computes for each outer validation fold:
    - k_fold derived from fixed capacity fraction (5,000 / 45,211)
    - Ranking metrics at k_fold (Conversions@k, Precision@k, Recall@k, Lift@k, PR-AUC, ROC-AUC)
    - Paired comparisons vs Raw RF (top-k overlap, changed leads, conversion delta, Spearman rho)
    - Aggregated metrics across outer folds (mean, std, min, max, total deltas)

    Args:
        df_oof_cal: OOF prediction dataframe containing 'fold', 'actual', and model scores.
        full_capacity: Full campaign call capacity (5,000).
        total_population: Total population size (45,211).

    Returns:
        Dict[str, Any]: Fold-by-fold metrics and aggregated summary.
    """
    folds = sorted(df_oof_cal["fold"].unique())
    fold_results: List[Dict[str, Any]] = []

    methods = [
        ("raw", "raw_score"),
        ("sigmoid", "sigmoid_prob"),
        ("isotonic", "isotonic_prob"),
    ]

    for f_idx in folds:
        df_fold = df_oof_cal[df_oof_cal["fold"] == f_idx].copy()
        n_val = len(df_fold)
        k_fold = compute_capacity_k(n_val, total_population, full_capacity)
        y_val = df_fold["actual"].values
        positives = int(y_val.sum())

        fold_entry: Dict[str, Any] = {
            "fold": int(f_idx),
            "n_val": n_val,
            "k_fold": k_fold,
            "positives": positives,
        }

        top_k_sets: Dict[str, set] = {}

        for m_key, col in methods:
            probs = df_fold[col].values
            ranked_idx = np.lexsort((np.arange(len(probs)), -probs))
            top_k = set(ranked_idx[:k_fold])
            top_k_sets[m_key] = top_k

            conv = int(y_val[list(top_k)].sum())
            prec = float(conv / k_fold) if k_fold > 0 else 0.0
            rec = float(conv / positives) if positives > 0 else 0.0
            prev = float(positives / n_val) if n_val > 0 else 0.0
            lift = float(prec / prev) if prev > 0 else 0.0

            roc = float(roc_auc_score(y_val, probs))
            pr = float(average_precision_score(y_val, probs))
            brier = float(brier_score_loss(y_val, probs))
            p_loss = np.clip(probs, 1e-15, 1.0 - 1e-15)
            loss = float(log_loss(y_val, p_loss))

            m_dict = {
                "conversions": conv,
                "precision": prec,
                "recall": rec,
                "lift": lift,
                "roc_auc": roc,
                "pr_auc": pr,
                "brier": brier,
                "log_loss": loss,
            }

            if m_key != "raw":
                raw_top_k = top_k_sets["raw"]
                raw_conv = fold_entry["raw"]["conversions"]
                raw_pr = fold_entry["raw"]["pr_auc"]
                raw_roc = fold_entry["raw"]["roc_auc"]

                shared = len(raw_top_k.intersection(top_k))
                changed = k_fold - shared
                delta_conv = conv - raw_conv
                rho = float(spearmanr(df_fold["raw_score"], df_fold[col]).statistic)

                m_dict.update({
                    "shared_leads": shared,
                    "changed_leads": changed,
                    "overlap_pct": float(shared / k_fold * 100),
                    "delta_conversions": delta_conv,
                    "spearman_rho": rho,
                    "delta_pr_auc": pr - raw_pr,
                    "delta_roc_auc": roc - raw_roc,
                })

            fold_entry[m_key] = m_dict

        fold_results.append(fold_entry)

    # Aggregations across folds
    fold_agg: Dict[str, Any] = {}
    for m_key, _ in methods:
        m_conv = [f[m_key]["conversions"] for f in fold_results]
        m_prec = [f[m_key]["precision"] for f in fold_results]
        m_rec = [f[m_key]["recall"] for f in fold_results]
        m_lift = [f[m_key]["lift"] for f in fold_results]
        m_roc = [f[m_key]["roc_auc"] for f in fold_results]
        m_pr = [f[m_key]["pr_auc"] for f in fold_results]
        m_brier = [f[m_key]["brier"] for f in fold_results]
        m_loss = [f[m_key]["log_loss"] for f in fold_results]

        fold_agg[m_key] = {
            "conversions": {"mean": float(np.mean(m_conv)), "std": float(np.std(m_conv, ddof=1)), "sum": int(np.sum(m_conv))},
            "precision": {"mean": float(np.mean(m_prec)), "std": float(np.std(m_prec, ddof=1))},
            "recall": {"mean": float(np.mean(m_rec)), "std": float(np.std(m_rec, ddof=1))},
            "lift": {"mean": float(np.mean(m_lift)), "std": float(np.std(m_lift, ddof=1))},
            "roc_auc": {"mean": float(np.mean(m_roc)), "std": float(np.std(m_roc, ddof=1))},
            "pr_auc": {"mean": float(np.mean(m_pr)), "std": float(np.std(m_pr, ddof=1))},
            "brier": {"mean": float(np.mean(m_brier)), "std": float(np.std(m_brier, ddof=1))},
            "log_loss": {"mean": float(np.mean(m_loss)), "std": float(np.std(m_loss, ddof=1))},
        }

    paired_summary: Dict[str, Any] = {}
    for m_key in ["sigmoid", "isotonic"]:
        deltas_conv = [f[m_key]["delta_conversions"] for f in fold_results]
        overlaps = [f[m_key]["overlap_pct"] for f in fold_results]
        changed = [f[m_key]["changed_leads"] for f in fold_results]
        rhos = [f[m_key]["spearman_rho"] for f in fold_results]
        delta_pr = [f[m_key]["delta_pr_auc"] for f in fold_results]
        delta_roc = [f[m_key]["delta_roc_auc"] for f in fold_results]

        paired_summary[m_key] = {
            "delta_conversions": {"mean": float(np.mean(deltas_conv)), "std": float(np.std(deltas_conv, ddof=1)), "total": int(np.sum(deltas_conv))},
            "overlap_pct": {"mean": float(np.mean(overlaps)), "std": float(np.std(overlaps, ddof=1))},
            "changed_leads": {"mean": float(np.mean(changed)), "std": float(np.std(changed, ddof=1)), "total": int(np.sum(changed))},
            "spearman_rho": {"mean": float(np.mean(rhos)), "std": float(np.std(rhos, ddof=1))},
            "delta_pr_auc": {"mean": float(np.mean(delta_pr)), "std": float(np.std(delta_pr, ddof=1))},
            "delta_roc_auc": {"mean": float(np.mean(delta_roc)), "std": float(np.std(delta_roc, ddof=1))},
        }

    return {
        "fold_results": fold_results,
        "fold_agg": fold_agg,
        "paired_summary": paired_summary,
    }


def compare_calibration_methods(
    df_oof_cal: pd.DataFrame,
    k: int = 4000,
) -> Dict[str, Any]:
    """Perform exhaustive comparison between Uncalibrated, Sigmoid, and Isotonic models.

    Computes:
    - Pooled probability quality metrics (Brier, Log loss, Intercept, Slope)
    - Pooled discrimination metrics (ROC-AUC, PR-AUC)
    - Pooled capacity ranking metrics at k (Conversions@k, Precision@k, Recall@k, Lift@k)
    - Fold-by-fold ranking evaluation and preservation metrics across 5 outer folds.

    Args:
        df_oof_cal: DataFrame from generate_oof_calibrated_predictions.
        k: Evaluated pooled outreach capacity (default: 4,000).

    Returns:
        Dict[str, Any]: Comprehensive comparison dictionary.
    """
    methods = [
        ("Uncalibrated RF", "raw_score"),
        ("Sigmoid Calibrated", "sigmoid_prob"),
        ("Isotonic Calibrated", "isotonic_prob"),
    ]

    method_metrics: Dict[str, Dict[str, Any]] = {}
    top_k_indices: Dict[str, np.ndarray] = {}

    y_arr = df_oof_cal["actual"].values

    for label, col in methods:
        probs = df_oof_cal[col].values
        m = compute_calibration_metrics(y_arr, probs, k=k)
        method_metrics[label] = m

        # Deterministic descending rank sort
        ranked = np.lexsort((np.arange(len(probs)), -probs))
        top_k_indices[label] = ranked[:k]

    # Baseline comparisons against Uncalibrated RF
    raw_top_k = set(top_k_indices["Uncalibrated RF"])
    raw_conversions = method_metrics["Uncalibrated RF"]["conversions_at_k"]

    overlap_summary: Dict[str, Any] = {}
    for label in ["Sigmoid Calibrated", "Isotonic Calibrated"]:
        target_top_k = set(top_k_indices[label])
        target_conversions = method_metrics[label]["conversions_at_k"]

        shared = len(raw_top_k.intersection(target_top_k))
        changed_leads = k - shared
        overlap_pct = (shared / k) * 100.0
        delta_conversions = target_conversions - raw_conversions

        overlap_summary[label] = {
            "shared_leads": shared,
            "overlap_pct": overlap_pct,
            "changed_leads": changed_leads,
            "delta_conversions": delta_conversions,
        }

    # Fold-level ranking analysis if fold column is present
    fold_eval = evaluate_fold_level_ranking(df_oof_cal) if "fold" in df_oof_cal.columns else None

    return {
        "k_evaluated": k,
        "method_metrics": method_metrics,
        "overlap_summary": overlap_summary,
        "fold_level_evaluation": fold_eval,
    }


def generate_calibration_figures(
    df_oof_cal: pd.DataFrame,
    output_dir: Path | str,
    k: int = 4000,
) -> Dict[str, Path]:
    """Generate and save publication-quality diagnostic calibration figures.

    Figures generated:
    1. 10_calibration_reliability_curves.png: Multi-method calibration curves against the diagonal.
    2. 11_calibrated_probability_distributions.png: Comparative probability distributions.
    3. 12_calibration_binned_gaps.png: Binned calibration gap bar chart across deciles.

    Args:
        df_oof_cal: OOF prediction dataframe.
        output_dir: Output directory for image files.
        k: Evaluated capacity.

    Returns:
        Dict[str, Path]: Mapping of figure names to absolute file paths.
    """
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    generated_figures: Dict[str, Path] = {}

    y_arr = df_oof_cal["actual"].values
    raw = df_oof_cal["raw_score"].values
    sig = df_oof_cal["sigmoid_prob"].values
    iso = df_oof_cal["isotonic_prob"].values

    colors = {
        "Uncalibrated RF": "#1f77b4",    # Steel Blue
        "Sigmoid Calibrated": "#ff7f0e",  # Amber Orange
        "Isotonic Calibrated": "#2ca02c", # Emerald Green
    }

    # -------------------------------------------------------------
    # Figure 10: Multi-Method Reliability Curves
    # -------------------------------------------------------------
    fig, (ax1, ax2) = plt.subplots(
        nrows=2, ncols=1, figsize=(8, 9), sharex=True,
        gridspec_kw={"height_ratios": [3, 1], "hspace": 0.08}
    )

    # Perfect calibration reference line
    ax1.plot([0, 1], [0, 1], linestyle="--", color="#7F8C8D", linewidth=1.5, label="Perfect Calibration (y = x)")

    curves = [
        ("Uncalibrated RF", raw, colors["Uncalibrated RF"], "o-"),
        ("Sigmoid Calibrated", sig, colors["Sigmoid Calibrated"], "s-"),
        ("Isotonic Calibrated", iso, colors["Isotonic Calibrated"], "^-"),
    ]

    for label, probs, col, fmt in curves:
        prob_true, prob_pred = calibration_curve(y_arr, probs, n_bins=10, strategy="uniform")
        ax1.plot(prob_pred, prob_true, fmt, color=col, linewidth=2, markersize=6, label=label)

    ax1.set_ylabel("Observed Conversion Rate (Empirical)", fontsize=11, fontweight="bold")
    ax1.set_title("OOF Probability Calibration Reliability Curves (5-Fold Nested CV)", fontsize=13, fontweight="bold", pad=12)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="upper left", framealpha=0.95, fontsize=10)
    ax1.set_xlim(-0.02, 1.02)
    ax1.set_ylim(-0.02, 1.02)

    # Lower histogram showing probability density
    bins = np.linspace(0, 1, 21)
    ax2.hist(raw, bins=bins, color=colors["Uncalibrated RF"], alpha=0.35, label="Uncalibrated", density=True)
    ax2.hist(sig, bins=bins, color=colors["Sigmoid Calibrated"], alpha=0.35, label="Sigmoid", density=True)
    ax2.hist(iso, bins=bins, color=colors["Isotonic Calibrated"], alpha=0.35, label="Isotonic", density=True)
    ax2.set_xlabel("Predicted Probability / Score", fontsize=11, fontweight="bold")
    ax2.set_ylabel("Density", fontsize=10)
    ax2.grid(True, linestyle="--", alpha=0.4)
    ax2.legend(loc="upper right", framealpha=0.9, fontsize=8)

    f10_path = out_path / "10_calibration_reliability_curves.png"
    plt.savefig(f10_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_figures["reliability_curves"] = f10_path

    # -------------------------------------------------------------
    # Figure 11: Comparative Probability Distributions
    # -------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(9, 5))
    bins = np.linspace(0, 1, 41)

    ax.hist(raw, bins=bins, alpha=0.4, color=colors["Uncalibrated RF"], edgecolor="white", label="Uncalibrated RF (Raw Scores)")
    ax.hist(sig, bins=bins, alpha=0.4, color=colors["Sigmoid Calibrated"], edgecolor="white", label="Sigmoid Calibrated")
    ax.hist(iso, bins=bins, alpha=0.4, color=colors["Isotonic Calibrated"], edgecolor="white", label="Isotonic Calibrated")

    ax.set_title("Distribution of Out-of-Fold Scores and Calibrated Probabilities", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Probability / Score Value", fontsize=10, fontweight="bold")
    ax.set_ylabel("Number of Development Prospects", fontsize=10, fontweight="bold")
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)
    ax.set_xlim(-0.02, 1.02)

    f11_path = out_path / "11_calibrated_probability_distributions.png"
    plt.savefig(f11_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_figures["probability_distributions"] = f11_path

    # -------------------------------------------------------------
    # Figure 12: Binned Calibration Gaps
    # -------------------------------------------------------------
    rel_raw = build_reliability_table(y_arr, raw, n_bins=10, strategy="quantile")
    rel_sig = build_reliability_table(y_arr, sig, n_bins=10, strategy="quantile")
    rel_iso = build_reliability_table(y_arr, iso, n_bins=10, strategy="quantile")

    x_indices = np.arange(10)
    bar_width = 0.26

    fig, ax = plt.subplots(figsize=(11, 5.5))
    ax.bar(x_indices - bar_width, rel_raw["abs_calibration_gap"], width=bar_width, color=colors["Uncalibrated RF"], alpha=0.85, label="Uncalibrated RF")
    ax.bar(x_indices, rel_sig["abs_calibration_gap"], width=bar_width, color=colors["Sigmoid Calibrated"], alpha=0.85, label="Sigmoid Calibrated")
    ax.bar(x_indices + bar_width, rel_iso["abs_calibration_gap"], width=bar_width, color=colors["Isotonic Calibrated"], alpha=0.85, label="Isotonic Calibrated")

    ax.set_title("Absolute Calibration Gap (|Mean Pred - Observed Rate|) by Quantile Decile", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Decile Bin (Equal Frequency: ~3,617 Prospects/Bin)", fontsize=10, fontweight="bold")
    ax.set_ylabel("Absolute Gap (Lower is Better)", fontsize=10, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels([f"D{i+1}\n{r}" for i, r in enumerate(rel_raw["bin_range"])], rotation=0, ha="center", fontsize=8)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)

    f12_path = out_path / "12_calibration_binned_gaps.png"
    plt.savefig(f12_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_figures["binned_gaps"] = f12_path

    logger.info(f"Successfully generated and saved 3 calibration figures to {out_path}")
    return generated_figures
