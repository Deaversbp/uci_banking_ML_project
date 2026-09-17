"""Probability Calibration and Reliability Diagnostics for Supervised Models.

Methodological Guarantees:
- Strict Leakage Protection: Drops 'duration' prior to all feature engineering and calibration.
- Nested Cross-Validation: Evaluates calibration strictly on out-of-fold outer-validation splits.
  Calibrator and base classifier are fitted solely on outer-training splits using internal CV
  (via CalibratedClassifierCV with ensemble=False). Outer-validation data is never used for fitting.
- Partition Isolation: Operates strictly on the 80% development partition; holdout test partition is never accessed.
- Business Alignment: Quantifies impact on capacity-constrained lead ranking (5,000 / 45,211 capacity fraction).
"""
from pathlib import Path
from typing import Dict, Any, Tuple, Optional, List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.special import logit
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score, average_precision_score
from sklearn.model_selection import StratifiedKFold

from src.features.build_features import (
    create_pre_call_pipeline,
    drop_duration,
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
    - 5-fold outer StratifiedKFold cross-validation partitions the development data.
    - For each outer fold:
      1. Pipeline (feature engineer + preprocessor + RF) is trained on outer training data.
      2. CalibratedClassifierCV(..., ensemble=False, cv=inner_cv) trains the calibrator
         using out-of-fold predictions strictly within the outer training data.
      3. The outer validation fold is scored using:
         - Uncalibrated base pipeline
         - Sigmoid-calibrated classifier
         - Isotonic-calibrated classifier
      4. Outer validation labels are NEVER accessed during estimator or calibrator training.
    - Every development observation receives exactly one OOF prediction per method.

    Args:
        X_dev: Raw development feature DataFrame (excluding or including 'duration').
        y_dev: Development target Series ('yes' / 'no').
        n_outer_splits: Number of outer validation splits.
        n_inner_splits: Number of inner validation splits for calibration fitting.
        random_state: Random state for reproducible splitting.
        pos_label: Positive class label string.

    Returns:
        pd.DataFrame: Out-of-fold predictions with columns:
            ['orig_idx', 'actual_str', 'actual', 'raw_score', 'sigmoid_prob', 'isotonic_prob', 'fold']
    """
    X_clean = drop_duration(X_dev)
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
        f"Starting nested probability calibration across {n_outer_splits} outer folds "
        f"and {n_inner_splits} inner folds on {n_dev:,} development records..."
    )

    for fold_idx, (train_idx, val_idx) in enumerate(skf_outer.split(X_clean, y_dev)):
        X_fold_train = X_clean.iloc[train_idx]
        y_fold_train = y_dev.iloc[train_idx]
        X_fold_val = X_clean.iloc[val_idx]
        y_fold_val = y_dev.iloc[val_idx]

        # Construct leak-free pipeline
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
    assert "duration" not in df_oof.columns, "Leakage detected: 'duration' found in output"

    return df_oof


def build_reliability_table(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10,
    strategy: str = "uniform",
    min_count: int = 30,
) -> pd.DataFrame:
    """Build a detailed reliability / calibration table across score bins.

    Args:
        y_true: Ground truth binary targets (0 or 1).
        y_prob: Predicted probabilities.
        n_bins: Number of probability bins (default: 10).
        strategy: Binning strategy: 'uniform' (equal width) or 'quantile' (equal frequency).
        min_count: Minimum sample threshold to flag unstable bin estimates.

    Returns:
        pd.DataFrame: Table containing bin intervals, counts, mean predicted prob,
            observed rate, calibration gap, and small-sample flags.
    """
    y_arr = np.asarray(y_true).astype(int)
    p_arr = np.asarray(y_prob, dtype=float)
    n_samples = len(y_arr)

    if strategy == "uniform":
        bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        # Digitize into 1..n_bins
        bin_indices = np.digitize(p_arr, bin_edges, right=True)
        bin_indices = np.clip(bin_indices, 1, n_bins)
    elif strategy == "quantile":
        quantiles = np.linspace(0, 1, n_bins + 1)
        bin_edges = np.percentile(p_arr, quantiles * 100)
        bin_edges = np.unique(bin_edges)
        if len(bin_edges) <= 2:
            bin_edges = np.linspace(0.0, 1.0, n_bins + 1)
        bin_indices = np.digitize(p_arr, bin_edges[1:], right=True) + 1
        bin_indices = np.clip(bin_indices, 1, len(bin_edges) - 1)
        n_bins = len(bin_edges) - 1
    else:
        raise ValueError(f"Unsupported strategy '{strategy}'. Use 'uniform' or 'quantile'.")

    rows: List[Dict[str, Any]] = []
    for b in range(1, n_bins + 1):
        mask = (bin_indices == b)
        count = int(mask.sum())
        low = float(bin_edges[b - 1])
        high = float(bin_edges[b])

        if count == 0:
            rows.append({
                "bin_idx": b,
                "bin_range": f"[{low:.2f}, {high:.2f}]",
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
            "bin_range": f"[{low:.2f}, {high:.2f}]",
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
    return df_table


def compare_calibration_methods(
    df_oof_cal: pd.DataFrame,
    k: int = 4000,
) -> Dict[str, Any]:
    """Perform exhaustive comparison between Uncalibrated, Sigmoid, and Isotonic models.

    Computes:
    - Probability quality metrics (Brier, Log loss, Intercept, Slope)
    - Discrimination metrics (ROC-AUC, PR-AUC)
    - Capacity ranking metrics (Conversions@k, Precision@k, Recall@k, Lift@k)
    - Prospect selection overlap, churn count, and conversion differences.

    Args:
        df_oof_cal: DataFrame from generate_oof_calibrated_predictions.
        k: Evaluated outreach capacity.

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

        # Rank descending and record top-k
        ranked = np.argsort(-probs)
        top_k_indices[label] = ranked[:k]

    # Baseline comparisons against Uncalibrated RF
    raw_top_k = set(top_k_indices["Uncalibrated RF"])
    raw_conversions = method_metrics["Uncalibrated RF"]["conversions_at_k"]

    overlap_summary: Dict[str, Any] = {}
    for label in ["Sigmoid Calibrated", "Isotonic Calibrated"]:
        target_top_k = set(top_k_indices[label])
        target_conversions = method_metrics[label]["conversions_at_k"]

        shared = len(raw_top_k.intersection(target_top_k))
        changed_leads = len(raw_top_k.symmetric_difference(target_top_k)) // 2
        overlap_pct = (shared / k) * 100.0
        delta_conversions = target_conversions - raw_conversions

        overlap_summary[label] = {
            "shared_leads": shared,
            "overlap_pct": overlap_pct,
            "changed_leads": changed_leads,
            "delta_conversions": delta_conversions,
        }

    return {
        "k_evaluated": k,
        "method_metrics": method_metrics,
        "overlap_summary": overlap_summary,
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
    3. 12_calibration_binned_gaps.png: Binned calibration gap bar chart highlighting over/underconfidence.

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
        "Uncalibrated RF": "#4A90E2",    # Clean Blue
        "Sigmoid Calibrated": "#E2841A",  # Warm Amber/Orange
        "Isotonic Calibrated": "#2ECC71", # Vibrant Green
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
    rel_raw = build_reliability_table(y_arr, raw, n_bins=10, strategy="uniform")
    rel_sig = build_reliability_table(y_arr, sig, n_bins=10, strategy="uniform")
    rel_iso = build_reliability_table(y_arr, iso, n_bins=10, strategy="uniform")

    x_indices = np.arange(10)
    bar_width = 0.26

    fig, ax = plt.subplots(figsize=(10, 5.5))
    ax.bar(x_indices - bar_width, rel_raw["abs_calibration_gap"], width=bar_width, color=colors["Uncalibrated RF"], alpha=0.85, label="Uncalibrated RF")
    ax.bar(x_indices, rel_sig["abs_calibration_gap"], width=bar_width, color=colors["Sigmoid Calibrated"], alpha=0.85, label="Sigmoid Calibrated")
    ax.bar(x_indices + bar_width, rel_iso["abs_calibration_gap"], width=bar_width, color=colors["Isotonic Calibrated"], alpha=0.85, label="Isotonic Calibrated")

    ax.set_title("Absolute Calibration Gap (|Mean Pred - Observed Rate|) by Probability Decile", fontsize=12, fontweight="bold", pad=12)
    ax.set_xlabel("Probability Decile Bin", fontsize=10, fontweight="bold")
    ax.set_ylabel("Absolute Gap (Lower is Better)", fontsize=10, fontweight="bold")
    ax.set_xticks(x_indices)
    ax.set_xticklabels(rel_raw["bin_range"], rotation=35, ha="right", fontsize=9)
    ax.grid(True, linestyle="--", alpha=0.5)
    ax.legend(loc="upper right", framealpha=0.95, fontsize=10)

    f12_path = out_path / "12_calibration_binned_gaps.png"
    plt.savefig(f12_path, dpi=300, bbox_inches="tight")
    plt.close()
    generated_figures["binned_gaps"] = f12_path

    logger.info(f"Successfully generated and saved 3 calibration figures to {out_path}")
    return generated_figures
