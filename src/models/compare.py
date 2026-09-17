"""Supervised model comparison module for lead ranking optimization.

Executes development-only repeated stratified cross-validation on the 80% development
partition to evaluate candidate model families (Logistic Regression vs. Random Forest)
and class-weight sensitivity ('balanced' vs None) without touching the frozen 20% test set.

Enforces:
- Pre-call leakage protection (excludes 'duration' from all models and preprocessing).
- Proportional capacity constraint: capacity_fraction = 5,000 / 45,211 (~11.059%).
- Primary decision metric: Conversions@capacity.
- Secondary diagnostics: Precision@k, Recall@k, Lift@k, PR-AUC, ROC-AUC.
- Paired fold comparison: delta_conversions = RF Conversions@k - LR Conversions@k.
"""
from typing import Dict, Any, Optional, List, Tuple
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data.load_data import load_bank_marketing_data
from src.features.build_features import (
    drop_duration,
    split_data,
    create_pre_call_pipeline,
)
from src.models.evaluate import evaluate_model
from src.models.train import compute_capacity_k
from src.utils.logger import setup_logger, load_config

logger = setup_logger(__name__)


def get_comparison_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """Define the 4 controlled candidate model variants for comparison.

    Variants:
    - LogisticRegression (unweighted): class_weight=None
    - LogisticRegression (balanced): class_weight='balanced'
    - RandomForest (unweighted): class_weight=None
    - RandomForest (balanced): class_weight='balanced'

    Args:
        random_state: Random seed for deterministic initialization.

    Returns:
        Dict[str, Any]: Mapping of variant name to un-fitted scikit-learn estimator.
    """
    return {
        "LogisticRegression (unweighted)": LogisticRegression(
            max_iter=1000,
            class_weight=None,
            random_state=random_state,
        ),
        "LogisticRegression (balanced)": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state,
        ),
        "RandomForest (unweighted)": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight=None,
            random_state=random_state,
            n_jobs=-1,
        ),
        "RandomForest (balanced)": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1,
        ),
    }


def run_repeated_validation(
    X_dev: pd.DataFrame,
    y_dev: pd.Series,
    candidate_models: Optional[Dict[str, Any]] = None,
    n_splits: int = 5,
    n_repeats: int = 3,
    random_state: int = 42,
    full_capacity: int = 5000,
    total_population: int = 45211,
    pos_label: str = "yes",
) -> pd.DataFrame:
    """Run repeated stratified cross-validation strictly on the development partition.

    For each fold:
    1. Preprocessing and feature engineering are fit solely on the fold's training split.
    2. Candidate models are trained on the fold's training split.
    3. Outreach capacity k is computed proportionally from the fold's validation size.
    4. Lead-ranking and diagnostic metrics are evaluated on the fold's validation split.

    Args:
        X_dev: Feature dataframe for development partition (80% of data).
        y_dev: Target series for development partition.
        candidate_models: Optional dictionary of candidate models.
        n_splits: Number of CV folds per repeat (default: 5).
        n_repeats: Number of CV repeats (default: 3).
        random_state: Random state for CV split reproducibility.
        full_capacity: Full campaign call capacity constraint (default: 5,000).
        total_population: Total eligible population size (default: 45,211).
        pos_label: Positive class label (default: 'yes').

    Returns:
        pd.DataFrame: Fold-level evaluation results.
    """
    if candidate_models is None:
        candidate_models = get_comparison_candidate_models(random_state=random_state)

    # Ensure leakage protection
    X_dev_clean = drop_duration(X_dev)

    rskf = RepeatedStratifiedKFold(
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
    )

    records: List[Dict[str, Any]] = []
    total_folds = n_splits * n_repeats

    logger.info(
        f"Starting Repeated Stratified CV on {len(X_dev_clean):,} dev records: "
        f"{n_splits} folds × {n_repeats} repeats = {total_folds} evaluation splits."
    )

    for fold_idx, (train_idx, val_idx) in enumerate(rskf.split(X_dev_clean, y_dev)):
        repeat_idx = fold_idx // n_splits
        split_within_repeat = fold_idx % n_splits

        X_fold_train = X_dev_clean.iloc[train_idx]
        y_fold_train = y_dev.iloc[train_idx]
        X_fold_val = X_dev_clean.iloc[val_idx]
        y_fold_val = y_dev.iloc[val_idx]

        k_val = compute_capacity_k(
            n_samples=len(y_fold_val),
            total_population=total_population,
            full_capacity=full_capacity,
        )

        for model_name, clf_template in candidate_models.items():
            # Clone/re-instantiate estimator with same hyperparameters
            from sklearn.base import clone
            clf = clone(clf_template)

            pipeline = create_pre_call_pipeline(classifier=clf, raw_feature_df=X_fold_train)
            pipeline.fit(X_fold_train, y_fold_train)

            metrics = evaluate_model(
                pipeline,
                X_fold_val,
                y_fold_val,
                pos_label=pos_label,
                k=k_val,
            )

            records.append({
                "fold_id": fold_idx,
                "repeat": repeat_idx + 1,
                "fold_within_repeat": split_within_repeat + 1,
                "model": model_name,
                "k_evaluated": k_val,
                "val_samples": len(y_fold_val),
                "conversions_at_k": metrics["conversions_at_k"],
                "precision_at_k": metrics["precision_at_k"],
                "recall_at_k": metrics["recall_at_k"],
                "lift_at_k": metrics["lift_at_k"],
                "pr_auc": metrics.get("pr_auc"),
                "roc_auc": metrics.get("roc_auc"),
            })

        logger.info(f"Completed fold {fold_idx + 1}/{total_folds} (val_size={len(y_fold_val):,}, k_val={k_val:,})")

    return pd.DataFrame(records)


def compute_paired_delta_statistics(
    fold_results: pd.DataFrame,
    model_a: str,
    model_b: str,
    metric: str = "conversions_at_k",
) -> Dict[str, Any]:
    """Calculate paired fold-by-fold differences and win/loss stability.

    Delta = Model A Metric - Model B Metric

    Args:
        fold_results: DataFrame of fold-level records from run_repeated_validation.
        model_a: Name of Model A (e.g. 'RandomForest (balanced)').
        model_b: Name of Model B (e.g. 'LogisticRegression (balanced)').
        metric: Metric column to compute delta on (default: 'conversions_at_k').

    Returns:
        Dict[str, Any]: Summary statistics of paired deltas and win rates.
    """
    sub_a = fold_results[fold_results["model"] == model_a].set_index("fold_id")[metric]
    sub_b = fold_results[fold_results["model"] == model_b].set_index("fold_id")[metric]

    if len(sub_a) == 0:
        raise ValueError(f"Model '{model_a}' not found in fold results.")
    if len(sub_b) == 0:
        raise ValueError(f"Model '{model_b}' not found in fold results.")

    deltas = sub_a - sub_b
    n_folds = len(deltas)

    wins_a = int((deltas > 0).sum())
    wins_b = int((deltas < 0).sum())
    ties = int((deltas == 0).sum())

    return {
        "model_a": model_a,
        "model_b": model_b,
        "metric": metric,
        "n_folds": n_folds,
        "model_a_mean": float(sub_a.mean()),
        "model_a_median": float(sub_a.median()),
        "model_a_std": float(sub_a.std()),
        "model_a_min": float(sub_a.min()),
        "model_a_max": float(sub_a.max()),
        "model_b_mean": float(sub_b.mean()),
        "model_b_median": float(sub_b.median()),
        "model_b_std": float(sub_b.std()),
        "model_b_min": float(sub_b.min()),
        "model_b_max": float(sub_b.max()),
        "delta_mean": float(deltas.mean()),
        "delta_median": float(deltas.median()),
        "delta_std": float(deltas.std()),
        "delta_min": float(deltas.min()),
        "delta_max": float(deltas.max()),
        "model_a_wins": wins_a,
        "model_a_win_rate": float(wins_a / n_folds) if n_folds > 0 else 0.0,
        "model_b_wins": wins_b,
        "model_b_win_rate": float(wins_b / n_folds) if n_folds > 0 else 0.0,
        "ties": ties,
        "tie_rate": float(ties / n_folds) if n_folds > 0 else 0.0,
        "deltas": deltas.tolist(),
    }


def compute_aggregate_metrics(
    fold_results: pd.DataFrame,
    benchmark_model: str = "LogisticRegression (balanced)",
) -> pd.DataFrame:
    """Compute aggregate mean and std metrics across folds with paired fold win rate.

    Args:
        fold_results: Fold-level results DataFrame.
        benchmark_model: Baseline reference model to compute fold win rate against.

    Returns:
        pd.DataFrame: Formatted aggregate metrics table per candidate model.
    """
    models = fold_results["model"].unique()
    rows = []

    for m in models:
        df_m = fold_results[fold_results["model"] == m]
        
        # Calculate win rate vs benchmark model on conversions_at_k
        if m == benchmark_model:
            win_rate_pct = "-"
        else:
            paired = compute_paired_delta_statistics(fold_results, model_a=m, model_b=benchmark_model)
            win_rate_pct = f"{paired['model_a_win_rate'] * 100:.1f}%"

        rows.append({
            "model": m,
            "mean_conversions_at_k": df_m["conversions_at_k"].mean(),
            "std_conversions_at_k": df_m["conversions_at_k"].std(),
            "mean_precision_at_k": df_m["precision_at_k"].mean(),
            "std_precision_at_k": df_m["precision_at_k"].std(),
            "mean_recall_at_k": df_m["recall_at_k"].mean(),
            "std_recall_at_k": df_m["recall_at_k"].std(),
            "mean_lift_at_k": df_m["lift_at_k"].mean(),
            "std_lift_at_k": df_m["lift_at_k"].std(),
            "mean_pr_auc": df_m["pr_auc"].mean(),
            "std_pr_auc": df_m["pr_auc"].std(),
            "mean_roc_auc": df_m["roc_auc"].mean(),
            "std_roc_auc": df_m["roc_auc"].std(),
            "win_rate_vs_benchmark": win_rate_pct,
        })

    return pd.DataFrame(rows)


def compare_supervised_candidates(
    full_capacity: int = 5000,
    random_state: int = 42,
    n_splits: int = 5,
    n_repeats: int = 3,
) -> Dict[str, Any]:
    """Execute end-to-end supervised candidate comparison on the development partition.

    Loads the dataset, enforces leakage protection, separates the 80% development set,
    runs repeated stratified cross-validation, and computes comparative statistics.
    Never accesses or evaluates the 20% holdout test partition.

    Args:
        full_capacity: Campaign outreach capacity constraint (default: 5,000).
        random_state: Master random state for split and model reproducibility.
        n_splits: Folds per repeat (default: 5).
        n_repeats: Number of repeats (default: 3).

    Returns:
        Dict[str, Any]: Comparison results including fold data, paired statistics, and summaries.
    """
    config = load_config()
    pos_label = config.get("dataset", {}).get("positive_class", "yes")

    logger.info("Loading Bank Marketing dataset...")
    X, y = load_bank_marketing_data()
    n_total = len(X)

    # 1. Enforce leakage protection
    X_clean = drop_duration(X)

    # 2. Outer 80/20 train/test split (test partition is strictly isolated and not accessed)
    X_dev, X_test_frozen, y_dev, y_test_frozen = split_data(
        X_clean,
        y,
        test_size=0.2,
        random_state=random_state,
    )
    # Explicitly discard test partition references to guarantee zero data snooping
    del X_test_frozen
    del y_test_frozen

    logger.info(
        f"Development partition isolated: {len(X_dev):,} records (80.0% of total). "
        "The held-out test set was not revisited during this comparison phase; its previously recorded single evaluation remains frozen."
    )

    # 3. Repeated stratified cross validation on 80% dev partition
    candidate_models = get_comparison_candidate_models(random_state=random_state)
    fold_results = run_repeated_validation(
        X_dev=X_dev,
        y_dev=y_dev,
        candidate_models=candidate_models,
        n_splits=n_splits,
        n_repeats=n_repeats,
        random_state=random_state,
        full_capacity=full_capacity,
        total_population=n_total,
        pos_label=pos_label,
    )

    # 4. Paired delta statistics
    paired_balanced = compute_paired_delta_statistics(
        fold_results,
        model_a="RandomForest (balanced)",
        model_b="LogisticRegression (balanced)",
    )
    paired_unweighted = compute_paired_delta_statistics(
        fold_results,
        model_a="RandomForest (unweighted)",
        model_b="LogisticRegression (unweighted)",
    )
    paired_rf_weighting = compute_paired_delta_statistics(
        fold_results,
        model_a="RandomForest (unweighted)",
        model_b="RandomForest (balanced)",
    )

    # 5. Aggregate metrics
    aggregate_df = compute_aggregate_metrics(fold_results, benchmark_model="LogisticRegression (balanced)")

    # 6. Print formatted comparative reports
    print("\n" + "=" * 105)
    print("   TABLE 1: SUPERVISED MODEL CANDIDATES — REPEATED STRATIFIED VALIDATION (15 FOLDS)   ")
    print("=" * 105)
    header = (
        f"{'Model / Configuration':<32} | "
        f"{'Mean Conv@k':<12} | "
        f"{'Std Conv@k':<10} | "
        f"{'Mean Prec@k':<11} | "
        f"{'Mean Lift@k':<11} | "
        f"{'Mean PR-AUC':<11} | "
        f"{'Mean ROC-AUC':<12} | "
        f"{'Fold Win Rate':<13}"
    )
    print(header)
    print("-" * 105)
    for _, row in aggregate_df.iterrows():
        print(
            f"{row['model']:<32} | "
            f"{row['mean_conversions_at_k']:>12.2f} | "
            f"{row['std_conversions_at_k']:>10.2f} | "
            f"{row['mean_precision_at_k']*100:>10.2f}% | "
            f"{row['mean_lift_at_k']:>10.2f}x | "
            f"{row['mean_pr_auc']:>11.4f} | "
            f"{row['mean_roc_auc']:>12.4f} | "
            f"{row['win_rate_vs_benchmark']:>13}"
        )
    print("=" * 105)

    print("\n" + "=" * 80)
    print("   TABLE 2: PAIRED DELTA-CONVERSIONS (Random Forest vs Logistic Regression)   ")
    print("=" * 80)
    for label, paired in [("Configured / Balanced Weighting", paired_balanced), ("Unweighted Variants", paired_unweighted)]:
        print(f"\n--- {label} ---")
        print(f"Comparison: {paired['model_a']} minus {paired['model_b']}")
        print(f"Mean Delta Conversions@k   : {paired['delta_mean']:+.2f} conversions")
        print(f"Median Delta Conversions@k : {paired['delta_median']:+.2f} conversions")
        print(f"Std Dev of Delta           : {paired['delta_std']:.2f}")
        print(f"Min / Max Delta            : {paired['delta_min']:+.2f} / {paired['delta_max']:+.2f}")
        print(f"Random Forest Wins         : {paired['model_a_wins']}/{paired['n_folds']} ({paired['model_a_win_rate']*100:.1f}%)")
        print(f"Logistic Regression Wins   : {paired['model_b_wins']}/{paired['n_folds']} ({paired['model_b_win_rate']*100:.1f}%)")
        print(f"Ties                       : {paired['ties']}/{paired['n_folds']} ({paired['tie_rate']*100:.1f}%)")
    print("=" * 80 + "\n")

    print("\n" + "=" * 80)
    print("   TABLE 3: PAIRED RANDOM FOREST WEIGHTING COMPARISON (Unweighted vs Balanced)   ")
    print("=" * 80)
    print(f"Comparison: {paired_rf_weighting['model_a']} minus {paired_rf_weighting['model_b']}")
    print(f"Mean Delta Conversions@k   : {paired_rf_weighting['delta_mean']:+.2f} conversions")
    print(f"Median Delta Conversions@k : {paired_rf_weighting['delta_median']:+.2f} conversions")
    print(f"Std Dev of Delta           : {paired_rf_weighting['delta_std']:.2f}")
    print(f"Min / Max Delta            : {paired_rf_weighting['delta_min']:+.2f} / {paired_rf_weighting['delta_max']:+.2f}")
    print(f"Unweighted Wins            : {paired_rf_weighting['model_a_wins']}/{paired_rf_weighting['n_folds']} ({paired_rf_weighting['model_a_win_rate']*100:.1f}%)")
    print(f"Balanced Wins              : {paired_rf_weighting['model_b_wins']}/{paired_rf_weighting['n_folds']} ({paired_rf_weighting['model_b_win_rate']*100:.1f}%)")
    print(f"Ties                       : {paired_rf_weighting['ties']}/{paired_rf_weighting['n_folds']} ({paired_rf_weighting['tie_rate']*100:.1f}%)")
    print("=" * 80 + "\n")

    return {
        "fold_results": fold_results,
        "aggregate_df": aggregate_df,
        "paired_balanced": paired_balanced,
        "paired_unweighted": paired_unweighted,
        "paired_rf_weighting": paired_rf_weighting,
    }


if __name__ == "__main__":
    logger.info("Starting supervised candidate model comparison...")
    compare_supervised_candidates()
