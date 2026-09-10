"""Train baseline models on the Bank Marketing dataset.

Methodological Hardening & Validation Design:
- Leakage Protection: Excludes 'duration' from all candidate features and pipelines.
- Clean Validation Design:
  1. Partitions data into full training (80%) and an untouched holdout test set (20%).
  2. Sub-partitions full training data into train (60% of total) and validation (20% of total).
  3. Compares candidates strictly on validation data at capacity k_val.
  4. Selects winning model using Conversions@capacity / Precision@capacity as the primary decision metric.
  5. Refits winning pipeline on full training data (80%).
  6. Evaluates selected model on the untouched test set (20%) exactly once.
- Fixed-Capacity Decision Alignment:
  - capacity_fraction = 5,000 / full_eligible_population
  - k_val = round(len(y_val) * capacity_fraction)
  - k_test = round(len(y_test) * capacity_fraction)
  - Primary selection metric: Conversions@capacity (Precision@capacity)
  - Secondary ranking diagnostics: PR-AUC, ROC-AUC, Lift@k
- Two-Tier Baseline Benchmarks on Test Set:
  - Random Selection Benchmark
  - Business-Rule Heuristic Benchmark
"""
from pathlib import Path
from typing import Dict, Any, Tuple
import joblib
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from src.data.load_data import load_bank_marketing_data
from src.features.build_features import (
    create_pre_call_pipeline,
    split_data,
    drop_duration,
)
from src.models.evaluate import evaluate_model
from src.models.baseline import (
    evaluate_random_baseline,
    evaluate_business_rule_baseline,
)
from src.utils.logger import setup_logger, load_config, get_project_root

logger = setup_logger(__name__)


def get_candidate_models(random_state: int = 42) -> Dict[str, Any]:
    """Define candidate baseline classifiers."""
    return {
        "LogisticRegression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=random_state
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=100,
            max_depth=12,
            class_weight="balanced",
            random_state=random_state,
            n_jobs=-1
        ),
    }


def compute_capacity_k(n_samples: int, total_population: int, full_capacity: int = 5000) -> int:
    """Derive partition capacity k from the fixed-capacity fraction.

    Args:
        n_samples: Number of observations in partition.
        total_population: Total eligible population size.
        full_capacity: Global call capacity constraint (5,000).

    Returns:
        int: Proportional outreach capacity k.
    """
    capacity_fraction = full_capacity / total_population
    return int(round(n_samples * capacity_fraction))


def train_baseline_models(
    save_best: bool = True,
    full_capacity: int = 5000
) -> Dict[str, Any]:
    """End-to-end model training, validation selection, and test evaluation pipeline.

    Args:
        save_best: If True, serializes the selected pipeline to disk.
        full_capacity: Global outreach capacity constraint (default: 5,000 calls).

    Returns:
        Dict[str, Any]: Validation results, test results, and selected pipeline.
    """
    config = load_config()
    random_state = config.get("model", {}).get("random_state", 42)
    pos_label = config.get("dataset", {}).get("positive_class", "yes")

    logger.info("Loading Bank Marketing dataset...")
    X, y = load_bank_marketing_data()
    n_total = len(X)
    logger.info(f"Dataset loaded: {n_total:,} rows, {X.shape[1]} columns")

    # 1. Enforce leakage protection for the supervised pre-call pipeline
    X_clean = drop_duration(X)

    # 2. Split into full training (80%) and untouched holdout test set (20%)
    X_train_full, X_test, y_train_full, y_test = split_data(X_clean, y, test_size=0.2, random_state=random_state)
    k_test = compute_capacity_k(len(X_test), n_total, full_capacity)

    # 3. Sub-partition training data into train (60% of total) and validation (20% of total)
    # Using test_size=0.25 on the 80% train_full partition yields 20% of total dataset for validation
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_full,
        y_train_full,
        test_size=0.25,
        random_state=random_state,
        stratify=y_train_full
    )
    k_val = compute_capacity_k(len(X_val), n_total, full_capacity)

    logger.info(
        f"Validation Partition Design: Train={len(X_train):,}, Validation={len(X_val):,} (k_val={k_val:,}), "
        f"Untouched Test={len(X_test):,} (k_test={k_test:,})"
    )

    # 4. Compare Candidate Models on the Validation Partition
    candidates = get_candidate_models(random_state=random_state)
    val_results = {}
    best_model_name = None
    best_val_conversions = -1

    for name, clf in candidates.items():
        logger.info(f"Fitting candidate pipeline for {name} on training partition...")
        pipeline = create_pre_call_pipeline(classifier=clf, raw_feature_df=X_train)
        pipeline.fit(X_train, y_train)

        val_metrics = evaluate_model(pipeline, X_val, y_val, pos_label=pos_label, k=k_val)
        val_results[name] = val_metrics

        # Primary selection metric: Conversions@capacity on the validation set
        conv_k = val_metrics.get("conversions_at_k", 0)
        if conv_k > best_val_conversions:
            best_val_conversions = conv_k
            best_model_name = name

    # Print Table 1: Validation Set Candidate Comparison
    print("\n" + "=" * 85)
    print(f"   TABLE 1: VALIDATION CANDIDATE COMPARISON & SELECTION (Validation Set k={k_val:,})   ")
    print("=" * 85)
    header = f"{'Candidate Model':<25} | {'Conversions@k (Primary)':<24} | {'Precision@k':<12} | {'Lift@k':<8} | {'PR-AUC':<8} | {'ROC-AUC':<8}"
    print(header)
    print("-" * 85)
    for name, m in val_results.items():
        selected_mark = " <-- SELECTED" if name == best_model_name else ""
        print(
            f"{name:<25} | "
            f"{m.get('conversions_at_k', 0):>24,} | "
            f"{m.get('precision_at_k', 0.0)*100:>11.2f}% | "
            f"{m.get('lift_at_k', 0.0):>7.2f}x | "
            f"{(m.get('pr_auc') or 0.0):>8.4f} | "
            f"{(m.get('roc_auc') or 0.0):>8.4f}"
            f"{selected_mark}"
        )
    print("=" * 85)
    logger.info(f"Model selected from validation results: {best_model_name} (Conversions@k: {best_val_conversions:,})")

    # 5. Refit Selected Model on Full Training Data (80%)
    logger.info(f"Refitting selected model ({best_model_name}) on full training partition ({len(X_train_full):,} records)...")
    selected_clf = get_candidate_models(random_state=random_state)[best_model_name]
    best_pipeline = create_pre_call_pipeline(classifier=selected_clf, raw_feature_df=X_train_full)
    best_pipeline.fit(X_train_full, y_train_full)

    # 6. Evaluate Selected Model on the Untouched Test Set EXACTLY ONCE
    logger.info(f"Evaluating selected model on untouched holdout test set ({len(X_test):,} records, k_test={k_test:,})...")
    test_metrics = evaluate_model(best_pipeline, X_test, y_test, pos_label=pos_label, k=k_test)

    # 7. Evaluate Non-ML Benchmarks on the Test Set
    random_bench = evaluate_random_baseline(y_test, k=k_test, pos_label=pos_label)
    business_bench = evaluate_business_rule_baseline(X_test, y_test, k=k_test, pos_label=pos_label)

    # Print Table 2: Final Test Set Evaluation
    print("\n" + "=" * 85)
    print(f"   TABLE 2: FINAL UNTOUCHED TEST SET EVALUATION (Holdout Test Set k={k_test:,})   ")
    print("=" * 85)
    header = f"{'Strategy / Selected Model':<32} | {'Conversions@k':<14} | {'Precision@k':<12} | {'Lift@k':<8} | {'PR-AUC':<8} | {'ROC-AUC':<8}"
    print(header)
    print("-" * 85)

    # Random baseline
    print(
        f"{random_bench['baseline_name']:<32} | "
        f"{random_bench['conversions_at_k']:>14.1f} | "
        f"{random_bench['precision_at_k']*100:>11.2f}% | "
        f"{random_bench['lift_at_k']:>7.2f}x | "
        f"{random_bench['pr_auc']:>8.4f} | "
        f"{random_bench['roc_auc']:>8.4f}"
    )

    # Business-rule baseline
    print(
        f"{business_bench['baseline_name']:<32} | "
        f"{business_bench['conversions_at_k']:>14,} | "
        f"{business_bench['precision_at_k']*100:>11.2f}% | "
        f"{business_bench['lift_at_k']:>7.2f}x | "
        f"{(business_bench['pr_auc'] or 0.0):>8.4f} | "
        f"{(business_bench['roc_auc'] or 0.0):>8.4f}"
    )

    # Selected model evaluated once
    print(
        f"{best_model_name + ' (Selected ML)':<32} | "
        f"{test_metrics.get('conversions_at_k', 0):>14,} | "
        f"{test_metrics.get('precision_at_k', 0.0)*100:>11.2f}% | "
        f"{test_metrics.get('lift_at_k', 0.0):>7.2f}x | "
        f"{(test_metrics.get('pr_auc') or 0.0):>8.4f} | "
        f"{(test_metrics.get('roc_auc') or 0.0):>8.4f}"
    )
    print("=" * 85 + "\n")

    # 8. Serialize Best Model Pipeline
    if save_best and best_pipeline is not None:
        root = get_project_root()
        models_dir = root / config.get("paths", {}).get("models_dir", "models")
        models_dir.mkdir(parents=True, exist_ok=True)
        save_path = models_dir / f"best_{best_model_name.lower()}_pipeline.joblib"
        joblib.dump(best_pipeline, save_path)
        logger.info(f"Saved best model pipeline to {save_path}")

    return {
        "capacity_fraction": full_capacity / n_total,
        "k_val": k_val,
        "k_test": k_test,
        "validation_results": val_results,
        "selected_model": best_model_name,
        "test_results": {
            "random_baseline": random_bench,
            "business_rule_baseline": business_bench,
            "selected_model": test_metrics,
        },
        "pipeline": best_pipeline,
    }


if __name__ == "__main__":
    logger.info("Starting hardened baseline training, validation selection, and single test run...")
    train_baseline_models()
