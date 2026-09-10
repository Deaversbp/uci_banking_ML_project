"""Train baseline models on the Bank Marketing dataset.

Enforces pre-call leakage prevention by excluding 'duration', incorporates
domain feature engineering via PreCallFeatureEngineer, evaluates using lead-ranking
metrics (PR-AUC, ROC-AUC, Lift@k, Conversions@k), and benchmarks models against
both the Random Selection Baseline and the Business-Rule Heuristic Baseline.
"""
from pathlib import Path
from typing import Dict, Any
import joblib
import pandas as pd
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


def train_baseline_models(
    save_best: bool = True,
    k_capacity: int = 5000
) -> Dict[str, Any]:
    """End-to-end baseline training, benchmarking, and ranking evaluation pipeline.

    Args:
        save_best: If True, serializes the best performing pipeline to disk.
        k_capacity: Overall call capacity constraint (default: 5,000 calls).

    Returns:
        Dict[str, Any]: Benchmark results and candidate model evaluations.
    """
    config = load_config()
    random_state = config.get("model", {}).get("random_state", 42)
    pos_label = config.get("dataset", {}).get("positive_class", "yes")

    logger.info("Loading Bank Marketing dataset...")
    X, y = load_bank_marketing_data()
    logger.info(f"Dataset loaded: {X.shape[0]} rows, {X.shape[1]} columns")

    # Explicitly verify duration removal for pre-call validity
    X_clean = drop_duration(X)

    X_train, X_test, y_train, y_test = split_data(X_clean, y, random_state=random_state)
    logger.info(f"Split completed: Train={len(X_train):,}, Test={len(X_test):,}")

    # Scale k proportionally for the test partition (e.g. 20% of 5,000 = 1,000 calls)
    k_test = int(round(k_capacity * (len(X_test) / len(X))))

    # 1. Non-ML Benchmarks
    random_bench = evaluate_random_baseline(y_test, k=k_test, pos_label=pos_label)
    business_bench = evaluate_business_rule_baseline(X_test, y_test, k=k_test, pos_label=pos_label)

    candidates = get_candidate_models(random_state=random_state)
    model_results = {}
    best_model_name = None
    best_pr_auc = -1.0
    best_pipeline = None

    # 2. Train Candidate Models
    for name, clf in candidates.items():
        logger.info(f"Training pre-call pipeline for {name}...")
        pipeline = create_pre_call_pipeline(classifier=clf, raw_feature_df=X_train)

        pipeline.fit(X_train, y_train)
        metrics = evaluate_model(pipeline, X_test, y_test, pos_label=pos_label, k=k_test)
        model_results[name] = {"metrics": metrics, "pipeline": pipeline}

        # Model selection based on PR-AUC (lead ranking efficacy) rather than thresholded F1
        pr_auc = metrics.get("pr_auc") or 0.0
        if pr_auc > best_pr_auc:
            best_pr_auc = pr_auc
            best_model_name = name
            best_pipeline = pipeline

    # 3. Print Comparison Table
    print("\n" + "=" * 80)
    print(f"   PRE-CALL LEAD RANKING BENCHMARKS & MODEL COMPARISON (Test Set k={k_test:,})   ")
    print("=" * 80)
    header = f"{'Strategy / Model':<28} | {'Conversions@k':<14} | {'Precision@k':<12} | {'Lift@k':<8} | {'PR-AUC':<8} | {'ROC-AUC':<8}"
    print(header)
    print("-" * 80)

    # Random baseline row
    print(
        f"{random_bench['baseline_name']:<28} | "
        f"{random_bench['conversions_at_k']:>14.1f} | "
        f"{random_bench['precision_at_k']*100:>11.2f}% | "
        f"{random_bench['lift_at_k']:>7.2f}x | "
        f"{random_bench['pr_auc']:>8.4f} | "
        f"{random_bench['roc_auc']:>8.4f}"
    )

    # Business-rule baseline row
    print(
        f"{business_bench['baseline_name']:<28} | "
        f"{business_bench['conversions_at_k']:>14,} | "
        f"{business_bench['precision_at_k']*100:>11.2f}% | "
        f"{business_bench['lift_at_k']:>7.2f}x | "
        f"{(business_bench['pr_auc'] or 0.0):>8.4f} | "
        f"{(business_bench['roc_auc'] or 0.0):>8.4f}"
    )

    # ML candidate rows
    for name, res in model_results.items():
        m = res["metrics"]
        print(
            f"{name:<28} | "
            f"{m.get('conversions_at_k', 0):>14,} | "
            f"{m.get('precision_at_k', 0.0)*100:>11.2f}% | "
            f"{m.get('lift_at_k', 0.0):>7.2f}x | "
            f"{(m.get('pr_auc') or 0.0):>8.4f} | "
            f"{(m.get('roc_auc') or 0.0):>8.4f}"
        )
    print("=" * 80 + "\n")

    logger.info(f"Best candidate selected by PR-AUC: {best_model_name} (PR-AUC: {best_pr_auc:.4f})")

    # 4. Serialize best pipeline
    if save_best and best_pipeline is not None:
        root = get_project_root()
        models_dir = root / config.get("paths", {}).get("models_dir", "models")
        models_dir.mkdir(parents=True, exist_ok=True)
        save_path = models_dir / f"best_{best_model_name.lower()}_pipeline.joblib"
        joblib.dump(best_pipeline, save_path)
        logger.info(f"Saved best model pipeline to {save_path}")

    return {
        "benchmarks": {
            "random": random_bench,
            "business_rule": business_bench,
        },
        "models": model_results,
        "best_model_name": best_model_name,
        "best_pipeline": best_pipeline,
    }


if __name__ == "__main__":
    logger.info("Starting baseline training and ranking benchmark run...")
    train_baseline_models()
