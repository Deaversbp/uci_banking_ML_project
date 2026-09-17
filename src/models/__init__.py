"""Model training, baseline benchmarking, evaluation, and comparison package."""
from .evaluate import evaluate_model, compute_ranking_metrics_at_k
from .baseline import (
    evaluate_random_baseline,
    business_rule_baseline_score,
    evaluate_business_rule_baseline,
)
from .compare import (
    get_comparison_candidate_models,
    run_repeated_validation,
    compute_paired_delta_statistics,
    compute_aggregate_metrics,
    compare_supervised_candidates,
)

__all__ = [
    "train_baseline_models",
    "get_candidate_models",
    "evaluate_model",
    "compute_ranking_metrics_at_k",
    "evaluate_random_baseline",
    "business_rule_baseline_score",
    "evaluate_business_rule_baseline",
    "get_comparison_candidate_models",
    "run_repeated_validation",
    "compute_paired_delta_statistics",
    "compute_aggregate_metrics",
    "compare_supervised_candidates",
]

def __getattr__(name):
    if name in ("train_baseline_models", "get_candidate_models"):
        from . import train
        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

