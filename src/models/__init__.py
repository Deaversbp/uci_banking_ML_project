"""Model training, baseline benchmarking, and evaluation package."""
from .evaluate import evaluate_model, compute_ranking_metrics_at_k
from .baseline import (
    evaluate_random_baseline,
    business_rule_baseline_score,
    evaluate_business_rule_baseline,
)

__all__ = [
    "train_baseline_models",
    "get_candidate_models",
    "evaluate_model",
    "compute_ranking_metrics_at_k",
    "evaluate_random_baseline",
    "business_rule_baseline_score",
    "evaluate_business_rule_baseline",
]

def __getattr__(name):
    if name in ("train_baseline_models", "get_candidate_models"):
        from . import train
        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
