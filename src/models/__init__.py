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
from .diagnostics import (
    generate_oof_predictions,
    compute_capacity_diagnostics,
    analyze_subgroups,
    analyze_false_positives_and_missed_positives,
    compute_permutation_feature_importance,
    compare_rf_vs_business_rule,
    generate_diagnostic_figures,
    run_full_diagnostics,
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
    "generate_oof_predictions",
    "compute_capacity_diagnostics",
    "analyze_subgroups",
    "analyze_false_positives_and_missed_positives",
    "compute_permutation_feature_importance",
    "compare_rf_vs_business_rule",
    "generate_diagnostic_figures",
    "run_full_diagnostics",
]

def __getattr__(name):
    if name in ("train_baseline_models", "get_candidate_models"):
        from . import train
        return getattr(train, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

