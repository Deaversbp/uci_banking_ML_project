"""Unit tests for supervised model comparison, repeated validation,
paired delta-conversion calculations, and leakage contracts.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

import src.models as models_pkg
from src.models.train import compute_capacity_k
from src.models.compare import (
    get_comparison_candidate_models,
    run_repeated_validation,
    compute_paired_delta_statistics,
    compute_aggregate_metrics,
)
from src.features.build_features import create_pre_call_pipeline


def test_compare_package_exports():
    """Verify that all comparison functions are exported properly from src.models."""
    assert hasattr(models_pkg, "get_comparison_candidate_models")
    assert hasattr(models_pkg, "run_repeated_validation")
    assert hasattr(models_pkg, "compute_paired_delta_statistics")
    assert hasattr(models_pkg, "compute_aggregate_metrics")
    assert hasattr(models_pkg, "compare_supervised_candidates")


def test_fold_capacity_k_calculation():
    """Verify that fold capacity k is derived strictly from the fixed-capacity fraction."""
    total_pop = 45211
    global_cap = 5000

    # For 5-fold CV on 36,168 records: folds are 7,233 or 7,234 records
    k_7233 = compute_capacity_k(7233, total_pop, global_cap)
    k_7234 = compute_capacity_k(7234, total_pop, global_cap)

    assert k_7233 == 800
    assert k_7234 == 800

    # Edge cases
    assert compute_capacity_k(0, total_pop, global_cap) == 0
    assert compute_capacity_k(total_pop, total_pop, global_cap) == 5000
    assert compute_capacity_k(9042, total_pop, global_cap) == 1000


def test_duration_never_reaches_candidate_models():
    """Verify that 'duration' (post-call leakage) never reaches candidate pipelines or classifiers."""
    df_synthetic = pd.DataFrame({
        "age": [30, 45, 28, 52, 40, 60, 35, 48],
        "job": ["admin.", "technician", "services", "retired", "admin.", "management", "blue-collar", "technician"],
        "marital": ["single", "married", "single", "married", "married", "divorced", "single", "married"],
        "education": ["secondary", "tertiary", "secondary", "primary", "tertiary", "secondary", "secondary", "tertiary"],
        "default": ["no"] * 8,
        "balance": [1000, 2500, -50, 15000, 400, 8000, 120, 3100],
        "housing": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        "loan": ["no", "no", "yes", "no", "no", "no", "no", "no"],
        "contact": ["cellular"] * 8,
        "day_of_week": [1, 15, 20, 5, 10, 25, 12, 18],
        "month": ["may", "jul", "aug", "jun", "nov", "aug", "may", "jul"],
        "duration": [500, 1200, 50, 800, 300, 450, 60, 700],  # Leakage column!
        "campaign": [1, 2, 1, 3, 2, 1, 4, 2],
        "pdays": [-1, 120, -1, 30, -1, -1, 90, -1],
        "previous": [0, 2, 0, 1, 0, 0, 3, 0],
        "poutcome": [None, "success", None, "failure", None, None, "success", None],
    })
    y_synthetic = pd.Series(["no", "yes", "no", "yes", "no", "no", "yes", "no"])

    # 1. Pipeline fitting test
    pipe = create_pre_call_pipeline(LogisticRegression(), raw_feature_df=df_synthetic)
    pipe.fit(df_synthetic, y_synthetic)

    # Assert preprocessor features do NOT include duration
    preprocessor = pipe.named_steps["preprocessor"]
    num_cols = preprocessor.transformers[0][2]
    cat_cols = preprocessor.transformers[1][2]
    assert "duration" not in num_cols
    assert "duration" not in cat_cols

    # 2. Repeated validation test with duration in input
    models = {
        "TestLR": LogisticRegression(max_iter=200, random_state=42),
    }
    results = run_repeated_validation(
        X_dev=df_synthetic,
        y_dev=y_synthetic,
        candidate_models=models,
        n_splits=2,
        n_repeats=1,
        random_state=42,
        full_capacity=2,
        total_population=len(df_synthetic),
    )
    assert len(results) == 2
    assert "conversions_at_k" in results.columns


def test_compute_paired_delta_statistics():
    """Verify paired delta conversions calculation, aggregate stats, and win/tie rates."""
    # Construct synthetic fold results
    records = [
        {"fold_id": 0, "model": "ModelA", "conversions_at_k": 380},
        {"fold_id": 0, "model": "ModelB", "conversions_at_k": 360},  # delta = +20 (A wins)
        {"fold_id": 1, "model": "ModelA", "conversions_at_k": 370},
        {"fold_id": 1, "model": "ModelB", "conversions_at_k": 370},  # delta = 0 (Tie)
        {"fold_id": 2, "model": "ModelA", "conversions_at_k": 365},
        {"fold_id": 2, "model": "ModelB", "conversions_at_k": 375},  # delta = -10 (B wins)
        {"fold_id": 3, "model": "ModelA", "conversions_at_k": 390},
        {"fold_id": 3, "model": "ModelB", "conversions_at_k": 380},  # delta = +10 (A wins)
    ]
    df = pd.DataFrame(records)

    stats = compute_paired_delta_statistics(df, model_a="ModelA", model_b="ModelB")

    assert stats["n_folds"] == 4
    assert stats["model_a_mean"] == pytest.approx((380 + 370 + 365 + 390) / 4)
    assert stats["model_b_mean"] == pytest.approx((360 + 370 + 375 + 380) / 4)
    assert stats["delta_mean"] == pytest.approx((20 + 0 - 10 + 10) / 4)  # 5.0
    assert stats["delta_median"] == pytest.approx(5.0)  # median of [20, 10, 0, -10] = 5.0
    assert stats["delta_min"] == -10
    assert stats["delta_max"] == 20
    assert stats["model_a_wins"] == 2
    assert stats["model_a_win_rate"] == pytest.approx(0.50)
    assert stats["model_b_wins"] == 1
    assert stats["model_b_win_rate"] == pytest.approx(0.25)
    assert stats["ties"] == 1
    assert stats["tie_rate"] == pytest.approx(0.25)

    # Test error on invalid model name
    with pytest.raises(ValueError):
        compute_paired_delta_statistics(df, model_a="NonExistentModel", model_b="ModelB")


def test_compute_aggregate_metrics():
    """Verify aggregate summary metrics formatting and calculations."""
    records = [
        {"fold_id": 0, "model": "ModelA", "conversions_at_k": 380, "precision_at_k": 0.475, "recall_at_k": 0.38, "lift_at_k": 4.1, "pr_auc": 0.45, "roc_auc": 0.79},
        {"fold_id": 1, "model": "ModelA", "conversions_at_k": 370, "precision_at_k": 0.462, "recall_at_k": 0.37, "lift_at_k": 4.0, "pr_auc": 0.43, "roc_auc": 0.78},
        {"fold_id": 0, "model": "Benchmark", "conversions_at_k": 360, "precision_at_k": 0.450, "recall_at_k": 0.36, "lift_at_k": 3.9, "pr_auc": 0.41, "roc_auc": 0.77},
        {"fold_id": 1, "model": "Benchmark", "conversions_at_k": 350, "precision_at_k": 0.438, "recall_at_k": 0.35, "lift_at_k": 3.8, "pr_auc": 0.40, "roc_auc": 0.76},
    ]
    df = pd.DataFrame(records)
    agg = compute_aggregate_metrics(df, benchmark_model="Benchmark")

    assert len(agg) == 2
    row_a = agg[agg["model"] == "ModelA"].iloc[0]
    assert row_a["mean_conversions_at_k"] == pytest.approx(375.0)
    assert row_a["std_conversions_at_k"] == pytest.approx(np.std([380, 370], ddof=1))
    assert row_a["win_rate_vs_benchmark"] == "100.0%"

    row_bench = agg[agg["model"] == "Benchmark"].iloc[0]
    assert row_bench["mean_conversions_at_k"] == pytest.approx(355.0)
    assert row_bench["win_rate_vs_benchmark"] == "-"


def test_deterministic_results_given_fixed_seed():
    """Verify that repeated validation produces strictly deterministic results when given fixed random_state."""
    df_synthetic = pd.DataFrame({
        "age": [25, 40, 35, 50, 60, 22, 45, 33, 55, 29] * 3,
        "job": ["admin.", "technician", "services", "retired", "admin.", "management", "blue-collar", "technician", "services", "retired"] * 3,
        "marital": ["single", "married", "single", "married", "married", "divorced", "single", "married", "married", "single"] * 3,
        "education": ["secondary", "tertiary", "secondary", "primary", "tertiary", "secondary", "secondary", "tertiary", "primary", "secondary"] * 3,
        "default": ["no"] * 30,
        "balance": [100, 2000, -30, 8000, 500, 12000, 50, 4500, 900, 150] * 3,
        "housing": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"] * 3,
        "loan": ["no"] * 30,
        "contact": ["cellular"] * 30,
        "day_of_week": [1, 5, 10, 15, 20, 25, 3, 8, 14, 22] * 3,
        "month": ["may", "jul", "aug", "jun", "nov", "aug", "may", "jul", "jun", "nov"] * 3,
        "campaign": [1, 2, 1, 3, 2, 1, 4, 2, 1, 2] * 3,
        "pdays": [-1, 100, -1, 30, -1, -1, 80, -1, 20, -1] * 3,
        "previous": [0, 1, 0, 1, 0, 0, 2, 0, 1, 0] * 3,
        "poutcome": [None, "success", None, "failure", None, None, "success", None, "failure", None] * 3,
    })
    y_synthetic = pd.Series(["no", "yes", "no", "yes", "no", "no", "yes", "no", "yes", "no"] * 3)

    candidates = {
        "LR": LogisticRegression(max_iter=200, random_state=42),
    }

    res1 = run_repeated_validation(
        X_dev=df_synthetic,
        y_dev=y_synthetic,
        candidate_models=candidates,
        n_splits=2,
        n_repeats=2,
        random_state=42,
        full_capacity=3,
        total_population=len(df_synthetic),
    )

    res2 = run_repeated_validation(
        X_dev=df_synthetic,
        y_dev=y_synthetic,
        candidate_models=candidates,
        n_splits=2,
        n_repeats=2,
        random_state=42,
        full_capacity=3,
        total_population=len(df_synthetic),
    )

    pd.testing.assert_frame_equal(res1, res2)
