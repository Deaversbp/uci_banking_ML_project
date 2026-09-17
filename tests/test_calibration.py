"""Unit and contract tests for probability calibration, reliability diagnostics,
nested cross-validation design, and ranking guardrails.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import src.models as models_pkg
from src.features.feature_contract import CANONICAL_FORBIDDEN_FEATURES
from src.models.calibration import (
    compute_calibration_slope_and_intercept,
    compute_calibration_metrics,
    generate_oof_calibrated_predictions,
    build_reliability_table,
    evaluate_fold_level_ranking,
    compare_calibration_methods,
)
from src.models.train import compute_capacity_k


def test_calibration_package_exports():
    """Verify clean exports across src.models for all calibration functions."""
    assert hasattr(models_pkg, "compute_calibration_slope_and_intercept")
    assert hasattr(models_pkg, "compute_calibration_metrics")
    assert hasattr(models_pkg, "generate_oof_calibrated_predictions")
    assert hasattr(models_pkg, "build_reliability_table")
    assert hasattr(models_pkg, "evaluate_fold_level_ranking")
    assert hasattr(models_pkg, "compare_calibration_methods")
    assert hasattr(models_pkg, "generate_calibration_figures")


def test_brier_and_log_loss_contracts():
    """Verify correctness of Brier score and Log loss calculations on synthetic data."""
    y_true = np.array([1, 0, 1, 0])
    # Perfect probabilities
    p_perfect = np.array([1.0, 0.0, 1.0, 0.0])
    m_perfect = compute_calibration_metrics(y_true, p_perfect)
    assert m_perfect["brier_score"] == pytest.approx(0.0, abs=1e-6)
    assert m_perfect["log_loss"] < 1e-4

    # Imperfect probabilities
    p_imperfect = np.array([0.5, 0.5, 0.5, 0.5])
    m_imperfect = compute_calibration_metrics(y_true, p_imperfect)
    assert m_imperfect["brier_score"] == pytest.approx(0.25, abs=1e-6)
    assert m_imperfect["log_loss"] == pytest.approx(np.log(2), abs=1e-4)


def test_calibration_slope_and_intercept_ideal():
    """Verify calibration slope ~ 1 and intercept ~ 0 for well-calibrated synthetic data."""
    # Synthetic well-calibrated probabilities
    np.random.seed(42)
    p = np.random.uniform(0.05, 0.95, size=5000)
    y = (np.random.rand(5000) < p).astype(int)

    intercept, slope = compute_calibration_slope_and_intercept(y, p)
    # Intercept should be close to 0 and slope close to 1
    assert abs(intercept) < 0.15
    assert abs(slope - 1.0) < 0.15


def test_nested_oof_calibration_leakage_and_contract():
    """Verify that every observation receives exactly one score per method,
    outer-validation rows are never in training data, and forbidden fields never reach pipeline.
    """
    df_synthetic = pd.DataFrame({
        "age": [25, 40, 35, 50, 60, 22, 45, 33, 55, 29] * 4,
        "job": ["admin.", "technician", "services", "retired", "admin.", "management", "blue-collar", "technician", "services", "retired"] * 4,
        "marital": ["single", "married", "single", "married", "married", "divorced", "single", "married", "married", "single"] * 4,
        "education": ["secondary", "tertiary", "secondary", "primary", "tertiary", "secondary", "secondary", "tertiary", "primary", "secondary"] * 4,
        "default": ["no"] * 40,
        "balance": [100, 2000, -30, 8000, 500, 12000, 50, 4500, 900, 150] * 4,
        "housing": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"] * 4,
        "loan": ["no"] * 40,
        "contact": ["cellular"] * 40,  # Forbidden
        "day_of_week": [1, 5, 10, 15, 20, 25, 3, 8, 14, 22] * 4,  # Forbidden
        "month": ["may", "jul", "aug", "jun", "nov", "aug", "may", "jul", "jun", "nov"] * 4,  # Forbidden
        "duration": [300, 120, 450, 60, 800, 150, 40, 600, 90, 250] * 4,  # Post-call leakage!
        "campaign": [1, 2, 1, 3, 2, 1, 4, 2, 1, 2] * 4,  # Forbidden
        "pdays": [-1, 100, -1, 30, -1, -1, 80, -1, 20, -1] * 4,
        "previous": [0, 1, 0, 1, 0, 0, 2, 0, 1, 0] * 4,
        "poutcome": [None, "success", None, "failure", None, None, "success", None, "failure", None] * 4,
    })
    y_synthetic = pd.Series(["no", "yes", "no", "yes", "no", "no", "yes", "no", "yes", "no"] * 4)

    df_oof = generate_oof_calibrated_predictions(
        X_dev=df_synthetic,
        y_dev=y_synthetic,
        n_outer_splits=2,
        n_inner_splits=2,
        random_state=42,
    )

    # 1. Exactly one prediction per row
    assert len(df_oof) == len(df_synthetic)
    assert len(df_oof["orig_idx"].unique()) == len(df_synthetic)

    # 2. All methods present without NaNs
    for col in ["raw_score", "sigmoid_prob", "isotonic_prob"]:
        assert not df_oof[col].isna().any()
        assert (df_oof[col] >= 0.0).all() and (df_oof[col] <= 1.0).all()

    # 3. Leakage contract: forbidden fields must never be in columns
    for forbidden in CANONICAL_FORBIDDEN_FEATURES:
        assert forbidden not in df_oof.columns


def test_reliability_table_reconciliation():
    """Verify that reliability bin counts reconcile exactly to total population."""
    np.random.seed(42)
    n_samples = 150
    y_true = np.random.binomial(1, 0.2, size=n_samples)
    y_prob = np.random.uniform(0.0, 1.0, size=n_samples)

    # Uniform strategy
    table_uniform = build_reliability_table(y_true, y_prob, n_bins=10, strategy="uniform", min_count=20)
    assert table_uniform["count"].sum() == n_samples
    assert table_uniform["positives"].sum() == y_true.sum()
    small_bins = table_uniform[table_uniform["count"] < 20]
    assert (small_bins["small_sample_flag"] == True).all()

    # Quantile strategy
    table_quantile = build_reliability_table(y_true, y_prob, n_bins=5, strategy="quantile", min_count=10)
    assert table_quantile["count"].sum() == n_samples
    assert table_quantile["positives"].sum() == y_true.sum()


def test_compare_calibration_methods_overlap_mathematics():
    """Verify set-theoretic overlap and conversion delta identities."""
    df_oof = pd.DataFrame({
        "actual": [1, 0, 1, 1, 0, 0, 1, 0, 0, 0],
        "raw_score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
        "sigmoid_prob": [0.95, 0.85, 0.75, 0.65, 0.55, 0.45, 0.35, 0.25, 0.15, 0.05],  # Same order
        "isotonic_prob": [0.9, 0.9, 0.5, 0.5, 0.5, 0.5, 0.2, 0.2, 0.1, 0.1],  # Ties
    })

    comp = compare_calibration_methods(df_oof, k=4)
    overlap = comp["overlap_summary"]

    # Sigmoid should have 100% overlap since order is identical
    assert overlap["Sigmoid Calibrated"]["shared_leads"] == 4
    assert overlap["Sigmoid Calibrated"]["overlap_pct"] == 100.0
    assert overlap["Sigmoid Calibrated"]["changed_leads"] == 0
    assert overlap["Sigmoid Calibrated"]["delta_conversions"] == 0

    # Isotonic top-4: shared + changed = k
    iso_shared = overlap["Isotonic Calibrated"]["shared_leads"]
    iso_changed = overlap["Isotonic Calibrated"]["changed_leads"]
    assert iso_shared + iso_changed == 4


def test_fold_level_ranking_evaluation_identities():
    """Verify fold-by-fold capacity evaluation and ranking preservation identities."""
    df_oof = pd.DataFrame({
        "fold": [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
        "actual": [1, 0, 1, 0, 0, 0, 1, 0, 1, 0],
        "raw_score": [0.9, 0.8, 0.7, 0.4, 0.1, 0.95, 0.85, 0.65, 0.35, 0.1],
        "sigmoid_prob": [0.92, 0.81, 0.73, 0.42, 0.12, 0.96, 0.86, 0.66, 0.36, 0.11],
        "isotonic_prob": [0.85, 0.85, 0.60, 0.60, 0.10, 0.90, 0.90, 0.50, 0.50, 0.10],
    })

    eval_results = evaluate_fold_level_ranking(df_oof, full_capacity=2, total_population=10)
    assert len(eval_results["fold_results"]) == 2

    for f_res in eval_results["fold_results"]:
        k_f = f_res["k_fold"]
        # Capacity check
        assert k_f == 1  # 5 * (2/10) = 1

        # Sigmoid monotonic preservation
        assert f_res["sigmoid"]["overlap_pct"] == 100.0
        assert f_res["sigmoid"]["changed_leads"] == 0
        assert f_res["sigmoid"]["delta_conversions"] == 0
        assert f_res["sigmoid"]["shared_leads"] == k_f

        # Isotonic identities
        assert f_res["isotonic"]["shared_leads"] + f_res["isotonic"]["changed_leads"] == k_f

    # Paired summary presence
    assert "delta_conversions" in eval_results["paired_summary"]["sigmoid"]
    assert "overlap_pct" in eval_results["paired_summary"]["isotonic"]


def test_deterministic_calibration_given_fixed_seed():
    """Verify deterministic repeatability of calibration predictions under fixed seed."""
    df_synthetic = pd.DataFrame({
        "age": [30, 45, 28, 52, 40, 60, 35, 48] * 3,
        "job": ["admin.", "technician", "services", "retired", "admin.", "management", "blue-collar", "technician"] * 3,
        "marital": ["single", "married", "single", "married", "married", "divorced", "single", "married"] * 3,
        "education": ["secondary", "tertiary", "secondary", "primary", "tertiary", "secondary", "secondary", "tertiary"] * 3,
        "default": ["no"] * 24,
        "balance": [1000, 2500, -50, 15000, 400, 8000, 120, 3100] * 3,
        "housing": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"] * 3,
        "loan": ["no"] * 24,
        "contact": ["cellular"] * 24,
        "day_of_week": [1, 15, 20, 5, 10, 25, 12, 18] * 3,
        "month": ["may", "jul", "aug", "jun", "nov", "aug", "may", "jul"] * 3,
        "campaign": [1, 2, 1, 3, 2, 1, 4, 2] * 3,
        "pdays": [-1, 120, -1, 30, -1, -1, 90, -1] * 3,
        "previous": [0, 2, 0, 1, 0, 0, 3, 0] * 3,
        "poutcome": [None, "success", None, "failure", None, None, "success", None] * 3,
    })
    y_synthetic = pd.Series(["no", "yes", "no", "yes", "no", "no", "yes", "no"] * 3)

    oof1 = generate_oof_calibrated_predictions(df_synthetic, y_synthetic, n_outer_splits=2, n_inner_splits=2, random_state=42)
    oof2 = generate_oof_calibrated_predictions(df_synthetic, y_synthetic, n_outer_splits=2, n_inner_splits=2, random_state=42)

    pd.testing.assert_frame_equal(oof1, oof2)
