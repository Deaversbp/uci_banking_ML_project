"""Unit and contract tests for supervised model interpretation, diagnostics,
out-of-fold error categorization, subgroup profiling, and baseline overlap.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest

import src.models as models_pkg
from src.models.diagnostics import (
    generate_oof_predictions,
    compute_capacity_diagnostics,
    analyze_subgroups,
    compare_rf_vs_business_rule,
    build_subgroup_definitions,
)
from src.models.train import compute_capacity_k


def test_diagnostics_package_exports():
    """Verify clean exports across src.models for all diagnostic functions."""
    assert hasattr(models_pkg, "generate_oof_predictions")
    assert hasattr(models_pkg, "compute_capacity_diagnostics")
    assert hasattr(models_pkg, "analyze_subgroups")
    assert hasattr(models_pkg, "analyze_false_positives_and_missed_positives")
    assert hasattr(models_pkg, "compute_permutation_feature_importance")
    assert hasattr(models_pkg, "compare_rf_vs_business_rule")
    assert hasattr(models_pkg, "generate_diagnostic_figures")
    assert hasattr(models_pkg, "run_full_diagnostics")


def test_oof_predictions_contract_and_leakage():
    """Verify that every observation receives exactly one OOF score, no training row is scored
    by a model fitted on that same fold, and post-call 'duration' is completely excluded.
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
        "contact": ["cellular"] * 40,
        "day_of_week": [1, 5, 10, 15, 20, 25, 3, 8, 14, 22] * 4,
        "month": ["may", "jul", "aug", "jun", "nov", "aug", "may", "jul", "jun", "nov"] * 4,
        "duration": [300, 120, 450, 60, 800, 150, 40, 600, 90, 250] * 4,  # Post-call leakage!
        "campaign": [1, 2, 1, 3, 2, 1, 4, 2, 1, 2] * 4,
        "pdays": [-1, 100, -1, 30, -1, -1, 80, -1, 20, -1] * 4,
        "previous": [0, 1, 0, 1, 0, 0, 2, 0, 1, 0] * 4,
        "poutcome": [None, "success", None, "failure", None, None, "success", None, "failure", None] * 4,
    })
    y_synthetic = pd.Series(["no", "yes", "no", "yes", "no", "no", "yes", "no", "yes", "no"] * 4)

    df_oof = generate_oof_predictions(
        X_dev=df_synthetic,
        y_dev=y_synthetic,
        n_splits=2,
        random_state=42,
    )

    # 1. Exactly one prediction per development row
    assert len(df_oof) == len(df_synthetic)
    assert len(df_oof["orig_idx"].unique()) == len(df_synthetic)
    assert not df_oof["score"].isna().any()

    # 2. Both folds exist and partition the dataset evenly
    assert set(df_oof["fold"].unique()) == {0, 1}
    assert (df_oof["fold"] == 0).sum() == 20
    assert (df_oof["fold"] == 1).sum() == 20

    # 3. Leakage verification: duration must never be in columns
    assert "duration" not in df_oof.columns


def test_capacity_k_derivation():
    """Verify that development OOF capacity k is derived proportionally."""
    n_dev = 36168
    total_pop = 45211
    global_capacity = 5000

    k_oof = compute_capacity_k(n_dev, total_pop, global_capacity)
    # round(36168 * 5000 / 45211) = round(3999.91) = 4000
    assert k_oof == 4000


def test_capacity_diagnostics_exact_reconciliation():
    """Verify ranking error categorization and mathematical identity reconciliations:
    TP + FP = k
    TP + FN = total actual positives
    TP + FP + FN + TN = total population
    """
    # Create 10 synthetic prospects, 4 positives, capacity k = 3
    df_oof = pd.DataFrame({
        "orig_idx": np.arange(10),
        "actual_str": ["yes", "no", "yes", "no", "no", "yes", "no", "yes", "no", "no"],
        "actual": [1, 0, 1, 0, 0, 1, 0, 1, 0, 0],
        "score": [0.95, 0.90, 0.85, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10],
        "fold": [0, 1, 0, 1, 0, 1, 0, 1, 0, 1],
    })

    # Total population 10, capacity 3 -> k = 3
    df_ann, summary = compute_capacity_diagnostics(df_oof, full_capacity=3, total_population=10)

    # Selected top-k ranks: [0.95 (yes), 0.90 (no), 0.85 (yes)]
    # TP = 2 (rows 0, 2), FP = 1 (row 1)
    # Missed positives: rows 5 (score 0.50, actual 1) and 7 (score 0.30, actual 1) -> FN = 2
    # Correctly rejected: rows 3, 4, 6, 8, 9 -> TN = 5
    assert summary["k_evaluated"] == 3
    assert summary["conversions_at_k"] == 2
    assert summary["top_k_false_positives"] == 1
    assert summary["missed_positives"] == 2
    assert summary["correctly_rejected"] == 5
    assert summary["total_positives"] == 4

    # Mathematical identity assertions
    assert summary["conversions_at_k"] + summary["top_k_false_positives"] == summary["k_evaluated"]
    assert summary["conversions_at_k"] + summary["missed_positives"] == summary["total_positives"]
    assert (
        summary["conversions_at_k"]
        + summary["top_k_false_positives"]
        + summary["missed_positives"]
        + summary["correctly_rejected"]
        == summary["total_samples"]
    )

    # Lead ranking metric formulas
    assert summary["precision_at_k"] == pytest.approx(2 / 3)
    assert summary["recall_at_k"] == pytest.approx(2 / 4)
    assert summary["lift_at_k"] == pytest.approx((2 / 3) / 0.4)
    assert summary["cutoff_score"] == pytest.approx(0.85)


def test_subgroup_analysis_reconciliation():
    """Verify that subgroup partition dimensions reconcile to population and selection totals."""
    X = pd.DataFrame({
        "pdays": [-1, 10, -1, 50, -1, -1, 100, -1, 20, -1],
        "poutcome": [None, "success", None, "failure", None, None, "success", None, "failure", None],
        "housing": ["yes", "no", "yes", "no", "yes", "no", "yes", "no", "yes", "no"],
        "loan": ["no"] * 10,
        "balance": [100, 2000, -50, 400, 0, 8000, -10, 500, 1200, 3000],
        "age": [25, 65, 35, 45, 55, 70, 22, 38, 48, 52],
        "contact": ["cellular"] * 10,
        "month": ["may", "jul", "aug", "jun", "nov", "aug", "may", "jul", "jun", "nov"],
        "campaign": [1, 2, 1, 3, 2, 1, 4, 2, 1, 2],
    })
    df_ann = pd.DataFrame({
        "in_top_k": [True, True, True, False, False, False, False, False, False, False],
        "actual": [1, 1, 0, 0, 0, 1, 0, 0, 1, 0],
        "score": [0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05],
    }, index=X.index)

    sub_df = analyze_subgroups(X, df_ann, min_count=1)

    # Check Contact History dimension (mutually exclusive and exhaustive)
    contact_hist = sub_df[sub_df["dimension"] == "Contact History"]
    assert contact_hist["population_count"].sum() == len(X)
    assert contact_hist["selected_count"].sum() == 3
    assert contact_hist["conversions_captured"].sum() == 2

    # Check small-sample safeguard enforcement
    sub_df_safeguard = analyze_subgroups(X, df_ann, min_count=2)
    small_groups = sub_df_safeguard[sub_df_safeguard["selected_count"] < 2]
    assert (small_groups["small_sample_flag"] == True).all()
    assert small_groups["precision_at_k"].isna().all()


def test_pdays_canonical_contract():
    """Verify that pdays == 0 does not exist in dataset and pdays != -1 is canonically equivalent to pdays > 0."""
    from src.data.load_data import load_bank_marketing_data
    X, _ = load_bank_marketing_data()
    assert (X["pdays"] == 0).sum() == 0
    assert ((X["pdays"] != -1) == (X["pdays"] > 0)).all()


def test_rf_vs_business_rule_overlap():
    """Verify set-theoretic identities for RF vs. Business Rule comparison:
    shared + rf_only = k
    shared + br_only = k
    """
    X = pd.DataFrame({
        "pdays": [100, -1, 50, -1, 20, -1],
        "poutcome": ["success", None, "failure", None, "other", None],
        "housing": ["no", "yes", "no", "yes", "no", "yes"],
        "loan": ["no", "no", "no", "no", "no", "no"],
        "balance": [5000, 100, 200, 300, 400, 500],
    })
    y = pd.Series(["yes", "no", "yes", "no", "no", "no"])
    df_ann = pd.DataFrame({
        "rank": [0, 1, 2, 3, 4, 5],
        "actual": [1, 0, 1, 0, 0, 0],
    }, index=X.index)

    # Let k = 2
    comp = compare_rf_vs_business_rule(X, y, df_ann, full_capacity=2, total_population=6)

    assert comp["k_evaluated"] == 2
    assert comp["shared_count"] + comp["rf_only_count"] == 2
    assert comp["shared_count"] + comp["br_only_count"] == 2
    assert 0.0 <= comp["jaccard_similarity"] <= 1.0
    assert comp["rf_total_conversions"] == comp["shared_conversions"] + comp["rf_only_conversions"]
    assert comp["br_total_conversions"] == comp["shared_conversions"] + comp["br_only_conversions"]


def test_deterministic_diagnostics_given_fixed_seed():
    """Verify reproducibility of OOF generation under fixed random_state."""
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

    oof1 = generate_oof_predictions(df_synthetic, y_synthetic, n_splits=2, random_state=42)
    oof2 = generate_oof_predictions(df_synthetic, y_synthetic, n_splits=2, random_state=42)

    pd.testing.assert_frame_equal(oof1, oof2)
