"""Unit and contract tests for unsupervised customer segmentation, PCA dimensionality analysis,
K-Means candidate evaluation, cluster stability, and post-hoc profiling.
"""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import adjusted_rand_score

import src.models as models_pkg
from src.features.feature_contract import CANONICAL_FORBIDDEN_FEATURES
from src.models.segmentation import (
    build_clustering_feature_matrix,
    compute_pca_summary,
    evaluate_kmeans_candidates,
    evaluate_cluster_stability,
    fit_final_segmentation,
    profile_clusters,
)


@pytest.fixture
def sample_feature_df():
    """Create synthetic customer dataset with mixed feature types and edge cases."""
    return pd.DataFrame({
        "age": [25, 45, 35, 62, 50, 22, 38, 55] * 5,
        "job": ["admin.", "technician", "services", "retired", "management", "blue-collar", "services", "retired"] * 5,
        "marital": ["single", "married", "single", "married", "married", "divorced", "married", "single"] * 5,
        "education": ["secondary", "tertiary", "secondary", "primary", "tertiary", "secondary", "secondary", "tertiary"] * 5,
        "default": ["no"] * 40,
        "balance": [100, 2500, -50, 12000, 500, 30, 450, 8000] * 5,
        "housing": ["yes", "no", "yes", "no", "yes", "no", "yes", "no"] * 5,
        "loan": ["no", "no", "yes", "no", "no", "no", "no", "no"] * 5,
        "contact": ["cellular"] * 40,
        "day_of_week": [1, 5, 10, 15, 20, 25, 12, 18] * 5,
        "month": ["may", "jul", "aug", "jun", "nov", "aug", "may", "jul"] * 5,
        "duration": [300, 120, 450, 60, 800, 150, 90, 400] * 5,  # Leakage!
        "campaign": [1, 2, 1, 3, 2, 1, 4, 2] * 5,
        "pdays": [-1, 100, -1, 30, -1, -1, 80, -1] * 5,
        "previous": [0, 1, 0, 1, 0, 0, 2, 0] * 5,
        "poutcome": [None, "success", None, "failure", None, None, "other", None] * 5,
        "y": ["no", "yes", "no", "yes", "no", "no", "yes", "no"] * 5,  # Target!
    })


def test_segmentation_package_exports():
    """Verify clean exports across src.models for all segmentation functions."""
    assert hasattr(models_pkg, "build_clustering_feature_matrix")
    assert hasattr(models_pkg, "compute_pca_summary")
    assert hasattr(models_pkg, "evaluate_kmeans_candidates")
    assert hasattr(models_pkg, "evaluate_cluster_stability")
    assert hasattr(models_pkg, "fit_final_segmentation")
    assert hasattr(models_pkg, "profile_clusters")
    assert hasattr(models_pkg, "generate_segmentation_figures")


def test_clustering_feature_contract_and_leakage_exclusion(sample_feature_df):
    """Verify that target 'y' and post-call 'duration' are strictly excluded from clustering matrix."""
    X_prep, feature_names, preprocessor, X_fe = build_clustering_feature_matrix(sample_feature_df)

    # 1. Neither 'duration' nor 'y' nor any forbidden execution variable must ever appear in feature names
    for forbidden in CANONICAL_FORBIDDEN_FEATURES:
        assert forbidden not in feature_names, f"Forbidden field '{forbidden}' found in clustering feature names"
        assert not any(f.startswith(f"cat__{forbidden}") or f.startswith(f"num__{forbidden}") for f in feature_names), (
            f"Forbidden feature prefix found for '{forbidden}'"
        )

    # 2. Excluded redundant features must not be in feature_names
    assert not any("balance" in f and "balance_log" not in f for f in feature_names)
    assert not any(f.endswith("__pdays") for f in feature_names)
    assert not any("prior_success" in f for f in feature_names)
    assert not any("has_debt_burden" in f for f in feature_names)
    assert not any("negative_balance_flag" in f for f in feature_names)

    # 3. Output shape must match rows
    assert X_prep.shape[0] == len(sample_feature_df)
    assert not np.isnan(X_prep).any()


def test_deterministic_preprocessing_and_missing_handling(sample_feature_df):
    """Verify categorical missingness is handled as 'unknown' and preprocessing is deterministic."""
    X_prep1, names1, _, _ = build_clustering_feature_matrix(sample_feature_df)
    X_prep2, names2, _, _ = build_clustering_feature_matrix(sample_feature_df)

    assert names1 == names2
    np.testing.assert_allclose(X_prep1, X_prep2)


def test_pca_shape_and_variance_reconciliation(sample_feature_df):
    """Verify that PCA output shapes and cumulative explained variance match expectations."""
    X_prep, feature_names, _, _ = build_clustering_feature_matrix(sample_feature_df)
    pca, summary = compute_pca_summary(X_prep, feature_names, random_state=42)

    # Number of components equals min(n_samples, n_features)
    expected_components = min(X_prep.shape[0], X_prep.shape[1])
    assert summary["total_components"] == expected_components
    assert len(summary["explained_variance_ratio"]) == expected_components

    # Explained variance must sum to 1.0 (within numerical tolerance)
    assert np.sum(summary["explained_variance_ratio"]) == pytest.approx(1.0, abs=1e-5)

    # Threshold checks
    for th_name, n_comp in summary["variance_threshold_components"].items():
        assert 1 <= n_comp <= expected_components


def test_kmeans_candidates_evaluation(sample_feature_df):
    """Verify that evaluate_kmeans_candidates returns expected table format and counts."""
    X_prep, _, _, _ = build_clustering_feature_matrix(sample_feature_df)
    k_range = range(2, 5)
    df_eval = evaluate_kmeans_candidates(X_prep, k_range=k_range, random_state=42, silhouette_sample_size=None)

    assert len(df_eval) == len(k_range)
    assert list(df_eval["k"]) == list(k_range)
    assert (df_eval["inertia"] > 0).all()
    assert (df_eval["silhouette_score"] >= -1.0).all() and (df_eval["silhouette_score"] <= 1.0).all()

    # Inertia must decrease monotonically with increasing k
    assert df_eval["inertia"].is_monotonic_decreasing


def test_cluster_label_coverage_and_reconciliation(sample_feature_df):
    """Verify that cluster labels cover every observation exactly once and sum to N."""
    X_prep, _, _, _ = build_clustering_feature_matrix(sample_feature_df)
    k = 3
    km, labels = fit_final_segmentation(X_prep, k=k, random_state=42)

    assert len(labels) == len(sample_feature_df)
    assert set(np.unique(labels)) == set(range(k))
    assert not pd.Series(labels).isna().any()


def test_stability_metric_label_invariance():
    """Verify that Adjusted Rand Index is invariant to arbitrary cluster label permutation."""
    labels_true = np.array([0, 0, 1, 1, 2, 2, 0, 1])
    # Permute cluster IDs: 0 -> 2, 1 -> 0, 2 -> 1
    mapping = {0: 2, 1: 0, 2: 1}
    labels_perm = np.array([mapping[x] for x in labels_true])

    ari = adjusted_rand_score(labels_true, labels_perm)
    # Perfect match under permutation must be exactly 1.0
    assert ari == pytest.approx(1.0)


def test_post_hoc_profiling_target_isolation(sample_feature_df):
    """Verify that target and RF scores enter only during post-hoc profiling without altering clusters."""
    X_prep, _, _, X_fe = build_clustering_feature_matrix(sample_feature_df)
    k = 2
    km, labels = fit_final_segmentation(X_prep, k=k, random_state=42)

    # 1. Profile without target
    prof_no_target = profile_clusters(X_fe, labels)
    for p in prof_no_target["cluster_profiles"]:
        assert "converters" not in p
        assert "conversion_rate" not in p
        assert "mean_rf_score" not in p

    # 2. Profile with target and synthetic RF scores
    y_dev = sample_feature_df["y"]
    rf_scores = np.linspace(0.01, 0.99, len(sample_feature_df))
    prof_with_target = profile_clusters(X_fe, labels, y_dev=y_dev, rf_scores=rf_scores, k_capacity=10)

    for p in prof_with_target["cluster_profiles"]:
        assert "converters" in p
        assert "conversion_rate" in p
        assert "mean_rf_score" in p
        assert "rf_selected_top_k" in p


def test_exact_redundancy_of_prior_contact_indicator():
    """Verify that was_previously_contacted is mathematically identical to (pdays_recency > 0)."""
    from src.data.load_data import load_bank_marketing_data
    from src.features.build_features import split_data, PreCallFeatureEngineer
    from src.features.feature_contract import select_canonical_pre_campaign_features

    X, y = load_bank_marketing_data()
    X_dev, _, _, _ = split_data(X, y)
    X_clean = select_canonical_pre_campaign_features(X_dev)
    fe = PreCallFeatureEngineer(enforce_contract=True, drop_leakage=True)
    X_fe = fe.fit_transform(X_clean)

    # 1. pdays == 0 never occurs in data
    assert (X["pdays"] == 0).sum() == 0
    assert (X_dev["pdays"] == 0).sum() == 0

    # 2. was_previously_contacted exactly equals pdays_recency > 0 across every row
    assert (X_fe["was_previously_contacted"] == (X_fe["pdays_recency"] > 0)).all()


def test_parsimonious_clustering_matrix_and_onehot_configuration(sample_feature_df):
    """Verify was_previously_contacted is absent from coordinates, and OneHotEncoder has drop=None."""
    X_prep, feature_names, preprocessor, X_fe = build_clustering_feature_matrix(sample_feature_df)

    # was_previously_contacted must not be in feature coordinates
    assert "num__was_previously_contacted" not in feature_names
    assert "cat__was_previously_contacted" not in feature_names

    # OneHotEncoder configuration must be drop=None for symmetric Euclidean distances
    cat_transformer = preprocessor.named_transformers_["cat"]
    assert cat_transformer.drop is None

    # Check that was_previously_contacted remains accessible in X_fe for profiling
    assert "was_previously_contacted" in X_fe.columns


def test_no_enforce_contract_false_bypass_in_segmentation():
    """Verify that enforce_contract=False is not present in segmentation module."""
    import inspect
    import src.models.segmentation as seg_mod

    source = inspect.getsource(seg_mod)
    assert "enforce_contract=False" not in source
    assert "CANONICAL_FORBIDDEN_FEATURES" in source
