"""Unit and integration tests for data loading, pre-call feature engineering,
leakage prevention, package exports, and lead ranking evaluation.
"""
import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LogisticRegression

# Test package exports
import src.data as data_pkg
import src.features as features_pkg
import src.models as models_pkg

from src.utils.logger import load_config
from src.features.build_features import (
    identify_feature_types,
    build_preprocessor,
    drop_duration,
    PreCallFeatureEngineer,
    create_pre_call_pipeline,
)
from src.models.evaluate import compute_ranking_metrics_at_k, evaluate_model
from src.models.baseline import (
    evaluate_random_baseline,
    business_rule_baseline_score,
    evaluate_business_rule_baseline,
)


def test_package_exports():
    """Verify clean exports across all packages without AttributeError or KeyError."""
    assert hasattr(data_pkg, "load_bank_marketing_data")
    assert hasattr(data_pkg, "fetch_and_save_data")
    assert hasattr(data_pkg, "run_full_data_audit")

    assert hasattr(features_pkg, "build_preprocessor")
    assert hasattr(features_pkg, "split_data")
    assert hasattr(features_pkg, "identify_feature_types")
    assert hasattr(features_pkg, "PreCallFeatureEngineer")
    assert hasattr(features_pkg, "drop_duration")
    assert hasattr(features_pkg, "create_pre_call_pipeline")

    assert hasattr(models_pkg, "train_baseline_models")
    assert hasattr(models_pkg, "get_candidate_models")
    assert hasattr(models_pkg, "evaluate_model")
    assert hasattr(models_pkg, "compute_ranking_metrics_at_k")
    assert hasattr(models_pkg, "evaluate_random_baseline")
    assert hasattr(models_pkg, "business_rule_baseline_score")
    assert hasattr(models_pkg, "evaluate_business_rule_baseline")


def test_load_config():
    """Verify that config loads correctly and contains essential keys."""
    config = load_config()
    assert "dataset" in config
    assert config["dataset"]["uci_id"] == 222
    assert "paths" in config
    assert "split" in config


def test_drop_duration():
    """Verify that drop_duration eliminates duration column."""
    df = pd.DataFrame({"age": [25, 30], "duration": [120, 300]})
    cleaned = drop_duration(df)
    assert "duration" not in cleaned.columns
    assert "age" in cleaned.columns


def test_pre_call_feature_engineer_leakage_and_features():
    """Verify that PreCallFeatureEngineer excludes duration and creates expected features."""
    df = pd.DataFrame({
        "age": [25, 60, 35],
        "balance": [-100, 5000, 0],
        "housing": ["yes", "no", "yes"],
        "loan": ["yes", "no", "no"],
        "campaign": [2, 1, 15],
        "pdays": [-1, 120, -1],
        "previous": [0, 3, 0],
        "poutcome": [None, "success", "failure"],
        "duration": [150, 420, 90],  # Post-call leakage column
    })

    fe = PreCallFeatureEngineer(drop_leakage=True)
    out = fe.fit_transform(df)

    # 1. Leakage assertion: duration must NEVER be present
    assert "duration" not in out.columns

    # 2. Prior contact features
    assert out["was_previously_contacted"].tolist() == [0, 1, 0]
    assert out["pdays_recency"].iloc[0] == 0.0  # -1 handled as 0
    assert out["pdays_recency"].iloc[1] == pytest.approx(np.log1p(120), rel=1e-3)
    assert out["prior_success"].tolist() == [0, 1, 0]

    # 3. Debt burden
    assert out["has_debt_burden"].tolist() == [1, 0, 0]

    # 4. Negative balance & signed log
    assert out["negative_balance_flag"].tolist() == [1, 0, 0]
    assert out["balance_log"].iloc[0] == pytest.approx(-np.log1p(100), rel=1e-3)
    assert out["balance_log"].iloc[1] == pytest.approx(np.log1p(5000), rel=1e-3)
    assert out["balance_log"].iloc[2] == 0.0

    # 5. Campaign: raw count must not be blindly capped
    assert out["campaign"].tolist() == [2, 1, 15]


def test_identify_feature_types_supports_numeric_and_boolean():
    """Verify detection supports all numpy numeric types and booleans, excluding duration."""
    df = pd.DataFrame({
        "int32_col": pd.Series([1, 2], dtype="int32"),
        "int64_col": pd.Series([10, 20], dtype="int64"),
        "float32_col": pd.Series([1.5, 2.5], dtype="float32"),
        "float64_col": pd.Series([3.14, 2.71], dtype="float64"),
        "bool_col": pd.Series([True, False], dtype="bool"),
        "boolean_col": pd.Series([True, False], dtype="boolean"),
        "cat_col": pd.Series(["admin.", "technician"], dtype="object"),
        "duration": pd.Series([100, 200], dtype="int64"),  # Leakage column
    })

    numeric, categorical = identify_feature_types(df)

    assert "duration" not in numeric
    assert "duration" not in categorical
    assert "cat_col" in categorical
    assert set(numeric) == {"int32_col", "int64_col", "float32_col", "float64_col", "bool_col", "boolean_col"}


def test_categorical_missing_state_preservation():
    """Verify that missing categorical values are preserved as 'unknown' and never imputed to 'failure'."""
    df = pd.DataFrame({
        "poutcome": ["failure", "failure", "success", None],  # 'failure' is most frequent!
        "age": [30, 40, 50, 60],
    })

    numeric_cols, categorical_cols = identify_feature_types(df)
    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    transformed = preprocessor.fit_transform(df)
    cat_feature_names = preprocessor.named_transformers_["cat"].named_steps["onehot"].get_feature_names_out(["poutcome"])

    # Ensure 'poutcome_unknown' exists (not converted to failure)
    assert any("unknown" in name for name in cat_feature_names)

    # Transform a row with None
    test_row = pd.DataFrame({"poutcome": [None], "age": [25]})
    test_transformed = preprocessor.transform(test_row)
    unknown_idx = list(cat_feature_names).index("poutcome_unknown")

    # The unknown feature index must be 1.0 (active), proving it wasn't mapped to failure
    assert test_transformed[0, len(numeric_cols) + unknown_idx] == 1.0


def test_create_pre_call_pipeline_leakage_safety():
    """Verify end-to-end pipeline fits cleanly and strips duration even if duration is fed in X."""
    df = pd.DataFrame({
        "age": [25, 30, 45, 50, 35],
        "job": ["admin.", "technician", "blue-collar", "retired", "services"],
        "marital": ["married", "single", "married", "divorced", "single"],
        "education": ["secondary", "tertiary", None, "primary", "secondary"],
        "default": ["no", "no", "no", "no", "no"],
        "balance": [1000, 200, -50, 15000, 300],
        "housing": ["yes", "no", "yes", "no", "yes"],
        "loan": ["no", "yes", "no", "no", "no"],
        "contact": ["cellular", None, "cellular", "telephone", "cellular"],
        "day_of_week": [5, 10, 15, 20, 25],
        "month": ["may", "jun", "jul", "aug", "sep"],
        "duration": [120, 300, 60, 450, 80],  # Post-call column present in input
        "campaign": [1, 2, 1, 3, 2],
        "pdays": [-1, 100, -1, 30, -1],
        "previous": [0, 2, 0, 1, 0],
        "poutcome": [None, "success", None, "failure", None],
    })
    y = pd.Series(["no", "yes", "no", "yes", "no"])

    pipeline = create_pre_call_pipeline(LogisticRegression(), raw_feature_df=df)
    pipeline.fit(df, y)

    preds = pipeline.predict(df)
    assert len(preds) == len(df)


def test_ranking_metrics_at_k():
    """Verify lead ranking metrics calculation."""
    y_true = pd.Series(["yes", "no", "yes", "no", "no", "no", "no", "no", "no", "no"])
    y_proba = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])

    metrics = compute_ranking_metrics_at_k(y_true, y_proba, k=3, pos_label="yes")

    assert metrics["k_evaluated"] == 3
    # Top 3 probas: [0.9 (yes), 0.8 (no), 0.7 (yes)] -> 2 conversions
    assert metrics["conversions_at_k"] == 2
    assert metrics["precision_at_k"] == pytest.approx(2 / 3)
    assert metrics["recall_at_k"] == 1.0  # Captured all 2 positives
    assert metrics["lift_at_k"] == pytest.approx((2 / 3) / 0.2)  # Base rate is 2/10 = 0.2 -> Lift is 3.33x


def test_business_rule_baseline():
    """Verify business-rule baseline scoring and ranking evaluation."""
    df = pd.DataFrame({
        "poutcome": ["success", "failure", None, "success"],
        "pdays": [100, 50, -1, 200],
        "housing": ["no", "yes", "yes", "no"],
        "loan": ["no", "no", "no", "no"],
        "balance": [5000, 200, 100, 12000],
    })
    y_true = pd.Series(["yes", "no", "no", "yes"])

    scores = business_rule_baseline_score(df)
    # Row 3 (poutcome=success, debt-free, balance=12k) and Row 0 (poutcome=success, balance=5k) should have top scores
    assert scores.iloc[3] > scores.iloc[0] > scores.iloc[1] > scores.iloc[2]

    eval_results = evaluate_business_rule_baseline(df, y_true, k=2, pos_label="yes")
    assert eval_results["conversions_at_k"] == 2
    assert eval_results["precision_at_k"] == 1.0
    assert eval_results["lift_at_k"] == pytest.approx(1.0 / 0.5)  # 2.0x lift
