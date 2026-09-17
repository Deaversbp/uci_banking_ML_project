"""Unit tests for the canonical prediction-time feature availability contract."""
import pytest
import pandas as pd

from src.features import (
    CANONICAL_PRE_CAMPAIGN_RAW_FEATURES,
    CANONICAL_FORBIDDEN_FEATURES,
    CANONICAL_PRE_CAMPAIGN_ENGINEERED_FEATURES,
    validate_pre_campaign_feature_contract,
)
from src.models.baseline import business_rule_baseline_score


def test_canonical_raw_features_pass_contract():
    """Verify that all canonical pre-campaign raw features pass validation."""
    is_valid, violations = validate_pre_campaign_feature_contract(
        CANONICAL_PRE_CAMPAIGN_RAW_FEATURES, raise_on_violation=True
    )
    assert is_valid is True
    assert len(violations) == 0


def test_canonical_engineered_features_pass_contract():
    """Verify that all canonical pre-campaign engineered features pass validation."""
    is_valid, violations = validate_pre_campaign_feature_contract(
        CANONICAL_PRE_CAMPAIGN_ENGINEERED_FEATURES, raise_on_violation=True
    )
    assert is_valid is True
    assert len(violations) == 0


def test_canonical_forbidden_features_fail_contract_individually():
    """Verify that every forbidden feature triggers contract failure individually."""
    for forbidden in CANONICAL_FORBIDDEN_FEATURES:
        is_valid, violations = validate_pre_campaign_feature_contract(
            [forbidden], raise_on_violation=False
        )
        assert is_valid is False
        assert forbidden in violations

        with pytest.raises(ValueError, match="Pre-campaign feature contract violation"):
            validate_pre_campaign_feature_contract([forbidden], raise_on_violation=True)


def test_forbidden_features_detected_in_mixed_pool():
    """Verify that forbidden features mixed with valid features are detected and flagged."""
    candidate_features = [
        "age",
        "balance",
        "housing",
        "contact",               # Forbidden (operational channel)
        "was_previously_contacted",
        "month",                 # Forbidden (calendar timing)
        "duration",              # Forbidden (post-call leakage)
        "campaign",              # Forbidden (contact count during campaign)
        "contact_day_of_month",  # Forbidden (calendar timing)
    ]

    is_valid, violations = validate_pre_campaign_feature_contract(
        candidate_features, raise_on_violation=False
    )
    assert is_valid is False
    assert set(violations) == {
        "contact",
        "month",
        "duration",
        "campaign",
        "contact_day_of_month",
    }

    with pytest.raises(ValueError, match="Pre-campaign feature contract violation"):
        validate_pre_campaign_feature_contract(candidate_features, raise_on_violation=True)


def test_business_rule_baseline_adheres_to_canonical_contract():
    """Verify that business-rule baseline relies strictly on pre-campaign features."""
    # Features inspected by business_rule_baseline_score in src/models/baseline.py:
    # 1. poutcome (Tier 1 bonus)
    # 2. pdays & loan (Tier 2 bonus)
    # 3. housing & loan (Tier 3 bonus)
    # 4. balance (continuous tier bonus)
    business_rule_features = ["poutcome", "pdays", "loan", "housing", "balance"]

    is_valid, violations = validate_pre_campaign_feature_contract(
        business_rule_features, raise_on_violation=True
    )
    assert is_valid is True
    assert len(violations) == 0

    # Ensure no forbidden features are present in business rule inputs
    for forbidden in CANONICAL_FORBIDDEN_FEATURES:
        assert forbidden not in business_rule_features


def test_uci_raw_features_exact_partition():
    """Verify that the 16 raw features from the UCI Bank Marketing dataset are cleanly partitioned."""
    all_raw_uci_features = [
        "age",
        "job",
        "marital",
        "education",
        "default",
        "balance",
        "housing",
        "loan",
        "contact",
        "day_of_week",  # Normalized as contact_day_of_month (1-31)
        "month",
        "duration",
        "campaign",
        "pdays",
        "previous",
        "poutcome",
    ]

    valid_subset = [f for f in all_raw_uci_features if f in CANONICAL_PRE_CAMPAIGN_RAW_FEATURES]
    forbidden_subset = [f for f in all_raw_uci_features if f in CANONICAL_FORBIDDEN_FEATURES]

    # Exactly 11 raw features are valid pre-campaign
    assert len(valid_subset) == 11
    assert set(valid_subset) == {
        "age", "job", "marital", "education", "default",
        "balance", "housing", "loan", "pdays", "previous", "poutcome"
    }

    # Exactly 5 raw features are forbidden (duration, contact, day_of_week, month, campaign)
    assert len(forbidden_subset) == 5
    assert set(forbidden_subset) == {
        "contact", "day_of_week", "month", "duration", "campaign"
    }

    # Complete partition: 11 + 5 = 16
    assert len(valid_subset) + len(forbidden_subset) == len(all_raw_uci_features)


def test_pipeline_quarantines_forbidden_features_from_raw_dataframe():
    """Verify that create_pre_call_pipeline completely bars forbidden features from reaching ColumnTransformer or model."""
    from sklearn.linear_model import LogisticRegression
    from src.features.build_features import create_pre_call_pipeline

    df_full_raw = pd.DataFrame({
        "age": [30, 45, 28, 52],
        "job": ["admin.", "technician", "services", "retired"],
        "marital": ["single", "married", "single", "married"],
        "education": ["secondary", "tertiary", "secondary", "primary"],
        "default": ["no", "no", "no", "no"],
        "balance": [1000, 2500, -50, 15000],
        "housing": ["yes", "no", "yes", "no"],
        "loan": ["no", "no", "yes", "no"],
        "contact": ["cellular", "telephone", "cellular", "unknown"],
        "day_of_week": [1, 15, 20, 5],
        "month": ["may", "jul", "aug", "jun"],
        "duration": [500, 1200, 50, 800],
        "campaign": [1, 2, 1, 3],
        "pdays": [-1, 120, -1, 30],
        "previous": [0, 2, 0, 1],
        "poutcome": [None, "success", None, "failure"],
    })
    y_raw = pd.Series(["no", "yes", "no", "yes"])

    pipe = create_pre_call_pipeline(LogisticRegression(), raw_feature_df=df_full_raw)
    pipe.fit(df_full_raw, y_raw)

    preprocessor = pipe.named_steps["preprocessor"]
    num_cols = preprocessor.transformers[0][2]
    cat_cols = preprocessor.transformers[1][2]
    all_preproc_cols = list(num_cols) + list(cat_cols)

    # NONE of the forbidden fields may be present
    for forbidden in CANONICAL_FORBIDDEN_FEATURES:
        assert forbidden not in all_preproc_cols, f"Forbidden feature '{forbidden}' reached preprocessor!"

    # Verify all 11 valid raw features and 6 approved engineered features are represented
    for raw_feat in CANONICAL_PRE_CAMPAIGN_RAW_FEATURES:
        assert raw_feat in all_preproc_cols, f"Valid raw feature '{raw_feat}' missing from preprocessor!"

    for eng_feat in CANONICAL_PRE_CAMPAIGN_ENGINEERED_FEATURES:
        assert eng_feat in all_preproc_cols, f"Approved engineered feature '{eng_feat}' missing from preprocessor!"


def test_pre_call_feature_engineer_produces_only_approved_features():
    """Verify PreCallFeatureEngineer fit_transform produces exactly 11 raw + 6 engineered features and 0 forbidden features."""
    from src.features.build_features import PreCallFeatureEngineer

    df_full_raw = pd.DataFrame({
        "age": [30, 45],
        "job": ["admin.", "technician"],
        "marital": ["single", "married"],
        "education": ["secondary", "tertiary"],
        "default": ["no", "no"],
        "balance": [1000, -50],
        "housing": ["yes", "no"],
        "loan": ["no", "yes"],
        "contact": ["cellular", "telephone"],
        "day_of_week": [1, 15],
        "month": ["may", "jul"],
        "duration": [500, 1200],
        "campaign": [1, 2],
        "pdays": [-1, 120],
        "previous": [0, 2],
        "poutcome": [None, "success"],
    })

    fe = PreCallFeatureEngineer(enforce_contract=True)
    out = fe.fit_transform(df_full_raw)

    # Output must have exactly 11 valid raw + 6 approved engineered = 17 features
    expected_features = set(CANONICAL_PRE_CAMPAIGN_RAW_FEATURES + CANONICAL_PRE_CAMPAIGN_ENGINEERED_FEATURES)
    assert set(out.columns) == expected_features
    assert len(out.columns) == 17

    # No forbidden features present
    for forbidden in CANONICAL_FORBIDDEN_FEATURES:
        assert forbidden not in out.columns


def test_business_rule_executable_dependency_audit():
    """Formally verify that the executable business-rule baseline accesses only legitimate pre-campaign features and never reads campaign."""
    accessed_columns = set()

    class TrackingDataFrame:
        def __init__(self, df):
            self._df = df

        @property
        def columns(self):
            return self._df.columns

        @property
        def index(self):
            return self._df.index

        def __getitem__(self, item):
            accessed_columns.add(item)
            return self._df[item]

    df = pd.DataFrame({
        "poutcome": ["success", "failure"],
        "pdays": [100, -1],
        "housing": ["no", "yes"],
        "loan": ["no", "no"],
        "balance": [5000, 100],
        "campaign": [1, 5],  # Forbidden column present in DataFrame
        "contact": ["cellular", "unknown"],
        "month": ["may", "jul"],
        "duration": [200, 100],
    })

    tracking_df = TrackingDataFrame(df)
    scores = business_rule_baseline_score(tracking_df)

    assert "campaign" not in accessed_columns
    assert "duration" not in accessed_columns
    assert "contact" not in accessed_columns
    assert "month" not in accessed_columns
    assert accessed_columns.issubset(set(CANONICAL_PRE_CAMPAIGN_RAW_FEATURES))
    assert len(scores) == 2

