"""Canonical Pre-Campaign Feature Contract.

Enforces strict prediction-time feature availability for outbound telemarketing
prospect prioritization.

Business Decision to Enforce:
"Before outreach begins for a new campaign, rank an eligible prospect pool
and choose which prospects should receive the limited outbound call capacity."

Prediction Timestamp:
IMMEDIATELY BEFORE any contact in the new/current campaign occurs.
"""
from typing import Sequence, Tuple, List, Set
import pandas as pd


# Canonical set of valid raw features available pre-campaign in bank CRM / core banking
CANONICAL_PRE_CAMPAIGN_RAW_FEATURES: List[str] = [
    "age",
    "job",
    "marital",
    "education",
    "default",
    "balance",
    "housing",
    "loan",
    "pdays",
    "previous",
    "poutcome",
]

# Canonical set of forbidden features (post-call leakage or current-campaign operational variables)
CANONICAL_FORBIDDEN_FEATURES: List[str] = [
    "duration",
    "contact",
    "month",
    "day_of_week",
    "contact_day_of_month",
    "day",
    "campaign",
    "y",
]

# Canonical set of legitimate engineered features derived strictly from pre-campaign variables
CANONICAL_PRE_CAMPAIGN_ENGINEERED_FEATURES: List[str] = [
    "was_previously_contacted",
    "pdays_recency",
    "prior_success",
    "has_debt_burden",
    "negative_balance_flag",
    "balance_log",
]


def select_canonical_pre_campaign_features(X: pd.DataFrame) -> pd.DataFrame:
    """Select strictly the canonical pre-campaign raw features from X.

    Strips any forbidden current-campaign fields (duration, contact, month,
    day_of_week, contact_day_of_month, day, campaign, y) and returns
    a dataframe containing only columns present in CANONICAL_PRE_CAMPAIGN_RAW_FEATURES.

    Args:
        X: Raw or incoming feature DataFrame.

    Returns:
        pd.DataFrame: Cleaned dataframe containing only valid pre-campaign raw features.
    """
    valid_cols = [col for col in CANONICAL_PRE_CAMPAIGN_RAW_FEATURES if col in X.columns]
    return X[valid_cols].copy()


def validate_pre_campaign_feature_contract(
    feature_names: Sequence[str],
    raise_on_violation: bool = True,
) -> Tuple[bool, List[str]]:
    """Validate that a collection of feature names contains no forbidden current-campaign fields.

    Args:
        feature_names: List or sequence of feature names to validate.
        raise_on_violation: If True, raises ValueError upon detecting any forbidden feature.

    Returns:
        Tuple[bool, List[str]]: (is_valid, list_of_forbidden_features_detected)

    Raises:
        ValueError: If forbidden features are detected and raise_on_violation is True.
    """
    forbidden_detected = [f for f in feature_names if f in CANONICAL_FORBIDDEN_FEATURES]
    is_valid = len(forbidden_detected) == 0

    if not is_valid and raise_on_violation:
        raise ValueError(
            f"Pre-campaign feature contract violation! The following forbidden current-campaign "
            f"or post-call leakage features were detected: {forbidden_detected}. "
            f"Prediction timestamp is strictly IMMEDIATELY BEFORE outreach begins for the new campaign."
        )

    return is_valid, forbidden_detected

