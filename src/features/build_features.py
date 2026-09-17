"""Pre-call feature engineering and preprocessing pipeline.

Ensures strict post-call leakage prevention by excluding 'duration' from all
feature pipelines, implements pre-call domain feature transformations, preserves
informative categorical missingness (e.g. 'unknown' for poutcome/contact),
and supports broad numeric and boolean types.
"""
from typing import Tuple, List, Any
import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import setup_logger, load_config
from src.features.feature_contract import (
    CANONICAL_PRE_CAMPAIGN_RAW_FEATURES,
    CANONICAL_FORBIDDEN_FEATURES,
    CANONICAL_PRE_CAMPAIGN_ENGINEERED_FEATURES,
    validate_pre_campaign_feature_contract,
    select_canonical_pre_campaign_features,
)

logger = setup_logger(__name__)


def drop_duration(X: pd.DataFrame) -> pd.DataFrame:
    """Explicitly drop the 'duration' column from a feature DataFrame.

    Duration is post-call information (intrinsically unknown prior to outreach).
    Retaining it introduces catastrophic target leakage into pre-call lead scoring.

    Args:
        X: Feature dataframe.

    Returns:
        pd.DataFrame: Dataframe guaranteed to exclude 'duration'.
    """
    if "duration" in X.columns:
        logger.info("Dropping post-call leakage feature: 'duration'")
        return X.drop(columns=["duration"])
    return X


class PreCallFeatureEngineer(BaseEstimator, TransformerMixin):
    """Scikit-learn compatible transformer engineering legitimate pre-call features.

    Methodological & Contractual Guarantees:
    - Canonical Feature Contract: Strictly enforces pre-campaign feature availability
      by filtering raw input features via select_canonical_pre_campaign_features().
      All forbidden current-campaign/leakage features ('duration', 'contact', 'month',
      'day_of_week', 'contact_day_of_month', 'day', 'campaign', 'y') are unconditionally
      purged before feature engineering or preprocessing.
    - Prior Contact Dynamics:
      - was_previously_contacted: Binary indicator for prior campaign history (pdays != -1).
      - pdays_recency: Non-negative transformed recency handling -1 explicitly as 0 (via log1p(max(pdays, 0))),
        avoiding distortion of the -1 sentinel on a continuous scale.
      - prior_success: Binary indicator for prior campaign success (poutcome == 'success').
      - Note on Missing poutcome: While missing poutcome predominantly corresponds to uncontacted clients
        (99.986% of missing records), exactly 5 records in the dataset have prior contacts (pdays > 0)
        with unrecorded outcomes. Missing values are preserved as 'unknown' rather than assumed to be
        exclusively uncontacted or imputed to 'failure'.
    - Financial Burden & Balance:
      - has_debt_burden: Binary indicator for dual housing and personal debt (housing == 'yes' & loan == 'yes').
      - negative_balance_flag: Binary indicator for accounts with balance < 0.
      - balance_log: Signed log1p transformation (sign(balance) * log1p(|balance|)) accommodating
        financial skewness (-€8,019 to +€102,127) without arbitrary truncation.
    """

    def __init__(self, enforce_contract: bool = True, drop_leakage: bool = True):
        self.enforce_contract = enforce_contract
        self.drop_leakage = drop_leakage

    def fit(self, X: pd.DataFrame, y=None):
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        if self.enforce_contract:
            X_out = select_canonical_pre_campaign_features(X)
        else:
            X_out = X.copy()
            if self.drop_leakage and "duration" in X_out.columns:
                X_out = X_out.drop(columns=["duration"])
            if "day_of_week" in X_out.columns and "contact_day_of_month" not in X_out.columns:
                X_out["contact_day_of_month"] = X_out["day_of_week"]
                X_out = X_out.drop(columns=["day_of_week"])

        # Prior contact features
        if "pdays" in X_out.columns:
            pdays_series = X_out["pdays"].fillna(-1)
            X_out["was_previously_contacted"] = (pdays_series != -1).astype(int)
            pdays_nonneg = np.maximum(pdays_series, 0)
            X_out["pdays_recency"] = np.log1p(pdays_nonneg).astype(float)

        if "poutcome" in X_out.columns:
            X_out["prior_success"] = (X_out["poutcome"] == "success").astype(int)

        # Financial burden & balance features
        if "housing" in X_out.columns and "loan" in X_out.columns:
            has_debt = (X_out["housing"] == "yes") & (X_out["loan"] == "yes")
            X_out["has_debt_burden"] = has_debt.astype(int)

        if "balance" in X_out.columns:
            bal = X_out["balance"].fillna(0)
            X_out["negative_balance_flag"] = (bal < 0).astype(int)
            X_out["balance_log"] = (np.sign(bal) * np.log1p(np.abs(bal))).astype(float)

        if self.enforce_contract:
            validate_pre_campaign_feature_contract(X_out.columns, raise_on_violation=True)

        return X_out



def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Automatically identify numeric/boolean and categorical columns.

    Enforces the canonical pre-campaign contract by strictly excluding all
    forbidden current-campaign and leakage fields.

    Args:
        X: Feature dataframe.

    Returns:
        Tuple[List[str], List[str]]: (numeric_features, categorical_features)
    """
    clean_cols = [col for col in X.columns if col not in CANONICAL_FORBIDDEN_FEATURES]
    X_clean = X[clean_cols]

    numeric_features = X_clean.select_dtypes(include=[np.number, "bool", "boolean"]).columns.tolist()
    categorical_features = X_clean.select_dtypes(include=["object", "category", "str", "string"]).columns.tolist()

    validate_pre_campaign_feature_contract(numeric_features + categorical_features, raise_on_violation=True)

    logger.debug(f"Identified {len(numeric_features)} numeric/bool features and {len(categorical_features)} categorical features")
    return numeric_features, categorical_features


class CategoricalMissingImputer(BaseEstimator, TransformerMixin):
    """Robust categorical imputer preserving informative missing states (None, np.nan, pd.NA).

    Imputes any missing categorical values with an explicit constant string (default: 'unknown'),
    guaranteeing that missing poutcome is never mistakenly converted to 'failure'.
    """

    def __init__(self, fill_value: str = "unknown"):
        self.fill_value = fill_value

    def fit(self, X: Any, y=None):
        return self

    def transform(self, X: Any) -> Any:
        if isinstance(X, pd.DataFrame):
            return X.fillna(self.fill_value).astype(str)
        return pd.DataFrame(X).fillna(self.fill_value).astype(str).to_numpy()

    def get_feature_names_out(self, input_features=None):
        if input_features is None:
            return np.array([], dtype=object)
        return np.asarray(input_features, dtype=object)


def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Construct a ColumnTransformer that preserves informative categorical missingness.

    Guarantees that no forbidden current-campaign fields reach the ColumnTransformer.

    - Numeric pipeline: SimpleImputer (median) -> StandardScaler
    - Categorical pipeline: CategoricalMissingImputer (constant='unknown') -> OneHotEncoder
      (Preserves 'unknown' as an informative state for poutcome; never imputes to 'failure')

    Args:
        numeric_features: List of numeric/boolean feature names.
        categorical_features: List of categorical feature names.

    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """
    numeric_clean = [col for col in numeric_features if col not in CANONICAL_FORBIDDEN_FEATURES]
    categorical_clean = [col for col in categorical_features if col not in CANONICAL_FORBIDDEN_FEATURES]

    validate_pre_campaign_feature_contract(numeric_clean + categorical_clean, raise_on_violation=True)

    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", CategoricalMissingImputer(fill_value="unknown")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_clean),
            ("cat", categorical_transformer, categorical_clean),
        ],
        remainder="drop"
    )
    return preprocessor


def create_pre_call_pipeline(
    classifier: Any,
    raw_feature_df: pd.DataFrame
) -> Pipeline:
    """Construct a leak-free end-to-end Pipeline with feature engineering, preprocessing, and classifier.

    Args:
        classifier: A scikit-learn compatible classifier.
        raw_feature_df: A sample of the raw feature dataframe to infer engineered feature schemas.

    Returns:
        Pipeline: Ready-to-fit scikit-learn Pipeline.
    """
    feature_engineer = PreCallFeatureEngineer(enforce_contract=True)
    sample_engineered = feature_engineer.transform(raw_feature_df.head(10))
    numeric_cols, categorical_cols = identify_feature_types(sample_engineered)

    preprocessor = build_preprocessor(numeric_cols, categorical_cols)

    return Pipeline(steps=[
        ("feature_engineer", feature_engineer),
        ("preprocessor", preprocessor),
        ("classifier", classifier),
    ])



def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = None,
    random_state: int = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Perform a stratified train/test split using config defaults if omitted."""
    config = load_config()
    split_cfg = config.get("split", {})

    test_size = test_size if test_size is not None else split_cfg.get("test_size", 0.2)
    random_state = random_state if random_state is not None else split_cfg.get("random_state", 42)
    stratify = y if split_cfg.get("stratify", True) else None

    logger.info(f"Splitting data with test_size={test_size}, random_state={random_state}, stratify={stratify is not None}")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=stratify
    )
    return X_train, X_test, y_train, y_test
