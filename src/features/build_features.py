"""Feature engineering and preprocessing pipeline."""
from typing import Tuple, List
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils.logger import setup_logger, load_config

logger = setup_logger(__name__)

def identify_feature_types(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """Automatically separate numeric and categorical columns.

    Args:
        X: Feature dataframe.

    Returns:
        Tuple[List[str], List[str]]: (numeric_features, categorical_features)
    """
    numeric_features = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_features = X.select_dtypes(include=["object", "category", "str", "string"]).columns.tolist()
    logger.debug(f"Numeric features ({len(numeric_features)}): {numeric_features}")
    logger.debug(f"Categorical features ({len(categorical_features)}): {categorical_features}")
    return numeric_features, categorical_features

def build_preprocessor(numeric_features: List[str], categorical_features: List[str]) -> ColumnTransformer:
    """Construct a scikit-learn ColumnTransformer for preprocessing.

    - Numeric pipeline: SimpleImputer (median) -> StandardScaler
    - Categorical pipeline: SimpleImputer (most_frequent) -> OneHotEncoder

    Returns:
        ColumnTransformer: Preprocessing pipeline.
    """
    numeric_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_transformer = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ],
        remainder="drop"
    )
    return preprocessor

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
