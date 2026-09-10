"""Fetch and manage the UCI Bank Marketing dataset (ID 222)."""
from pathlib import Path
from typing import Tuple
import pandas as pd
from ucimlrepo import fetch_ucirepo

from src.utils.logger import setup_logger, load_config, get_project_root

logger = setup_logger(__name__)

def fetch_and_save_data(force_download: bool = False) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Fetch the UCI Bank Marketing dataset (id=222) and cache locally as CSV files.

    Args:
        force_download: If True, re-downloads even if cached files exist.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (X features, y targets)
    """
    config = load_config()
    root = get_project_root()

    features_path = root / config["paths"]["raw_features_file"]
    targets_path = root / config["paths"]["raw_targets_file"]
    raw_dir = root / config["paths"]["raw_data_dir"]
    raw_dir.mkdir(parents=True, exist_ok=True)

    if not force_download and features_path.exists() and targets_path.exists():
        logger.info(f"Loading cached dataset from {features_path} and {targets_path}")
        X = pd.read_csv(features_path)
        y = pd.read_csv(targets_path)
        return X, y

    dataset_id = config["dataset"]["uci_id"]
    logger.info(f"Fetching dataset (ID: {dataset_id}) from UCI ML repository...")
    dataset = fetch_ucirepo(id=dataset_id)

    X = dataset.data.features
    y = dataset.data.targets

    logger.info(f"Saving raw dataset to {features_path} and {targets_path}")
    X.to_csv(features_path, index=False)
    y.to_csv(targets_path, index=False)

    logger.info(f"Successfully cached data: X shape={X.shape}, y shape={y.shape}")
    return X, y

def load_bank_marketing_data(force_download: bool = False) -> Tuple[pd.DataFrame, pd.Series]:
    """Convenience function returning X features and y as a 1D Series.

    Args:
        force_download: If True, bypass cache and re-download.

    Returns:
        Tuple[pd.DataFrame, pd.Series]: (X, y) where y is target column 'y'.
    """
    config = load_config()
    target_col = config["dataset"]["target_column"]

    X, y_df = fetch_and_save_data(force_download=force_download)
    if isinstance(y_df, pd.DataFrame):
        if target_col in y_df.columns:
            y = y_df[target_col]
        else:
            y = y_df.iloc[:, 0]
    else:
        y = y_df

    return X, y

if __name__ == "__main__":
    logger.info("Running load_data directly as script...")
    X, y = fetch_and_save_data()
    print("\n--- Dataset Overview ---")
    print(f"Features: {X.shape[0]} rows, {X.shape[1]} columns")
    print(f"Target distribution:\n{y.value_counts(normalize=True)}")
    print("\nFirst 3 rows of features:")
    print(X.head(3))
