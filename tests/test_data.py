"""Unit and integration tests for data loading and preprocessing."""
import pandas as pd
import pytest

from src.utils.logger import load_config
from src.features.build_features import identify_feature_types, build_preprocessor

def test_load_config():
    """Verify that config loads correctly and contains essential keys."""
    config = load_config()
    assert "dataset" in config
    assert config["dataset"]["uci_id"] == 222
    assert "paths" in config
    assert "split" in config

def test_identify_feature_types():
    """Test automatic column type segregation."""
    df = pd.DataFrame({
        "age": [25, 30, 45],
        "job": ["admin.", "technician", "blue-collar"],
        "balance": [1200.5, 340.0, 50.2],
        "housing": ["yes", "no", "yes"]
    })
    numeric, categorical = identify_feature_types(df)
    assert set(numeric) == {"age", "balance"}
    assert set(categorical) == {"job", "housing"}

def test_preprocessor_pipeline():
    """Test that ColumnTransformer handles missing values and produces expected output."""
    df = pd.DataFrame({
        "age": [25, 30, None],
        "job": ["admin.", "technician", None],
    })
    numeric, categorical = identify_feature_types(df)
    preprocessor = build_preprocessor(numeric, categorical)
    transformed = preprocessor.fit_transform(df)
    assert transformed.shape[0] == 3
    assert transformed.shape[1] > 0
