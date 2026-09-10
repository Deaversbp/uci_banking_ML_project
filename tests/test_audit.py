"""Unit tests for the data audit module."""
import pandas as pd
import pytest

from src.data.audit_dataset import (
    compute_target_prevalence,
    audit_duration_leakage,
    audit_campaign_fatigue,
    audit_prior_contacts,
    audit_data_quality_and_duplicates,
)

@pytest.fixture
def sample_df():
    """Create a minimal synthetic dataset for testing audit functions."""
    return pd.DataFrame({
        "age": [30, 40, 50, 30],
        "job": ["admin.", "technician", "blue-collar", "admin."],
        "marital": ["married", "single", "divorced", "married"],
        "education": ["secondary", "tertiary", None, "secondary"],
        "default": ["no", "no", "no", "no"],
        "balance": [100, 200, -50, 100],
        "housing": ["yes", "no", "yes", "yes"],
        "loan": ["no", "yes", "no", "no"],
        "contact": ["cellular", None, "cellular", "cellular"],
        "day_of_week": [5, 10, 15, 5],
        "month": ["may", "jun", "jul", "may"],
        "duration": [60, 300, 0, 60],
        "campaign": [1, 2, 8, 1],
        "pdays": [-1, 100, -1, 50],
        "previous": [0, 2, 0, 1],
        "poutcome": [None, "success", None, None],
        "target": ["no", "yes", "no", "no"],
    })

def test_compute_target_prevalence(sample_df):
    stats = compute_target_prevalence(sample_df)
    assert stats["total_records"] == 4
    assert stats["positive_count"] == 1
    assert stats["negative_count"] == 3
    assert stats["prevalence_rate"] == 0.25
    assert stats["expected_conversions_random"] == 1250

def test_audit_duration_leakage(sample_df):
    stats = audit_duration_leakage(sample_df)
    assert "duration_roc_auc" in stats
    assert stats["zero_duration_count"] == 1
    assert stats["zero_duration_conv_rate"] == 0.0

def test_audit_campaign_fatigue(sample_df):
    fatigue = audit_campaign_fatigue(sample_df)
    assert isinstance(fatigue, pd.DataFrame)
    assert "campaign_tier" in fatigue.columns
    assert "conversion_rate" in fatigue.columns
    assert "record_count" in fatigue.columns

def test_audit_prior_contacts(sample_df):
    prior = audit_prior_contacts(sample_df)
    assert prior["never_contacted_count"] == 2
    assert prior["never_contacted_pct"] == 0.50
    # Record at index 3 has pdays=50 != -1 but poutcome is None (exception)
    assert prior["exception_count"] == 1
    assert prior["exception_indices"] == [3]

def test_audit_data_quality_and_duplicates(sample_df):
    quality = audit_data_quality_and_duplicates(sample_df)
    assert "education" in quality["nan_counts"]
    assert quality["repeated_demographic_profiles"] == 1
    assert quality["duplicate_demographic_profiles"] == 1
