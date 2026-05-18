"""
Unit Tests — Preprocessing Module
"""

import pytest
import pandas as pd
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from data_generator import generate_churn_dataset
from preprocessing import preprocess


@pytest.fixture
def sample_df():
    return generate_churn_dataset(n_customers=500)


def test_dataset_shape(sample_df):
    assert sample_df.shape[0] == 500
    assert "churn" in sample_df.columns


def test_churn_rate_reasonable(sample_df):
    churn_rate = sample_df["churn"].mean()
    assert 0.10 <= churn_rate <= 0.60, f"Unexpected churn rate: {churn_rate:.2%}"


def test_preprocess_returns_split(sample_df):
    X_train, X_test, y_train, y_test, encoders, scaler = preprocess(sample_df)
    assert X_train.shape[0] > 0
    assert X_test.shape[0] > 0
    assert len(encoders) > 0
    assert scaler is not None


def test_no_nulls_after_preprocessing(sample_df):
    X_train, X_test, y_train, y_test, _, _ = preprocess(sample_df)
    assert X_train.isnull().sum().sum() == 0
    assert X_test.isnull().sum().sum() == 0
