"""
Preprocessing Module
Feature encoding, scaling, train/test split, and SMOTE oversampling.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from imblearn.over_sampling import SMOTE
from pathlib import Path
import joblib
import logging

logger = logging.getLogger(__name__)

CATEGORICAL_COLS = ["contract", "internet_service", "phone_service", "partner", "dependents", "payment_method", "paperless_billing"]
NUMERIC_COLS = ["tenure", "monthly_charges", "total_charges", "support_tickets", "senior_citizen"]
TARGET = "churn"
DROP_COLS = ["customer_id"]


def preprocess(df: pd.DataFrame, fit: bool = True, encoders: dict = None, scaler=None):
    """
    Encode categoricals, scale numerics, apply SMOTE.
    
    Args:
        df: Input DataFrame
        fit: If True, fit and return new encoders/scaler; else transform using provided ones
        encoders: Dict of LabelEncoders per categorical column
        scaler: Fitted StandardScaler
    
    Returns:
        X_train, X_test, y_train, y_test, encoders, scaler
    """
    df = df.drop(columns=DROP_COLS, errors="ignore").copy()
    
    if fit:
        encoders = {}
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = le
        scaler = StandardScaler()
        df[NUMERIC_COLS] = scaler.fit_transform(df[NUMERIC_COLS])
    else:
        for col in CATEGORICAL_COLS:
            df[col] = encoders[col].transform(df[col].astype(str))
        df[NUMERIC_COLS] = scaler.transform(df[NUMERIC_COLS])
    
    X = df.drop(columns=[TARGET])
    y = df[TARGET]
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)
    logger.info(f"After SMOTE — Train: {X_train.shape}, Test: {X_test.shape}")
    
    return X_train, X_test, y_train, y_test, encoders, scaler
