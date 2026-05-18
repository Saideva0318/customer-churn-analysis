"""Feature Engineering and Preprocessing Pipeline for Churn Analysis."""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
import joblib
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

CATEGORICAL_COLS = [
    'partner', 'dependents', 'phone_service', 'multiple_lines',
    'internet_service', 'online_security', 'tech_support',
    'contract', 'paperless_billing', 'payment_method'
]


class ChurnPreprocessor:
    def __init__(self):
        self.encoders = {}
        self.scaler = StandardScaler()
        self.feature_cols = None

    def fit_transform(self, df: pd.DataFrame):
        logger.info("Starting preprocessing pipeline...")
        data = df.copy()
        data.drop(columns=['customer_id'], inplace=True)

        # Encode categoricals
        for col in CATEGORICAL_COLS:
            le = LabelEncoder()
            data[col] = le.fit_transform(data[col].astype(str))
            self.encoders[col] = le

        # Feature engineering
        data['charge_per_month_tenure'] = data['total_charges'] / (data['tenure'] + 1)
        data['high_value'] = (data['monthly_charges'] > data['monthly_charges'].median()).astype(int)

        self.feature_cols = [c for c in data.columns if c != 'churn']
        X = data[self.feature_cols]
        y = data['churn']

        # Scale
        X_scaled = self.scaler.fit_transform(X)
        X_scaled = pd.DataFrame(X_scaled, columns=self.feature_cols)

        # SMOTE
        logger.info(f"Before SMOTE - Churn: {y.sum()}, Non-Churn: {(y==0).sum()}")
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_scaled, y)
        logger.info(f"After SMOTE  - Churn: {y_res.sum()}, Non-Churn: {(y_res==0).sum()}")

        X_train, X_test, y_train, y_test = train_test_split(X_res, y_res, test_size=0.2, random_state=42)
        logger.info(f"Train: {len(X_train)}, Test: {len(X_test)}")

        # Save artifacts
        os.makedirs('models', exist_ok=True)
        joblib.dump(self.scaler, 'models/scaler.pkl')
        joblib.dump(self.encoders, 'models/encoders.pkl')
        joblib.dump(self.feature_cols, 'models/feature_cols.pkl')

        return X_train, X_test, y_train, y_test


if __name__ == '__main__':
    df = pd.read_csv('data/raw/customer_churn.csv')
    processor = ChurnPreprocessor()
    X_train, X_test, y_train, y_test = processor.fit_transform(df)
    print(f"\n✅ Preprocessing complete!")
    print(f"Train shape: {X_train.shape}, Test shape: {X_test.shape}")
