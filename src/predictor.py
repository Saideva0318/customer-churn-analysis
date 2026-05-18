"""Inference script for Customer Churn Prediction."""

import pandas as pd
import numpy as np
import joblib
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class ChurnPredictor:
    def __init__(self, model_path: str = 'models/XGBoost.pkl'):
        self.model = joblib.load(model_path)
        self.scaler = joblib.load('models/scaler.pkl')
        self.encoders = joblib.load('models/encoders.pkl')
        self.feature_cols = joblib.load('models/feature_cols.pkl')
        logger.info(f"Model loaded from {model_path}")

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        data = df.copy()
        for col, encoder in self.encoders.items():
            if col in data.columns:
                data[col] = encoder.transform(data[col].astype(str))
        data['charge_per_month_tenure'] = data['total_charges'] / (data['tenure'] + 1)
        data['high_value'] = (data['monthly_charges'] > 65).astype(int)
        return data[self.feature_cols]

    def predict(self, df: pd.DataFrame) -> pd.DataFrame:
        X = self.preprocess(df)
        X_scaled = self.scaler.transform(X)
        probs = self.model.predict_proba(X_scaled)[:, 1]
        preds = (probs >= 0.5).astype(int)
        result = df[['customer_id']].copy()
        result['churn_probability'] = probs.round(4)
        result['churn_prediction'] = preds
        result['risk_level'] = pd.cut(probs, bins=[0, 0.3, 0.6, 1.0],
                                      labels=['Low', 'Medium', 'High'])
        return result


if __name__ == '__main__':
    df = pd.read_csv('data/raw/customer_churn.csv')
    sample = df.drop(columns=['churn']).head(20)
    predictor = ChurnPredictor()
    predictions = predictor.predict(sample)
    print("\n🔮 Churn Predictions (Sample):")
    print(predictions.to_string(index=False))
    print(f"\nHigh Risk Customers: {(predictions['risk_level'] == 'High').sum()}")
