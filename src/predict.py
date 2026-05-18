"""
Prediction Module
Load saved model and run inference on new customer data.
"""

import pandas as pd
import joblib
import argparse
import logging
from pathlib import Path

from preprocessing import CATEGORICAL_COLS, NUMERIC_COLS, preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def predict(input_path: str, output_path: str = "predictions.csv"):
    """
    Run churn prediction on new customer file.
    
    Args:
        input_path: Path to input CSV
        output_path: Path to save predictions
    """
    model = joblib.load("models/best_model.pkl")
    encoders = joblib.load("models/encoders.pkl")
    scaler = joblib.load("models/scaler.pkl")
    
    df = pd.read_csv(input_path)
    customer_ids = df["customer_id"].copy() if "customer_id" in df.columns else None
    
    # Add dummy churn column for preprocessing compatibility
    df["churn"] = 0
    _, X_test, _, _, _, _ = preprocess(df, fit=False, encoders=encoders, scaler=scaler)
    
    proba = model.predict_proba(X_test)[:, 1]
    predictions = (proba > 0.5).astype(int)
    
    results = pd.DataFrame({
        "churn_probability": proba.round(4),
        "predicted_churn": predictions,
        "risk_tier": pd.cut(proba, bins=[0, 0.3, 0.6, 1.0], labels=["Low", "Medium", "High"])
    })
    
    if customer_ids is not None:
        results.insert(0, "customer_id", customer_ids.values)
    
    results.to_csv(output_path, index=False)
    logger.info(f"Predictions saved to {output_path} | High Risk Customers: {(predictions == 1).sum()}")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Path to input CSV")
    parser.add_argument("--output", default="predictions.csv", help="Path to output CSV")
    args = parser.parse_args()
    results = predict(args.input, args.output)
    print(results.head())
