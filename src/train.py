"""
Training Module
Train Logistic Regression, Random Forest, and XGBoost; compare via ROC-AUC.
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
from sklearn.inspection import permutation_importance
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
from pathlib import Path
import logging

from data_generator import generate_churn_dataset
from preprocessing import preprocess

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

MODELS_DIR = Path("models")
MODELS_DIR.mkdir(exist_ok=True)


def train_and_evaluate():
    """Full training pipeline: data → preprocess → train → evaluate → save."""
    
    # Load or generate data
    raw_path = Path("data/raw/customers.csv")
    if raw_path.exists():
        df = pd.read_csv(raw_path)
    else:
        df = generate_churn_dataset()
        raw_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(raw_path, index=False)
    
    logger.info(f"Dataset loaded: {df.shape} | Churn rate: {df['churn'].mean():.2%}")
    
    X_train, X_test, y_train, y_test, encoders, scaler = preprocess(df)
    
    models = {
        "logistic_regression": LogisticRegression(max_iter=1000, random_state=42),
        "random_forest": RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
        "xgboost": XGBClassifier(n_estimators=100, random_state=42, eval_metric="logloss", use_label_encoder=False)
    }
    
    results = {}
    best_auc, best_name, best_model = 0, None, None
    
    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred_proba = model.predict_proba(X_test)[:, 1]
        y_pred = model.predict(X_test)
        auc = roc_auc_score(y_test, y_pred_proba)
        report = classification_report(y_test, y_pred, output_dict=True)
        results[name] = {"roc_auc": auc, "report": report}
        logger.info(f"{name} — ROC-AUC: {auc:.4f}")
        
        if auc > best_auc:
            best_auc, best_name, best_model = auc, name, model
    
    # Save best model and preprocessing artifacts
    joblib.dump(best_model, MODELS_DIR / "best_model.pkl")
    joblib.dump(encoders, MODELS_DIR / "encoders.pkl")
    joblib.dump(scaler, MODELS_DIR / "scaler.pkl")
    logger.info(f"Best model: {best_name} (ROC-AUC: {best_auc:.4f}) saved to models/")
    
    # Print results summary
    print("\n" + "="*50)
    print("MODEL COMPARISON RESULTS")
    print("="*50)
    for name, res in results.items():
        print(f"  {name:25s} ROC-AUC: {res['roc_auc']:.4f}")
    print(f"\n  Best Model: {best_name}")
    
    return best_model, encoders, scaler, results


if __name__ == "__main__":
    train_and_evaluate()
