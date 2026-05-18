"""Model Training, Evaluation, and Comparison for Customer Churn."""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    roc_auc_score, classification_report, confusion_matrix,
    roc_curve, precision_recall_curve
)
from xgboost import XGBClassifier
import joblib
import os
import logging

from preprocessor import ChurnPreprocessor

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

MODELS = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1),
    'XGBoost': XGBClassifier(n_estimators=200, learning_rate=0.05, max_depth=6,
                              eval_metric='logloss', random_state=42)
}


def evaluate_model(model, X_test, y_test, model_name: str) -> dict:
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, y_prob)
    report = classification_report(y_test, y_pred, output_dict=True)
    logger.info(f"{model_name} | ROC-AUC: {auc:.4f} | Precision: {report['1']['precision']:.3f} | Recall: {report['1']['recall']:.3f}")
    return {'model': model_name, 'auc': auc, 'report': report, 'y_pred': y_pred, 'y_prob': y_prob}


def plot_feature_importance(model, feature_cols, model_name, output_dir='reports'):
    os.makedirs(output_dir, exist_ok=True)
    if hasattr(model, 'feature_importances_'):
        importance = pd.Series(model.feature_importances_, index=feature_cols)
        top15 = importance.nlargest(15)
        plt.figure(figsize=(10, 6))
        top15.sort_values().plot(kind='barh', color='steelblue')
        plt.title(f'Top 15 Feature Importances - {model_name}')
        plt.tight_layout()
        plt.savefig(f'{output_dir}/feature_importance_{model_name.replace(" ", "_")}.png', dpi=150)
        plt.close()


def plot_confusion_matrix(y_test, y_pred, model_name, output_dir='reports'):
    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=['No Churn', 'Churn'], yticklabels=['No Churn', 'Churn'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/confusion_matrix_{model_name.replace(" ", "_")}.png', dpi=150)
    plt.close()


if __name__ == '__main__':
    df = pd.read_csv('data/raw/customer_churn.csv')
    processor = ChurnPreprocessor()
    X_train, X_test, y_train, y_test = processor.fit_transform(df)
    feature_cols = processor.feature_cols

    results = []
    for name, model in MODELS.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        result = evaluate_model(model, X_test, y_test, name)
        results.append(result)
        plot_feature_importance(model, feature_cols, name)
        plot_confusion_matrix(y_test, result['y_pred'], name)
        joblib.dump(model, f"models/{name.replace(' ', '_')}.pkl")

    # Summary
    summary = pd.DataFrame([{'Model': r['model'], 'ROC-AUC': r['auc']} for r in results])
    print("\n=== Model Comparison ===")
    print(summary.sort_values('ROC-AUC', ascending=False).to_string(index=False))
    print("\n✅ All models trained and saved to models/")
