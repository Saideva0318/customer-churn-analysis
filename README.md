# 🔄 Customer Churn Analysis

## Problem Statement
Subscription-based businesses lose significant revenue when customers churn. This project builds a machine learning pipeline to predict which customers are at risk of churning, enabling proactive retention strategies.

## Approach
1. **Data Generation** — Synthetic telecom customer dataset with behavioral and demographic features
2. **EDA** — Churn rate analysis, feature distributions, correlation analysis
3. **Preprocessing** — Encoding, scaling, SMOTE for class imbalance
4. **Modeling** — Logistic Regression, Random Forest, XGBoost with cross-validation
5. **Evaluation** — ROC-AUC, Precision-Recall, Confusion Matrix, Feature Importance
6. **Insights** — Top churn drivers and retention recommendations

## Tech Stack
| Tool | Purpose |
|------|---------|
| Python 3.10+ | Core language |
| Pandas / NumPy | Data manipulation |
| Scikit-learn | ML models & preprocessing |
| XGBoost | Gradient boosting classifier |
| Imbalanced-learn | SMOTE oversampling |
| Matplotlib / Seaborn | Visualizations |
| Plotly | Interactive charts |
| Jupyter Notebook | Analysis & reporting |

## Project Structure
```
customer-churn-analysis/
├── data/
│   ├── raw/               # Raw customer dataset
│   └── processed/         # Feature-engineered data
├── notebooks/
│   ├── 01_eda.ipynb       # Exploratory Analysis
│   └── 02_modeling.ipynb  # Model Training & Evaluation
├── src/
│   ├── data_generator.py  # Synthetic data creation
│   ├── preprocessor.py    # Feature engineering pipeline
│   ├── model_trainer.py   # Model training & evaluation
│   └── predictor.py       # Inference on new data
├── models/                # Saved model artifacts
├── reports/               # Evaluation reports & charts
├── requirements.txt
└── README.md
```

## Getting Started
```bash
git clone https://github.com/Saideva0318/customer-churn-analysis.git
cd customer-churn-analysis
pip install -r requirements.txt

# Generate data
python src/data_generator.py

# Preprocess
python src/preprocessor.py

# Train models
python src/model_trainer.py

# Predict on new data
python src/predictor.py
```

## Results
- Best Model: XGBoost (ROC-AUC ~0.89)
- Top Churn Factors: Contract type, tenure, monthly charges, tech support usage
- Actionable: Customers with month-to-month contracts + high charges are highest risk
