# 🔄 Customer Churn Analysis

## Problem Statement
Customer churn is one of the costliest problems for subscription-based businesses. This project builds a full ML pipeline to predict which customers are likely to churn, enabling proactive retention strategies.

## Approach
1. **Data Generation** – Realistic telecom customer dataset with 10,000+ records
2. **EDA** – Churn rate analysis, feature distributions, correlation heatmap
3. **Preprocessing** – Handle class imbalance (SMOTE), encode categoricals, scale numerics
4. **Modeling** – Logistic Regression, Random Forest, XGBoost comparison
5. **Evaluation** – ROC-AUC, Precision-Recall, Confusion Matrix, Feature Importance
6. **Insights** – Key churn drivers and business recommendations

## Tech Stack
| Layer | Technology |
|-------|------------|
| Language | Python 3.10+ |
| Data Processing | Pandas, NumPy |
| ML Framework | Scikit-learn, XGBoost |
| Imbalance Handling | imbalanced-learn (SMOTE) |
| Visualization | Matplotlib, Seaborn, Plotly |
| Model Persistence | joblib |

## Project Structure
```
customer-churn-analysis/
├── data/
│   ├── raw/                     # Raw customer dataset
│   └── processed/               # Encoded & scaled features
├── models/                      # Saved model artifacts (.pkl)
├── notebooks/
│   └── eda_and_modeling.ipynb
├── src/
│   ├── __init__.py
│   ├── data_generator.py         # Mock data generation
│   ├── preprocessing.py          # Feature engineering & encoding
│   ├── train.py                  # Model training & evaluation
│   └── predict.py                # Inference on new data
├── tests/
│   └── test_preprocessing.py
├── requirements.txt
├── .gitignore
└── README.md
```

## Getting Started
```bash
git clone https://github.com/Saideva0318/customer-churn-analysis.git
cd customer-churn-analysis
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt

# Train models and evaluate
python src/train.py

# Predict on new data
python src/predict.py --input data/raw/new_customers.csv
```

## Key Results (Mock Data Benchmark)
| Model | ROC-AUC | Precision | Recall | F1 |
|-------|---------|-----------|--------|----|
| Logistic Regression | 0.81 | 0.72 | 0.69 | 0.70 |
| Random Forest | 0.88 | 0.81 | 0.75 | 0.78 |
| XGBoost | 0.91 | 0.84 | 0.79 | 0.81 |

## Top Churn Drivers
1. Contract type (month-to-month vs annual)
2. Tenure (shorter = higher churn risk)
3. Monthly charges relative to service tier
4. Number of support tickets filed
5. Internet service type

## License
MIT
