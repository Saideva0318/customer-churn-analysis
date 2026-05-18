"""
Data Generator Module
Generates realistic mock telecom customer churn dataset.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def generate_churn_dataset(n_customers: int = 10000, seed: int = 42) -> pd.DataFrame:
    """
    Generate realistic telecom customer dataset with churn label.
    
    Args:
        n_customers: Number of customer records
        seed: Random seed
    
    Returns:
        DataFrame with customer features and churn label
    """
    np.random.seed(seed)
    
    customer_ids = [f"CUST_{i:06d}" for i in range(1, n_customers + 1)]
    tenure = np.random.exponential(scale=30, size=n_customers).clip(1, 72).astype(int)
    contract = np.random.choice(["Month-to-Month", "One Year", "Two Year"], n_customers, p=[0.55, 0.25, 0.20])
    internet_service = np.random.choice(["Fiber Optic", "DSL", "No Internet"], n_customers, p=[0.45, 0.35, 0.20])
    phone_service = np.random.choice(["Yes", "No"], n_customers, p=[0.90, 0.10])
    
    base_charge = np.where(internet_service == "Fiber Optic", 70, np.where(internet_service == "DSL", 45, 25))
    monthly_charges = (base_charge + np.random.normal(0, 10, n_customers)).clip(18, 120).round(2)
    total_charges = (monthly_charges * tenure + np.random.normal(0, 50, n_customers)).clip(0).round(2)
    
    support_tickets = np.random.poisson(lam=1.5, size=n_customers)
    senior_citizen = np.random.choice([0, 1], n_customers, p=[0.83, 0.17])
    partner = np.random.choice(["Yes", "No"], n_customers, p=[0.48, 0.52])
    dependents = np.random.choice(["Yes", "No"], n_customers, p=[0.30, 0.70])
    payment_method = np.random.choice(
        ["Electronic Check", "Mailed Check", "Bank Transfer", "Credit Card"],
        n_customers, p=[0.34, 0.22, 0.22, 0.22]
    )
    paperless_billing = np.random.choice(["Yes", "No"], n_customers, p=[0.59, 0.41])
    
    # Churn probability model (domain-inspired logic)
    churn_score = (
        0.3 * (contract == "Month-to-Month").astype(float)
        + 0.2 * (1 / (tenure + 1))
        + 0.15 * (internet_service == "Fiber Optic").astype(float)
        + 0.1 * (payment_method == "Electronic Check").astype(float)
        + 0.1 * (support_tickets / 10)
        + 0.05 * senior_citizen
        + np.random.normal(0, 0.05, n_customers)
    ).clip(0, 1)
    churn = (churn_score > 0.35).astype(int)
    
    df = pd.DataFrame({
        "customer_id": customer_ids,
        "tenure": tenure,
        "contract": contract,
        "internet_service": internet_service,
        "phone_service": phone_service,
        "monthly_charges": monthly_charges,
        "total_charges": total_charges,
        "support_tickets": support_tickets,
        "senior_citizen": senior_citizen,
        "partner": partner,
        "dependents": dependents,
        "payment_method": payment_method,
        "paperless_billing": paperless_billing,
        "churn": churn
    })
    
    churn_rate = churn.mean() * 100
    logger.info(f"Generated {n_customers} customer records | Churn Rate: {churn_rate:.1f}%")
    return df


if __name__ == "__main__":
    df = generate_churn_dataset()
    output_path = Path("data/raw/customers.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"Dataset saved to {output_path}")
    print(df["churn"].value_counts(normalize=True).mul(100).round(1).rename({0: "Retained", 1: "Churned"}))
