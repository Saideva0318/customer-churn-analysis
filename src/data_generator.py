"""Synthetic Customer Churn Dataset Generator."""

import pandas as pd
import numpy as np
import os
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

SEED = 42
N_CUSTOMERS = 7000


def generate_churn_data(n: int = N_CUSTOMERS) -> pd.DataFrame:
    np.random.seed(SEED)
    logger.info(f"Generating {n} customer records...")

    tenure = np.random.randint(1, 73, n)  # months 1-72
    monthly_charges = np.round(np.random.uniform(18.95, 118.75, n), 2)
    total_charges = np.round(tenure * monthly_charges * np.random.uniform(0.9, 1.1, n), 2)

    contract = np.random.choice(['Month-to-month', 'One year', 'Two year'], n, p=[0.55, 0.25, 0.20])
    internet_service = np.random.choice(['DSL', 'Fiber optic', 'No'], n, p=[0.34, 0.44, 0.22])
    tech_support = np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.29, 0.49, 0.22])
    payment_method = np.random.choice(
        ['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card'], n
    )
    senior_citizen = np.random.choice([0, 1], n, p=[0.84, 0.16])
    partner = np.random.choice(['Yes', 'No'], n)
    dependents = np.random.choice(['Yes', 'No'], n, p=[0.30, 0.70])
    phone_service = np.random.choice(['Yes', 'No'], n, p=[0.90, 0.10])
    multiple_lines = np.where(phone_service == 'Yes',
                              np.random.choice(['Yes', 'No'], n),
                              'No phone service')
    paperless_billing = np.random.choice(['Yes', 'No'], n, p=[0.60, 0.40])
    online_security = np.random.choice(['Yes', 'No', 'No internet service'], n, p=[0.28, 0.50, 0.22])

    # Churn probability based on realistic drivers
    churn_prob = (
        0.05
        + 0.25 * (contract == 'Month-to-month').astype(float)
        - 0.10 * (contract == 'Two year').astype(float)
        + 0.15 * (internet_service == 'Fiber optic').astype(float)
        - 0.08 * (tech_support == 'Yes').astype(float)
        + 0.01 * (monthly_charges / 100)
        - 0.003 * (tenure / 12)
        + 0.05 * (senior_citizen).astype(float)
        + np.random.normal(0, 0.05, n)
    )
    churn_prob = np.clip(churn_prob, 0.01, 0.95)
    churn = (np.random.uniform(0, 1, n) < churn_prob).astype(int)

    df = pd.DataFrame({
        'customer_id': [f'CUST-{str(i).zfill(5)}' for i in range(1, n + 1)],
        'senior_citizen': senior_citizen,
        'partner': partner,
        'dependents': dependents,
        'tenure': tenure,
        'phone_service': phone_service,
        'multiple_lines': multiple_lines,
        'internet_service': internet_service,
        'online_security': online_security,
        'tech_support': tech_support,
        'contract': contract,
        'paperless_billing': paperless_billing,
        'payment_method': payment_method,
        'monthly_charges': monthly_charges,
        'total_charges': total_charges,
        'churn': churn
    })

    churn_rate = churn.mean() * 100
    logger.info(f"Generated {n} records. Churn rate: {churn_rate:.1f}%")
    return df


if __name__ == '__main__':
    df = generate_churn_data()
    os.makedirs('data/raw', exist_ok=True)
    df.to_csv('data/raw/customer_churn.csv', index=False)
    print(f"\n✅ Data saved. Shape: {df.shape}")
    print(f"Churn Rate: {df['churn'].mean()*100:.1f}%")
    print(df.head())
