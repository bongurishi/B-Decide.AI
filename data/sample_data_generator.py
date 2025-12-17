"""
Sample Data Generator for B-Decide AI
Creates a synthetic dataset for testing if IBM Telco dataset is not available
"""

import pandas as pd
import numpy as np
from typing import Tuple


def generate_sample_churn_data(n_samples: int = 1000, random_state: int = 42) -> pd.DataFrame:
    """
    Generate synthetic customer churn data for testing.
    
    Args:
        n_samples: Number of samples to generate
        random_state: Random seed for reproducibility
        
    Returns:
        DataFrame with synthetic customer data
    """
    np.random.seed(random_state)
    
    # Generate customer IDs
    customer_ids = [f"CUST_{i:05d}" for i in range(n_samples)]
    
    # Generate features
    data = {
        'customerID': customer_ids,
        
        # Demographics
        'gender': np.random.choice(['Male', 'Female'], n_samples),
        'SeniorCitizen': np.random.choice([0, 1], n_samples, p=[0.85, 0.15]),
        'Partner': np.random.choice(['Yes', 'No'], n_samples, p=[0.5, 0.5]),
        'Dependents': np.random.choice(['Yes', 'No'], n_samples, p=[0.3, 0.7]),
        
        # Services
        'tenure': np.random.exponential(24, n_samples).astype(int).clip(1, 72),
        'PhoneService': np.random.choice(['Yes', 'No'], n_samples, p=[0.9, 0.1]),
        'MultipleLines': np.random.choice(['Yes', 'No', 'No phone service'], n_samples, p=[0.4, 0.5, 0.1]),
        'InternetService': np.random.choice(['DSL', 'Fiber optic', 'No'], n_samples, p=[0.35, 0.45, 0.2]),
        'OnlineSecurity': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'OnlineBackup': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.35, 0.45, 0.2]),
        'DeviceProtection': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.35, 0.45, 0.2]),
        'TechSupport': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.3, 0.5, 0.2]),
        'StreamingTV': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
        'StreamingMovies': np.random.choice(['Yes', 'No', 'No internet service'], n_samples, p=[0.4, 0.4, 0.2]),
        
        # Billing
        'Contract': np.random.choice(['Month-to-month', 'One year', 'Two year'], n_samples, p=[0.55, 0.25, 0.2]),
        'PaperlessBilling': np.random.choice(['Yes', 'No'], n_samples, p=[0.6, 0.4]),
        'PaymentMethod': np.random.choice(
            ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'],
            n_samples,
            p=[0.35, 0.15, 0.25, 0.25]
        ),
    }
    
    df = pd.DataFrame(data)
    
    # Generate monetary features based on tenure and services
    base_charge = 20
    service_charges = (
        (df['InternetService'] == 'Fiber optic') * 30 +
        (df['InternetService'] == 'DSL') * 20 +
        (df['PhoneService'] == 'Yes') * 15 +
        (df['StreamingTV'] == 'Yes') * 10 +
        (df['StreamingMovies'] == 'Yes') * 10
    )
    
    df['MonthlyCharges'] = (base_charge + service_charges + np.random.normal(0, 5, n_samples)).clip(18, 120).round(2)
    df['TotalCharges'] = (df['MonthlyCharges'] * df['tenure'] + np.random.normal(0, 100, n_samples)).clip(0).round(2)
    
    # Generate churn based on logical rules
    churn_probability = (
        0.05 +  # Base churn rate
        (df['tenure'] < 6) * 0.4 +  # New customers more likely to churn
        (df['Contract'] == 'Month-to-month') * 0.3 +  # Month-to-month more likely
        (df['MonthlyCharges'] > 80) * 0.2 +  # High charges increase churn
        (df['TechSupport'] == 'No') * 0.15 +  # No tech support increases churn
        (df['OnlineSecurity'] == 'No') * 0.1  # No security increases churn
    ).clip(0, 1)
    
    df['Churn'] = (np.random.random(n_samples) < churn_probability).astype(int)
    df['Churn'] = df['Churn'].map({0: 'No', 1: 'Yes'})
    
    return df


def save_sample_data(output_path: str = 'data/raw/telco_churn.csv'):
    """
    Generate and save sample data to CSV.
    
    Args:
        output_path: Path to save the CSV file
    """
    print("Generating sample customer churn data...")
    df = generate_sample_churn_data(n_samples=2000)
    
    # Create directory if it doesn't exist
    import os
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    # Save to CSV
    df.to_csv(output_path, index=False)
    print(f"✓ Sample data saved to {output_path}")
    print(f"  Total customers: {len(df)}")
    print(f"  Churn rate: {(df['Churn'] == 'Yes').mean():.2%}")
    print(f"  Columns: {len(df.columns)}")
    
    return df


if __name__ == "__main__":
    # Generate sample data
    df = save_sample_data()
    
    # Display summary
    print("\nSample Data Summary:")
    print(df.head())
    print("\nChurn Distribution:")
    print(df['Churn'].value_counts())

