"""
Generate sample mobile money transaction data for testing
Creates a synthetic dataset similar to PaySim
"""

import pandas as pd
import numpy as np
from datetime import datetime

def generate_sample_data(n_samples=1000, fraud_rate=0.1):
    """
    Generate synthetic mobile money transactions.
    
    Args:
        n_samples: Number of transactions to generate
        fraud_rate: Proportion of fraudulent transactions
        
    Returns:
        DataFrame with synthetic transactions
    """
    np.random.seed(42)
    
    # Transaction types
    types = ['CASH-IN', 'CASH-OUT', 'PAYMENT', 'TRANSFER', 'DEBIT']
    
    data = {
        'step': np.random.randint(1, 744, n_samples),  # ~1 month in hours
        'type': np.random.choice(types, n_samples, p=[0.2, 0.3, 0.2, 0.25, 0.05]),
        'amount': np.random.exponential(scale=100000, size=n_samples),
        'nameOrig': [f'C{np.random.randint(1000000, 9999999)}' for _ in range(n_samples)],
        'oldbalanceOrg': np.random.uniform(0, 500000, n_samples),
        'newbalanceOrig': np.zeros(n_samples),
        'nameDest': [f'C{np.random.randint(1000000, 9999999)}' for _ in range(n_samples)],
        'oldbalanceDest': np.random.uniform(0, 500000, n_samples),
        'newbalanceDest': np.zeros(n_samples),
    }
    
    df = pd.DataFrame(data)
    
    # Calculate balances based on transaction type
    for i in range(n_samples):
        if df.loc[i, 'type'] in ['CASH-OUT', 'PAYMENT', 'TRANSFER', 'DEBIT']:
            # Money leaving origin account
            df.loc[i, 'newbalanceOrig'] = max(0, df.loc[i, 'oldbalanceOrg'] - df.loc[i, 'amount'])
            if df.loc[i, 'type'] == 'TRANSFER':
                # Money entering destination account
                df.loc[i, 'newbalanceDest'] = df.loc[i, 'oldbalanceDest'] + df.loc[i, 'amount']
            else:
                df.loc[i, 'newbalanceDest'] = df.loc[i, 'oldbalanceDest']
        elif df.loc[i, 'type'] == 'CASH-IN':
            # Money entering origin account
            df.loc[i, 'newbalanceOrig'] = df.loc[i, 'oldbalanceOrg'] + df.loc[i, 'amount']
            df.loc[i, 'newbalanceDest'] = df.loc[i, 'oldbalanceDest']
    
    # Add some fraudulent patterns
    n_fraud = int(n_samples * fraud_rate)
    fraud_indices = np.random.choice(n_samples, n_fraud, replace=False)
    
    df['isFraud'] = 0
    df['isFlaggedFraud'] = 0
    
    for idx in fraud_indices:
        df.loc[idx, 'isFraud'] = 1
        
        # Fraudulent patterns:
        # 1. Large amounts
        df.loc[idx, 'amount'] = np.random.uniform(200000, 1000000)
        
        # 2. Balance errors (money disappears)
        if df.loc[idx, 'type'] in ['TRANSFER', 'CASH-OUT']:
            df.loc[idx, 'newbalanceOrig'] = max(0, df.loc[idx, 'oldbalanceOrg'] - df.loc[idx, 'amount'] - np.random.uniform(0, 50000))
            df.loc[idx, 'newbalanceDest'] = 0  # Money doesn't arrive
            
        # 3. Flag very large transactions
        if df.loc[idx, 'amount'] > 500000:
            df.loc[idx, 'isFlaggedFraud'] = 1
    
    return df


if __name__ == "__main__":
    # Generate training data
    print("Generating training data...")
    df_train = generate_sample_data(n_samples=10000, fraud_rate=0.08)
    df_train.to_csv('train_data.csv', index=False)
    print(f"✓ Created train_data.csv with {len(df_train)} transactions")
    
    # Generate test data
    print("Generating test data...")
    df_test = generate_sample_data(n_samples=500, fraud_rate=0.12)
    df_test.to_csv('test_data.csv', index=False)
    print(f"✓ Created test_data.csv with {len(df_test)} transactions")
    
    # Generate small demo data
    print("Generating demo data...")
    df_demo = generate_sample_data(n_samples=50, fraud_rate=0.15)
    df_demo.to_csv('demo_data.csv', index=False)
    print(f"✓ Created demo_data.csv with {len(df_demo)} transactions")
    
    print("\n📊 Summary:")
    print(f"Training data: {len(df_train)} transactions, {df_train['isFraud'].sum()} fraudulent")
    print(f"Test data: {len(df_test)} transactions, {df_test['isFraud'].sum()} fraudulent")
    print(f"Demo data: {len(df_demo)} transactions, {df_demo['isFraud'].sum()} fraudulent")
