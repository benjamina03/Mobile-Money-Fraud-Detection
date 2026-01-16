"""
Preprocessing module for Mobile Money Fraud Detection
Handles data loading, feature engineering, and scaling
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from typing import Tuple, Optional


def load_paysim_data(filepath: str, drop_fraud_labels: bool = True) -> pd.DataFrame:
    """
    Load PaySim dataset from CSV file.
    
    Args:
        filepath: Path to the CSV file
        drop_fraud_labels: If True, drops isFraud and isFlaggedFraud columns
        
    Returns:
        DataFrame with loaded data
    """
    df = pd.read_csv(filepath)
    
    # Drop fraud labels for unsupervised learning
    if drop_fraud_labels:
        fraud_cols = ['isFraud', 'isFlaggedFraud']
        df = df.drop(columns=[col for col in fraud_cols if col in df.columns], errors='ignore')
    
    return df


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create derived features for fraud detection.
    
    Features created:
    - Transaction Velocity: Number of transactions by nameOrig in the last hour (step)
    - Balance Error: Difference between expected and actual new balance
    - Type Encoding: One-hot encoding for transaction types
    
    Args:
        df: DataFrame with raw transaction data
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # 1. Transaction Velocity
    # Count transactions by nameOrig within each step (representing hourly windows)
    velocity = df.groupby(['step', 'nameOrig']).size().reset_index(name='transaction_velocity')
    df = df.merge(velocity, on=['step', 'nameOrig'], how='left')
    df['transaction_velocity'] = df['transaction_velocity'].fillna(1)
    
    # 2. Balance Error
    # Expected new balance = old balance - amount (for sender)
    df['balance_error_orig'] = (df['oldbalanceOrg'] - df['amount']) - df['newbalanceOrig']
    
    # For destination accounts (expected new balance = old balance + amount)
    df['balance_error_dest'] = (df['oldbalanceDest'] + df['amount']) - df['newbalanceDest']
    
    # 3. Type Encoding (One-Hot) - ensure all categories are present
    # Define all possible transaction types
    all_types = ['CASH-IN', 'CASH-OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']
    df = pd.get_dummies(df, columns=['type'], prefix='type')
    
    # Ensure all type columns exist (add missing ones with 0 values)
    for trans_type in all_types:
        col_name = f'type_{trans_type}'
        if col_name not in df.columns:
            df[col_name] = 0
    
    # Additional useful features
    df['amount_to_oldbalance_ratio'] = df['amount'] / (df['oldbalanceOrg'] + 1)
    df['dest_balance_change'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    return df


def select_features(df: pd.DataFrame, feature_columns: Optional[list] = None) -> pd.DataFrame:
    """
    Select relevant features for model training.
    
    Args:
        df: DataFrame with all features
        feature_columns: List of column names to select. If None, uses default set.
        
    Returns:
        DataFrame with selected features only
    """
    if feature_columns is None:
        # Default feature set (order matters for consistency)
        feature_columns = [
            'step', 'amount', 'oldbalanceOrg', 'newbalanceOrig',
            'oldbalanceDest', 'newbalanceDest', 'transaction_velocity',
            'balance_error_orig', 'balance_error_dest',
            'amount_to_oldbalance_ratio', 'dest_balance_change',
            # Type columns in fixed order
            'type_CASH-IN', 'type_CASH-OUT', 'type_DEBIT', 'type_PAYMENT', 'type_TRANSFER'
        ]
    
    # Only select columns that exist in the dataframe
    available_features = [col for col in feature_columns if col in df.columns]
    
    return df[available_features]


def scale_features(df: pd.DataFrame, scaler=None, scaler_type: str = 'standard') -> Tuple[np.ndarray, object]:
    """
    Scale features using StandardScaler or MinMaxScaler.
    
    Args:
        df: DataFrame with features to scale
        scaler: Pre-fitted scaler object. If None, a new one is created and fitted.
        scaler_type: Type of scaler ('standard' or 'minmax')
        
    Returns:
        Tuple of (scaled_data, fitted_scaler)
    """
    if scaler is None:
        if scaler_type == 'standard':
            scaler = StandardScaler()
        elif scaler_type == 'minmax':
            scaler = MinMaxScaler()
        else:
            raise ValueError(f"Unknown scaler_type: {scaler_type}")
        
        scaled_data = scaler.fit_transform(df)
    else:
        scaled_data = scaler.transform(df)
    
    return scaled_data, scaler


def preprocess_data(df: pd.DataFrame, scaler=None, feature_columns: Optional[list] = None) -> Tuple[np.ndarray, object, pd.DataFrame]:
    """
    Complete preprocessing pipeline: feature engineering -> selection -> scaling.
    
    Args:
        df: Raw DataFrame
        scaler: Pre-fitted scaler (for inference). If None, creates new one.
        feature_columns: Specific features to use
        
    Returns:
        Tuple of (scaled_data, scaler, processed_df)
    """
    # Engineer features
    df_featured = engineer_features(df)
    
    # Select features
    df_selected = select_features(df_featured, feature_columns)
    
    # Scale features
    scaled_data, fitted_scaler = scale_features(df_selected, scaler)
    
    return scaled_data, fitted_scaler, df_featured


def preprocess_single_transaction(transaction: pd.Series, scaler, transaction_history: pd.DataFrame = None) -> np.ndarray:
    """
    Preprocess a single transaction for real-time prediction.
    
    Args:
        transaction: Series representing a single transaction
        scaler: Pre-fitted scaler
        transaction_history: Historical transactions for velocity calculation
        
    Returns:
        Scaled feature array for the transaction
    """
    # Convert to DataFrame
    df = pd.DataFrame([transaction])
    
    # If we have transaction history, merge to calculate velocity
    if transaction_history is not None and len(transaction_history) > 0:
        # Calculate velocity from history
        relevant_history = transaction_history[
            (transaction_history['step'] == transaction['step']) &
            (transaction_history['nameOrig'] == transaction['nameOrig'])
        ]
        velocity = len(relevant_history) + 1  # +1 for current transaction
    else:
        velocity = 1
    
    # Engineer features
    df_featured = engineer_features(df)
    df_featured['transaction_velocity'] = velocity
    
    # Select features
    df_selected = select_features(df_featured)
    
    # Scale
    scaled_data, _ = scale_features(df_selected, scaler)
    
    return scaled_data
