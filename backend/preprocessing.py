"""
Preprocessing module for PaySim dataset.
Handles data cleaning, encoding, and feature scaling.
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from typing import Tuple

# Transaction types in PaySim dataset
TRANSACTION_TYPES = ['CASH_IN', 'CASH_OUT', 'DEBIT', 'PAYMENT', 'TRANSFER']

# Small epsilon value for division by zero protection
EPSILON = 1e-10


def load_and_clean_dataset(file) -> pd.DataFrame:
    """
    Load CSV file and perform initial cleaning.
    
    Args:
        file: Uploaded file object or file path
        
    Returns:
        Cleaned pandas DataFrame
    """
    # Read CSV file
    df = pd.read_csv(file)
    
    # Remove null values
    df = df.dropna()
    
    # Remove duplicate rows
    df = df.drop_duplicates()
    
    # Reset index
    df = df.reset_index(drop=True)
    
    return df


def encode_categorical_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical features (transaction type).
    
    Args:
        df: Input DataFrame
        
    Returns:
        DataFrame with encoded categorical features
    """
    df = df.copy()
    
    # Encode transaction type
    type_encoder = LabelEncoder()
    if 'type' in df.columns:
        df['type_encoded'] = type_encoder.fit_transform(df['type'])
    
    return df


def extract_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract and engineer features for fraud detection.
    
    Args:
        df: Input DataFrame with encoded features
        
    Returns:
        DataFrame with engineered features
    """
    df = df.copy()
    
    # Calculate balance differences
    df['orig_balance_diff'] = df['oldbalanceOrg'] - df['newbalanceOrig']
    df['dest_balance_diff'] = df['newbalanceDest'] - df['oldbalanceDest']
    
    # Calculate ratio features (with division by zero handling using EPSILON)
    df['amount_orig_ratio'] = df['amount'] / (df['oldbalanceOrg'] + EPSILON)
    df['amount_dest_ratio'] = df['amount'] / (df['oldbalanceDest'] + EPSILON)
    
    # Flag for zero balances (suspicious patterns)
    df['zero_orig_new_balance'] = (df['newbalanceOrig'] == 0).astype(int)
    df['zero_dest_old_balance'] = (df['oldbalanceDest'] == 0).astype(int)
    
    # Transaction matching (amount equals balance difference)
    df['amount_matches_diff'] = (
        np.abs(df['amount'] - df['orig_balance_diff']) < 1
    ).astype(int)
    
    return df


def scale_features(df: pd.DataFrame, feature_columns: list) -> Tuple[np.ndarray, StandardScaler]:
    """
    Scale numerical features using StandardScaler.
    
    Args:
        df: Input DataFrame
        feature_columns: List of column names to scale
        
    Returns:
        Tuple of (scaled_features_array, fitted_scaler)
    """
    scaler = StandardScaler()
    
    # Select only the feature columns
    features = df[feature_columns].values
    
    # Fit and transform
    scaled_features = scaler.fit_transform(features)
    
    return scaled_features, scaler


def generate_feature_matrix(df: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame, list]:
    """
    Generate the complete feature matrix for model training.
    
    Args:
        df: Raw input DataFrame
        
    Returns:
        Tuple of (scaled_feature_matrix, processed_dataframe, feature_column_names)
    """
    # Encode categorical features
    df_encoded = encode_categorical_features(df)
    
    # Extract engineered features
    df_features = extract_features(df_encoded)
    
    # Define feature columns for model training
    feature_columns = [
        'step',
        'type_encoded',
        'amount',
        'oldbalanceOrg',
        'newbalanceOrig',
        'oldbalanceDest',
        'newbalanceDest',
        'orig_balance_diff',
        'dest_balance_diff',
        'amount_orig_ratio',
        'amount_dest_ratio',
        'zero_orig_new_balance',
        'zero_dest_old_balance',
        'amount_matches_diff'
    ]
    
    # Scale features
    scaled_features, scaler = scale_features(df_features, feature_columns)
    
    return scaled_features, df_features, feature_columns


def preprocess_dataset(file) -> Tuple[np.ndarray, pd.DataFrame, list]:
    """
    Complete preprocessing pipeline for PaySim dataset.
    
    Args:
        file: Uploaded file object or file path
        
    Returns:
        Tuple of (scaled_feature_matrix, processed_dataframe, feature_columns)
    """
    # Load and clean
    df = load_and_clean_dataset(file)
    
    # Generate feature matrix
    return generate_feature_matrix(df)


def get_dataset_summary(df: pd.DataFrame, fraud_predictions: np.ndarray) -> dict:
    """
    Generate dataset summary statistics.
    
    Args:
        df: Processed DataFrame
        fraud_predictions: Array of fraud predictions (1 = fraud, 0 = valid)
        
    Returns:
        Dictionary with dataset summary
    """
    total = len(df)
    fraudulent = int(np.sum(fraud_predictions))
    valid = total - fraudulent
    fraud_rate = fraudulent / total if total > 0 else 0
    
    return {
        "total_transactions": total,
        "fraudulent_transactions": fraudulent,
        "valid_transactions": valid,
        "fraud_rate": f"{fraud_rate * 100:.2f}%"
    }
