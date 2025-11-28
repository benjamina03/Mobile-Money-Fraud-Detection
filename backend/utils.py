"""
Utility functions for file validation, dataset checking, and JSON formatting.
"""

import pandas as pd
from typing import Tuple, Dict, Any, List

# Required columns for PaySim dataset format
REQUIRED_COLUMNS = ['step', 'type', 'amount', 'nameOrig', 'oldbalanceOrg', 
                    'newbalanceOrig', 'nameDest', 'oldbalanceDest', 'newbalanceDest']


def validate_file(file) -> Tuple[bool, str]:
    """
    Validate uploaded file.
    
    Args:
        file: Uploaded file object
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    if file is None:
        return False, "No file uploaded"
    
    if not file.filename:
        return False, "No file selected"
    
    if not file.filename.lower().endswith('.csv'):
        return False, "File must be a CSV file"
    
    return True, ""


def validate_dataset_columns(df: pd.DataFrame) -> Tuple[bool, str]:
    """
    Check if the dataset has the required columns for PaySim format.
    
    Args:
        df: Pandas DataFrame
        
    Returns:
        Tuple of (is_valid, error_message)
    """
    missing_columns = [col for col in REQUIRED_COLUMNS if col not in df.columns]
    
    if missing_columns:
        return False, f"Missing required columns: {', '.join(missing_columns)}"
    
    return True, ""


def format_percentage(value: float) -> str:
    """
    Format a decimal value as a percentage string.
    
    Args:
        value: Decimal value (e.g., 0.012)
        
    Returns:
        Formatted percentage string (e.g., "1.2%")
    """
    return f"{value * 100:.2f}%"


def format_metrics(metrics: Dict[str, float]) -> Dict[str, Any]:
    """
    Format model metrics for JSON response.
    
    Args:
        metrics: Dictionary of metric names and values
        
    Returns:
        Formatted metrics dictionary
    """
    formatted = {}
    for key, value in metrics.items():
        if isinstance(value, float):
            formatted[key] = round(value, 4)
        else:
            formatted[key] = value
    return formatted


def format_confusion_matrix(matrix: List[List[int]]) -> Dict[str, int]:
    """
    Format confusion matrix for JSON response.
    
    Args:
        matrix: 2x2 confusion matrix as nested list
        
    Returns:
        Dictionary with TP, TN, FP, FN values
    """
    return {
        "TN": int(matrix[0][0]),
        "FP": int(matrix[0][1]),
        "FN": int(matrix[1][0]),
        "TP": int(matrix[1][1])
    }


def format_flagged_transactions(df: pd.DataFrame, 
                                 if_scores: List[float],
                                 lof_scores: List[float],
                                 ae_scores: List[float],
                                 final_results: List[str],
                                 limit: int = 100) -> List[Dict[str, Any]]:
    """
    Format flagged transactions for JSON response.
    
    Args:
        df: Original DataFrame
        if_scores: Isolation Forest anomaly scores
        lof_scores: LOF anomaly scores
        ae_scores: Autoencoder reconstruction errors (normalized)
        final_results: Final hybrid ensemble results
        limit: Maximum number of flagged transactions to return
        
    Returns:
        List of flagged transaction dictionaries
    """
    flagged = []
    
    for i, result in enumerate(final_results):
        if result == "Fraud" and len(flagged) < limit:
            transaction = {
                "id": i,
                "amount": round(float(df.iloc[i]['amount']), 2),
                "if_score": round(float(if_scores[i]), 4),
                "lof_score": round(float(lof_scores[i]), 4),
                "ae_score": round(float(ae_scores[i]), 4),
                "final_result": result
            }
            flagged.append(transaction)
    
    return flagged


def create_error_response(message: str, status_code: int = 400) -> Tuple[Dict[str, Any], int]:
    """
    Create a standardized error response.
    
    Args:
        message: Error message
        status_code: HTTP status code
        
    Returns:
        Tuple of (error_dict, status_code)
    """
    return {"error": message, "success": False}, status_code


def create_success_response(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Create a standardized success response.
    
    Args:
        data: Response data dictionary
        
    Returns:
        Success response dictionary
    """
    data["success"] = True
    return data
