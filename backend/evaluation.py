"""
Evaluation module for fraud detection models.
Computes metrics, confusion matrices, and hybrid ensemble logic.
"""

import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from typing import Dict, Any, List, Tuple


def convert_predictions_to_binary(predictions: np.ndarray) -> np.ndarray:
    """
    Convert model predictions to binary format.
    Isolation Forest and LOF use -1 for outliers (fraud), 1 for inliers (valid).
    
    Args:
        predictions: Array of model predictions (-1 or 1)
        
    Returns:
        Binary array (1 for fraud, 0 for valid)
    """
    return (predictions == -1).astype(int)


def generate_pseudo_labels(model_results: Dict[str, Any], 
                           consensus_threshold: int = 2) -> np.ndarray:
    """
    Generate pseudo-labels using model consensus.
    A transaction is labeled as fraud if at least `consensus_threshold` models agree.
    
    Args:
        model_results: Dictionary containing all model outputs
        consensus_threshold: Minimum number of models that must agree for fraud label
        
    Returns:
        Binary pseudo-labels (1 for fraud, 0 for valid)
    """
    if_binary = convert_predictions_to_binary(model_results['isolation_forest']['predictions'])
    lof_binary = convert_predictions_to_binary(model_results['lof']['predictions'])
    ae_binary = convert_predictions_to_binary(model_results['autoencoder']['predictions'])
    
    # Count votes for fraud
    fraud_votes = if_binary + lof_binary + ae_binary
    
    # Apply consensus threshold
    pseudo_labels = (fraud_votes >= consensus_threshold).astype(int)
    
    return pseudo_labels


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with accuracy, precision, recall, and F1 score
    """
    # Handle case where all predictions are the same class
    if len(np.unique(y_pred)) == 1 or len(np.unique(y_true)) == 1:
        return {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': 0.0,
            'recall': 0.0,
            'f1': 0.0
        }
    
    return {
        'accuracy': round(accuracy_score(y_true, y_pred), 4),
        'precision': round(precision_score(y_true, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_true, y_pred, zero_division=0), 4),
        'f1': round(f1_score(y_true, y_pred, zero_division=0), 4)
    }


def compute_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, int]:
    """
    Compute and format confusion matrix.
    
    Args:
        y_true: Ground truth labels
        y_pred: Predicted labels
        
    Returns:
        Dictionary with TN, FP, FN, TP values
    """
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    
    return {
        'TN': int(cm[0, 0]),
        'FP': int(cm[0, 1]),
        'FN': int(cm[1, 0]),
        'TP': int(cm[1, 1])
    }


def compute_fraud_probability(if_score: float, lof_score: float, 
                               ae_score: float, 
                               weights: Tuple[float, float, float] = (0.35, 0.35, 0.30)) -> float:
    """
    Compute fraud probability using weighted ensemble.
    
    Args:
        if_score: Isolation Forest anomaly score
        lof_score: LOF anomaly score
        ae_score: Autoencoder reconstruction error (normalized)
        weights: Tuple of weights for (IF, LOF, AE)
        
    Returns:
        Fraud probability (0-1)
    """
    weighted_score = (
        weights[0] * if_score +
        weights[1] * lof_score +
        weights[2] * ae_score
    )
    return round(weighted_score, 4)


def hybrid_ensemble(model_results: Dict[str, Any], 
                    threshold: float = 0.5,
                    weights: Tuple[float, float, float] = (0.35, 0.35, 0.30)) -> Dict[str, Any]:
    """
    Compute hybrid ensemble predictions using weighted voting.
    
    Args:
        model_results: Dictionary containing all model outputs
        threshold: Threshold for fraud classification
        weights: Tuple of weights for (IF, LOF, AE)
        
    Returns:
        Dictionary with ensemble results
    """
    if_scores = model_results['isolation_forest']['scores']
    lof_scores = model_results['lof']['scores']
    ae_scores = model_results['autoencoder']['errors']
    
    n_samples = len(if_scores)
    
    # Compute weighted ensemble scores
    ensemble_scores = np.zeros(n_samples)
    for i in range(n_samples):
        ensemble_scores[i] = compute_fraud_probability(
            if_scores[i], lof_scores[i], ae_scores[i], weights
        )
    
    # Generate predictions based on threshold
    predictions = (ensemble_scores >= threshold).astype(int)
    
    # Generate result labels
    results = ['Fraud' if p == 1 else 'Valid' for p in predictions]
    
    # Compute overall ensemble score (average of individual accuracies weighted)
    overall_score = round(np.mean(ensemble_scores[predictions == 1]) if np.any(predictions == 1) 
                          else 0.0, 4)
    
    return {
        'scores': ensemble_scores,
        'predictions': predictions,
        'results': results,
        'overall_score': overall_score,
        'threshold': threshold
    }


def evaluate_all_models(model_results: Dict[str, Any], 
                        original_labels: np.ndarray = None) -> Dict[str, Any]:
    """
    Evaluate all models and compute comprehensive metrics.
    
    Args:
        model_results: Dictionary containing all model outputs
        original_labels: Optional ground truth labels (if available in dataset)
        
    Returns:
        Dictionary with evaluation results for all models
    """
    # Generate pseudo-labels for evaluation (using consensus)
    pseudo_labels = generate_pseudo_labels(model_results)
    
    # Convert model predictions to binary
    if_binary = convert_predictions_to_binary(model_results['isolation_forest']['predictions'])
    lof_binary = convert_predictions_to_binary(model_results['lof']['predictions'])
    ae_binary = convert_predictions_to_binary(model_results['autoencoder']['predictions'])
    
    # Use original labels if available, otherwise use pseudo-labels
    y_true = original_labels if original_labels is not None else pseudo_labels
    
    # Compute metrics for each model
    if_metrics = compute_metrics(y_true, if_binary)
    lof_metrics = compute_metrics(y_true, lof_binary)
    
    # Compute hybrid ensemble
    ensemble_results = hybrid_ensemble(model_results)
    hybrid_metrics = compute_metrics(y_true, ensemble_results['predictions'])
    
    # Compute confusion matrices
    if_cm = compute_confusion_matrix(y_true, if_binary)
    lof_cm = compute_confusion_matrix(y_true, lof_binary)
    ae_cm = compute_confusion_matrix(y_true, ae_binary)
    
    return {
        'performance': {
            'isolation_forest': if_metrics,
            'lof': lof_metrics,
            'autoencoder': {
                'reconstruction_error': round(float(np.mean(model_results['autoencoder']['errors'])), 4),
                'threshold': round(model_results['autoencoder']['threshold'], 4),
                'loss': round(model_results['autoencoder']['loss'], 4)
            },
            'hybrid': {
                'score': round(np.mean(ensemble_results['scores']), 4),
                **hybrid_metrics
            }
        },
        'confusion_matrices': {
            'IF': if_cm,
            'LOF': lof_cm,
            'AE': ae_cm
        },
        'ensemble_results': ensemble_results,
        'pseudo_labels': pseudo_labels
    }


def get_anomaly_scores_over_time(model_results: Dict[str, Any], 
                                  step_values: np.ndarray,
                                  sample_size: int = 100) -> List[Dict[str, Any]]:
    """
    Get anomaly scores aggregated over time steps.
    
    Args:
        model_results: Dictionary containing all model outputs
        step_values: Array of time step values
        sample_size: Number of time points to return
        
    Returns:
        List of dictionaries with time-based anomaly data
    """
    if_scores = model_results['isolation_forest']['scores']
    lof_scores = model_results['lof']['scores']
    ae_scores = model_results['autoencoder']['errors']
    
    # Get unique steps and sample if too many
    unique_steps = np.unique(step_values)
    if len(unique_steps) > sample_size:
        step_indices = np.linspace(0, len(unique_steps) - 1, sample_size, dtype=int)
        sampled_steps = unique_steps[step_indices]
    else:
        sampled_steps = unique_steps
    
    time_series_data = []
    for step in sampled_steps:
        mask = step_values == step
        time_series_data.append({
            'step': int(step),
            'if_avg': round(float(np.mean(if_scores[mask])), 4),
            'lof_avg': round(float(np.mean(lof_scores[mask])), 4),
            'ae_avg': round(float(np.mean(ae_scores[mask])), 4)
        })
    
    return time_series_data
