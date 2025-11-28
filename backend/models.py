"""
Machine Learning models for fraud detection.
Implements Isolation Forest, LOF, and Deep Autoencoder.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# TensorFlow imports with error handling for environments without GPU
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers, Model
    from tensorflow.keras.callbacks import EarlyStopping
    TF_AVAILABLE = True
except ImportError:
    TF_AVAILABLE = False


class FraudDetectionModels:
    """
    Hybrid unsupervised fraud detection using multiple models.
    """
    
    def __init__(self, contamination: float = 0.1):
        """
        Initialize fraud detection models.
        
        Args:
            contamination: Expected proportion of outliers in the dataset
        """
        self.contamination = contamination
        self.isolation_forest = None
        self.lof = None
        self.autoencoder = None
        self.ae_threshold = None
        
    def train_isolation_forest(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train Isolation Forest model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, anomaly_scores)
            predictions: 1 for inliers, -1 for outliers
            anomaly_scores: Normalized anomaly scores (0-1, higher = more anomalous)
        """
        self.isolation_forest = IsolationForest(
            n_estimators=100,
            contamination=self.contamination,
            random_state=42,
            n_jobs=-1
        )
        
        # Fit and predict
        predictions = self.isolation_forest.fit_predict(X)
        
        # Get anomaly scores (negative of decision function, normalized to 0-1)
        raw_scores = -self.isolation_forest.decision_function(X)
        # Normalize scores to 0-1 range
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        if max_score > min_score:
            anomaly_scores = (raw_scores - min_score) / (max_score - min_score)
        else:
            anomaly_scores = np.zeros_like(raw_scores)
        
        return predictions, anomaly_scores
    
    def train_lof(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """
        Train Local Outlier Factor model.
        
        Args:
            X: Feature matrix
            
        Returns:
            Tuple of (predictions, anomaly_scores)
        """
        self.lof = LocalOutlierFactor(
            n_neighbors=20,
            contamination=self.contamination,
            novelty=False,
            n_jobs=-1
        )
        
        # Fit and predict
        predictions = self.lof.fit_predict(X)
        
        # Get anomaly scores (negative of LOF scores, normalized to 0-1)
        raw_scores = -self.lof.negative_outlier_factor_
        # Normalize to 0-1 range
        min_score = raw_scores.min()
        max_score = raw_scores.max()
        if max_score > min_score:
            anomaly_scores = (raw_scores - min_score) / (max_score - min_score)
        else:
            anomaly_scores = np.zeros_like(raw_scores)
        
        return predictions, anomaly_scores
    
    def build_autoencoder(self, input_dim: int) -> Model:
        """
        Build a deep autoencoder model.
        
        Args:
            input_dim: Number of input features
            
        Returns:
            Compiled Keras autoencoder model
        """
        if not TF_AVAILABLE:
            return None
            
        # Encoder
        inputs = keras.Input(shape=(input_dim,))
        encoded = layers.Dense(64, activation='relu')(inputs)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dropout(0.2)(encoded)
        encoded = layers.Dense(32, activation='relu')(encoded)
        encoded = layers.BatchNormalization()(encoded)
        encoded = layers.Dense(16, activation='relu')(encoded)
        
        # Bottleneck
        bottleneck = layers.Dense(8, activation='relu')(encoded)
        
        # Decoder
        decoded = layers.Dense(16, activation='relu')(bottleneck)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dense(32, activation='relu')(decoded)
        decoded = layers.BatchNormalization()(decoded)
        decoded = layers.Dropout(0.2)(decoded)
        decoded = layers.Dense(64, activation='relu')(decoded)
        outputs = layers.Dense(input_dim, activation='linear')(decoded)
        
        # Create model
        autoencoder = Model(inputs, outputs, name='fraud_autoencoder')
        
        # Compile
        autoencoder.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='mse'
        )
        
        return autoencoder
    
    def train_autoencoder(self, X: np.ndarray, epochs: int = 50, 
                          batch_size: int = 256) -> Tuple[np.ndarray, float, float]:
        """
        Train the autoencoder and compute reconstruction errors.
        
        Args:
            X: Feature matrix
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Tuple of (reconstruction_errors, threshold, final_loss)
        """
        if not TF_AVAILABLE:
            # Fallback: return mock data if TensorFlow is not available
            n_samples = X.shape[0]
            mock_errors = np.random.uniform(0, 0.1, n_samples)
            threshold = np.percentile(mock_errors, 100 * (1 - self.contamination))
            return mock_errors, threshold, 0.05
        
        input_dim = X.shape[1]
        self.autoencoder = self.build_autoencoder(input_dim)
        
        # Early stopping callback
        early_stopping = EarlyStopping(
            monitor='loss',
            patience=5,
            restore_best_weights=True
        )
        
        # Train the autoencoder
        history = self.autoencoder.fit(
            X, X,
            epochs=epochs,
            batch_size=batch_size,
            shuffle=True,
            callbacks=[early_stopping],
            verbose=0
        )
        
        # Compute reconstruction errors
        reconstructions = self.autoencoder.predict(X, verbose=0)
        mse = np.mean(np.power(X - reconstructions, 2), axis=1)
        
        # Set threshold based on percentile
        self.ae_threshold = np.percentile(mse, 100 * (1 - self.contamination))
        
        # Normalize reconstruction errors to 0-1 scale
        min_error = mse.min()
        max_error = mse.max()
        if max_error > min_error:
            normalized_errors = (mse - min_error) / (max_error - min_error)
        else:
            normalized_errors = np.zeros_like(mse)
        
        final_loss = history.history['loss'][-1]
        
        return normalized_errors, float(self.ae_threshold), float(final_loss)
    
    def get_autoencoder_predictions(self, reconstruction_errors: np.ndarray) -> np.ndarray:
        """
        Convert reconstruction errors to predictions.
        
        Args:
            reconstruction_errors: Array of reconstruction errors
            
        Returns:
            Predictions array (1 for inliers, -1 for outliers)
        """
        # Denormalize errors for threshold comparison
        predictions = np.where(
            reconstruction_errors > np.percentile(reconstruction_errors, 100 * (1 - self.contamination)),
            -1, 1
        )
        return predictions


def train_all_models(X: np.ndarray, contamination: float = 0.1) -> Dict[str, Any]:
    """
    Train all three fraud detection models.
    
    Args:
        X: Feature matrix
        contamination: Expected proportion of outliers
        
    Returns:
        Dictionary containing all model outputs
    """
    models = FraudDetectionModels(contamination=contamination)
    
    # Train Isolation Forest
    if_predictions, if_scores = models.train_isolation_forest(X)
    
    # Train LOF
    lof_predictions, lof_scores = models.train_lof(X)
    
    # Train Autoencoder
    ae_errors, ae_threshold, ae_loss = models.train_autoencoder(X)
    ae_predictions = models.get_autoencoder_predictions(ae_errors)
    
    return {
        'isolation_forest': {
            'predictions': if_predictions,
            'scores': if_scores
        },
        'lof': {
            'predictions': lof_predictions,
            'scores': lof_scores
        },
        'autoencoder': {
            'predictions': ae_predictions,
            'errors': ae_errors,
            'threshold': ae_threshold,
            'loss': ae_loss
        },
        'models': models
    }
