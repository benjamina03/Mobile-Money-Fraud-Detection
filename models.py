"""
Models module for Mobile Money Fraud Detection
Implements Isolation Forest, Autoencoder, and DBSCAN with hybrid scoring
"""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.ensemble import IsolationForest
from sklearn.cluster import DBSCAN
from sklearn.preprocessing import MinMaxScaler
import joblib
import os
from typing import Tuple, Dict


class Autoencoder(nn.Module):
    """
    PyTorch Autoencoder for anomaly detection.
    High reconstruction error indicates anomalies.
    """
    
    def __init__(self, input_dim: int, encoding_dim: int = 10):
        """
        Initialize the Autoencoder.
        
        Args:
            input_dim: Number of input features
            encoding_dim: Size of the encoded representation
        """
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 32),
            nn.ReLU(),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, encoding_dim),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(encoding_dim, 16),
            nn.ReLU(),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, input_dim)
        )
    
    def forward(self, x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded


class AutoencoderModel:
    """Wrapper class for Autoencoder training and inference."""
    
    def __init__(self, input_dim: int, encoding_dim: int = 10):
        self.model = Autoencoder(input_dim, encoding_dim)
        self.criterion = nn.MSELoss()
        self.optimizer = None
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        
    def train(self, X_train: np.ndarray, epochs: int = 50, learning_rate: float = 0.001):
        """
        Train the autoencoder.
        
        Args:
            X_train: Training data
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
        """
        self.optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)
        X_tensor = torch.FloatTensor(X_train)
        
        self.model.train()
        for epoch in range(epochs):
            # Forward pass
            outputs = self.model(X_tensor)
            loss = self.criterion(outputs, X_tensor)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
            if (epoch + 1) % 10 == 0:
                print(f'Autoencoder Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')
    
    def get_reconstruction_errors(self, X: np.ndarray) -> np.ndarray:
        """
        Calculate reconstruction errors for anomaly detection.
        
        Args:
            X: Input data
            
        Returns:
            Array of reconstruction errors
        """
        self.model.eval()
        with torch.no_grad():
            X_tensor = torch.FloatTensor(X)
            reconstructed = self.model(X_tensor)
            errors = torch.mean((X_tensor - reconstructed) ** 2, dim=1)
            return errors.numpy()
    
    def predict(self, X: np.ndarray, threshold_percentile: float = 95) -> np.ndarray:
        """
        Predict anomalies based on reconstruction error.
        
        Args:
            X: Input data
            threshold_percentile: Percentile for anomaly threshold
            
        Returns:
            Binary predictions (1 for anomaly, 0 for normal)
        """
        errors = self.get_reconstruction_errors(X)
        threshold = np.percentile(errors, threshold_percentile)
        return (errors > threshold).astype(int)


class IsolationForestModel:
    """Wrapper for Sklearn's Isolation Forest."""
    
    def __init__(self, contamination: float = 0.1, random_state: int = 42):
        """
        Initialize Isolation Forest.
        
        Args:
            contamination: Expected proportion of outliers
            random_state: Random seed
        """
        self.model = IsolationForest(
            contamination=contamination,
            random_state=random_state,
            n_estimators=100
        )
    
    def train(self, X_train: np.ndarray):
        """
        Train the Isolation Forest.
        
        Args:
            X_train: Training data
        """
        self.model.fit(X_train)
        print("Isolation Forest training complete")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (-1 for anomaly, 1 for normal).
        Converts to (1 for anomaly, 0 for normal).
        
        Args:
            X: Input data
            
        Returns:
            Binary predictions
        """
        predictions = self.model.predict(X)
        # Convert: -1 (anomaly) -> 1, 1 (normal) -> 0
        return (predictions == -1).astype(int)
    
    def score_samples(self, X: np.ndarray) -> np.ndarray:
        """
        Get anomaly scores (lower score = more anomalous).
        
        Args:
            X: Input data
            
        Returns:
            Anomaly scores
        """
        return self.model.score_samples(X)


class DBSCANModel:
    """Wrapper for Sklearn's DBSCAN clustering."""
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        Initialize DBSCAN.
        
        Args:
            eps: Maximum distance between samples
            min_samples: Minimum samples in a neighborhood
        """
        self.model = DBSCAN(eps=eps, min_samples=min_samples)
        self.labels_ = None
    
    def train(self, X_train: np.ndarray):
        """
        Fit DBSCAN (no training, just clustering).
        
        Args:
            X_train: Training data
        """
        self.labels_ = self.model.fit_predict(X_train)
        n_clusters = len(set(self.labels_)) - (1 if -1 in self.labels_ else 0)
        n_noise = list(self.labels_).count(-1)
        print(f"DBSCAN complete: {n_clusters} clusters, {n_noise} noise points")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict anomalies (points labeled as noise/-1).
        
        Args:
            X: Input data
            
        Returns:
            Binary predictions (1 for anomaly/noise, 0 for normal)
        """
        labels = self.model.fit_predict(X)
        # Points labeled -1 are noise (anomalies)
        return (labels == -1).astype(int)


class HybridModel:
    """
    Hybrid ensemble combining Isolation Forest, Autoencoder, and DBSCAN.
    """
    
    def __init__(self, input_dim: int):
        """
        Initialize all three models.
        
        Args:
            input_dim: Number of input features
        """
        self.iso_forest = IsolationForestModel(contamination=0.1)
        self.autoencoder = AutoencoderModel(input_dim=input_dim)
        self.dbscan = DBSCANModel(eps=0.5, min_samples=5)
        self.input_dim = input_dim
        
    def train(self, X_train: np.ndarray, ae_epochs: int = 50):
        """
        Train all three models in parallel.
        
        Args:
            X_train: Training data
            ae_epochs: Epochs for autoencoder training
        """
        print("Training Isolation Forest...")
        self.iso_forest.train(X_train)
        
        print("\nTraining Autoencoder...")
        self.autoencoder.train(X_train, epochs=ae_epochs)
        
        print("\nFitting DBSCAN...")
        self.dbscan.train(X_train)
        
        print("\n✓ All models trained successfully!")
    
    def calculate_hybrid_score(self, X: np.ndarray) -> Tuple[np.ndarray, Dict[str, np.ndarray]]:
        """
        Calculate hybrid anomaly scores.
        
        Formula: Hybrid_Score = 0.4 * IsoForest + 0.4 * Autoencoder + 0.2 * DBSCAN
        All scores normalized to [0, 1] range.
        
        Args:
            X: Input data
            
        Returns:
            Tuple of (hybrid_scores, individual_scores_dict)
        """
        # Get individual scores
        iso_scores = self.iso_forest.score_samples(X)
        ae_errors = self.autoencoder.get_reconstruction_errors(X)
        dbscan_preds = self.dbscan.predict(X)
        
        # Normalize to [0, 1] where 1 = more anomalous
        # Isolation Forest: lower scores are more anomalous, so invert
        iso_normalized = 1 - MinMaxScaler().fit_transform(iso_scores.reshape(-1, 1)).flatten()
        
        # Autoencoder: higher errors are more anomalous
        ae_normalized = MinMaxScaler().fit_transform(ae_errors.reshape(-1, 1)).flatten()
        
        # DBSCAN: already binary (0 or 1)
        dbscan_normalized = dbscan_preds.astype(float)
        
        # Calculate weighted hybrid score
        hybrid_scores = (
            0.4 * iso_normalized +
            0.4 * ae_normalized +
            0.2 * dbscan_normalized
        )
        
        individual_scores = {
            'isolation_forest': iso_normalized,
            'autoencoder': ae_normalized,
            'dbscan': dbscan_normalized
        }
        
        return hybrid_scores, individual_scores
    
    def predict(self, X: np.ndarray, threshold: float = 0.75) -> Tuple[np.ndarray, np.ndarray, Dict[str, np.ndarray]]:
        """
        Predict anomalies using hybrid scoring.
        
        Args:
            X: Input data
            threshold: Threshold for blocking (default 0.75)
            
        Returns:
            Tuple of (predictions, hybrid_scores, individual_scores)
        """
        hybrid_scores, individual_scores = self.calculate_hybrid_score(X)
        predictions = (hybrid_scores > threshold).astype(int)
        
        return predictions, hybrid_scores, individual_scores


def save_models(hybrid_model: HybridModel, scaler, filepath: str = 'trained_models'):
    """
    Save trained models and scaler to disk.
    
    Args:
        hybrid_model: Trained HybridModel instance
        scaler: Fitted scaler object
        filepath: Directory path to save models
    """
    os.makedirs(filepath, exist_ok=True)
    
    # Save Isolation Forest
    joblib.dump(hybrid_model.iso_forest, os.path.join(filepath, 'isolation_forest.pkl'))
    
    # Save Autoencoder
    torch.save({
        'model_state_dict': hybrid_model.autoencoder.model.state_dict(),
        'input_dim': hybrid_model.autoencoder.input_dim,
        'encoding_dim': hybrid_model.autoencoder.encoding_dim
    }, os.path.join(filepath, 'autoencoder.pth'))
    
    # Save DBSCAN
    joblib.dump(hybrid_model.dbscan, os.path.join(filepath, 'dbscan.pkl'))
    
    # Save scaler
    joblib.dump(scaler, os.path.join(filepath, 'scaler.pkl'))
    
    print(f"✓ Models saved to {filepath}/")


def load_models(filepath: str = 'trained_models') -> Tuple[HybridModel, object]:
    """
    Load trained models and scaler from disk.
    
    Args:
        filepath: Directory path containing saved models
        
    Returns:
        Tuple of (HybridModel, scaler)
    """
    # Load scaler first to get input dimension
    scaler = joblib.load(os.path.join(filepath, 'scaler.pkl'))
    
    # Load Autoencoder checkpoint
    ae_checkpoint = torch.load(os.path.join(filepath, 'autoencoder.pth'))
    input_dim = ae_checkpoint['input_dim']
    
    # Create HybridModel instance
    hybrid_model = HybridModel(input_dim=input_dim)
    
    # Load Isolation Forest
    hybrid_model.iso_forest = joblib.load(os.path.join(filepath, 'isolation_forest.pkl'))
    
    # Load Autoencoder state
    hybrid_model.autoencoder.model.load_state_dict(ae_checkpoint['model_state_dict'])
    hybrid_model.autoencoder.model.eval()
    
    # Load DBSCAN
    hybrid_model.dbscan = joblib.load(os.path.join(filepath, 'dbscan.pkl'))
    
    print(f"✓ Models loaded from {filepath}/")
    
    return hybrid_model, scaler
