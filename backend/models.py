"""
Machine Learning models for fraud detection.
Implements Isolation Forest, LOF, and Deep Autoencoder using PyTorch.
"""

import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from typing import Tuple, Dict, Any
import warnings
warnings.filterwarnings('ignore')

# PyTorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


class Autoencoder(nn.Module):
    """PyTorch Autoencoder for anomaly detection."""
    
    def __init__(self, input_dim: int):
        super(Autoencoder, self).__init__()
        
        # Encoder
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.BatchNorm1d(64),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Linear(16, 8),
            nn.ReLU()
        )
        
        # Decoder
        self.decoder = nn.Sequential(
            nn.Linear(8, 16),
            nn.ReLU(),
            nn.BatchNorm1d(16),
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.BatchNorm1d(32),
            nn.Dropout(0.2),
            nn.Linear(32, 64),
            nn.ReLU(),
            nn.Linear(64, input_dim)
        )
    
    def encode(self, x):
        """Encode input to latent representation."""
        return self.encoder(x)
    
    def decode(self, z):
        """Decode latent representation to reconstruction."""
        return self.decoder(z)
    
    def forward(self, x):
        """Forward pass returning both encoded and decoded outputs."""
        encoded = self.encode(x)
        decoded = self.decode(encoded)
        return decoded


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
        # Set device with graceful fallback to CPU
        try:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        except Exception:
            self.device = torch.device('cpu')
        
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
    
    def train_autoencoder(self, X: np.ndarray, epochs: int = 50, 
                          batch_size: int = 256) -> Tuple[np.ndarray, float, float]:
        """
        Train the PyTorch autoencoder and compute reconstruction errors.
        
        Args:
            X: Feature matrix
            epochs: Number of training epochs
            batch_size: Training batch size
            
        Returns:
            Tuple of (reconstruction_errors, threshold, final_loss)
        """
        input_dim = X.shape[1]
        self.autoencoder = Autoencoder(input_dim)
        
        # Move model to device with error handling
        try:
            self.autoencoder = self.autoencoder.to(self.device)
        except Exception:
            self.device = torch.device('cpu')
            self.autoencoder = self.autoencoder.to(self.device)
        
        # Convert to PyTorch tensors
        X_tensor = torch.FloatTensor(X).to(self.device)
        dataset = TensorDataset(X_tensor, X_tensor)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        
        # Loss and optimizer
        criterion = nn.MSELoss()
        optimizer = optim.Adam(self.autoencoder.parameters(), lr=0.001)
        
        # Training loop with early stopping
        best_loss = float('inf')
        patience = 5
        patience_counter = 0
        final_loss = 0.0
        
        self.autoencoder.train()
        for epoch in range(epochs):
            epoch_loss = 0.0
            for batch_x, _ in dataloader:
                optimizer.zero_grad()
                outputs = self.autoencoder(batch_x)
                loss = criterion(outputs, batch_x)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            
            epoch_loss /= len(dataloader)
            final_loss = epoch_loss
            
            # Early stopping check
            if epoch_loss < best_loss:
                best_loss = epoch_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    break
        
        # Compute reconstruction errors
        self.autoencoder.eval()
        with torch.no_grad():
            reconstructions = self.autoencoder(X_tensor).cpu().numpy()
        
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
        
        return normalized_errors, float(self.ae_threshold), float(final_loss)
    
    def get_autoencoder_predictions(self, reconstruction_errors: np.ndarray) -> np.ndarray:
        """
        Convert reconstruction errors to predictions.
        
        Args:
            reconstruction_errors: Array of reconstruction errors
            
        Returns:
            Predictions array (1 for inliers, -1 for outliers)
        """
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
