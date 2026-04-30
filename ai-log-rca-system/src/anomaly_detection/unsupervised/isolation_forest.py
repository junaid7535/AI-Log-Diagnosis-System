import numpy as np
from sklearn.ensemble import IsolationForest
from typing import List, Tuple
import pickle
from pathlib import Path

class IsolationForestDetector:
    def __init__(self, contamination: float = 0.1, n_estimators: int = 100):
        self.model = IsolationForest(
            contamination=contamination,
            n_estimators=n_estimators,
            random_state=42,
            bootstrap=True
        )
        self.is_fitted = False
        
    def fit(self, features: np.ndarray):
        """Train isolation forest on normal data"""
        self.model.fit(features)
        self.is_fitted = True
        
    def detect(self, features: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return anomaly scores and labels (-1=anomaly, 1=normal)"""
        if not self.is_fitted:
            raise ValueError("Model not fitted")
        
        scores = self.model.score_samples(features)
        labels = self.model.predict(features)
        
        # Convert to binary (1=anomaly, 0=normal)
        anomaly_labels = (labels == -1).astype(int)
        anomaly_scores = 1 - (scores - scores.min()) / (scores.max() - scores.min())
        
        return anomaly_scores, anomaly_labels
    
    def save(self, path: Path):
        with open(path, 'wb') as f:
            pickle.dump(self.model, f)
    
    def load(self, path: Path):
        with open(path, 'rb') as f:
            self.model = pickle.load(f)
        self.is_fitted = True