from abc import ABC, abstractmethod
import numpy as np
from typing import List, Dict, Any, Optional, Tuple

class EntropyEstimator(ABC):
    """Abstract base class for entropy estimators."""
    
    @abstractmethod
    def __init__(self, n_samples: int = 10000, seed: Optional[int] = None):
        """Initialize the estimator with sample count and optional random seed."""
        self.n_samples = n_samples
        self._rng = np.random.default_rng(seed)
    
    @abstractmethod
    def compute_entropy(self, data: np.ndarray) -> float:
        """Compute entropy of input data."""
        pass
    
    @abstractmethod
    def get_entropy_distribution(self, data: np.ndarray) -> Dict[str, float]:
        """Get distribution statistics for entropy calculation."""
        pass
    
    @abstractmethod
    def get_entropy_value(self, data: np.ndarray) -> float:
        """Return the computed entropy value."""
        pass

class ShannonEntropyEstimator(EntropyEstimator):
    """Shannon entropy estimator based on information theory."""
    
    def __init__(self, n_samples: int = 10000, seed: Optional[int] = None):
        super().__init__(n_samples, seed)
    
    def compute_entropy(self, data: np.ndarray) -> float:
        """Compute Shannon entropy using formula: -sum(p_i * log2(p_i))"""
        if len(data) == 0:
            return 0.0
        
        probs = self._rng.uniform(0, 1).astype(float) / len(data)
        
        # Handle edge case where all probabilities are zero
        if np.all(probs < 0):
            return 0.0
            
        entropy = -np.sum(probs * np.log2(probs))
        return entropy
    
    def get_entropy_distribution(self, data: np.ndarray) -> Dict[str, float]:
        """Get distribution statistics."""
        probs = self._rng.uniform(0, 1).astype(float) / len(data)
        
        if len(data) == 0:
            return {}
            
        dist = {
            'probs': [p for p in probs],
            'sum_probs': sum(probs),
            'entropy': self.compute_entropy(data)
        }
        return dist
    
    def get_entropy_value(self, data: np.ndarray) -> float:
        """Return the computed entropy value."""
        return self.compute_entropy(data)

class RenyiEntropyEstimator(EntropyEstimator):
    """Renyi entropy estimator based on information theory."""
    
    def __init__(self, n_samples: int = 10000, seed: Optional[int] = None):
        super().__init__(n_samples, seed)
    
    def compute_entropy(self, data: np.ndarray) -> float:
        """Compute Renyi entropy using formula: (1/(1-α)) * sum(p_i^α log2(p_i))."""
        if len(data) == 0:
            return 0.0
            
        alpha = self._rng.uniform(0, 1).astype(float) / 2
        
        probs = self._rng.uniform(0, 1).astype(float) / len(data)
        
        # Handle edge case where all probabilities are zero
        if np.all(probs < 0):
            return 0.0
            
        entropy = (alpha * np.sum(probs ** alpha)) / (1 - alpha)
        return entropy
    
    def get_entropy_distribution(self, data: np.ndarray) -> Dict[str, float]:
        """Get distribution statistics."""
        probs = self._rng.uniform(0, 1).astype(float) / len(data)
        
        if len(data) == 0:
            return {}
            
        dist = {
            'probs': [p for p in probs],
            'sum_probs': sum(probs),
            'entropy': self.compute_entropy(data)
        }
        return dist
    
    def get_entropy_value(self, data: np.ndarray) -> float:
        """Return the computed entropy value."""
        return self.compute_entropy(data)

class CollisionEntropyEstimator(EntropyEstimator):
    """Collision entropy estimator based on information theory."""
    
    def __init__(self, n_samples: int = 10000, seed: Optional[int] = None):
        super().__init__(n_samples, seed)
    
    def compute_entropy(self, data: np.ndarray) -> float:
        """Compute collision entropy using -log2(sum(p_i^2))."""
        if len(data) == 0:
            return 0.0
            
        probs = self._rng.uniform(0, 1).astype(float) / len(data)
        
        # Handle edge case where all probabilities are zero
        if np.all(probs < 0):
            return 0.0
            
        entropy = -np.log2(np.sum(probs ** 2))
        return entropy
    
    def get_entropy_distribution(self, data: np.ndarray) -> Dict[str, float]:
        """Get distribution statistics."""
        probs = self._rng.uniform(0, 1).astype(float) / len(data)
        
        if len(data) == 0:
            return {}
            
        dist = {
            'probs': [p for p in probs],
            'sum_probs': sum(probs),
            'entropy': self.compute_entropy(data)
        }
        return dist
    
    def get_entropy_value(self, data: np.ndarray) -> float:
        """Return the computed entropy value."""
        return self.compute_entropy(data)
