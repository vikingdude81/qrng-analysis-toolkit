"""
Comprehensive Entropy/Chaos/Consciousness Metrics Module for QRNG Analysis Toolkit

This module provides a unified interface for computing various entropy, chaos,
and consciousness metrics with standardized normalization and configurable parameters.
Designed for cross-project compatibility with Helios-Trajectory-Analysis.

Author: QRNG Analysis Toolkit Team
License: MIT
"""

import numpy as np
from scipy.stats import entropy
from nolds import sample_entropy, fuzzy_entropy, permutation_entropy, lyapunov_spectrum, epiplexity
from typing import Dict, Optional, List


class MetricsConfig:
    """Configuration class for metrics computation."""
    
    def __init__(
        self,
        normalization: str = 'zscore',  # 'minmax' or 'zscore'
        window_size: int = 100,
        embedding_dim: int = 5,
        time_lag: int = 2,
        sample_entropy_m: int = 3,
        sample_entropy_r: float = 0.1
    ):
        """
        Initialize metrics configuration.
        
        Args:
            normalization: Normalization method ('minmax' or 'zscore')
            window_size: Fixed window size for analysis
            embedding_dim: Embedding dimension m for permutation entropy
            time_lag: Time lag τ for permutation entropy
            sample_entropy_m: Embedding dimension for sample entropy
            sample_entropy_r: Tolerance factor for sample entropy
        """
        self.normalization = normalization
        self.window_size = window_size
        self.embedding_dim = embedding_dim
        self.time_lag = time_lag
        self.sample_entropy_m = sample_entropy_m
        self.sample_entropy_r = sample_entropy_r
    
    def normalize(self, data: np.ndarray) -> np.ndarray:
        """Normalize the input data."""
        if self.normalization == 'zscore':
            mean = np.mean(data)
            std = np.std(data)
            return (data - mean) / std
        elif self.normalization == 'minmax':
            min_val = np.min(data)
            max_val = np.max(data)
            return (data - min_val) / (max_val - min_val)
        else:
            raise ValueError(f"Unknown normalization method: {self.normalization}")


def compute_all_metrics(
    time_series: np.ndarray,
    config: Optional[MetricsConfig] = None
) -> Dict[str, float]:
    """Compute various entropy metrics for a time series.

    Args:
        time_series: A 1D numpy array representing the time series.
        config: Optional configuration object. If None, uses default settings.
        
    Returns:
        A dictionary with keys 'shannon_entropy', 'sample_entropy', 'fuzzy_entropy',
        'permutation_entropy', 'lyapunov_spectrum', 'epiplexity' and values as computed.
    """
    # Normalize the time series
    if config is None:
        config = MetricsConfig()
    
    normalized_series = config.normalize(time_series)
    
    # Compute Shannon entropy
    shannon_entropy = entropy(normalized_series, base=2)
    
    # Compute sample entropy
    sample_entropy_value = sample_entropy(
        normalized_series,
        m=config.sample_entropy_m,
        r=config.sample_entropy_r
    )
    
    # Compute fuzzy entropy
    fuzzy_entropy_value = fuzzy_entropy(normalized_series)
    
    # Compute permutation entropy
    permutation_entropy_value = permutation_entropy(
        normalized_series,
        embedding_dim=config.embedding_dim,
        tau=config.time_lag
    )
    
    # Compute Lyapunov spectrum (approximation)
    lyapunov_spectrum_value = lyapunov_spectrum(normalized_series, method='approx')
    
    # Compute epiplexity
    epiplexity_value = epiplexity(normalized_series)
    
    return {
        'shannon_entropy': float(shannon_entropy),
        'sample_entropy': float(sample_entropy_value),
        'fuzzy_entropy': float(fuzzy_entropy_value),
        'permutation_entropy': float(permutation_entropy_value),
        'lyapunov_spectrum': float(lyapunov_spectrum_value),
        'epiplexity': float(epiplexity_value)
    }


def compute_metrics_batch(
    time_series_list: List[np.ndarray],
    config: Optional[MetricsConfig] = None
) -> Dict[str, np.ndarray]:
    """Compute metrics for a batch of time series.

    Args:
        time_series_list: List of 1D numpy arrays representing time series.
        config: Optional configuration object. If None, uses default settings.
        
    Returns:
        Dictionary with metric names as keys and arrays of values as values.
    """
    if config is None:
        config = MetricsConfig()
    
    results = {
        'shannon_entropy': [],
        'sample_entropy': [],
        'fuzzy_entropy': [],
        'permutation_entropy': [],
        'lyapunov_spectrum': [],
        'epiplexity': []
    }
    
    for series in time_series_list:
        metrics = compute_all_metrics(series, config)
        for key, value in metrics.items():
            results[key].append(value)
    
    # Convert lists to arrays
    return {key: np.array(values) for key, values in results.items()}
