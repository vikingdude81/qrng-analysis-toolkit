"""
Multiscale entropy analysis for trajectory data.
Addresses gap: No multiscale entropy - missing analysis across multiple time scales.

This module provides comprehensive multiscale entropy (MSE) analysis,
especially important for understanding hierarchical structure in consciousness-related signals.
"""

import numpy as np
from typing import List, Optional, Tuple
import warnings

warnings.warn('This module implements novel multiscale entropy capabilities', DeprecationWarning)

def calculate_multiscale_entropy(
    time_series: np.ndarray,
    scale_factor: int = 10,
    max_scale: int = 20,
    embedding_dim: int = 3,
    tau: int = 1
) -> Tuple[float, List[float]]:
    """
    Calculate multiscale entropy (MSE) of a time series.
    
    Analyzes complexity across multiple temporal scales to reveal hierarchical structure.
    Essential for understanding consciousness-related signals with nested patterns.
    
    Args:
        time_series: 1D array of time series data
        scale_factor: Factor by which to downsample at each scale (default: 10)
        max_scale: Maximum number of scales to analyze (default: 20)
        embedding_dim: Embedding dimension for permutation entropy (default: 3)
        tau: Time delay for permutation entropy (default: 1)
    
    Returns:
        Tuple of (average_entropy_across_scales, list_of_entropies_per_scale)
    """
    if len(time_series) < embedding_dim + 10:
        raise ValueError("Time series too short for MSE analysis")
    
    entropies = []
    
    for scale in range(1, max_scale + 1):
        # Downsample at current scale
        start_idx = (scale - 1) * len(time_series) // scale_factor
        if start_idx >= len(time_series):
            break
        
        downsampled = time_series[start_idx::scale_factor]
        
        if len(downsampled) < embedding_dim + 5:
            continue
        
        # Calculate permutation entropy at this scale
        entropy = _calculate_permutation_entropy(
            downsampled, embedding_dim, tau
        )
        entropies.append(entropy)
    
    if not entropies:
        return 0.0, []
    
    avg_entropy = np.mean(entropies)
    return float(avg_entropy), entropies

def calculate_multiscale_sample_entropy(
    time_series: np.ndarray,
    scale_factor: int = 10,
    max_scale: int = 20,
    m: int = 2,
    r: float = 0.2
) -> Tuple[float, List[float]]:
    """
    Calculate multiscale sample entropy (MSE).
    
    Uses sample entropy which is less sensitive to data length than permutation entropy.
    
    Args:
        time_series: 1D array of time series data
        scale_factor: Factor by which to downsample at each scale
        max_scale: Maximum number of scales
        m: Pattern length (default: 2)
        r: Tolerance parameter as fraction of SD (default: 0.2)
    
    Returns:
        Tuple of (average_entropy, list_of_entropies_per_scale)
    """
    if len(time_series) < m + 10:
        raise ValueError("Time series too short for MSE analysis")
    
    entropies = []
    
    for scale in range(1, max_scale + 1):
        start_idx = (scale - 1) * len(time_series) // scale_factor
        if start_idx >= len(time_series):
            break
        
        downsampled = time_series[start_idx::scale_factor]
        
        if len(downsampled) < m + 5:
            continue
        
        # Calculate sample entropy at this scale
        entropy = _calculate_sample_entropy(
            downsampled, m, r,
            embedding_dim=3, tau=1
        )
        entropies.append(entropy)
    
    if not entropies:
        return 0.0, []
    
    avg_entropy = np.mean(entropies)
    return float(avg_entropy), entropies

def _calculate_permutation_entropy(
    time_series: np.ndarray,
    embedding_dim: int,
    tau: int
) -> float:
    """
    Calculate permutation entropy for a single scale.
    """
    if len(time_series) < embedding_dim + 1:
        return 0.0
    
    n = len(time_series)
    patterns = []
    
    for i in range(n - (embedding_dim - 1) * tau):
        pattern = time_series[i:i + embedding_dim * tau]
        sorted_indices = np.argsort(pattern)
        perm_pattern = [np.where(sorted_indices == j)[0][0] for j in range(embedding_dim)]
        patterns.append(tuple(perm_pattern))
    
    from collections import Counter
    pattern_counts = Counter(patterns)
    total = len(patterns)
    probs = np.array([count / total for count in pattern_counts.values()])
    
    # Filter out zero probabilities
    probs = probs[probs > 0]
    
    entropy = -np.sum(probs * np.log2(probs))
    return float(entropy)

def _calculate_sample_entropy(
    time_series: np.ndarray,
    m: int,
    r: float,
    embedding_dim: int = 3,
    tau: int = 1
) -> float:
    """
    Calculate sample entropy for a single scale.
    Uses pattern matching with tolerance parameter.
    """
    if len(time_series) < m + 10:
        return 0.0
    
    n = len(time_series)
    std = np.std(time_series)
    r_scaled = r * std
    
    # Calculate embedding
    embedded = _delay_embedding(time_series, embedding_dim, tau)
    if len(embedded) < m:
        return 0.0
    
    # Count similar patterns
    A_m = 0
    B_m = 0
    
    for i in range(len(embedded) - m + 1):
        pattern_i = embedded[i:i + m]
        
        count_A = 0
        count_B = 0
        
        for j in range(i + 1, len(embedded) - m + 1):
            pattern_j = embedded[j:j + m]
            
            if _calculate_distance(pattern_i, pattern_j) <= r_scaled:
                count_A += 1
            
            if _calculate_distance(embedded[i:i+m], embedded[j:j+m]) <= r_scaled:
                count_B += 1
        
        if count_B > 0 and count_A > 0:
            A_m += -np.log(count_A / count_B)
        elif count_B == 0:
            A_m += 0
    
    if A_m == 0:
        return 0.0
    
    # Calculate B_m (same but without offset)
    for i in range(len(embedded) - m + 1):
        pattern_i = embedded[i:i + m]
        
        count_B = 0
        for j in range(i + 1, len(embedded) - m + 1):
            if _calculate_distance(pattern_i, embedded[j:j+m]) <= r_scaled:
                count_B += 1
        
        if count_B > 0:
            B_m += -np.log(count_B)
    
    if B_m == 0:
        return 0.0
    
    sample_entropy = (A_m / (len(embedded) - m)) - (B_m / (len(embedded) - m))
    return float(sample_entropy)

def _delay_embedding(
    time_series: np.ndarray,
    embedding_dim: int,
    tau: int
) -> np.ndarray:
    """
    Create delay embedding of time series.
    """
    n = len(time_series)
    embedded = []
    
    for i in range(n - (embedding_dim - 1) * tau):
        point = tuple(time_series[i + k * tau] for k in range(embedding_dim))
        embedded.append(point)
    
    return np.array(embedded)

def _calculate_distance(
    point1: Tuple[float, ...],
    point2: Tuple[float, ...]
) -> float:
    """
    Calculate Euclidean distance between two points.
    """
    return np.sqrt(np.sum(np.array(point1) - np.array(point2)) ** 2)

def analyze_trajectory_complexity(
    trajectory_data: np.ndarray
) -> dict:
    """
    Comprehensive complexity analysis of trajectory data.
    
    Args:
        trajectory_data: Time series from trajectory analysis
    
    Returns:
        Dictionary with entropy measures across scales
    """
    result = {
        'multiscale_entropy': calculate_multiscale_entropy(trajectory_data),
        'sample_entropy': calculate_multiscale_sample_entropy(trajectory_data),
        'single_scale_entropy': _calculate_permutation_entropy(
            trajectory_data, 3, 1
        ),
    }
    
    return result
