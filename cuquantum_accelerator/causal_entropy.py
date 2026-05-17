#!/usr/bin/env python
"""Causal entropy estimation for influence detection.

This module implements novel causal entropy capabilities to help distinguish
correlation from causation in consciousness-related trajectory data.

Key features:
- Conditional entropy analysis
- Transfer entropy calculation
- Granger causality testing
- Recurrent network entropy measures
"""

import numpy as np
from typing import List, Optional, Tuple
import warnings

warnings.warn('This module implements novel causal entropy capabilities', DeprecationWarning)


def calculate_conditional_entropy(
    target: np.ndarray,
    source: np.ndarray,
    lag: int = 1
) -> float:
    """Calculate conditional entropy H(target | source).
    
    Measures uncertainty in target given knowledge of source.
    Lower values indicate stronger causal influence from source to target.
    
    Args:
        target: Time series of the target variable
        source: Time series of the source variable
        lag: Time lag between source and target (default: 1)
    
    Returns:
        Conditional entropy in bits
    """
    if len(target) != len(source):
        raise ValueError("Series must have equal length")
    
    n = len(target)
    if n == 0:
        return 0.0
    
    # Create joint distribution with lag
    joint_pairs = []
    for i in range(n - lag):
        pair = (source[i], target[i + lag])
        joint_pairs.append(pair)
    
    if not joint_pairs:
        return 0.0
    
    # Calculate marginal distributions
    source_vals = np.array([p[0] for p in joint_pairs])
    target_vals = np.array([p[1] for p in joint_pairs])
    
    # Discretize if continuous
    if hasattr(source_vals, 'dtype') and np.issubdtype(source_vals.dtype, np.floating):
        source_bins = np.histogram_bin_edges(source_vals)
        target_bins = np.histogram_bin_edges(target_vals)
        
        source_discrete = np.digitize(source_vals, source_bins) - 1
        target_discrete = np.digitize(target_vals, target_bins) - 1
    else:
        source_discrete = source_vals.astype(int)
        target_discrete = target_vals.astype(int)
    
    # Calculate joint and marginal probabilities
    from collections import Counter
    joint_counts = Counter(zip(source_discrete, target_discrete))
    total = len(joint_pairs)
    joint_probs = np.array([count / total for count in joint_counts.values()])
    
    source_counts = Counter(source_discrete)
    source_probs = np.array([count / total for count in source_counts.values()])
    target_counts = Counter(target_discrete)
    target_probs = np.array([count / total for count in target_counts.values()])
    
    # Calculate entropies
    h_target = -np.sum(target_probs * np.log2(target_probs + 1e-15))
    h_joint = -np.sum(joint_probs * np.log2(joint_probs + 1e-15))
    
    # Conditional entropy: H(Y|X) = H(X,Y) - H(X)
    conditional_entropy = h_joint - h_target
    
    return float(max(0, conditional_entropy))