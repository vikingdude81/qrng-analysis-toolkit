# cuquantum_accelerator/entropy/renyi_entropy.py

import math

def renyi_entropy(p, alpha=1.0):
    """Calculate Rényi entropy of a probability distribution.
    
    Args:
        p: List of probabilities that sum to 1.0
        alpha: Order parameter for Rényi entropy (default 1.0 gives Shannon)
    
    Returns:
        Rényi entropy in bits
    """
    if alpha == 1.0:
        # Limit case: returns Shannon entropy
        return -sum(p[i] * math.log2(p[i]) for i in range(len(p)) if p[i] > 0)
    
    sum_alpha = sum(p[i]**alpha for i in range(len(p)) if p[i] > 0)
    return -1 / (alpha - 1) * math.log2(sum_alpha)
