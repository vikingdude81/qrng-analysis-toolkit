# cuquantum_accelerator/entropy/shannon_entropy.py

import math

def shannon_entropy(p):
    """Calculate Shannon entropy of a probability distribution.
    
    Args:
        p: List of probabilities that sum to 1.0
    
    Returns:
        Shannon entropy in bits (natural log base e, converted to bits)
    """
    return -sum(p[i] * math.log2(p[i]) for i in range(len(p)) if p[i] > 0)
