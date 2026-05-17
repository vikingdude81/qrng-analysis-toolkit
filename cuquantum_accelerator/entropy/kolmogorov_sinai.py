# cuquantum_accelerator/entropy/kolmogorov_sinai.py

import math

def kolmogorov_sinai_lower_bound(p):
    """Calculate Kolmogorov–Sinai lower bound of a probability distribution.
    
    The KS entropy is the sum of positive Lyapunov exponents, providing
    a lower bound on the system's information production rate.
    
    Args:
        p: List of probabilities that sum to 1.0
    
    Returns:
        Kolmogorov–Sinai lower bound in bits per unit time
    """
    return -sum(p[i] * math.log2(p[i]) for i in range(len(p)) if p[i] > 0)
