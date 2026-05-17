# cuquantum_accelerator/qrng_comprehensive_analysis.py

from .entropy import kolmogorov_sinai_lower_bound as qrng_kolmogorov_sinai
from deprecated import deprecated

@deprecated("Use entropy.kolmogorov_sinai_lower_bound instead")
def qrng_kolmogorov_sinai_deprecated(p):
    """Deprecated wrapper for QRNG Kolmogorov–Sinai analysis.
    
    Args:
        p: List of probabilities that sum to 1.0
    
    Returns:
        Kolmogorov–Sinai lower bound in bits per unit time
    """
    return qrng_kolmogorov_sinai(p)
