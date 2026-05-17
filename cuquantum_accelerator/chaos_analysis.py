# cuquantum_accelerator/chaos_analysis.py

from .entropy import shannon_entropy as chaos_shannon_entropy
from deprecated import deprecated

@deprecated("Use entropy.shannon_entropy instead")
def chaos_shannon_entropy_deprecated(p):
    """Deprecated wrapper for chaos Shannon entropy calculation.
    
    Args:
        p: List of probabilities that sum to 1.0
    
    Returns:
        Shannon entropy in bits
    """
    return chaos_shannon_entropy(p)
