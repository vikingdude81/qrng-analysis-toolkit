# cuquantum_accelerator/consciousness_metrics.py

from .entropy import renyi_entropy as consciousness_renyi_entropy
from deprecated import deprecated

@deprecated("Use entropy.renyi_entropy instead")
def consciousness_renyi_entropy_deprecated(p, alpha=1.0):
    """Deprecated wrapper for consciousness Rényi entropy calculation.
    
    Args:
        p: List of probabilities that sum to 1.0
        alpha: Order parameter for Rényi entropy (default 1.0 gives Shannon)
    
    Returns:
        Rényi entropy in bits, used as a consciousness metric
    """
    return consciousness_renyi_entropy(p, alpha)
