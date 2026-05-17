# core.py

import numpy as np
try:
    import cupy as cp
except ImportError:
    cp = None

def requires_cupy(func):
    def wrapper(*args, **kwargs):
        if cp is None:
            raise ImportError("CuPy is not installed. Please install CuPy to use this function.")
        return func(*args, **kwargs)
    return wrapper

@requires_cupy
def entropy_gpu(data: np.ndarray) -> float:
    """
    Calculate the entropy of a dataset using GPU.

    Args:
        data (np.ndarray): Input data array.

    Returns:
        float: Entropy value.
    """
    # GPU-specific implementation
    pass

def entropy_cpu(data: np.ndarray) -> float:
    """
    Calculate the entropy of a dataset using CPU.

    Args:
        data (np.ndarray): Input data array.

    Returns:
        float: Entropy value.
    """
    # CPU-specific implementation
    pass
