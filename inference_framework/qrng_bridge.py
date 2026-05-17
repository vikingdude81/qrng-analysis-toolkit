from typing import List, Optional
import numpy as np
from entropy_estimators import EntropyEstimator


def generate_entropy(data: List[bytes]) -> int:
    """Generate random entropy using quantum methods.
    
    Args:
        data: List of byte sequences to process for entropy estimation.
        
    Returns:
        Integer representation of computed entropy value.
        
    Raises:
        RuntimeError: If entropy estimation fails.
    """
    try:
        estimator = EntropyEstimator()
        return int(estimator.estimate(data))
    except Exception as e:
        raise RuntimeError(f"Entropy estimation failed: {str(e)}") from e


def compute_entropy_from_bytes(byte_data: bytes) -> float:
    """Compute entropy directly from byte data.
    
    Args:
        byte_data: Raw byte sequence to analyze.
        
    Returns:
        Shannon entropy value as a float.
    """
    try:
        estimator = EntropyEstimator()
        return float(estimator.estimate(np.frombuffer(byte_data, dtype=np.uint8)))
    except Exception as e:
        raise ValueError(f"Entropy computation from bytes failed: {str(e)}") from e


def batch_entropy_estimation(data_list: List[bytes]) -> List[float]:
    """Compute entropy for multiple byte sequences.
    
    Args:
        data_list: List of byte sequences to process.
        
    Returns:
        List of entropy values for each input sequence.
    """
    try:
        estimator = EntropyEstimator()
        results = []
        for data in data_list:
            entropy = estimator.estimate(np.frombuffer(data, dtype=np.uint8))
            results.append(float(entropy))
        return results
    except Exception as e:
        raise RuntimeError(f"Batch entropy estimation failed: {str(e)}") from e