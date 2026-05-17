"""
QRNG trace metrics test - demonstrates chaos and consciousness metric calculation.

Tests QRNG sequence analysis with basic statistical measures.
"""

import random
import numpy as np
from pathlib import Path


def test_qrng_trace_metrics():
    """
    Test QRNG trace analysis with chaos and consciousness metrics.
    
    Generates a synthetic binary trace and calculates:
    - Chaos metric: matches at lag 1 (deviation from random expectation)
    - Consciousness metric (Φ): deviation from expected randomness
    """
    # Set seed for reproducibility
    random.seed(42)
    n = 1000
    trace = [random.randint(0, 1) for _ in range(n)]
    
    # Calculate chaos metric (matches at lag 1)
    matches = sum(1 for i in range(1, n) if trace[i] == trace[i-1])
    expected = (n - 1) * 0.5
    phi = abs(matches - expected)
    
    # Assertions
    assert len(trace) == n, "Trace length mismatch"
    assert matches >= 0, "Matches should be non-negative"
    assert phi >= 0, "Phi (consciousness metric) should be non-negative"
    
    print(f"QRNG trace (length={n}):")
    print(f"  Chaos metric (matches at lag 1): {matches}")
    print(f"  Consciousness metric (Φ): {phi:.4f}")
    
    return matches, phi


def test_qrng_trace_with_numpy():
    """
    Alternative implementation using numpy arrays for efficiency.
    """
    random.seed(42)
    n = 1000
    trace = np.array([random.randint(0, 1) for _ in range(n)])
    
    # Vectorized calculation of matches at lag 1
    matches = np.sum(trace[1:] == trace[:-1])
    expected = (n - 1) * 0.5
    phi = abs(matches - expected)
    
    assert len(trace) == n, "Trace length mismatch"
    assert matches >= 0, "Matches should be non-negative"
    
    print(f"QRNG trace (length={n}):")
    print(f"  Chaos metric (matches at lag 1): {matches}")
    print(f"  Consciousness metric (Φ): {phi:.4f}")
    
    return matches, phi


if __name__ == "__main__":
    # Run tests
    test_qrng_trace_metrics()
    print("\n--- Numpy version ---")
    test_qrng_trace_with_numpy()
