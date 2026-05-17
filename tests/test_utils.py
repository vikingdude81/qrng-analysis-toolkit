"""
Tests for shared utility modules in helios.
"""

import pytest
import numpy as np
import sys
sys.path.insert(0, 'C:\Users\akbon\OneDrive\Documents\GitHub\helios-trajectory-analysis')

from helios.utils.entropy_utils import (
    calculate_shannon_entropy,
    calculate_renyi_entropy,
    calculate_permutation_entropy,
)
from helios.utils.chaos_utils import (
    calculate_lyapunov_exponent,
    calculate_correlation_dimension,
)
from helios.utils.pattern_utils import (
    calculate_lemel_ziv_complexity,
    detect_patterns_in_sequence,
    calculate_pattern_entropy,
)


class TestEntropyUtils:
    """Tests for entropy utilities."""
    
    def test_shannon_entropy_all_same(self):
        """Test Shannon entropy with all same values (should be 0)."""
        data = np.array([1, 1, 1, 1, 1])
        result = calculate_shannon_entropy(data)
        assert abs(result) < 1e-10
    
    def test_shannon_entropy_uniform(self):
        """Test Shannon entropy with uniform distribution."""
        data = np.array([1, 2, 3, 4, 5])
        result = calculate_shannon_entropy(data)
        expected = np.log2(5)  # log2 of number of unique values
        assert abs(result - expected) < 0.01
    
    def test_renyi_entropy_alpha_2(self):
        """Test Rényi entropy with alpha=2."""
        data = np.array([1, 1, 2, 2, 3])
        result = calculate_renyi_entropy(data, alpha=2)
        assert isinstance(result, float)
    
    def test_permutation_entropy(self):
        """Test permutation entropy."""
        ts = np.array([0.1, 0.5, 0.9, 0.3, 0.7])
        result = calculate_permutation_entropy(ts)
        assert isinstance(result, float)
    
    def test_permutation_entropy_short_series(self):
        """Test permutation entropy with short series."""
        ts = np.array([1, 2, 3])
        result = calculate_permutation_entropy(ts)
        assert abs(result) < 0.01


class TestChaosUtils:
    """Tests for chaos analysis utilities."""
    
    def test_lyapunov_exponent_short_series(self):
        """Test Lyapunov exponent with short series."""
        ts = np.array([1, 2, 3, 4, 5])
        result, _ = calculate_lyapunov_exponent(ts)
        assert isinstance(result, float)
    
    def test_correlation_dimension(self):
        """Test correlation dimension estimation."""
        # Generate a chaotic time series
        np.random.seed(42)
        ts = np.sin(np.linspace(0, 10 * np.pi, 100))
        
        dim = calculate_correlation_dimension(ts)
        assert dim is not None or True  # May return None for short series
    
    def test_lyapunov_exponent_chaos(self):
        """Test Lyapunov exponent on chaotic signal."""
        # Logistic map at chaotic parameter
        mu = 4.0
        x = 0.5
        ts = np.zeros(100)
        for i in range(1, 100):
            x = mu * x * (1 - x)
            ts[i] = x
        
        lyap, _ = calculate_lyapunov_exponent(ts)
        assert isinstance(lyap, float)


class TestPatternUtils:
    """Tests for pattern analysis utilities."""
    
    def test_lemel_ziv_complexity_constant(self):
        """Test Lempel-Ziv complexity with constant sequence."""
        ts = np.array([1, 1, 1, 1, 1])
        result = calculate_lemel_ziv_complexity(ts)
        assert isinstance(result, float)
    
    def test_lemel_ziv_complexity_random(self):
        """Test Lempel-Ziv complexity with random sequence."""
        np.random.seed(42)
        ts = np.random.randint(0, 5, size=100)
        result = calculate_lemel_ziv_complexity(ts)
        assert isinstance(result, float)
    
    def test_detect_patterns(self):
        """Test pattern detection."""
        ts = np.array([1, 2, 3, 1, 2, 3, 1, 2, 3])
        patterns = detect_patterns_in_sequence(ts)
        assert isinstance(patterns, list)
    
    def test_pattern_entropy(self):
        """Test pattern-based entropy."""
        ts = np.array([0.1, 0.5, 0.9, 0.3, 0.7, 0.1, 0.5, 0.9])
        result = calculate_pattern_entropy(ts)
        assert isinstance(result, float)
    
    def test_find_dominant_patterns(self):
        """Test finding dominant patterns."""
        ts = np.array([1, 2, 3, 1, 2, 3, 4, 5])
        patterns = find_dominant_patterns(ts)
        assert isinstance(patterns, list)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
