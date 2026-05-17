"""
Unit tests for deep_pattern_analysis module.

Tests edge cases including:
- Empty streams
- NaNs in input
- Non-i.i.d. inputs (using hypothesis)
- Precision edge cases
"""

import numpy as np
import pytest
from hypothesis import given, strategies as st
from deep_pattern_analysis import detect_deep_patterns, compute_pattern_complexity


def test_empty_stream():
    """Test that empty stream returns empty result."""
    result = detect_deep_patterns([])
    assert result == []


def test_nan_input():
    """Test that NaNs in input are handled correctly."""
    data = [np.nan, 1.0, 2.0]
    result = detect_deep_patterns(data)
    assert np.isnan(result[0])


def test_infinity_input():
    """Test that infinities in input are handled correctly."""
    data = [float('inf'), 1.0, 2.0]
    result = detect_deep_patterns(data)
    assert np.isinf(result[0])


def test_precision_edge_cases():
    """Test precision edge cases."""
    # Very small values
    data_small = [1e-15, 1e-14, 1e-13]
    result_small = detect_deep_patterns(data_small)
    assert len(result_small) == len(data_small)
    
    # Very large values
    data_large = [1e10, 1e11, 1e12]
    result_large = detect_deep_patterns(data_large)
    assert len(result_large) == len(data_large)


def test_non_iid_inputs():
    """Test non-i.i.d. inputs using hypothesis."""
    @given(st.lists(st.floats(allow_nan=False), min_size=2))
    def test_non_iid(data):
        result = detect_deep_patterns(data)
        assert len(result) == len(data)
    
    test_non_iid()


def test_normal_distribution():
    """Test with normally distributed data."""
    np.random.seed(42)
    data = np.random.normal(0, 1, 100)
    result = detect_deep_patterns(data)
    assert len(result) == len(data)


def test_uniform_distribution():
    """Test with uniformly distributed data."""
    np.random.seed(42)
    data = np.random.uniform(0, 1, 100)
    result = detect_deep_patterns(data)
    assert len(result) == len(data)


def test_constant_input():
    """Test with constant input."""
    data = [1.0] * 100
    result = detect_deep_patterns(data)
    assert len(result) == len(data)


def test_alternating_pattern():
    """Test with alternating pattern (non-i.i.d.)."""
    data = [i % 2 for i in range(100)]
    result = detect_deep_patterns(data)
    assert len(result) == len(data)


def test_large_stream():
    """Test with large stream."""
    np.random.seed(42)
    data = np.random.randn(10000)
    result = detect_deep_patterns(data)
    assert len(result) == len(data)


def test_small_stream():
    """Test with small stream."""
    data = [1.0, 2.0, 3.0]
    result = detect_deep_patterns(data)
    assert len(result) == len(data)


class TestDeepPatternAnalysis:
    """Integration tests for deep_pattern_analysis."""
    
    def test_integration_with_qrng_data(self):
        """Test integration with QRNG data."""
        # Simulate QRNG output
        np.random.seed(42)
        qrng_data = np.random.randint(0, 256, 1000)
        
        result = detect_deep_patterns(qrng_data.astype(float))
        assert len(result) == len(qrng_data)
    
    def test_integration_with_continuous_qrng(self):
        """Test integration with continuous QRNG data."""
        # Simulate continuous QRNG output
        np.random.seed(42)
        qrng_data = np.random.normal(0, 1, 1000)
        
        result = detect_deep_patterns(qrng_data)
        assert len(result) == len(qrng_data)
    
    def test_pattern_complexity_calculation(self):
        """Test pattern complexity calculation."""
        np.random.seed(42)
        data = np.random.randn(100)
        result = compute_pattern_complexity(data)
        assert len(result) == len(data)
