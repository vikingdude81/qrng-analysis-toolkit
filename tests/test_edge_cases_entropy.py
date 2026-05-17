#!/usr/bin/env python
"""
Unit Tests for Edge Cases in Entropy Estimation

Tests cover scenarios such as empty input sequences, constant sequences,
and sequences containing NaN/Inf values.

Author: AI Command Center Fan-Out Team
Date: 2024
"""

import numpy as np
import pytest
from robust_entropy_estimators import (
    RobustShannonEntropy,
    RobustSampleEntropy,
    RobustApproximateEntropy,
    entropy_estimators
)


class TestEmptyInput:
    """Test empty input sequences."""
    
    def test_shannon_empty(self):
        estimator = RobustShannonEntropy()
        result, metadata = estimator.estimate(np.array([]))
        assert result == 0.0
        assert metadata['error'] == 'empty_input'
    
    def test_sample_empty(self):
        estimator = RobustSampleEntropy()
        result, metadata = estimator.estimate(np.array([]))
        assert result == 0.0
        assert metadata['error'] == 'empty_input'
    
    def test_approximate_empty(self):
        estimator = RobustApproximateEntropy()
        result, metadata = estimator.estimate(np.array([]))
        assert result == 0.0
        assert metadata['error'] == 'empty_input'


class TestConstantSequence:
    """Test constant sequences."""
    
    def test_shannon_constant(self):
        data = np.ones(100)
        estimator = RobustShannonEntropy()
        result, metadata = estimator.estimate(data)
        assert result == 0.0
        assert metadata['error'] == 'constant_sequence'
    
    def test_sample_constant(self):
        data = np.ones(100)
        estimator = RobustSampleEntropy()
        result, metadata = estimator.estimate(data)
        assert result == 0.0
        assert metadata['error'] == 'constant_sequence'
    
    def test_approximate_constant(self):
        data = np.ones(100)
        estimator = RobustApproximateEntropy()
        result, metadata = estimator.estimate(data)
        assert result == 0.0
        assert metadata['error'] == 'constant_sequence'


class TestNaNValues:
    """Test sequences containing NaN values."""
    
    def test_shannon_nan(self):
        data = np.concatenate([np.ones(90), np.nan, np.ones(10)])
        estimator = RobustShannonEntropy()
        result, metadata = estimator.estimate(data)
        # Should handle NaN gracefully
        assert isinstance(result, float)
    
    def test_sample_nan(self):
        data = np.concatenate([np.ones(90), np.nan, np.ones(10)])
        estimator = RobustSampleEntropy()
        result, metadata = estimator.estimate(data)
        # Should handle NaN gracefully
        assert isinstance(result, float)
    
    def test_approximate_nan(self):
        data = np.concatenate([np.ones(90), np.nan, np.ones(10)])
        estimator = RobustApproximateEntropy()
        result, metadata = estimator.estimate(data)
        # Should handle NaN gracefully
        assert isinstance(result, float)


class TestInfValues:
    """Test sequences containing Inf values."""
    
    def test_shannon_posinf(self):
        data = np.concatenate([np.ones(90), np.inf, np.ones(10)])
        estimator = RobustShannonEntropy()
        result, metadata = estimator.estimate(data)
        # Should handle Inf gracefully
        assert isinstance(result, float)
    
    def test_shannon_neginf(self):
        data = np.concatenate([np.ones(90), -np.inf, np.ones(10)])
        estimator = RobustShannonEntropy()
        result, metadata = estimator.estimate(data)
        # Should handle Inf gracefully
        assert isinstance(result, float)
    
    def test_sample_posinf(self):
        data = np.concatenate([np.ones(90), np.inf, np.ones(10)])
        estimator = RobustSampleEntropy()
        result, metadata = estimator.estimate(data)
        # Should handle Inf gracefully
        assert isinstance(result, float)
    
    def test_approximate_posinf(self):
        data = np.concatenate([np.ones(90), np.inf, np.ones(10)])
        estimator = RobustApproximateEntropy()
        result, metadata = estimator.estimate(data)
        # Should handle Inf gracefully
        assert isinstance(result, float)


class TestSmallSampleSize:
    """Test small sample sizes."""
    
    def test_shannon_small(self):
        data = np.random.randn(5)
        estimator = RobustShannonEntropy()
        result, metadata = estimator.estimate(data)
        # Should return a valid entropy estimate
        assert isinstance(result, float)
    
    def test_sample_small(self):
        data = np.random.randn(10)
        estimator = RobustSampleEntropy()
        result, metadata = estimator.estimate(data)
        # May return 0.0 if sample too small
        assert isinstance(result, float)
    
    def test_approximate_small(self):
        data = np.random.randn(10)
        estimator = RobustApproximateEntropy()
        result, metadata = estimator.estimate(data)
        # May return 0.0 if sample too small
        assert isinstance(result, float)


class TestBootstrapFallback:
    """Test bootstrap fallback mechanism."""
    
    def test_shannon_bootstrap(self):
        data = np.random.randn(100)
        estimator = RobustShannonEntropy()
        result, metadata = estimator.estimate(data)
        # Should return a valid entropy estimate
        assert isinstance(result, float)
        assert result >= 0.0
    
    def test_sample_bootstrap(self):
        data = np.random.randn(200)
        estimator = RobustSampleEntropy()
        result, metadata = estimator.estimate(data)
        # Should return a valid entropy estimate
        assert isinstance(result, float)
        assert result >= 0.0
    
    def test_approximate_bootstrap(self):
        data = np.random.randn(200)
        estimator = RobustApproximateEntropy()
        result, metadata = estimator.estimate(data)
        # Should return a valid entropy estimate
        assert isinstance(result, float)
        assert result >= 0.0


class TestEntropyEstimatorsFunction:
    """Test the entropy_estimators convenience function."""
    
    def test_all_estimators(self):
        data = np.random.randn(1000)
        results = entropy_estimators(data)
        # Should return results for all estimators
        assert 'shannon' in results
        assert 'sample' in results
        assert 'approximate' in results
    
    def test_specific_estimators(self):
        data = np.random.randn(1000)
        results = entropy_estimators(data, estimators=['shannon', 'sample'])
        # Should return results for specified estimators
        assert 'shannon' in results
        assert 'sample' in results
        assert 'approximate' not in results
    
    def test_empty_data(self):
        data = np.array([])
        results = entropy_estimators(data)
        # Should handle empty data gracefully
        assert isinstance(results, dict)


class TestEdgeCases:
    """Test various edge cases."""
    
    def test_single_value(self):
        data = np.array([1.0])
        estimator = RobustShannonEntropy()
        result, metadata = estimator.estimate(data)
        # Should handle single value gracefully
        assert isinstance(result, float)
    
    def test_two_values(self):
        data = np.array([1.0, 2.0])
        estimator = RobustShannonEntropy()
        result, metadata = estimator.estimate(data)
        # Should handle two values gracefully
        assert isinstance(result, float)
    
    def test_large_values(self):
        data = np.random.randn(100) * 1e6
        estimator = RobustShannonEntropy()
        result, metadata = estimator.estimate(data)
        # Should handle large values gracefully
        assert isinstance(result, float)
    
    def test_mixed_signs(self):
        data = np.concatenate([np.random.randn(50), -np.random.randn(50)])
        estimator = RobustShannonEntropy()
        result, metadata = estimator.estimate(data)
        # Should handle mixed signs gracefully
        assert isinstance(result, float)
    
    def test_monotonic_increasing(self):
        data = np.arange(1000, dtype=float)
        estimator = RobustShannonEntropy()
        result, metadata = estimator.estimate(data)
        # Should handle monotonic increasing gracefully
        assert isinstance(result, float)
    
    def test_monotonic_decreasing(self):
        data = -np.arange(1000, dtype=float)
        estimator = RobustShannonEntropy()
        result, metadata = estimator.estimate(data)
        # Should handle monotonic decreasing gracefully
        assert isinstance(result, float)


class TestReproducibility:
    """Test reproducibility of estimates."""
    
    def test_shannon_reproducible(self):
        np.random.seed(42)
        data = np.cumsum(np.random.randn(1000))
        estimator = RobustShannonEntropy()
        result1, _ = estimator.estimate(data)
        result2, _ = estimator.estimate(data)
        assert abs(result1 - result2) < 1e-10
    
    def test_sample_reproducible(self):
        np.random.seed(42)
        data = np.cumsum(np.random.randn(1000))
        estimator = RobustSampleEntropy()
        result1, _ = estimator.estimate(data)
        result2, _ = estimator.estimate(data)
        assert abs(result1 - result2) < 1e-10
    
    def test_approximate_reproducible(self):
        np.random.seed(42)
        data = np.cumsum(np.random.randn(1000))
        estimator = RobustApproximateEntropy()
        result1, _ = estimator.estimate(data)
        result2, _ = estimator.estimate(data)
        assert abs(result1 - result2) < 1e-10


if __name__ == "__main__":
    # Run tests manually
    print("Running edge case tests for entropy estimation...")
    
    test_empty_input = TestEmptyInput()
    test_constant_sequence = TestConstantSequence()
    test_nan_values = TestNaNValues()
    test_inf_values = TestInfValues()
    test_small_sample_size = TestSmallSampleSize()
    test_bootstrap_fallback = TestBootstrapFallback()
    test_entropy_estimators_function = TestEntropyEstimatorsFunction()
    test_edge_cases = TestEdgeCases()
    test_reproducibility = TestReproducibility()
    
    all_tests = [
        test_empty_input,
        test_constant_sequence,
        test_nan_values,
        test_inf_values,
        test_small_sample_size,
        test_bootstrap_fallback,
        test_entropy_estimators_function,
        test_edge_cases,
        test_reproducibility
    ]
    
    failed = 0
    for test_class in all_tests:
        for method_name in dir(test_class):
            if method_name.startswith('test_'):
                method = getattr(test_class, method_name)
                try:
                    method()
                    print(f"PASS: {test_class.__name__}.{method_name}")
                except AssertionError as e:
                    print(f"FAIL: {test_class.__name__}.{method_name}: {e}")
                    failed += 1
    
    if failed == 0:
        print("\nAll edge case tests passed!")
    else:
        print(f"\n{failed} tests failed.")