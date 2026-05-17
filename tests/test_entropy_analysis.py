"""
Tests for entropy analysis functions.
"""

import pytest
import numpy as np
from typing import List

from entropy_analysis import (
    sample_entropy,
    approximate_entropy,
    multiscale_entropy,
    permutation_entropy,
    fuzzy_entropy,
    renyi_entropy,
    shannon_entropy,
    validate_entropy_bounds,
    compute_all_entropies,
)


class TestEntropyAnalysis:
    """Test cases for entropy analysis functions."""

    def test_sample_entropy_basic(self):
        """Test basic sample entropy computation."""
        data = np.random.randn(100)
        se = sample_entropy(data, window_size=5, scaling_factor=1.5)
        assert isinstance(se, float)
        assert not np.isnan(se)

    def test_sample_entropy_short_series(self):
        """Test sample entropy with short series returns nan."""
        data = np.random.randn(10)
        se = sample_entropy(data, window_size=5, scaling_factor=1.5)
        assert np.isnan(se)

    def test_approximate_entropy_basic(self):
        """Test basic approximate entropy computation."""
        data = np.random.randn(100)
        ae = approximate_entropy(data, window_size=5, scaling_factor=1.5)
        assert isinstance(ae, float)
        assert not np.isnan(ae)

    def test_multiscale_entropy(self):
        """Test multiscale entropy computation."""
        data = np.random.randn(200)
        scale_factors = [1, 2, 3]
        mse = multiscale_entropy(data, scale_factors=scale_factors)
        assert isinstance(mse, list)
        assert len(mse) == 3
        for e in mse:
            assert isinstance(e, float)

    def test_multiscale_entropy_no_scale_factors(self):
        """Test multiscale entropy with default scale factors."""
        data = np.random.randn(200)
        mse = multiscale_entropy(data)
        assert isinstance(mse, list)
        assert len(mse) > 0

    def test_permutation_entropy(self):
        """Test permutation entropy computation."""
        data = np.random.randn(100)
        pe = permutation_entropy(data, n_perm=100)
        assert isinstance(pe, float)
        assert not np.isnan(pe)

    def test_fuzzy_entropy(self):
        """Test fuzzy entropy computation."""
        data = np.random.randn(100)
        fe = fuzzy_entropy(data, window_size=5, scaling_factor=1.5)
        assert isinstance(fe, float)
        assert not np.isnan(fe)

    def test_renyi_entropy(self):
        """Test Renyi entropy computation."""
        data = np.random.randn(100)
        re = renyi_entropy(data, alpha=2.0)
        assert isinstance(re, float)
        assert not np.isnan(re)

    def test_shannon_entropy(self):
        """Test Shannon entropy computation."""
        data = np.random.randn(100)
        se = shannon_entropy(data)
        assert isinstance(se, float)
        assert not np.isnan(se)

    def test_validate_entropy_bounds_valid(self):
        """Test entropy bounds validation with valid value."""
        entropy_value = 2.5
        data_length = 100
        result = validate_entropy_bounds(entropy_value, data_length)
        assert result is True

    def test_validate_entropy_bounds_invalid(self):
        """Test entropy bounds validation with invalid value."""
        entropy_value = 10.0  # Exceeds log2(100) ≈ 6.64
        data_length = 100
        result = validate_entropy_bounds(entropy_value, data_length)
        assert result is False

    def test_compute_all_entropies(self):
        """Test computing all entropy measures."""
        data = np.random.randn(200)
        entropies = compute_all_entropies(data, window_size=5, scaling_factor=1.5)
        assert isinstance(entropies, dict)
        assert "sample_entropy" in entropies
        assert "approximate_entropy" in entropies
        assert "permutation_entropy" in entropies
        assert "shannon_entropy" in entropies
        assert "multiscale_entropy" in entropies

    def test_compute_all_entropies_no_scale_factors(self):
        """Test computing all entropy measures without scale factors."""
        data = np.random.randn(200)
        entropies = compute_all_entropies(data, window_size=5, scaling_factor=1.5)
        assert entropies["multiscale_entropy"] is None
