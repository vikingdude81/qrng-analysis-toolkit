"""
Tests for chaos analysis functions.
"""

import pytest
import numpy as np
from typing import List, Optional

from chaos_analysis import (
    wolf_lyapunov_exponent,
    rosenstein_lyapunov_exponent,
    multiscale_entropy,
    permutation_entropy,
    fuzzy_entropy,
    renyi_entropy,
    shannon_entropy,
    validate_entropy_bounds,
    compute_all_entropies,
)


class TestChaosAnalysis:
    """Test cases for chaos analysis functions."""

    def test_wolf_lyapunov_exponent_basic(self):
        """Test basic Wolf Lyapunov exponent computation."""
        data = np.random.randn(200)
        le = wolf_lyapunov_exponent(data, max_lag=100, min_lag=5)
        assert isinstance(le, (float, type(None)))

    def test_rosenstein_lyapunov_exponent_basic(self):
        """Test basic Rosenstein Lyapunov exponent computation."""
        data = np.random.randn(200)
        le = rosenstein_lyapunov_exponent(data, max_lag=100, min_lag=5)
        assert isinstance(le, (float, type(None)))

    def test_multiscale_entropy(self):
        """Test multiscale entropy computation."""
        data = np.random.randn(200)
        scale_factors = [1, 2, 3]
        mse = multiscale_entropy(data, scale_factors=scale_factors)
        assert isinstance(mse, list)
        assert len(mse) == 3

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

    def test_fuzzy_entropy(self):
        """Test fuzzy entropy computation."""
        data = np.random.randn(100)
        fe = fuzzy_entropy(data, window_size=5, scaling_factor=1.5)
        assert isinstance(fe, float)

    def test_renyi_entropy(self):
        """Test Renyi entropy computation."""
        data = np.random.randn(100)
        re = renyi_entropy(data, alpha=2.0)
        assert isinstance(re, float)

    def test_shannon_entropy(self):
        """Test Shannon entropy computation."""
        data = np.random.randn(100)
        se = shannon_entropy(data)
        assert isinstance(se, float)

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
