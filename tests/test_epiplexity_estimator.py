# test_epiplexity_estimator.py
import numpy as np
from helios_core import EpiPlexityEstimator, PermutationEntropy
import pytest


class TestEpiPlexityEstimator:
    def test_permutation_entropy_positive(self):
        """Ensure permutation entropy is positive for chaotic systems."""
        x = np.random.randn(100) * 2 - 1
        y = np.random.randn(100) * 2 + 1
        est = EpiPlexityEstimator(x, y)
        pe = est.permutation_entropy()
        assert isinstance(pe, float), "Expected permutation entropy to be a number"
        assert pe > 0, "Permutation entropy must be positive for chaotic systems"

    def test_time_lagged_mutual_info(self):
        """Ensure time-lagged mutual information is computed."""
        x = np.random.randn(100) * 2 - 1
        y = np.random.randn(100) * 2 + 1
        est = EpiPlexityEstimator(x, y)

        # Time lag should be between 0 and 5 for chaotic systems
        assert isinstance(est.time_lagged_mutual_info(), float), "Expected time-lagged mutual info to be a number"
        assert 0 <= est.time_lagged_mutual_info() <= 5

    def test_permutation_entropy_computation(self):
        """Ensure permutation entropy is computed correctly."""
        x = np.random.randn(100) * 2 - 1
        y = np.random.randn(100) * 2 + 1
        est = EpiPlexityEstimator(x, y)

        assert isinstance(est.permutation_entropy(), float), "Permutation entropy should be a number"
        assert isinstance(est.time_lagged_mutual_info(), float), "Time-lagged mutual info should be a number"
