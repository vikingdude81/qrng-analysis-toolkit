import numpy as np
from test import *


class TestEntropyRobustness:
    def test_uniform_distribution(self):
        # Uniform distribution over [0, 1]
        x = np.random.uniform(0, 1, size=100)
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.5)

    def test_binary_spike_train(self):
        # Binary spike train: 0 or 1 with small noise
        x = np.random.binomial(1, 0.9, size=200) * 0.01 + np.random.randn(200) * 0.005
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.9)

    def test_all_zero_input(self):
        # All zeros input
        x = np.zeros(size=100)
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.0)

    def test_nan_inf_handling(self):
        # NaN and inf handling in input
        x = np.array([1e10, np.nan, -np.inf]) * 0.01 + np.random.randn(3) * 0.005
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.9)

    def test_large_sample_convergence(self):
        # Large sample size for convergence check
        x = np.random.uniform(0, 1, size=10000)
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.5)

    def test_large_sample_convergence_2(self):
        # Large sample size for convergence check with different distribution
        x = np.random.binomial(1, 0.9, size=10000) * 0.01 + np.random.randn(10000) * 0.005
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.9)

    def test_large_sample_convergence_3(self):
        # Large sample size for convergence check with all zeros
        x = np.zeros(size=10000)
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.0)

    def test_large_sample_convergence_4(self):
        # Large sample size for convergence check with NaN/inf
        x = np.array([1e10, np.nan, -np.inf]) * 0.01 + np.random.randn(3) * 0.005
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.9)

    def test_large_sample_convergence_5(self):
        # Large sample size for convergence check with uniform distribution
        x = np.random.uniform(0, 1, size=10000)
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.5)

    def test_large_sample_convergence_6(self):
        # Large sample size for convergence check with binary spike train
        x = np.random.binomial(1, 0.9, size=10000) * 0.01 + np.random.randn(10000) * 0.005
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.9)

    def test_large_sample_convergence_7(self):
        # Large sample size for convergence check with all zeros
        x = np.zeros(size=10000)
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.0)

    def test_large_sample_convergence_8(self):
        # Large sample size for convergence check with NaN/inf
        x = np.array([1e10, np.nan, -np.inf]) * 0.01 + np.random.randn(3) * 0.005
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.9)

    def test_large_sample_convergence_9(self):
        # Large sample size for convergence check with uniform distribution
        x = np.random.uniform(0, 1, size=10000)
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.5)

    def test_large_sample_convergence_10(self):
        # Large sample size for convergence check with binary spike train
        x = np.random.binomial(1, 0.9, size=10000) * 0.01 + np.random.randn(10000) * 0.005
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.9)

    def test_large_sample_convergence_11(self):
        # Large sample size for convergence check with all zeros
        x = np.zeros(size=10000)
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.0)

    def test_large_sample_convergence_12(self):
        # Large sample size for convergence check with NaN/inf
        x = np.array([1e10, np.nan, -np.inf]) * 0.01 + np.random.randn(3) * 0.005
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.9)

    def test_large_sample_convergence_13(self):
        # Large sample size for convergence check with uniform distribution
        x = np.random.uniform(0, 1, size=10000)
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.5)

    def test_large_sample_convergence_14(self):
        # Large sample size for convergence check with binary spike train
        x = np.random.binomial(1, 0.9, size=10000) * 0.01 + np.random.randn(10000) * 0.005
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.9)

    def test_large_sample_convergence_15(self):
        # Large sample size for convergence check with all zeros
        x = np.zeros(size=10000)
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.0)

    def test_large_sample_convergence_16(self):
        # Large sample size for convergence check with NaN/inf
        x = np.array([1e10, np.nan, -np.inf]) * 0.01 + np.random.randn(3) * 0.005
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.9)

    def test_large_sample_convergence_17(self):
        # Large sample size for convergence check with uniform distribution
        x = np.random.uniform(0, 1, size=10000)
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.5)

    def test_large_sample_convergence_18(self):
        # Large sample size for convergence check with binary spike train
        x = np.random.binomial(1, 0.9, size=10000) * 0.01 + np.random.randn(10000) * 0.005
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.9)

    def test_large_sample_convergence_19(self):
        # Large sample size for convergence check with all zeros
        x = np.zeros(size=10000)
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.0)

    def test_large_sample_convergence_20(self):
        # Large sample size for convergence check with NaN/inf
        x = np.array([1e10, np.nan, -np.inf]) * 0.01 + np.random.randn(3) * 0.005
        entropy = -np.sum(x ** 2) / (x.size + 1e-10)
        assert_allclose(entropy, 0.9)
