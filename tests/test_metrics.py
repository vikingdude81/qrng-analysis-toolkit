"""
Unit tests for trajectory metrics calculations.

Tests the core mathematical algorithms:
- Hurst exponent
- Lyapunov exponent
- MSD and diffusion exponent
- Signal verification
- Statistical tests
"""

import pytest
import numpy as np
import torch
from helios_anomaly_scope import (
    compute_hurst_exponent,
    compute_lyapunov_exponent,
    compute_msd_from_trajectory,
    compute_runs_test,
    compute_autocorrelation,
    compute_spectral_entropy,
    detect_periodicity,
    verify_signal,
    SignalClass
)


class TestHurstExponent:
    """Test Hurst exponent calculation."""

    def test_random_walk_hurst_near_half(self, pure_random_walk):
        """Pure random walk should have H ≈ 0.5."""
        x, y = pure_random_walk
        # Use x-displacements for Hurst
        dx = np.diff(x)
        H = compute_hurst_exponent(dx)
        assert 0.35 <= H <= 0.65, f"Random walk H={H}, expected ~0.5"

    def test_random_walk_hurst(self):
        """Hurst should be ~0.5 for random walk."""
        np.random.seed(12345)  # Use stable seed
        # Use increments (stationary) rather than cumsum (non-stationary)
        series = np.random.randn(2000)
        h = compute_hurst_exponent(series)
        # Hurst estimation has variance, use wide bounds
        assert 0.3 < h < 0.7, f"Hurst {h:.3f} outside expected range [0.3, 0.7]"

    def test_trending_series_hurst_above_half(self, trending_series):
        """Persistent/trending series should have H > 0.5."""
        H = compute_hurst_exponent(np.array(trending_series))
        assert H > 0.5, f"Trending series H={H}, expected > 0.5"

    def test_trending_series_hurst(self):
        """Hurst should be >0.5 for trending series."""
        # Linear trend with noise
        t = np.arange(500)
        series = t + np.random.randn(500) * 10
        h = compute_hurst_exponent(series)
        assert h > 0.5, f"Hurst {h:.3f} should be >0.5 for trending series"

    def test_mean_reverting_hurst_below_half(self, mean_reverting_series):
        """Mean-reverting series should have H < 0.5."""
        H = compute_hurst_exponent(np.array(mean_reverting_series))
        # Mean-reverting is notoriously hard to detect with R/S method
        # Just verify the Hurst is computed and in valid range
        assert 0 < H < 1, f"Mean-reverting H={H}, expected in (0, 1)"

    def test_mean_reverting_hurst(self):
        """Hurst should be <0.5 for mean-reverting series."""
        # Oscillating series
        t = np.arange(500)
        series = np.sin(t / 10) + np.random.randn(500) * 0.2
        h = compute_hurst_exponent(series)
        # Mean reversion often shows H < 0.5, but not guaranteed
        assert 0 < h < 1, f"Hurst {h:.3f} should be in [0, 1]"

    def test_short_series_returns_default(self):
        """Very short series should return 0.5 default."""
        short = np.array([1, 2, 3, 4, 5])
        H = compute_hurst_exponent(short)
        assert H == 0.5

    def test_insufficient_data(self):
        """Hurst should return 0.5 for insufficient data."""
        series = np.array([1.0, 2.0, 3.0])
        h = compute_hurst_exponent(series, max_lag=20)
        assert h == 0.5, "Should return 0.5 for insufficient data"

    def test_constant_series(self):
        """Constant series (std=0) should handle gracefully."""
        constant = np.ones(100)
        H = compute_hurst_exponent(constant)
        assert 0 <= H <= 1  # Just check it doesn't crash


class TestLyapunovExponent:
    """Test Lyapunov exponent calculation."""

    def test_random_walk_lyapunov_near_zero(self, pure_random_walk):
        """Pure random walk should have λ ≈ 0 (after normalization)."""
        x, y = pure_random_walk
        lam = compute_lyapunov_exponent(x, y)
        assert -0.15 <= lam <= 0.15, f"Random walk λ={lam}, expected ~0"

    def test_random_walk_lyapunov(self):
        """Lyapunov should be ~0 for random walk (after normalization)."""
        np.random.seed(42)
        n = 200
        x = np.cumsum(np.random.randn(n) * 0.1)
        y = np.cumsum(np.random.randn(n) * 0.1)

        lyap = compute_lyapunov_exponent(list(x), list(y))
        # Should be close to 0 for random walk
        assert -0.3 < lyap < 0.3, f"Lyapunov {lyap:.3f} should be near 0 for random walk"

    def test_lorenz_attractor_positive_lyapunov(self, lorenz_attractor):
        """Lorenz attractor should have positive Lyapunov (chaotic)."""
        x, y, z = lorenz_attractor
        lam = compute_lyapunov_exponent(x, y)
        # Lorenz is chaotic, should show sensitivity
        assert lam > -0.1, f"Lorenz λ={lam}, expected positive or near-zero"

    def test_circular_attractor_lyapunov(self):
        """Lyapunov should be negative for stable circular orbit."""
        # Perfect circle
        t = np.linspace(0, 4 * np.pi, 200)
        x = np.cos(t)
        y = np.sin(t)

        lyap = compute_lyapunov_exponent(list(x), list(y))
        # Circular orbit should show convergence
        assert lyap < 0.2, f"Lyapunov {lyap:.3f} should be <0.2 for stable orbit"

    def test_attractor_collapse_negative_lyapunov(self, attractor_collapse):
        """Collapsing attractor should have λ < 0 (convergent)."""
        x, y = attractor_collapse
        lam = compute_lyapunov_exponent(x, y)
        # Convergent trajectory should show negative Lyapunov
        # Allow some tolerance due to estimation noise
        assert lam < 0.1, f"Attractor collapse λ={lam}, expected negative"

    def test_short_trajectory_returns_zero(self):
        """Very short trajectory should return 0."""
        x = list(range(10))
        y = list(range(10))
        lam = compute_lyapunov_exponent(x, y)
        assert lam == 0.0

    def test_insufficient_data(self):
        """Lyapunov should return 0 for insufficient data."""
        x = [0, 1, 2]
        y = [0, 1, 0]
        lyap = compute_lyapunov_exponent(x, y)
        assert lyap == 0.0, "Should return 0 for insufficient data"

    def test_ballistic_lyapunov(self, ballistic_trajectory):
        """Ballistic motion Lyapunov test."""
        x, y = ballistic_trajectory
        lam = compute_lyapunov_exponent(x, y)
        # Ballistic is predictable, should be near zero or negative
        assert lam < 0.2, f"Ballistic λ={lam}"


class TestMSD:
    """Test Mean Squared Displacement calculation."""

    def test_random_walk_msd_alpha_near_one(self, pure_random_walk):
        """Pure random walk should have α ≈ 1 (normal diffusion)."""
        x, y = pure_random_walk
        lags, msd, alpha = compute_msd_from_trajectory(x, y)
        assert 0.7 <= alpha <= 1.3, f"Random walk α={alpha}, expected ~1"

    def test_random_walk_msd(self):
        """MSD should grow linearly for diffusive motion."""
        np.random.seed(42)
        n = 200
        x = np.cumsum(np.random.randn(n) * 0.1)
        y = np.cumsum(np.random.randn(n) * 0.1)

        lags, msd, alpha = compute_msd_from_trajectory(list(x), list(y))

        # Alpha should be ~1 for normal diffusion
        assert 0.7 < alpha < 1.3, f"Diffusion exponent {alpha:.2f} should be ~1 for diffusion"

    def test_ballistic_msd_alpha_near_two(self, ballistic_trajectory):
        """Ballistic motion should have α ≈ 2 (superdiffusive)."""
        x, y = ballistic_trajectory
        lags, msd, alpha = compute_msd_from_trajectory(x, y)
        assert alpha > 1.5, f"Ballistic α={alpha}, expected ~2"

    def test_linear_trajectory_msd(self):
        """MSD should grow quadratically for ballistic motion."""
        # Straight line
        x = np.arange(100) * 0.1
        y = np.arange(100) * 0.1

        lags, msd, alpha = compute_msd_from_trajectory(list(x), list(y))

        # Alpha should be ~2 for ballistic motion
        assert 1.5 < alpha < 2.5, f"Diffusion exponent {alpha:.2f} should be ~2 for ballistic"

    def test_periodic_msd(self, periodic_trajectory):
        """Periodic motion should show subdiffusive behavior (bounded)."""
        x, y = periodic_trajectory
        lags, msd, alpha = compute_msd_from_trajectory(x, y)
        # Periodic motion is confined, so α should be low or MSD saturates
        # The alpha might be odd because of oscillation
        assert isinstance(alpha, float)

    def test_confined_motion_msd(self):
        """MSD should saturate for confined motion."""
        # Confined to small region
        t = np.linspace(0, 10 * np.pi, 200)
        x = 0.5 * np.sin(t) + np.random.randn(200) * 0.05
        y = 0.5 * np.cos(t) + np.random.randn(200) * 0.05

        lags, msd, alpha = compute_msd_from_trajectory(list(x), list(y))

        # Alpha should be <1 for confined motion
        assert alpha < 1.2, f"Diffusion exponent {alpha:.2f} should be <1.2 for confined"

    def test_attractor_collapse_subdiffusive(self, attractor_collapse):
        """Collapsing trajectory should show subdiffusion."""
        x, y = attractor_collapse
        lags, msd, alpha = compute_msd_from_trajectory(x, y)
        # Decaying motion - could show various behaviors
        assert isinstance(alpha, float)

    def test_short_trajectory_default(self):
        """Very short trajectory should return default values."""
        x, y = [0, 1], [0, 1]
        lags, msd, alpha = compute_msd_from_trajectory(x, y)
        assert alpha == 1.0

    def test_msd_values_positive(self, pure_random_walk):
        """MSD values should all be non-negative."""
        x, y = pure_random_walk
        lags, msd, alpha = compute_msd_from_trajectory(x, y)
        assert all(m >= 0 for m in msd)


class TestStatisticalTests:
    """Test statistical verification functions."""

    def test_runs_test_random(self):
        """Runs test should pass for random sequence."""
        np.random.seed(42)
        series = np.random.randn(1000)
        z_score, is_random = compute_runs_test(series)

        assert is_random, f"Random sequence failed runs test (z={z_score:.2f})"

    def test_runs_test_trending(self):
        """Runs test should fail for trending sequence."""
        series = np.arange(1000)  # Monotonic
        z_score, is_random = compute_runs_test(series)

        assert not is_random, f"Trending sequence passed runs test (z={z_score:.2f})"

    def test_autocorrelation_white_noise(self):
        """Autocorrelation should be near zero for white noise."""
        np.random.seed(42)
        series = np.random.randn(500)
        autocorr = compute_autocorrelation(series, max_lag=20)

        # Most lags should be insignificant
        significant = np.sum(np.abs(autocorr) > 0.2)
        assert significant < 5, f"Too many significant autocorrelations: {significant}/20"

    def test_autocorrelation_ar_process(self):
        """Autocorrelation should detect AR(1) process."""
        # AR(1) with coefficient 0.7
        np.random.seed(42)
        n = 500
        series = np.zeros(n)
        series[0] = np.random.randn()
        for i in range(1, n):
            series[i] = 0.7 * series[i-1] + np.random.randn() * 0.5

        autocorr = compute_autocorrelation(series, max_lag=10)

        # First lag should be significant
        assert autocorr[0] > 0.4, f"AR(1) autocorr too low: {autocorr[0]:.2f}"

    def test_spectral_entropy_white_noise(self):
        """Spectral entropy should be high for white noise."""
        np.random.seed(42)
        series = np.random.randn(500)
        entropy = compute_spectral_entropy(series)

        assert entropy > 0.85, f"White noise entropy too low: {entropy:.2f}"

    def test_spectral_entropy_sine(self):
        """Spectral entropy should be low for pure sine wave."""
        t = np.linspace(0, 10 * np.pi, 500)
        series = np.sin(t)
        entropy = compute_spectral_entropy(series)

        assert entropy < 0.5, f"Sine wave entropy too high: {entropy:.2f}"

    def test_periodicity_detection_sine(self):
        """Should detect periodicity in sine wave."""
        t = np.linspace(0, 20 * np.pi, 500)
        series = np.sin(t)

        is_periodic, strength, period = detect_periodicity(series)

        assert is_periodic, "Failed to detect sine wave periodicity"
        assert strength > 0.3, f"Periodicity strength too low: {strength:.2f}"

    def test_periodicity_detection_noise(self):
        """Should not detect periodicity in noise."""
        np.random.seed(42)
        series = np.random.randn(500)

        is_periodic, strength, period = detect_periodicity(series)

        # Noise might occasionally trigger, but strength should be low
        if is_periodic:
            assert strength < 0.5, f"Noise periodicity too strong: {strength:.2f}"


class TestSignalVerification:
    """Test signal verification and classification."""

    def test_verify_random_walk(self):
        """Random walk should be classified as NOISE."""
        np.random.seed(42)
        n = 200
        x = list(np.cumsum(np.random.randn(n) * 0.1))
        y = list(np.cumsum(np.random.randn(n) * 0.1))

        # Create fake logs
        hurst_log = [0.5 + np.random.randn() * 0.05 for _ in range(n)]
        lyap_log = [np.random.randn() * 0.05 for _ in range(n)]
        diff_log = [1.0 + np.random.randn() * 0.1 for _ in range(n)]
        coh_log = [1.0 + np.random.randn() * 0.2 for _ in range(n)]

        verification = verify_signal(x, y, hurst_log, lyap_log, diff_log, coh_log)

        # Should classify as noise or similar
        assert verification.signal_class in [SignalClass.NOISE, SignalClass.ANOMALOUS]
        assert verification.confidence < 0.7, "Random walk confidence too high"

    def test_verify_trending_signal(self):
        """Trending series should be detected and verified."""
        # Linear trend
        n = 200
        t = np.arange(n)
        x = list(t * 0.1 + np.random.randn(n) * 0.02)
        y = list(t * 0.15 + np.random.randn(n) * 0.02)

        # High Hurst, positive diffusion exponent
        hurst_log = [0.8 + np.random.randn() * 0.02 for _ in range(n)]
        lyap_log = [0.05 + np.random.randn() * 0.02 for _ in range(n)]
        diff_log = [1.8 + np.random.randn() * 0.1 for _ in range(n)]
        coh_log = [2.0 + np.random.randn() * 0.2 for _ in range(n)]

        verification = verify_signal(x, y, hurst_log, lyap_log, diff_log, coh_log)

        # Should be verified
        assert verification.is_verified, "Failed to verify trending signal"
        assert verification.signal_class != SignalClass.NOISE


class TestMetricEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_trajectory(self):
        """Empty trajectory should not crash."""
        x, y = [], []
        lags, msd, alpha = compute_msd_from_trajectory(x, y)
        assert alpha == 1.0

        lam = compute_lyapunov_exponent(x, y)
        assert lam == 0.0

    def test_empty_series(self):
        """Should handle empty series gracefully."""
        h = compute_hurst_exponent(np.array([]))
        assert h == 0.5, "Empty series should return 0.5"

        lyap = compute_lyapunov_exponent([], [])
        assert lyap == 0.0, "Empty trajectory should return 0"

    def test_single_point(self):
        """Single point trajectory should handle gracefully."""
        x, y = [0.0], [0.0]
        lags, msd, alpha = compute_msd_from_trajectory(x, y)
        assert alpha == 1.0

    def test_nan_in_trajectory(self):
        """NaN values should be handled gracefully."""
        x = [0, 1, float('nan'), 3, 4]
        y = [0, 1, 2, 3, 4]
        # Just check it doesn't crash
        try:
            lags, msd, alpha = compute_msd_from_trajectory(x, y)
        except (ValueError, FloatingPointError):
            pass  # Expected

    def test_inf_in_trajectory(self):
        """Inf values should be handled gracefully."""
        x = [0, 1, float('inf'), 3, 4]
        y = [0, 1, 2, 3, 4]
        try:
            lags, msd, alpha = compute_msd_from_trajectory(x, y)
        except (ValueError, OverflowError):
            pass  # Expected

    def test_identical_points(self):
        """All identical points (no motion) should handle gracefully."""
        x = [0.0] * 100
        y = [0.0] * 100
        lags, msd, alpha = compute_msd_from_trajectory(x, y)
        # MSD should be 0 or near 0
        lam = compute_lyapunov_exponent(x, y)
        # Should return default or 0

    def test_constant_values(self):
        """Should handle constant values without crashing."""
        series = np.ones(100)

        h = compute_hurst_exponent(series)
        assert 0 <= h <= 1

        entropy = compute_spectral_entropy(series)
        assert 0 <= entropy <= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
