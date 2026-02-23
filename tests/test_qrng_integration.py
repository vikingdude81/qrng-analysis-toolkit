#!/usr/bin/env python3
"""
Integration Tests with Real QRNG Data
=====================================

Tests analysis pipeline using actual QRNG stream data to ensure:
1. Data loading works correctly
2. Metrics are within expected ranges
3. Different sources produce comparable quality
4. Pipeline handles real-world data gracefully
"""

import json
import pytest
import numpy as np
from pathlib import Path
from typing import Optional

# Import modules under test
from helios_anomaly_scope import (
    QRNGStreamScope,
    compute_hurst_exponent,
    compute_lyapunov_exponent,
    compute_msd_from_trajectory,
    SignalClass,
)


# Path to QRNG stream data
STREAMS_DIR = Path(__file__).parent.parent / "qrng_streams"


def find_stream_files() -> dict[str, list[Path]]:
    """Find all available QRNG stream files by source type."""
    sources = {
        "outshift": [],
        "anu": [],
        "cipherstone": [],
        "cpu_hwrng": [],
    }

    if not STREAMS_DIR.exists():
        return sources

    for filepath in STREAMS_DIR.glob("*.json"):
        name = filepath.name.lower()
        if "anu" in name:
            sources["anu"].append(filepath)
        elif "cipherstone" in name:
            sources["cipherstone"].append(filepath)
        elif "cpu" in name or "hwrng" in name:
            sources["cpu_hwrng"].append(filepath)
        elif "qrng_stream" in name:  # Default outshift files
            sources["outshift"].append(filepath)

    return sources


def load_stream_data(filepath: Path) -> Optional[list[float]]:
    """Load QRNG stream data from JSON file."""
    try:
        with open(filepath) as f:
            data = json.load(f)
        return data.get("floats", data.get("values", []))
    except (json.JSONDecodeError, IOError):
        return None


# ============================================================================
# Fixtures
# ============================================================================

@pytest.fixture(scope="module")
def available_streams():
    """Get available stream files."""
    return find_stream_files()


@pytest.fixture
def outshift_data(available_streams):
    """Load Outshift QRNG data."""
    files = available_streams.get("outshift", [])
    if not files:
        pytest.skip("No Outshift stream files available")
    return load_stream_data(files[0])


@pytest.fixture
def anu_data(available_streams):
    """Load ANU QRNG data."""
    files = available_streams.get("anu", [])
    if not files:
        pytest.skip("No ANU stream files available")
    return load_stream_data(files[0])


@pytest.fixture
def cipherstone_data(available_streams):
    """Load Cipherstone QRNG data."""
    files = available_streams.get("cipherstone", [])
    if not files:
        pytest.skip("No Cipherstone stream files available")
    return load_stream_data(files[0])


@pytest.fixture
def cpu_hwrng_data(available_streams):
    """Load CPU HWRNG data."""
    files = available_streams.get("cpu_hwrng", [])
    if not files:
        pytest.skip("No CPU HWRNG stream files available")
    return load_stream_data(files[0])


# ============================================================================
# Data Loading Tests
# ============================================================================

class TestDataLoading:
    """Test that QRNG data loads correctly."""

    def test_stream_directory_exists(self):
        """Verify stream directory exists."""
        # Note: This may fail in CI without data
        if not STREAMS_DIR.exists():
            pytest.skip("Stream directory not present")
        assert STREAMS_DIR.is_dir()

    def test_outshift_data_structure(self, outshift_data):
        """Test Outshift data has correct structure."""
        assert outshift_data is not None
        assert len(outshift_data) > 0
        assert all(isinstance(v, (int, float)) for v in outshift_data[:100])

    def test_anu_data_structure(self, anu_data):
        """Test ANU data has correct structure."""
        assert anu_data is not None
        assert len(anu_data) > 0
        assert all(isinstance(v, (int, float)) for v in anu_data[:100])

    def test_cipherstone_data_structure(self, cipherstone_data):
        """Test Cipherstone data has correct structure."""
        assert cipherstone_data is not None
        assert len(cipherstone_data) > 0
        assert all(isinstance(v, (int, float)) for v in cipherstone_data[:100])

    def test_data_range_normalized(self, outshift_data):
        """Test data is normalized to [0, 1]."""
        arr = np.array(outshift_data[:1000])
        assert arr.min() >= 0.0, f"Min value {arr.min()} below 0"
        assert arr.max() <= 1.0, f"Max value {arr.max()} above 1"


# ============================================================================
# Quality Metric Tests
# ============================================================================

class TestQRNGQuality:
    """Test QRNG quality metrics on real data."""

    def test_outshift_mean_near_half(self, outshift_data):
        """Outshift data should have mean near 0.5."""
        arr = np.array(outshift_data[:1000])
        mean = arr.mean()
        assert 0.45 <= mean <= 0.55, f"Mean {mean} outside [0.45, 0.55]"

    def test_anu_mean_near_half(self, anu_data):
        """ANU data should have mean near 0.5."""
        arr = np.array(anu_data[:1000])
        mean = arr.mean()
        assert 0.45 <= mean <= 0.55, f"Mean {mean} outside [0.45, 0.55]"

    def test_cipherstone_mean_near_half(self, cipherstone_data):
        """Cipherstone data should have mean near 0.5."""
        arr = np.array(cipherstone_data[:1000])
        mean = arr.mean()
        assert 0.45 <= mean <= 0.55, f"Mean {mean} outside [0.45, 0.55]"

    def test_outshift_uniform_distribution(self, outshift_data):
        """Outshift data should be approximately uniform."""
        arr = np.array(outshift_data[:1000])
        # Chi-squared test for uniformity (10 bins)
        hist, _ = np.histogram(arr, bins=10, range=(0, 1))
        expected = len(arr) / 10
        chi2 = np.sum((hist - expected)**2 / expected)
        # Chi-squared critical value for df=9, alpha=0.01 is ~21.67
        assert chi2 < 30, f"Chi-squared {chi2} suggests non-uniform distribution"

    def test_autocorrelation_low(self, outshift_data):
        """Autocorrelation at lag 1 should be low."""
        arr = np.array(outshift_data[:1000])
        # Centered
        centered = arr - arr.mean()
        # Lag-1 autocorrelation
        autocorr = np.corrcoef(centered[:-1], centered[1:])[0, 1]
        assert abs(autocorr) < 0.1, f"Autocorrelation {autocorr} too high"


# ============================================================================
# Trajectory Analysis Tests
# ============================================================================

class TestTrajectoryAnalysis:
    """Test trajectory analysis on real QRNG data."""

    def test_hurst_random_walk_range(self, outshift_data):
        """Hurst exponent should be near 0.5 for good QRNG."""
        arr = np.array(outshift_data[:500])
        # Compute on increments (centered)
        increments = np.diff(arr)
        H = compute_hurst_exponent(increments)
        # Relaxed range for real data
        assert 0.3 <= H <= 0.7, f"Hurst {H} outside [0.3, 0.7]"

    def test_lyapunov_near_zero(self, outshift_data):
        """Lyapunov exponent should be near zero for random walk."""
        arr = np.array(outshift_data[:500])
        # Build simple trajectory
        x = np.cumsum(arr - 0.5)
        y = np.cumsum(np.roll(arr, 1) - 0.5)
        trajectory = np.column_stack([x[1:], y[1:]])

        lyap = compute_lyapunov_exponent(trajectory)
        # Should be close to zero (within noise)
        assert -0.5 <= lyap <= 0.5, f"Lyapunov {lyap} outside [-0.5, 0.5]"

    def test_msd_normal_diffusion(self, outshift_data):
        """MSD exponent should be near 1 (normal diffusion)."""
        arr = np.array(outshift_data[:500])
        # Build trajectory
        x = np.cumsum(arr - 0.5)
        y = np.cumsum(np.roll(arr, 1) - 0.5)
        trajectory = np.column_stack([x[1:], y[1:]])

        alpha, _, _ = compute_msd_from_trajectory(trajectory)
        # Normal diffusion: alpha ≈ 1
        assert 0.5 <= alpha <= 1.5, f"MSD alpha {alpha} outside [0.5, 1.5]"

    def test_scope_processes_stream(self, outshift_data):
        """QRNGStreamScope should process real data without errors."""
        scope = QRNGStreamScope(history_len=100, walk_mode='angle')

        errors = []
        for i, value in enumerate(outshift_data[:200]):
            try:
                metrics = scope.update_from_stream(value)
                assert 'hurst' in metrics or i < scope.min_analysis_len
            except Exception as e:
                errors.append(f"Step {i}: {e}")

        assert len(errors) == 0, f"Errors during processing: {errors}"


# ============================================================================
# Cross-Source Comparison Tests
# ============================================================================

class TestSourceComparison:
    """Compare quality metrics across different QRNG sources."""

    def test_sources_have_similar_means(self, available_streams):
        """All sources should have means near 0.5."""
        means = {}

        for source, files in available_streams.items():
            if files:
                data = load_stream_data(files[0])
                if data:
                    means[source] = np.mean(data[:1000])

        if len(means) < 2:
            pytest.skip("Need at least 2 sources for comparison")

        for source, mean in means.items():
            assert 0.45 <= mean <= 0.55, f"{source} mean {mean} out of range"

    def test_sources_have_similar_variance(self, available_streams):
        """All sources should have variance near 1/12 (uniform)."""
        variances = {}
        expected_var = 1/12  # Variance of uniform [0,1]

        for source, files in available_streams.items():
            if files:
                data = load_stream_data(files[0])
                if data:
                    variances[source] = np.var(data[:1000])

        if len(variances) < 2:
            pytest.skip("Need at least 2 sources for comparison")

        for source, var in variances.items():
            # Allow 50% deviation from expected
            assert 0.5 * expected_var <= var <= 1.5 * expected_var, \
                f"{source} variance {var} far from expected {expected_var}"

    def test_no_source_bias_detected(self, available_streams):
        """No source should show significant bias."""
        for source, files in available_streams.items():
            if files:
                data = load_stream_data(files[0])
                if data and len(data) >= 1000:
                    arr = np.array(data[:1000])
                    # One-sample t-test against 0.5
                    from scipy import stats
                    _, p_value = stats.ttest_1samp(arr, 0.5)
                    # Should not reject null (mean = 0.5)
                    assert p_value > 0.01, \
                        f"{source} shows bias (p={p_value:.4f})"


# ============================================================================
# Signal Classification Tests
# ============================================================================

class TestSignalClassification:
    """Test signal classification on real QRNG data."""

    def test_qrng_classified_as_noise(self, outshift_data):
        """Good QRNG should typically classify as NOISE."""
        scope = QRNGStreamScope(history_len=100, walk_mode='angle')

        # Process data
        for value in outshift_data[:150]:
            scope.update_from_stream(value)

        # Verify signal
        verification = scope.verify_current_signal()

        # Should be NOISE or at most DRIFT (not ATTRACTOR/CHAOTIC/INFLUENCE)
        assert verification.classification in [
            SignalClass.NOISE,
            SignalClass.DRIFT,
            SignalClass.UNKNOWN
        ], f"Unexpected classification: {verification.classification}"

    def test_all_sources_classify_appropriately(self, available_streams):
        """All QRNG sources should classify as NOISE-like."""
        classifications = {}

        for source, files in available_streams.items():
            if files:
                data = load_stream_data(files[0])
                if data and len(data) >= 150:
                    scope = QRNGStreamScope(history_len=100, walk_mode='angle')
                    for value in data[:150]:
                        scope.update_from_stream(value)

                    verification = scope.verify_current_signal()
                    classifications[source] = verification.classification

        if not classifications:
            pytest.skip("No data available for classification test")

        for source, cls in classifications.items():
            assert cls in [
                SignalClass.NOISE,
                SignalClass.DRIFT,
                SignalClass.UNKNOWN
            ], f"{source} classified as {cls}"


# ============================================================================
# Robustness Tests
# ============================================================================

class TestRobustness:
    """Test analysis robustness on real data."""

    def test_handles_empty_file_gracefully(self, tmp_path):
        """Should handle empty or malformed files."""
        # Create empty JSON
        empty_file = tmp_path / "empty.json"
        empty_file.write_text("{}")

        data = load_stream_data(empty_file)
        assert data == [] or data is None

    def test_handles_missing_floats_key(self, tmp_path):
        """Should handle JSON without 'floats' key."""
        no_floats = tmp_path / "no_floats.json"
        no_floats.write_text('{"source": "test", "count": 0}')

        data = load_stream_data(no_floats)
        assert data == [] or data is None

    def test_processes_partial_data(self, outshift_data):
        """Should handle partial data without crashing."""
        scope = QRNGStreamScope(history_len=100, walk_mode='angle')

        # Process less than minimum required
        for value in outshift_data[:30]:
            metrics = scope.update_from_stream(value)

        # Should return metrics (possibly partial)
        assert isinstance(metrics, dict)

    def test_consistent_results_across_runs(self, outshift_data):
        """Same data should produce same results."""
        results = []

        for _ in range(3):
            scope = QRNGStreamScope(history_len=100, walk_mode='angle')
            for value in outshift_data[:150]:
                metrics = scope.update_from_stream(value)
            results.append(metrics.get('hurst', 0))

        # Results should be identical
        assert all(r == results[0] for r in results), "Inconsistent results"


# ============================================================================
# Performance Tests
# ============================================================================

@pytest.mark.slow
class TestPerformance:
    """Performance tests on larger datasets."""

    def test_processes_large_stream_efficiently(self, outshift_data):
        """Should process large streams without timeout."""
        import time

        scope = QRNGStreamScope(history_len=100, walk_mode='angle')

        start = time.perf_counter()
        for value in outshift_data:  # All data
            scope.update_from_stream(value)
        elapsed = time.perf_counter() - start

        # Should process at least 1000 samples/second
        rate = len(outshift_data) / elapsed if elapsed > 0 else float('inf')
        assert rate > 1000, f"Processing rate {rate:.0f}/sec too slow"

    def test_memory_stable_over_long_stream(self, outshift_data):
        """Memory usage should be bounded by history_len."""
        import sys

        scope = QRNGStreamScope(history_len=100, walk_mode='angle')

        # Process and check memory periodically
        for i, value in enumerate(outshift_data):
            scope.update_from_stream(value)

        # Check trajectory length is bounded
        assert len(scope.trajectory_history) <= scope.history_len + 1


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
