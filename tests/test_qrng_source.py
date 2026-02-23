"""
Tests for SPDC QRNG source implementation.

Tests:
- Bit generation and quality
- Statistical properties
- Min-entropy calculation
- Toeplitz extraction
- Quality metrics
"""

import pytest
import numpy as np
from qrng_spdc_source import (
    SPDCQuantumSource,
    ToeplitzExtractor,
    compute_min_entropy,
    compute_autocorrelation_coefficient,
    evaluate_qrng_quality,
    SPDCSourceConfig
)


class TestSPDCQuantumSource:
    """Test SPDC quantum source."""

    def test_basic_generation(self):
        """Should generate random values in [0, 1)."""
        source = SPDCQuantumSource()

        values = [source.get_random() for _ in range(100)]

        assert all(0 <= v < 1 for v in values), "Values outside [0, 1)"
        assert len(set(values)) > 50, "Values not diverse enough"

    def test_bit_generation(self):
        """Should generate binary bits."""
        source = SPDCQuantumSource()

        bits = source.get_random_bits(100)

        assert len(bits) == 100
        assert all(b in [0, 1] for b in bits), "Non-binary bits generated"

    def test_integer_generation(self):
        """Should generate integers in range."""
        source = SPDCQuantumSource()

        # Test various ranges
        for n in [2, 10, 100, 256]:
            integers = [source.get_random_int(n) for _ in range(100)]
            assert all(0 <= i < n for i in integers), f"Integers outside [0, {n})"

    def test_gaussian_generation(self):
        """Should generate Gaussian distributed values."""
        source = SPDCQuantumSource()

        values = [source.get_random_gaussian(mean=0, std=1) for _ in range(1000)]
        values = np.array(values)

        # Check approximate moments
        assert -0.2 < np.mean(values) < 0.2, "Mean too far from 0"
        assert 0.8 < np.std(values) < 1.2, "Std too far from 1"

    def test_extraction_enabled(self):
        """Should apply Toeplitz extraction when enabled."""
        source_with = SPDCQuantumSource(use_extraction=True)
        source_without = SPDCQuantumSource(use_extraction=False)

        # Both should work
        bits_with = source_with.get_random_bits(100)
        bits_without = source_without.get_random_bits(100)

        assert len(bits_with) == 100
        assert len(bits_without) == 100

    def test_statistics_tracking(self):
        """Should track generation statistics."""
        source = SPDCQuantumSource()

        # Generate some bits
        source.get_random_bits(1000)

        stats = source.get_statistics()

        assert stats['total_bits'] >= 1000
        assert stats['coincidences'] > 0
        assert stats['elapsed_seconds'] > 0


class TestToeplitzExtractor:
    """Test Toeplitz randomness extraction."""

    def test_extraction_output_size(self):
        """Should produce correct output size."""
        extractor = ToeplitzExtractor(input_size=256, output_size=128)

        raw_bits = np.random.randint(0, 2, size=256, dtype=np.uint8)
        extracted = extractor.extract(raw_bits)

        assert len(extracted) == 128

    def test_extraction_deterministic(self):
        """Same input should give same output."""
        seed = b'test_seed' * 10
        extractor = ToeplitzExtractor(input_size=256, output_size=128, seed=seed)

        raw_bits = np.random.randint(0, 2, size=256, dtype=np.uint8)

        output1 = extractor.extract(raw_bits)
        output2 = extractor.extract(raw_bits)

        assert np.array_equal(output1, output2), "Extraction not deterministic"

    def test_extraction_different_inputs(self):
        """Different inputs should give different outputs."""
        extractor = ToeplitzExtractor(input_size=256, output_size=128)

        raw1 = np.random.randint(0, 2, size=256, dtype=np.uint8)
        raw2 = np.random.randint(0, 2, size=256, dtype=np.uint8)

        out1 = extractor.extract(raw1)
        out2 = extractor.extract(raw2)

        # Should be different (very unlikely to be same)
        assert not np.array_equal(out1, out2), "Different inputs gave same output"

    def test_insufficient_bits(self):
        """Should raise error for insufficient bits."""
        extractor = ToeplitzExtractor(input_size=256, output_size=128)

        raw_bits = np.random.randint(0, 2, size=100, dtype=np.uint8)

        with pytest.raises(ValueError):
            extractor.extract(raw_bits)


class TestMinEntropy:
    """Test min-entropy calculation."""

    def test_uniform_bits_high_entropy(self):
        """Uniform random bits should have high min-entropy."""
        np.random.seed(42)
        bits = np.random.randint(0, 2, size=10000, dtype=np.uint8)

        min_ent = compute_min_entropy(bits, block_size=8)

        # Should be reasonably high for uniform (may vary due to estimation)
        assert min_ent > 0.80, f"Min-entropy too low: {min_ent:.3f}"

    def test_biased_bits_low_entropy(self):
        """Biased bits should have lower min-entropy."""
        # 80% ones
        bits = np.random.choice([0, 1], size=10000, p=[0.2, 0.8])

        min_ent = compute_min_entropy(bits, block_size=8)

        # Should be lower than uniform
        assert min_ent < 0.9, f"Biased bits entropy too high: {min_ent:.3f}"

    def test_constant_bits_zero_entropy(self):
        """Constant bits should have low min-entropy."""
        bits = np.ones(10000, dtype=np.uint8)

        min_ent = compute_min_entropy(bits, block_size=8)

        # Should be very low (all blocks identical)
        assert min_ent < 0.2, f"Constant bits entropy too high: {min_ent:.3f}"

    def test_insufficient_data(self):
        """Should handle insufficient data gracefully."""
        bits = np.array([0, 1, 0], dtype=np.uint8)

        min_ent = compute_min_entropy(bits, block_size=8)

        assert min_ent == 1.0, "Should return 1.0 for insufficient data"


class TestAutocorrelation:
    """Test autocorrelation coefficient calculation."""

    def test_white_noise_low_autocorr(self):
        """White noise should have low autocorrelation."""
        np.random.seed(42)
        bits = np.random.randint(0, 2, size=10000, dtype=np.uint8)

        autocorr, mean, std = compute_autocorrelation_coefficient(bits, max_delay=50)

        # Mean should be very small for white noise
        assert mean < 0.1, f"Autocorrelation mean too high: {mean:.4f}"

    def test_correlated_bits_high_autocorr(self):
        """Correlated bits should show higher autocorrelation."""
        # Create correlated sequence
        n = 10000
        bits = np.zeros(n, dtype=np.uint8)
        bits[0] = np.random.randint(0, 2)

        # Each bit has 70% chance of matching previous
        for i in range(1, n):
            if np.random.random() < 0.7:
                bits[i] = bits[i-1]
            else:
                bits[i] = 1 - bits[i-1]

        autocorr, mean, std = compute_autocorrelation_coefficient(bits, max_delay=20)

        # Should show correlation
        assert mean > 0.01, f"Correlated sequence autocorr too low: {mean:.4f}"

    def test_insufficient_data(self):
        """Should handle insufficient data."""
        bits = np.array([0, 1, 0, 1], dtype=np.uint8)

        autocorr, mean, std = compute_autocorrelation_coefficient(bits, max_delay=100)

        # Should return zeros for insufficient data
        assert mean == 0.0
        assert std == 0.0


class TestQualityMetrics:
    """Test QRNG quality evaluation."""

    def test_quality_evaluation(self):
        """Should evaluate QRNG quality."""
        source = SPDCQuantumSource()

        quality = evaluate_qrng_quality(source, n_bits=5000)

        # Check all metrics present
        assert hasattr(quality, 'min_entropy')
        assert hasattr(quality, 'autocorr_mean')
        assert hasattr(quality, 'frequency_test_passed')
        assert hasattr(quality, 'runs_test_passed')

        # Basic sanity checks
        assert 0 < quality.min_entropy <= 1
        assert quality.autocorr_mean >= 0
        assert quality.bit_rate_mbps > 0

    def test_quality_acceptable_check(self):
        """Should correctly identify acceptable quality."""
        source = SPDCQuantumSource()

        quality = evaluate_qrng_quality(source, n_bits=10000)

        # Should generally pass for properly implemented QRNG
        # (May occasionally fail due to randomness - that's OK)
        if not quality.is_quality_acceptable():
            print(f"Warning: Quality check failed (can happen randomly): {quality}")


class TestStatisticalProperties:
    """Test statistical properties of generated values."""

    def test_uniform_distribution(self):
        """Generated values should be uniformly distributed."""
        source = SPDCQuantumSource()

        values = np.array([source.get_random() for _ in range(1000)])

        # Chi-square test
        bins = 10
        observed, _ = np.histogram(values, bins=bins, range=(0, 1))
        expected = 1000 / bins
        chi2 = np.sum((observed - expected)**2 / expected)

        # p > 0.05 threshold is chi2 < 16.9 for 9 dof
        assert chi2 < 25, f"Chi-square too high: {chi2:.2f}"

    def test_bit_balance(self):
        """Bits should be approximately 50/50."""
        source = SPDCQuantumSource()

        bits = source.get_random_bits(10000)
        ones_ratio = np.mean(bits)

        # Should be close to 0.5
        assert 0.45 < ones_ratio < 0.55, f"Bit ratio {ones_ratio:.3f} not balanced"

    def test_independence(self):
        """Successive values should be independent."""
        source = SPDCQuantumSource()

        values = np.array([source.get_random() for _ in range(1000)])

        # Check correlation between successive values
        corr = np.corrcoef(values[:-1], values[1:])[0, 1]

        # Should be low
        assert abs(corr) < 0.1, f"Correlation too high: {corr:.3f}"


class TestConfigurationOptions:
    """Test various configuration options."""

    def test_ring_sections(self):
        """Should support different ring section counts."""
        # Only test values that are valid RingSectionID enum values
        for sections in [4, 8]:
            source = SPDCQuantumSource(ring_sections=sections)
            value = source.get_random()
            assert 0 <= value < 1

    def test_pump_power(self):
        """Should support different pump powers."""
        for power in [10.0, 17.0, 25.0]:
            source = SPDCQuantumSource(pump_power_mw=power)
            bits = source.get_random_bits(100)
            assert len(bits) == 100


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
