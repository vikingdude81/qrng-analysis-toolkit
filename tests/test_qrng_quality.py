"""
QRNG Quality Tests

Tests for SPDC quantum random source quality:
- Uniformity (Chi-square)
- Autocorrelation
- Runs test
- Basic NIST 800-22 subset
"""

import pytest
import numpy as np
from scipy import stats
import sys
from qrng_spdc_source import SPDCQuantumSource, get_quantum_random


class TestQRNGUniformity:
    """Test uniform distribution of QRNG output."""
    
    @pytest.fixture
    def samples(self):
        """Generate 10000 QRNG samples."""
        source = SPDCQuantumSource()
        return np.array([source.get_random() for _ in range(10000)])
    
    def test_mean_near_half(self, samples):
        """Mean should be ≈ 0.5."""
        mean = np.mean(samples)
        assert 0.48 <= mean <= 0.52, f"Mean={mean}, expected ~0.5"
    
    def test_std_near_uniform(self, samples):
        """Std should be ≈ 1/√12 ≈ 0.289 for uniform."""
        std = np.std(samples)
        expected = 1 / np.sqrt(12)  # 0.2887
        assert 0.27 <= std <= 0.31, f"Std={std}, expected ~{expected}"
    
    def test_chi_square_uniformity(self, samples):
        """Chi-square test for uniformity."""
        bins = 10
        observed, _ = np.histogram(samples, bins=bins, range=(0, 1))
        expected = len(samples) / bins
        chi2, p_value = stats.chisquare(observed, [expected] * bins)
        assert p_value > 0.01, f"Chi² p={p_value}, distribution not uniform"
    
    def test_ks_uniform(self, samples):
        """Kolmogorov-Smirnov test against uniform distribution."""
        stat, p_value = stats.kstest(samples, 'uniform')
        assert p_value > 0.01, f"KS test p={p_value}, not uniform"
    
    def test_range_zero_to_one(self, samples):
        """All values should be in [0, 1)."""
        assert np.min(samples) >= 0, "Values below 0 found"
        assert np.max(samples) < 1, "Values >= 1 found"


class TestQRNGAutocorrelation:
    """Test for absence of autocorrelation."""
    
    @pytest.fixture
    def samples(self):
        """Generate 2000 QRNG samples."""
        source = SPDCQuantumSource()
        return np.array([source.get_random() for _ in range(2000)])
    
    def test_no_significant_autocorrelation(self, samples):
        """No lag should have significant autocorrelation."""
        vals_c = samples - samples.mean()
        var = np.sum(vals_c**2)
        ci = 1.96 / np.sqrt(len(samples))  # 95% CI
        
        significant_count = 0
        for lag in [1, 5, 10, 20, 30, 50]:
            if lag >= len(samples):
                continue
            acf = np.sum(vals_c[:-lag] * vals_c[lag:]) / var
            if abs(acf) > ci:
                significant_count += 1
        
        # At most 1 significant lag expected by chance (5% rate)
        assert significant_count <= 2, f"{significant_count} significant autocorrelations"
    
    def test_lag1_autocorrelation_small(self, samples):
        """Lag-1 autocorrelation should be near zero."""
        vals_c = samples - samples.mean()
        var = np.sum(vals_c**2)
        acf1 = np.sum(vals_c[:-1] * vals_c[1:]) / var
        assert abs(acf1) < 0.1, f"Lag-1 ACF={acf1}, expected ~0"


class TestQRNGRunsTest:
    """Runs test for randomness."""
    
    @pytest.fixture
    def samples(self):
        """Generate samples."""
        source = SPDCQuantumSource()
        return np.array([source.get_random() for _ in range(1000)])
    
    def test_runs_test(self, samples):
        """Wald-Wolfowitz runs test."""
        median = np.median(samples)
        binary = (samples > median).astype(int)
        
        # Count runs
        runs = 1
        for i in range(1, len(binary)):
            if binary[i] != binary[i-1]:
                runs += 1
        
        n1 = np.sum(binary)
        n0 = len(binary) - n1
        
        if n0 == 0 or n1 == 0:
            pytest.skip("All values same side of median")
        
        # Expected runs and std
        mu_r = (2 * n0 * n1) / (n0 + n1) + 1
        sigma_r = np.sqrt((2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / 
                          ((n0 + n1)**2 * (n0 + n1 - 1)))
        
        z = (runs - mu_r) / sigma_r
        p_value = 2 * (1 - stats.norm.cdf(abs(z)))
        
        assert p_value > 0.01, f"Runs test p={p_value}, non-random pattern detected"


class TestNIST800_22Subset:
    """
    Subset of NIST SP 800-22 statistical tests.
    
    These are simplified versions for quick validation.
    Full NIST suite requires dedicated library.
    """
    
    @pytest.fixture
    def bit_string(self):
        """Generate bit string from QRNG."""
        source = SPDCQuantumSource()
        values = [source.get_random() for _ in range(1000)]
        # Convert to bits: value > 0.5 -> 1, else 0
        return np.array([1 if v > 0.5 else 0 for v in values])
    
    def test_frequency_monobit(self, bit_string):
        """
        NIST Test 1: Frequency (Monobit) Test
        Tests if number of 1s and 0s are approximately equal.
        """
        n = len(bit_string)
        s = np.sum(2 * bit_string - 1)  # Convert 0->-1, 1->+1
        s_obs = abs(s) / np.sqrt(n)
        p_value = 2 * (1 - stats.norm.cdf(s_obs))
        
        assert p_value > 0.01, f"Monobit test p={p_value}"
    
    def test_frequency_block(self, bit_string):
        """
        NIST Test 2: Frequency Test within a Block
        Tests if frequency of 1s in M-bit blocks is ≈ M/2.
        """
        M = 10  # Block size
        n = len(bit_string)
        N = n // M  # Number of blocks
        
        chi2 = 0
        for i in range(N):
            block = bit_string[i*M:(i+1)*M]
            pi = np.sum(block) / M
            chi2 += (pi - 0.5)**2
        
        chi2 *= 4 * M
        p_value = 1 - stats.chi2.cdf(chi2, N)
        
        # More lenient threshold for smaller sample
        assert p_value > 0.001, f"Block frequency test p={p_value}"
    
    def test_runs(self, bit_string):
        """
        NIST Test 3: Runs Test
        Tests oscillation between 0s and 1s.
        """
        n = len(bit_string)
        pi = np.sum(bit_string) / n
        
        # Pre-test
        tau = 2 / np.sqrt(n)
        if abs(pi - 0.5) >= tau:
            pytest.skip("Pre-test failed (too biased)")
        
        # Count runs
        runs = 1
        for i in range(1, n):
            if bit_string[i] != bit_string[i-1]:
                runs += 1
        
        # Compute p-value
        p_value = 2 * abs(runs - 2*n*pi*(1-pi)) / (2*np.sqrt(2*n)*pi*(1-pi))
        p_value = 2 * (1 - stats.norm.cdf(p_value))
        
        assert p_value > 0.01, f"NIST Runs test p={p_value}"
    
    def test_longest_run_of_ones(self, bit_string):
        """
        NIST Test 4: Longest Run of Ones in a Block
        (Simplified version)
        """
        # Find longest run of 1s
        max_run = 0
        current_run = 0
        for bit in bit_string:
            if bit == 1:
                current_run += 1
                max_run = max(max_run, current_run)
            else:
                current_run = 0
        
        n = len(bit_string)
        # For n=1000, expected longest run ≈ log2(n) ≈ 10
        expected_max = np.log2(n) + 1
        
        # Very loose bound - just sanity check
        assert max_run < 30, f"Longest run={max_run}, suspiciously long"
        assert max_run > 3, f"Longest run={max_run}, suspiciously short"


class TestSHA256Whitening:
    """Test SHA-256 whitening mode."""
    
    def test_whitening_produces_valid_output(self):
        """Whitened output should still be uniform."""
        source = SPDCQuantumSource(use_sha256_whitening=True)
        samples = np.array([source.get_random() for _ in range(1000)])
        
        assert 0.45 <= np.mean(samples) <= 0.55
        assert np.min(samples) >= 0
        assert np.max(samples) < 1
    
    def test_whitening_different_from_raw(self):
        """Whitened and raw outputs should differ."""
        # Reset sources with same conditions
        source_raw = SPDCQuantumSource(use_sha256_whitening=False)
        source_whitened = SPDCQuantumSource(use_sha256_whitening=True)
        
        raw = [source_raw.get_random() for _ in range(100)]
        whitened = [source_whitened.get_random() for _ in range(100)]
        
        # They should be different sequences
        assert raw != whitened
