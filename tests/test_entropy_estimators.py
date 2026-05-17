"""
Tests for hybrid entropy estimators.

Tests SymbolicWeightedWaveletEntropy, NonStationarySymbolicEntropy,
and EntropyEstimatorEnsemble classes.
"""

import numpy as np
import pytest
from src.entropy_estimators import (
    SymbolicWeightedWaveletEntropy,
    NonStationarySymbolicEntropy,
    EntropyEstimatorEnsemble
)


class TestSymbolicWeightedWaveletEntropy:
    """Tests for SWWE class."""
    
    def test_init(self):
        """Test initialization with default parameters."""
        swwe = SymbolicWeightedWaveletEntropy()
        assert swwe.embedding_dim == 3
        assert swwe.tau == 1
        assert swwe.wavelet_type == 'db4'
    
    def test_fit_transform(self):
        """Test fit and transform on sample data."""
        np.random.seed(42)
        X = np.random.randn(100)
        
        swwe = SymbolicWeightedWaveletEntropy()
        entropy = swwe.fit_transform(X)
        
        assert isinstance(entropy, float)
        assert -1.0 < entropy < 1.0
    
    def test_predict(self):
        """Test predict method."""
        np.random.seed(42)
        X_train = np.random.randn(50)
        X_test = np.random.randn(30)
        
        swwe = SymbolicWeightedWaveletEntropy()
        swwe.fit(X_train)
        entropy = swwe.predict(X_test)
        
        assert isinstance(entropy, float)
    
    def test_entropy_range(self):
        """Test that entropy values are in valid range."""
        np.random.seed(42)
        X = np.random.randn(100)
        
        swwe = SymbolicWeightedWaveletEntropy()
        entropy = swwe.fit_transform(X)
        
        # Sample entropy should be between 0 and log2(embedding_dim!)
        assert 0 <= entropy < 3.0
    
    def test_deterministic_output(self):
        """Test that same input gives same output."""
        np.random.seed(42)
        X = np.random.randn(100)
        
        swwe1 = SymbolicWeightedWaveletEntropy()
        swwe2 = SymbolicWeightedWaveletEntropy()
        
        entropy1 = swwe1.fit_transform(X)
        entropy2 = swwe2.fit_transform(X)
        
        assert np.isclose(entropy1, entropy2)


class TestNonStationarySymbolicEntropy:
    """Tests for NSAE class."""
    
    def test_init(self):
        """Test initialization with default parameters."""
        nsaes = NonStationarySymbolicEntropy()
        assert nsaes.embedding_dim == 3
        assert nsaes.tau == 1
        assert nsaes.transition_window == 10
    
    def test_fit_transform(self):
        """Test fit and transform on sample data."""
        np.random.seed(42)
        X = np.random.randn(100)
        
        nsaes = NonStationarySymbolicEntropy()
        entropy = nsaes.fit_transform(X)
        
        assert isinstance(entropy, float)
    
    def test_predict(self):
        """Test predict method."""
        np.random.seed(42)
        X_train = np.random.randn(50)
        X_test = np.random.randn(30)
        
        nsaes = NonStationarySymbolicEntropy()
        nsaes.fit(X_train)
        entropy = nsaes.predict(X_test)
        
        assert isinstance(entropy, float)
    
    def test_entropy_range(self):
        """Test that entropy values are in valid range."""
        np.random.seed(42)
        X = np.random.randn(100)
        
        nsaes = NonStationarySymbolicEntropy()
        entropy = nsaes.fit_transform(X)
        
        # Entropy should be non-negative
        assert entropy >= 0
    
    def test_deterministic_output(self):
        """Test that same input gives same output."""
        np.random.seed(42)
        X = np.random.randn(100)
        
        nsaes1 = NonStationarySymbolicEntropy()
        nsaes2 = NonStationarySymbolicEntropy()
        
        entropy1 = nsaes1.fit_transform(X)
        entropy2 = nsaes2.fit_transform(X)
        
        assert np.isclose(entropy1, entropy2)


class TestEntropyEstimatorEnsemble:
    """Tests for ensemble class."""
    
    def test_init(self):
        """Test initialization."""
        ensemble = EntropyEstimatorEnsemble()
        assert ensemble.sample_entropy is None
        assert ensemble.swwe is not None
        assert ensemble.nsaes is not None
    
    def test_fit_transform(self):
        """Test fit and transform on sample data."""
        np.random.seed(42)
        X = np.random.randn(100)
        
        ensemble = EntropyEstimatorEnsemble()
        entropies = ensemble.fit_transform(X)
        
        assert isinstance(entropies, np.ndarray)
        assert len(entropies) == 5
    
    def test_transform_after_fit(self):
        """Test transform after fit."""
        np.random.seed(42)
        X_train = np.random.randn(50)
        X_test = np.random.randn(30)
        
        ensemble = EntropyEstimatorEnsemble()
        ensemble.fit(X_train)
        entropies = ensemble.transform(X_test)
        
        assert isinstance(entropies, np.ndarray)
        assert len(entropies) == 5
    
    def test_entropy_values_valid(self):
        """Test that all entropy values are valid."""
        np.random.seed(42)
        X = np.random.randn(100)
        
        ensemble = EntropyEstimatorEnsemble()
        entropies = ensemble.fit_transform(X)
        
        # All entropies should be in reasonable range
        for e in entropies:
            assert -2.0 < e < 5.0
    
    def test_deterministic_output(self):
        """Test that same input gives same output."""
        np.random.seed(42)
        X = np.random.randn(100)
        
        ensemble1 = EntropyEstimatorEnsemble()
        ensemble2 = EntropyEstimatorEnsemble()
        
        entropies1 = ensemble1.fit_transform(X)
        entropies2 = ensemble2.fit_transform(X)
        
        assert np.allclose(entropies1, entropies2)


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_small_sample_size(self):
        """Test with small sample size (N < 20)."""
        np.random.seed(42)
        X = np.random.randn(15)  # Small sample
        
        swwe = SymbolicWeightedWaveletEntropy()
        entropy = swwe.fit_transform(X)
        
        assert isinstance(entropy, float)
    
    def test_constant_signal(self):
        """Test with constant signal (zero entropy expected)."""
        X = np.ones(100) * 5.0
        
        swwe = SymbolicWeightedWaveletEntropy()
        entropy = swwe.fit_transform(X)
        
        # Should have low entropy for constant signal
        assert entropy < 1.0
    
    def test_high_frequency_noise(self):
        """Test with high-frequency noise."""
        np.random.seed(42)
        X = np.random.randn(100) * 10  # High variance
        
        swwe = SymbolicWeightedWaveletEntropy()
        entropy = swwe.fit_transform(X)
        
        assert isinstance(entropy, float)
    
    def test_different_wavelet_types(self):
        """Test with different wavelet types."""
        np.random.seed(42)
        X = np.random.randn(100)
        
        for wavelet_type in ['db4', 'sym8', 'coif5']:
            swwe = SymbolicWeightedWaveletEntropy(wavelet_type=wavelet_type)
            entropy = swwe.fit_transform(X)
            assert isinstance(entropy, float)
    
    def test_different_transition_windows(self):
        """Test with different transition windows."""
        np.random.seed(42)
        X = np.random.randn(100)
        
        for window in [5, 10, 15]:
            nsaes = NonStationarySymbolicEntropy(transition_window=window)
            entropy = nsaes.fit_transform(X)
            assert isinstance(entropy, float)


if __name__ == '__main__':
    pytest.main([__file__, '-v'])
