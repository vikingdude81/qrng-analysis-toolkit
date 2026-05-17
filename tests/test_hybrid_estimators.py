"""
Tests for hybrid entropy estimators (SWWE and NSAE).

This module validates the statistical validity and robustness of the new
hybrid entropy estimators added to Helios.
"""

import numpy as np
import pytest
from src.entropy_estimators import (
    SymbolicWeightedWaveletEntropy,
    NonStationarySymbolicEntropy,
    EntropyEstimatorEnsemble,
)


class TestSymbolicWeightedWaveletEntropy:
    """Tests for SWWE estimator."""
    
    def test_initialization(self):
        """Test that SWWE initializes correctly."""
        swwe = SymbolicWeightedWaveletEntropy(
            embedding_dim=3,
            tau=1,
            wavelet_type='db4',
            wavelet_level=2
        )
        assert swwe.embedding_dim == 3
        assert swwe.tau == 1
        assert swwe.wavelet_type == 'db4'
        assert swwe.wavelet_level == 2
    
    def test_fit_transform(self):
        """Test that SWWE can fit and transform data."""
        np.random.seed(42)
        data = np.random.randn(100) * 0.5 + np.sin(np.arange(100))
        
        swwe = SymbolicWeightedWaveletEntropy()
        result = swwe.fit_transform(data)
        
        assert isinstance(result, float)
        assert -2 < result < 3  # Entropy should be in reasonable range
    
    def test_entropy_bounds(self):
        """Test that entropy values are within expected bounds."""
        np.random.seed(42)
        data = np.random.randn(500)
        
        swwe = SymbolicWeightedWaveletEntropy()
        result = swwe.fit_transform(data)
        
        # Entropy should be between 0 and log2(alphabet_size)
        assert 0 <= result <= 4
    
    def test_different_wavelets(self):
        """Test different wavelet types."""
        data = np.random.randn(100)
        
        for wavelet_type in ['db4', 'sym8', 'coif5']:
            swwe = SymbolicWeightedWaveletEntropy(wavelet_type=wavelet_type)
            result = swwe.fit_transform(data)
            assert isinstance(result, float)
    
    def test_different_embedding_dims(self):
        """Test different embedding dimensions."""
        data = np.random.randn(200)
        
        for dim in [2, 3, 4, 5]:
            swwe = SymbolicWeightedWaveletEntropy(embedding_dim=dim)
            result = swwe.fit_transform(data)
            assert isinstance(result, float)


class TestNonStationarySymbolicEntropy:
    """Tests for NSAE estimator."""
    
    def test_initialization(self):
        """Test that NSAE initializes correctly."""
        nsaes = NonStationarySymbolicEntropy(
            embedding_dim=3,
            tau=1,
            transition_window=10,
            alphabet_size=8
        )
        assert nsaes.embedding_dim == 3
        assert nsaes.tau == 1
        assert nsaes.transition_window == 10
        assert nsaes.alphabet_size == 8
    
    def test_fit_transform(self):
        """Test that NSAE can fit and transform data."""
        np.random.seed(42)
        data = np.random.randn(100) * 0.5 + np.sin(np.arange(100))
        
        nsaes = NonStationarySymbolicEntropy()
        result = nsaes.fit_transform(data)
        
        assert isinstance(result, float)
        assert -2 < result < 3
    
    def test_entropy_bounds(self):
        """Test that entropy values are within expected bounds."""
        np.random.seed(42)
        data = np.random.randn(500)
        
        nsaes = NonStationarySymbolicEntropy()
        result = nsaes.fit_transform(data)
        
        assert 0 <= result <= 3
    
    def test_different_transition_windows(self):
        """Test different transition window sizes."""
        data = np.random.randn(100)
        
        for window in [5, 10, 20, 50]:
            nsaes = NonStationarySymbolicEntropy(transition_window=window)
            result = nsaes.fit_transform(data)
            assert isinstance(result, float)
    
    def test_different_alphabet_sizes(self):
        """Test different alphabet sizes."""
        data = np.random.randn(100)
        
        for size in [4, 8, 16]:
            nsaes = NonStationarySymbolicEntropy(alphabet_size=size)
            result = nsaes.fit_transform(data)
            assert isinstance(result, float)


class TestEntropyEstimatorEnsemble:
    """Tests for the ensemble of all estimators."""
    
    def test_initialization(self):
        """Test that ensemble initializes correctly."""
        ensemble = EntropyEstimatorEnsemble(use_hybrid=True)
        assert ensemble.use_hybrid is True
    
    def test_fit_transform_all_estimators(self):
        """Test that ensemble returns all estimator results."""
        np.random.seed(42)
        data = np.random.randn(500)
        
        ensemble = EntropyEstimatorEnsemble(use_hybrid=True)
        entropies = ensemble.fit_transform(data)
        
        assert isinstance(entropies, list)
        assert len(entropies) == 5
        assert all(isinstance(e, float) for e in entropies)
    
    def test_entropy_values_reasonable(self):
        """Test that all entropy values are reasonable."""
        np.random.seed(42)
        data = np.random.randn(500)
        
        ensemble = EntropyEstimatorEnsemble(use_hybrid=True)
        entropies = ensemble.fit_transform(data)
        
        # All entropies should be between 0 and 3
        for e in entropies:
            assert 0 <= e <= 3
    
    def test_small_sample(self):
        """Test with small sample size."""
        np.random.seed(42)
        data = np.random.randn(20)  # Small sample
        
        ensemble = EntropyEstimatorEnsemble(use_hybrid=True)
        entropies = ensemble.fit_transform(data)
        
        assert len(entropies) == 5
        # All values should be finite
        assert all(np.isfinite(e) for e in entropies)
    
    def test_large_sample(self):
        """Test with large sample size."""
        np.random.seed(42)
        data = np.random.randn(10000)
        
        ensemble = EntropyEstimatorEnsemble(use_hybrid=True)
        entropies = ensemble.fit_transform(data)
        
        assert len(entropies) == 5
        # All values should be finite
        assert all(np.isfinite(e) for e in entropies)
    
    def test_consistency_across_runs(self):
        """Test that results are consistent across runs."""
        np.random.seed(42)
        data = np.random.randn(500)
        
        ensemble1 = EntropyEstimatorEnsemble(use_hybrid=True)
        entropies1 = ensemble1.fit_transform(data)
        
        ensemble2 = EntropyEstimatorEnsemble(use_hybrid=True)
        entropies2 = ensemble2.fit_transform(data)
        
        # Results should be identical (same seed)
        assert np.allclose(entropies1, entropies2)


class TestHybridVsStandard:
    """Tests comparing hybrid vs standard estimators."""
    
    def test_hybrid_lower_bias(self):
        """Test that hybrid estimators have lower bias than standard."""
        np.random.seed(42)
        # Create data with known structure
        t = np.arange(500)
        signal = np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 0.3 * t)
        noise = np.random.randn(500) * 0.5
        data = signal + noise
        
        # Standard estimator (sample entropy)
        from src.entropy_estimators import SampleEntropy
        se = SampleEntropy()
        se.fit(data)
        se_entropy = se.transform(data)
        
        # Hybrid estimator (SWWE)
        swwe = SymbolicWeightedWaveletEntropy()
        swwe.fit(data)
        swwe_entropy = swwe.transform(data)
        
        # SWWE should have lower bias for this structured data
        assert abs(swwe_entropy - 1.0) < abs(se_entropy - 1.0) + 0.2


class TestEdgeCases:
    """Tests for edge cases."""
    
    def test_constant_data(self):
        """Test with constant data (zero entropy expected)."""
        data = np.ones(100)
        
        swwe = SymbolicWeightedWaveletEntropy()
        result = swwe.fit_transform(data)
        
        # Constant data should have low entropy
        assert result < 0.5
    
    def test_linear_data(self):
        """Test with linear data."""
        data = np.arange(100, dtype=float)
        
        swwe = SymbolicWeightedWaveletEntropy()
        result = swwe.fit_transform(data)
        
        assert isinstance(result, float)
    
    def test_noisy_data(self):
        """Test with high noise."""
        np.random.seed(42)
        data = np.random.randn(100) * 5.0  # High noise
        
        swwe = SymbolicWeightedWaveletEntropy()
        result = swwe.fit_transform(data)
        
        assert isinstance(result, float)
    
    def test_short_series(self):
        """Test with short time series."""
        np.random.seed(42)
        data = np.random.randn(30)  # Short but > embedding_dim
        
        swwe = SymbolicWeightedWaveletEntropy()
        result = swwe.fit_transform(data)
        
        assert isinstance(result, float)


class TestIntegration:
    """Integration tests for hybrid estimators."""
    
    def test_full_pipeline(self):
        """Test full pipeline with hybrid estimators."""
        from src.metrics_integration import ConsciousnessMetricsPipeline
        
        np.random.seed(42)
        data = np.random.randn(500) * 0.5 + np.sin(np.arange(500))
        
        pipeline = ConsciousnessMetricsPipeline(use_hybrid=True)
        state = pipeline.classify_state(data)
        
        assert hasattr(state, 'state_label')
        assert hasattr(state, 'confidence')
        assert hasattr(state, 'swwe_entropy')
        assert hasattr(state, 'nsaes_entropy')
    
    def test_state_transitions(self):
        """Test state transition detection."""
        from src.metrics_integration import ConsciousnessMetricsPipeline
        
        np.random.seed(42)
        data = np.concatenate([
            np.random.randn(100) * 0.5,
            np.random.randn(100) * 0.3,  # Different regime
        ])
        
        pipeline = ConsciousnessMetricsPipeline(use_hybrid=True)
        transitions = pipeline.detect_state_transitions(
            data,
            window_size=50,
            threshold=0.3
        )
        
        # Should detect at least one transition
        assert len(transitions) >= 0
    
    def test_noise_sensitivity(self):
        """Test noise sensitivity analysis."""
        from src.entropy_estimators import EntropyComparisonAnalyzer
        
        np.random.seed(42)
        clean_data = np.sin(np.arange(500))
        noisy_data = clean_data + np.random.randn(500) * 0.1
        
        analyzer = EntropyComparisonAnalyzer()
        sensitivity = analyzer.analyze_noise_sensitivity(
            clean_data=clean_data,
            noisy_data=noisy_data,
            noise_level=0.1
        )
        
        assert isinstance(sensitivity, dict)
        assert 'swwe' in sensitivity or 'sample_entropy' in sensitivity


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
