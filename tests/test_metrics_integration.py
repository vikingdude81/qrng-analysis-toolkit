"""
Integration Tests for Metrics Module

These tests validate the complete metrics computation pipeline.
"""

import numpy as np
import pytest
from src.metrics.comprehensive_metrics import (
    compute_all_metrics,
    MetricsConfig,
    compute_metrics_batch
)


class TestMetricsIntegration:
    """Integration tests for metrics computation."""
    
    def test_single_series_computation(self):
        """Test computation on a single time series."""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        result = compute_all_metrics(data)
        
        assert 'shannon_entropy' in result
        assert 'sample_entropy' in result
        assert 'fuzzy_entropy' in result
        assert 'permutation_entropy' in result
        assert 'lyapunov_spectrum' in result
        assert 'epiplexity' in result
    
    def test_batch_computation(self):
        """Test computation on a batch of time series."""
        np.random.seed(42)
        data_list = [
            np.random.randn(100),
            np.random.randn(200),
            np.random.randn(300)
        ]
        
        result = compute_metrics_batch(data_list)
        
        assert 'shannon_entropy' in result
        assert len(result['shannon_entropy']) == 3
    
    def test_config_defaults(self):
        """Test default configuration."""
        config = MetricsConfig()
        
        assert config.normalization == 'zscore'
        assert config.embedding_dim == 5
        assert config.time_lag == 2
        assert config.window_size == 100
    
    def test_config_customization(self):
        """Test custom configuration."""
        config = MetricsConfig(
            normalization='minmax',
            embedding_dim=7,
            time_lag=3,
            window_size=200
        )
        
        assert config.normalization == 'minmax'
        assert config.embedding_dim == 7
        assert config.time_lag == 3
        assert config.window_size == 200
    
    def test_reproducibility(self):
        """Test that results are reproducible."""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        result1 = compute_all_metrics(data)
        result2 = compute_all_metrics(data)
        
        assert np.allclose(result1, result2)
    
    def test_different_seeds(self):
        """Test that different seeds produce different results."""
        data1 = np.random.randn(1000)
        data2 = np.random.randn(1000)
        
        result1 = compute_all_metrics(data1)
        result2 = compute_all_metrics(data2)
        
        assert not np.allclose(result1, result2)


class TestMetricsValues:
    """Tests for metric value ranges."""
    
    def test_shannon_entropy_range(self):
        """Test Shannon entropy is in valid range [0, 1]."""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        result = compute_all_metrics(data)
        
        assert 0 <= result['shannon_entropy'] <= 1
    
    def test_sample_entropy_range(self):
        """Test sample entropy is in valid range."""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        result = compute_all_metrics(data)
        
        assert isinstance(result['sample_entropy'], float)
    
    def test_permutation_entropy_range(self):
        """Test permutation entropy is in valid range."""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        result = compute_all_metrics(data)
        
        assert isinstance(result['permutation_entropy'], float)
    
    def test_epiplexity_range(self):
        """Test epiplexity is in valid range."""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        result = compute_all_metrics(data)
        
        assert isinstance(result['epiplexity'], float)


class TestMetricsWithDifferentData:
    """Tests with different types of data."""
    
    def test_constant_signal(self):
        """Test with a constant signal."""
        data = np.ones(1000) * 5.0
        
        result = compute_all_metrics(data)
        
        # Constant signal should have low entropy
        assert result['shannon_entropy'] < 0.1
    
    def test_linear_signal(self):
        """Test with a linear signal."""
        data = np.linspace(0, 10, 1000)
        
        result = compute_all_metrics(data)
        
        # Linear signal should have low entropy
        assert result['shannon_entropy'] < 0.5
    
    def test_random_signal(self):
        """Test with a random signal."""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        result = compute_all_metrics(data)
        
        # Random signal should have high entropy
        assert result['shannon_entropy'] > 0.5
    
    def test_sine_wave(self):
        """Test with a sine wave."""
        t = np.linspace(0, 10 * np.pi, 1000)
        data = np.sin(t)
        
        result = compute_all_metrics(data)
        
        # Sine wave should have moderate entropy
        assert 0.3 < result['shannon_entropy'] < 0.8
    
    def test_chaos_signal(self):
        """Test with a chaotic signal."""
        # Logistic map at chaotic regime
        x = 0.5
        for _ in range(1000):
            x = 4 * x * (1 - x)
        data = np.array([x])
        
        result = compute_all_metrics(data)
        
        # Chaotic signal should have high entropy
        assert result['shannon_entropy'] > 0.5


class TestMetricsNormalization:
    """Tests for different normalization methods."""
    
    def test_zscore_normalization(self):
        """Test Z-score normalization."""
        np.random.seed(42)
        data = np.random.randn(1000) + 5.0
        
        config = MetricsConfig(normalization='zscore')
        result = compute_all_metrics(data, config)
        
        assert not np.isnan(result['shannon_entropy'])
    
    def test_minmax_normalization(self):
        """Test min-max normalization."""
        np.random.seed(42)
        data = np.random.randn(1000) + 5.0
        
        config = MetricsConfig(normalization='minmax')
        result = compute_all_metrics(data, config)
        
        assert not np.isnan(result['shannon_entropy'])
    
    def test_normalization_affects_results(self):
        """Test that normalization affects results."""
        np.random.seed(42)
        data = np.random.randn(1000) + 5.0
        
        config_zscore = MetricsConfig(normalization='zscore')
        config_minmax = MetricsConfig(normalization='minmax')
        
        result_zscore = compute_all_metrics(data, config_zscore)
        result_minmax = compute_all_metrics(data, config_minmax)
        
        assert not np.allclose(result_zscore['shannon_entropy'],
                               result_minmax['shannon_entropy'])


class TestMetricsParameterSensitivity:
    """Tests for parameter sensitivity."""
    
    def test_embedding_dim_sensitivity(self):
        """Test sensitivity to embedding dimension."""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        config_m5 = MetricsConfig(embedding_dim=5, time_lag=2)
        config_m7 = MetricsConfig(embedding_dim=7, time_lag=1)
        
        result_m5 = compute_all_metrics(data, config_m5)
        result_m7 = compute_all_metrics(data, config_m7)
        
        # Results should be different
        assert not np.allclose(result_m5['permutation_entropy'],
                               result_m7['permutation_entropy'])
    
    def test_time_lag_sensitivity(self):
        """Test sensitivity to time lag."""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        config_tau2 = MetricsConfig(time_lag=2)
        config_tau3 = MetricsConfig(time_lag=3)
        
        result_tau2 = compute_all_metrics(data, config_tau2)
        result_tau3 = compute_all_metrics(data, config_tau3)
        
        # Results should be different
        assert not np.allclose(result_tau2['permutation_entropy'],
                               result_tau3['permutation_entropy'])
    
    def test_window_size_sensitivity(self):
        """Test sensitivity to window size."""
        np.random.seed(42)
        data = np.random.randn(1000)
        
        config_100 = MetricsConfig(window_size=100)
        config_200 = MetricsConfig(window_size=200)
        
        result_100 = compute_all_metrics(data, config_100)
        result_200 = compute_all_metrics(data, config_200)
        
        # Results should be similar (window_size affects batch processing more)
        assert np.isclose(result_100['shannon_entropy'],
                          result_200['shannon_entropy'])