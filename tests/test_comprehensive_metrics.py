"""
Tests for comprehensive metrics module.

This test suite validates the unified metrics computation functionality
across entropy, chaos, and consciousness metrics.
"""

import numpy as np
import pytest
from src.metrics.comprehensive_metrics import (
    MetricsConfig,
    compute_all_metrics,
    compute_metrics_batch
)


class TestMetricsConfig:
    """Tests for MetricsConfig class."""
    
    def test_default_config(self):
        """Test default configuration values."""
        config = MetricsConfig()
        assert config.normalization == 'zscore'
        assert config.window_size == 100
        assert config.embedding_dim == 5
        assert config.time_lag == 2
        assert config.sample_entropy_m == 3
        assert config.sample_entropy_r == 0.1
    
    def test_minmax_normalization_config(self):
        """Test min-max normalization configuration."""
        config = MetricsConfig(normalization='minmax')
        assert config.normalization == 'minmax'
    
    def test_custom_config(self):
        """Test custom configuration values."""
        config = MetricsConfig(
            normalization='zscore',
            window_size=200,
            embedding_dim=7,
            time_lag=1
        )
        assert config.window_size == 200
        assert config.embedding_dim == 7
        assert config.time_lag == 1


class TestComputeAllMetrics:
    """Tests for compute_all_metrics function."""
    
    def test_basic_computation(self):
        """Test basic metrics computation."""
        np.random.seed(42)
        time_series = np.random.randn(1000)
        
        result = compute_all_metrics(time_series)
        
        assert 'shannon_entropy' in result
        assert 'sample_entropy' in result
        assert 'fuzzy_entropy' in result
        assert 'permutation_entropy' in result
        assert 'lyapunov_spectrum' in result
        assert 'epiplexity' in result
    
    def test_return_types(self):
        """Test that all return values are floats."""
        np.random.seed(42)
        time_series = np.random.randn(1000)
        
        result = compute_all_metrics(time_series)
        
        for key, value in result.items():
            assert isinstance(value, float), f"{key} should be float, got {type(value)}"
    
    def test_empty_array(self):
        """Test computation with empty array."""
        time_series = np.array([])
        
        with pytest.raises(ValueError):
            compute_all_metrics(time_series)
    
    def test_constant_signal(self):
        """Test computation with constant signal."""
        time_series = np.ones(1000) * 5.0
        
        result = compute_all_metrics(time_series)
        
        # Shannon entropy should be close to 0 for constant signal
        assert abs(result['shannon_entropy']) < 1e-10
    
    def test_with_config(self):
        """Test computation with custom config."""
        np.random.seed(42)
        time_series = np.random.randn(1000)
        
        config = MetricsConfig(
            normalization='minmax',
            embedding_dim=7,
            time_lag=1
        )
        
        result = compute_all_metrics(time_series, config)
        
        assert 'shannon_entropy' in result
        assert len(result) == 6


class TestComputeMetricsBatch:
    """Tests for compute_metrics_batch function."""
    
    def test_basic_batch_computation(self):
        """Test basic batch metrics computation."""
        np.random.seed(42)
        time_series_list = [
            np.random.randn(100),
            np.random.randn(200),
            np.random.randn(300)
        ]
        
        result = compute_metrics_batch(time_series_list)
        
        assert 'shannon_entropy' in result
        assert len(result['shannon_entropy']) == 3
    
    def test_batch_return_types(self):
        """Test that batch results are numpy arrays."""
        np.random.seed(42)
        time_series_list = [
            np.random.randn(100),
            np.random.randn(200)
        ]
        
        result = compute_metrics_batch(time_series_list)
        
        for key, value in result.items():
            assert isinstance(value, np.ndarray)
    
    def test_batch_with_config(self):
        """Test batch computation with custom config."""
        np.random.seed(42)
        time_series_list = [
            np.random.randn(100),
            np.random.randn(200)
        ]
        
        config = MetricsConfig(normalization='minmax')
        result = compute_metrics_batch(time_series_list, config)
        
        assert len(result['shannon_entropy']) == 2


class TestMetricsConsistency:
    """Tests for metrics consistency across different inputs."""
    
    def test_different_signal_types(self):
        """Test metrics on different signal types."""
        np.random.seed(42)
        
        # White noise
        white_noise = np.random.randn(1000)
        
        # Sine wave
        t = np.linspace(0, 10, 1000)
        sine_wave = np.sin(t)
        
        # Random walk
        random_walk = np.cumsum(np.random.randn(1000))
        
        metrics_white = compute_all_metrics(white_noise)
        metrics_sine = compute_all_metrics(sine_wave)
        metrics_random = compute_all_metrics(random_walk)
        
        # White noise should have higher entropy than sine wave
        assert metrics_white['shannon_entropy'] > metrics_sine['shannon_entropy']
    
    def test_reproducibility(self):
        """Test that results are reproducible."""
        np.random.seed(42)
        time_series = np.random.randn(1000)
        
        result1 = compute_all_metrics(time_series)
        result2 = compute_all_metrics(time_series)
        
        assert np.allclose(result1, result2)


class TestUnifiedInterface:
    """Tests for unified interface recommendations."""
    
    def test_standardized_normalization(self):
        """Test standardized Z-score normalization."""
        np.random.seed(42)
        time_series = np.random.randn(1000) + 5.0  # Add offset
        
        config = MetricsConfig(normalization='zscore')
        result = compute_all_metrics(time_series, config)
        
        # Z-score normalization should handle offset correctly
        assert not np.isnan(result['shannon_entropy'])
    
    def test_dynamic_windowing(self):
        """Test dynamic windowing strategy."""
        np.random.seed(42)
        
        # Short signal
        short_signal = np.random.randn(100)
        
        # Long signal
        long_signal = np.random.randn(1000)
        
        config_short = MetricsConfig(window_size=50)
        config_long = MetricsConfig(window_size=200)
        
        result_short = compute_all_metrics(short_signal, config_short)
        result_long = compute_all_metrics(long_signal, config_long)
        
        assert len(result_short) == 6
        assert len(result_long) == 6
    
    def test_default_parameters(self):
        """Test default parameter set."""
        np.random.seed(42)
        time_series = np.random.randn(1000)
        
        # Default config should work
        result = compute_all_metrics(time_series)
        
        assert 'shannon_entropy' in result
        assert 'sample_entropy' in result
        assert 'fuzzy_entropy' in result
        assert 'permutation_entropy' in result
        assert 'lyapunov_spectrum' in result
        assert 'epiplexity' in result
    
    def test_compatibility(self):
        """Test compatibility with existing metrics modules."""
        np.random.seed(42)
        time_series = np.random.randn(1000)
        
        # Test that results are comparable to expected ranges
        result = compute_all_metrics(time_series)
        
        # Shannon entropy should be in reasonable range for normalized data
        assert 0 <= result['shannon_entropy'] <= 1.0
        
        # Sample entropy should be non-negative
        assert result['sample_entropy'] >= 0