"""
Cross-Project Compatibility Tests

This test suite validates compatibility between Helios-Trajectory-Analysis
and QRNG-Analysis-Toolkit metrics modules.
"""

import sys
import numpy as np
import pytest

# Test Helios metrics
sys.path.insert(0, 'src')
from src.metrics.comprehensive_metrics import (
    MetricsConfig as HeliosMetricsConfig,
    compute_all_metrics as helios_compute_all_metrics,
    compute_metrics_batch as helios_compute_metrics_batch
)

# Test QRNG Toolkit metrics (if available)
try:
    sys.path.insert(0, r'C:\Users\akbon\OneDrive\Documents\GitHub\qrng-analysis-toolkit')
    from metrics.comprehensive_metrics import (
        MetricsConfig as QRNGMetricsConfig,
        compute_all_metrics as qrng_compute_all_metrics,
        compute_metrics_batch as qrng_compute_metrics_batch
    )
    QRNG_AVAILABLE = True
except ImportError:
    QRNG_AVAILABLE = False


class TestHeliosMetrics:
    """Tests for Helios-Trajectory-Analysis metrics."""
    
    def test_basic_computation(self):
        """Test basic metrics computation in Helios."""
        np.random.seed(42)
        time_series = np.random.randn(1000)
        
        result = helios_compute_all_metrics(time_series)
        
        assert 'shannon_entropy' in result
        assert 'sample_entropy' in result
        assert 'fuzzy_entropy' in result
        assert 'permutation_entropy' in result
        assert 'lyapunov_spectrum' in result
        assert 'epiplexity' in result
    
    def test_config_creation(self):
        """Test MetricsConfig creation."""
        config = HeliosMetricsConfig()
        assert config.normalization == 'zscore'
        assert config.embedding_dim == 5
        assert config.time_lag == 2
    
    def test_batch_computation(self):
        """Test batch metrics computation."""
        np.random.seed(42)
        time_series_list = [
            np.random.randn(100),
            np.random.randn(200)
        ]
        
        result = helios_compute_metrics_batch(time_series_list)
        
        assert 'shannon_entropy' in result
        assert len(result['shannon_entropy']) == 2


class TestQRNGMetrics:
    """Tests for QRNG-Analysis-Toolkit metrics."""
    
    @pytest.mark.skipif(not QRNG_AVAILABLE, reason="QRNG toolkit not available")
    def test_basic_computation(self):
        """Test basic metrics computation in QRNG Toolkit."""
        np.random.seed(42)
        time_series = np.random.randn(1000)
        
        result = qrng_compute_all_metrics(time_series)
        
        assert 'shannon_entropy' in result
        assert 'sample_entropy' in result
        assert 'fuzzy_entropy' in result
        assert 'permutation_entropy' in result
        assert 'lyapunov_spectrum' in result
        assert 'epiplexity' in result
    
    @pytest.mark.skipif(not QRNG_AVAILABLE, reason="QRNG toolkit not available")
    def test_config_creation(self):
        """Test MetricsConfig creation."""
        config = QRNGMetricsConfig()
        assert config.normalization == 'zscore'
        assert config.embedding_dim == 5
        assert config.time_lag == 2
    
    @pytest.mark.skipif(not QRNG_AVAILABLE, reason="QRNG toolkit not available")
    def test_batch_computation(self):
        """Test batch metrics computation."""
        np.random.seed(42)
        time_series_list = [
            np.random.randn(100),
            np.random.randn(200)
        ]
        
        result = qrng_compute_metrics_batch(time_series_list)
        
        assert 'shannon_entropy' in result
        assert len(result['shannon_entropy']) == 2


class TestCrossProjectCompatibility:
    """Tests for cross-project compatibility."""
    
    @pytest.mark.skipif(not QRNG_AVAILABLE, reason="QRNG toolkit not available")
    def test_same_results(self):
        """Test that both projects produce same results."""
        np.random.seed(42)
        time_series = np.random.randn(1000)
        
        helios_result = helios_compute_all_metrics(time_series)
        qrng_result = qrng_compute_all_metrics(time_series)
        
        # Check that all keys match
        assert set(helios_result.keys()) == set(qrng_result.keys())
        
        # Check that values are close (allowing for minor numerical differences)
        for key in helios_result.keys():
            assert np.isclose(helios_result[key], qrng_result[key])
    
    @pytest.mark.skipif(not QRNG_AVAILABLE, reason="QRNG toolkit not available")
    def test_same_config(self):
        """Test that both projects use same config structure."""
        helios_config = HeliosMetricsConfig(
            normalization='zscore',
            embedding_dim=7,
            time_lag=1
        )
        
        qrng_config = QRNGMetricsConfig(
            normalization='zscore',
            embedding_dim=7,
            time_lag=1
        )
        
        assert helios_config.normalization == qrng_config.normalization
        assert helios_config.embedding_dim == qrng_config.embedding_dim
        assert helios_config.time_lag == qrng_config.time_lag
    
    @pytest.mark.skipif(not QRNG_AVAILABLE, reason="QRNG toolkit not available")
    def test_batch_compatibility(self):
        """Test batch computation compatibility."""
        np.random.seed(42)
        time_series_list = [
            np.random.randn(100),
            np.random.randn(200),
            np.random.randn(300)
        ]
        
        helios_result = helios_compute_metrics_batch(time_series_list)
        qrng_result = qrng_compute_metrics_batch(time_series_list)
        
        # Check that all keys match
        assert set(helios_result.keys()) == set(qrng_result.keys())
        
        # Check lengths match
        for key in helios_result.keys():
            assert len(helios_result[key]) == len(qrng_result[key])
    
    @pytest.mark.skipif(not QRNG_AVAILABLE, reason="QRNG toolkit not available")
    def test_config_normalization(self):
        """Test that both projects support same normalization methods."""
        helios_config = HeliosMetricsConfig(normalization='minmax')
        qrng_config = QRNGMetricsConfig(normalization='minmax')
        
        assert helios_config.normalization == 'minmax'
        assert qrng_config.normalization == 'minmax'


class TestUnifiedInterface:
    """Tests for unified interface recommendations."""
    
    def test_standardized_normalization(self):
        """Test standardized Z-score normalization."""
        np.random.seed(42)
        time_series = np.random.randn(1000) + 5.0
        
        config = HeliosMetricsConfig(normalization='zscore')
        result = helios_compute_all_metrics(time_series, config)
        
        assert not np.isnan(result['shannon_entropy'])
    
    def test_default_parameters(self):
        """Test default parameter set."""
        np.random.seed(42)
        time_series = np.random.randn(1000)
        
        result = helios_compute_all_metrics(time_series)
        
        assert len(result) == 6
    
    def test_reproducibility(self):
        """Test that results are reproducible."""
        np.random.seed(42)
        time_series = np.random.randn(1000)
        
        result1 = helios_compute_all_metrics(time_series)
        result2 = helios_compute_all_metrics(time_series)
        
        assert np.allclose(result1, result2)


class TestParameterSensitivity:
    """Tests for parameter sensitivity analysis."""
    
    def test_embedding_dim_sensitivity(self):
        """Test sensitivity to embedding dimension."""
        np.random.seed(42)
        time_series = np.random.randn(1000)
        
        config_m5 = HeliosMetricsConfig(embedding_dim=5, time_lag=2)
        config_m7 = HeliosMetricsConfig(embedding_dim=7, time_lag=1)
        
        result_m5 = helios_compute_all_metrics(time_series, config_m5)
        result_m7 = helios_compute_all_metrics(time_series, config_m7)
        
        # Results should be different due to different parameters
        assert not np.allclose(result_m5['permutation_entropy'], 
                               result_m7['permutation_entropy'])
    
    def test_normalization_sensitivity(self):
        """Test sensitivity to normalization method."""
        np.random.seed(42)
        time_series = np.random.randn(1000) + 5.0
        
        config_zscore = HeliosMetricsConfig(normalization='zscore')
        config_minmax = HeliosMetricsConfig(normalization='minmax')
        
        result_zscore = helios_compute_all_metrics(time_series, config_zscore)
        result_minmax = helios_compute_all_metrics(time_series, config_minmax)
        
        # Results should be different due to different normalization
        assert not np.allclose(result_zscore['shannon_entropy'],
                               result_minmax['shannon_entropy'])
    
    def test_window_size_sensitivity(self):
        """Test sensitivity to window size."""
        np.random.seed(42)
        time_series = np.random.randn(1000)
        
        config_100 = HeliosMetricsConfig(window_size=100)
        config_200 = HeliosMetricsConfig(window_size=200)
        
        result_100 = helios_compute_all_metrics(time_series, config_100)
        result_200 = helios_compute_all_metrics(time_series, config_200)
        
        # Results should be similar (window_size affects batch processing more)
        assert np.isclose(result_100['shannon_entropy'],
                          result_200['shannon_entropy'])