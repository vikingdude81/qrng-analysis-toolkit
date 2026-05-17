"""
Unit tests for TensorAnalyzer class
Tests edge cases: empty streams, NaNs in input, non-i.i.d. inputs
Uses pytest with hypothesis for property-based testing
"""
import numpy as np
import pytest
from hypothesis import given, strategies as st


class TestTensorAnalyzer:
    """Test suite for TensorAnalyzer class."""
    
    def test_empty_stream(self):
        """Test that empty stream returns empty result."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        analyzer = TensorAnalyzer()
        analyzer.fit([])
        result = analyzer.analyze()
        assert len(result) == 0 or 'empty' in str(result).lower()
    
    def test_nan_input(self):
        """Test handling of NaNs in input data."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        analyzer = TensorAnalyzer()
        data = [np.nan, 1.0, 2.0]
        analyzer.fit(data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_single_element(self):
        """Test with single element input."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        analyzer = TensorAnalyzer()
        data = [5.0]
        analyzer.fit(data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_all_zeros(self):
        """Test with all zeros input."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        analyzer = TensorAnalyzer()
        data = [0.0, 0.0, 0.0]
        analyzer.fit(data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_large_values(self):
        """Test with large magnitude values."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        analyzer = TensorAnalyzer()
        data = [1e10, -1e10, 5e9]
        analyzer.fit(data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_negative_values(self):
        """Test with negative values."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        analyzer = TensorAnalyzer()
        data = [-1.0, -2.0, -3.0]
        analyzer.fit(data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_mixed_signs(self):
        """Test with mixed positive and negative values."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        analyzer = TensorAnalyzer()
        data = [-1.0, 2.0, -3.0, 4.0]
        analyzer.fit(data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_non_iid_inputs(self):
        """Test with non-i.i.d. inputs using hypothesis."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        
        @given(st.lists(st.floats(allow_nan=False), min_size=2))
        def test(data):
            analyzer = TensorAnalyzer()
            analyzer.fit(data)
            result = analyzer.analyze()
            assert len(result) > 0
        
        test()
    
    def test_iid_inputs(self):
        """Test with i.i.d. inputs."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        np.random.seed(42)
        data = [np.random.randn() for _ in range(100)]
        analyzer = TensorAnalyzer()
        analyzer.fit(data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_repeated_values(self):
        """Test with repeated identical values."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        analyzer = TensorAnalyzer()
        data = [5.0, 5.0, 5.0, 5.0]
        analyzer.fit(data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_alternating_pattern(self):
        """Test with alternating pattern."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        analyzer = TensorAnalyzer()
        data = [1.0, -1.0, 2.0, -2.0, 3.0, -3.0]
        analyzer.fit(data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_monotonic_increasing(self):
        """Test with monotonically increasing sequence."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        analyzer = TensorAnalyzer()
        data = list(range(1, 21))
        analyzer.fit(data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_monotonic_decreasing(self):
        """Test with monotonically decreasing sequence."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        analyzer = TensorAnalyzer()
        data = list(range(20, 0, -1))
        analyzer.fit(data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_periodic_pattern(self):
        """Test with periodic pattern."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        analyzer = TensorAnalyzer()
        data = [np.sin(i) for i in range(50)]
        analyzer.fit(data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_spike_detection(self):
        """Test with spike values (anomalies)."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        analyzer = TensorAnalyzer()
        data = [1.0, 2.0, 3.0, 100.0, 4.0, 5.0]  # 100 is a spike
        analyzer.fit(data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_outlier_detection(self):
        """Test with outlier values."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        np.random.seed(42)
        data = [np.random.randn() for _ in range(50)]
        data.append(np.random.randn() * 10)  # Add an outlier
        analyzer = TensorAnalyzer()
        analyzer.fit(data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_infinity_handling(self):
        """Test handling of infinity values."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        analyzer = TensorAnalyzer()
        data = [np.inf, -np.inf, 1.0]
        analyzer.fit(data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_precision_edge_cases(self):
        """Test with precision edge cases."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        analyzer = TensorAnalyzer()
        data = [1e-15, 1e-10, 1e-5, 1e0, 1e5, 1e10]
        analyzer.fit(data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_qrng_data(self):
        """Test with typical QRNG data patterns."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        np.random.seed(42)
        # Simulate QRNG output (should be uniform-ish)
        qrng_data = [np.random.uniform(0, 1) for _ in range(1000)]
        analyzer = TensorAnalyzer()
        analyzer.fit(qrng_data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_drifting_mean(self):
        """Test with drifting mean (non-stationary)."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        drift_data = []
        for i in range(100):
            drift_data.extend(np.random.randn() + 0.1 * i / 50)
        analyzer = TensorAnalyzer()
        analyzer.fit(drift_data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_high_frequency_noise(self):
        """Test with high-frequency noise patterns."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        np.random.seed(42)
        freq_data = [np.sin(10 * i) + np.cos(5 * i) for i in range(100)]
        analyzer = TensorAnalyzer()
        analyzer.fit(freq_data)
        result = analyzer.analyze()
        assert len(result) > 0
    
    def test_low_frequency_signal(self):
        """Test with low-frequency signal patterns."""
        from cuquantum_accelerator.tensor_analysis import TensorAnalyzer
        freq_data = [np.sin(0.1 * i) for i in range(100)]
        analyzer = TensorAnalyzer()
        analyzer.fit(freq_data)
        result = analyzer.analyze()
        assert len(result) > 0