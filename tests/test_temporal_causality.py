# helios-trajectory-analysis/tests/test_temporal_causality.py
"""Tests for temporal causality analysis."""

import pytest
import numpy as np
from helios.analysis import temporal_causality


class TestTransferEntropy:
    """Tests for transfer entropy computation."""
    
    def test_basic_case(self):
        """Test basic transfer entropy estimation."""
        np.random.seed(42)
        series_a = np.random.randn(1000)
        series_b = np.random.randn(1000)
        te = temporal_causality.compute_transfer_entropy(series_a, series_b, k=3, m=2)
        assert isinstance(te, float), f"Transfer entropy should be float, got {type(te)}"
    
    def test_markov_series(self):
        """Test with Markov series."""
        # Generate a simple Markov process
        np.random.seed(42)
        states = np.zeros(1000)
        for i in range(1, len(states)):
            states[i] = 1 if np.random.rand() > 0.5 else 0
        
        te = temporal_causality.compute_transfer_entropy(states, states, k=2, m=1)
        assert isinstance(te, float), f"Transfer entropy should be float, got {type(te)}"
    
    def test_non_markov_series(self):
        """Test with non-Markov series."""
        np.random.seed(42)
        # Create a series with memory
        series = np.zeros(1000)
        for i in range(1, len(series)):
            series[i] = 0.5 * series[i-1] + np.random.randn() * 0.1
        
        te = temporal_causality.compute_transfer_entropy(series, series, k=3, m=2)
        assert isinstance(te, float), f"Transfer entropy should be float, got {type(te)}"
    
    def test_chaotic_series(self):
        """Test with chaotic series."""
        # Generate Lorenz attractor-like chaos
        np.random.seed(42)
        x = 1.0
        y = 1.0
        z = 1.0
        
        for _ in range(1000):
            x = 10 * (y - x)
            y = 28 * x - y - z
            z = x * y - (8/3) * z
            
        series_a = np.array([x, y, z])
        te = temporal_causality.compute_transfer_entropy(series_a, series_a, k=2, m=1)
        assert isinstance(te, float), f"Transfer entropy should be float, got {type(te)}"


class TestCausalAsymmetry:
    """Tests for causal asymmetry score."""
    
    def test_basic_case(self):
        """Test basic causal asymmetry estimation."""
        np.random.seed(42)
        timeseries = np.random.randn(500)
        score = temporal_causality.causal_asymmetry_score(timeseries, alpha=1.0)
        assert isinstance(score, float), f"Causal asymmetry should be float, got {type(score)}"
    
    def test_alpha_parameter(self):
        """Test with different alpha values."""
        np.random.seed(42)
        timeseries = np.random.randn(500)
        
        for alpha in [0.5, 1.0, 2.0, 3.0]:
            score = temporal_causality.causal_asymmetry_score(timeseries, alpha=alpha)
            assert isinstance(score, float), f"Causal asymmetry should be float for alpha={alpha}, got {type(score)}"
    
    def test_short_series(self):
        """Test with short series."""
        timeseries = np.random.randn(20)
        score = temporal_causality.causal_asymmetry_score(timeseries, alpha=1.0)
        assert isinstance(score, float), f"Causal asymmetry should be float, got {type(score)}"


class TestMemoryKernel:
    """Tests for memory kernel fitting."""
    
    def test_basic_case(self):
        """Test basic memory kernel fitting."""
        np.random.seed(42)
        series = np.random.randn(100)
        decay, fit = temporal_causality.memory_kernel_fit(series, lags=[1, 2, 3])
        assert isinstance(decay, float), f"Decay should be float, got {type(decay)}"
        assert isinstance(fit, float), f"Fit should be float, got {type(fit)}"
    
    def test_markov_series(self):
        """Test with Markov series."""
        np.random.seed(42)
        states = np.zeros(100)
        for i in range(1, len(states)):
            states[i] = 1 if np.random.rand() > 0.5 else 0
        
        decay, fit = temporal_causality.memory_kernel_fit(states, lags=[1, 2])
        assert isinstance(decay, float), f"Decay should be float, got {type(decay)}"
    
    def test_non_markov_series(self):
        """Test with non-Markov series."""
        np.random.seed(42)
        # Create a series with memory
        series = np.zeros(100)
        for i in range(1, len(series)):
            series[i] = 0.7 * series[i-1] + np.random.randn() * 0.1
        
        decay, fit = temporal_causality.memory_kernel_fit(series, lags=[1, 2, 3])
        assert isinstance(decay, float), f"Decay should be float, got {type(decay)}"
    
    def test_chaotic_series(self):
        """Test with chaotic series."""
        np.random.seed(42)
        x = 1.0
        y = 1.0
        z = 1.0
        
        for _ in range(100):
            x = 10 * (y - x)
            y = 28 * x - y - z
            z = x * y - (8/3) * z
            
        series = np.array([x, y, z])
        decay, fit = temporal_causality.memory_kernel_fit(series, lags=[1])
        assert isinstance(decay, float), f"Decay should be float, got {type(decay)}"
    
    def test_short_series(self):
        """Test with short series."""
        series = np.random.randn(10)
        decay, fit = temporal_causality.memory_kernel_fit(series, lags=[1])
        assert isinstance(decay, float), f"Decay should be float, got {type(decay)}"


class TestEdgeCases:
    """Tests for edge cases and error handling."""
    
    def test_empty_series(self):
        """Test with empty series."""
        series_a = np.array([])
        series_b = np.array([])
        
        with pytest.raises(Exception):
            temporal_causality.compute_transfer_entropy(series_a, series_b)
    
    def test_single_element(self):
        """Test with single element."""
        series_a = np.array([1.0])
        series_b = np.array([2.0])
        
        te = temporal_causality.compute_transfer_entropy(series_a, series_b)
        assert isinstance(te, float), f"Transfer entropy should be float, got {type(te)}"
    
    def test_constant_series(self):
        """Test with constant series."""
        series_a = np.ones(100) * 5.0
        series_b = np.ones(100) * 3.0
        
        te = temporal_causality.compute_transfer_entropy(series_a, series_b)
        assert isinstance(te, float), f"Transfer entropy should be float, got {type(te)}"


class TestIntegration:
    """Integration tests for combined functionality."""
    
    def test_full_analysis(self):
        """Test full temporal causality analysis pipeline."""
        np.random.seed(42)
        n_samples = 500
        
        # Generate synthetic data
        series_a = np.random.randn(n_samples)
        series_b = 0.5 * series_a + np.random.randn(n_samples) * 0.3
        
        # Compute all metrics
        te_ab = temporal_causality.compute_transfer_entropy(series_a, series_b, k=3, m=2)
        te_ba = temporal_causality.compute_transfer_entropy(series_b, series_a, k=3, m=2)
        asymmetry = temporal_causality.causal_asymmetry_score(np.concatenate([series_a, series_b]), alpha=1.0)
        decay, fit = temporal_causality.memory_kernel_fit(series_a, lags=[1, 2, 3])
        
        # Verify results
        assert isinstance(te_ab, float), f"TE(A->B) should be float, got {type(te_ab)}"
        assert isinstance(te_ba, float), f"TE(B->A) should be float, got {type(te_ba)}"
        assert isinstance(asymmetry, float), f"Asymmetry should be float, got {type(asymmetry)}"
        assert isinstance(decay, float), f"Decay should be float, got {type(decay)}"
        assert isinstance(fit, float), f"Fit should be float, got {type(fit)}"
