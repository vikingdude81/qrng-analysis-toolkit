import pytest

class TestHeliosCoreImports:
    """Test that helios core modules can be imported successfully."""
    
    def test_chaos_analysis_import(self):
        from helios_core import chaos_analysis
        assert chaos_analysis is not None
        
    def test_consciousness_metrics_import(self):
        from helios_core import consciousness_metrics
        assert consciousness_metrics is not None
        
    def test_epiplexity_estimator_import(self):
        from helios_core import epiplexity_estimator
        assert epiplexity_estimator is not None

class TestChaosAnalysis:
    """Test chaos analysis functions."""
    
    def test_analyze_trajectory_basic(self):
        from helios_core.chaos_analysis import analyze_trajectory
        import numpy as np
        data = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = analyze_trajectory(data)
        assert not np.isnan(result)
    
    def test_analyze_trajectory_with_dict(self):
        from helios_core.chaos_analysis import analyze_trajectory
        data = {'trajectory': np.array([1.0, 2.0, 3.0])}
        result = analyze_trajectory(data)
        assert not np.isnan(result)

class TestConsciousnessMetrics:
    """Test consciousness metrics functions."""
    
    def test_calculate_consciousness_valid(self):
        from helios_core.consciousness_metrics import calculate_consciousness
        result = calculate_consciousness(0.5)
        assert not np.isnan(result)
    
    def test_calculate_consciousness_invalid_alpha(self):
        from helios_core.consciousness_metrics import calculate_consciousness
        with pytest.raises(TypeError):
            calculate_consciousness(-1.0)

class TestEpiplexityEstimator:
    """Test epiplexity estimator."""
    
    def test_epiplexity_estimator_init(self):
        from helios_core.epiplexity_estimator import EpiplexityEstimator
        estimator = EpiplexityEstimator(entropy=0.5)
        assert isinstance(estimator, EpiplexityEstimator)
        assert hasattr(estimator, 'calculate')
    
    def test_epiplexity_estimator_calculate(self):
        from helios_core.epiplexity_estimator import EpiplexityEstimator
        estimator = EpiplexityEstimator(entropy=0.5)
        result = estimator.calculate()
        assert isinstance(result, (int, float))