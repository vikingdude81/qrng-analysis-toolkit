"""
Integration tests for the inference pipeline.

Tests QRNG bridge, experiment runner, anomaly detection, and model loading.
"""

import os
import sys
from pathlib import Path
from typing import List, Dict, Any
import pytest

# Add parent directories to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "inference_framework"))


class TestQRNGBridge:
    """Tests for QRNG bridge module."""
    
    def test_qrng_bridge_initialization(self):
        """Test that QRNGBridge initializes correctly."""
        from inference_framework.qrng_bridge import QRNGBridge
        
        bridge = QRNGBridge()
        assert bridge is not None
        
    def test_qrng_bridge_generate_random_uninitialized(self):
        """Test that generate_random raises error when QRNG not initialized."""
        from inference_framework.qrng_bridge import QRNGBridge
        
        bridge = QRNGBridge()
        with pytest.raises(RuntimeError, match="QRNG not initialized"):
            bridge.generate_random(10)
    
    def test_qrng_bridge_generate_random_initialized(self):
        """Test that generate_random works when QRNG is initialized."""
        from inference_framework.qrng_bridge import QRNGBridge
        
        # Create a mock QRNG instance
        class MockQRNG:
            def __init__(self):
                self._counter = 0
            
            def random(self) -> float:
                self._counter += 1
                return float(self._counter) / (self._counter + 1)
        
        bridge = QRNGBridge()
        bridge.qrng = MockQRNG()
        
        samples = bridge.generate_random(100)
        assert len(samples) == 100
        assert all(isinstance(s, float) for s in samples)
    
    def test_generate_random_function(self):
        """Test the generate_random function."""
        from inference_framework.qrng_bridge import generate_random
        
        # Create mock QRNG
        class MockQRNG:
            def __init__(self):
                self._counter = 0
            
            def random(self) -> float:
                self._counter += 1
                return float(self._counter) / (self._counter + 1)
        
        # Patch the QRNGBridge to use mock
        from inference_framework import qrng_bridge
        original_qrng = qrng_bridge.QRNGBridge.__init__
        
        def mock_init(self):
            self.qrng = MockQRNG()
        
        qrng_bridge.QRNGBridge.__init__ = mock_init
        
        try:
            samples = generate_random(10_000)
            assert len(samples) == 10_000
            assert all(isinstance(s, float) for s in samples)
        finally:
            qrng_bridge.QRNGBridge.__init__ = original_qrng


class TestExperimentRunner:
    """Tests for experiment runner module."""
    
    def test_experiment_runner_initialization(self):
        """Test that ExperimentRunner initializes correctly."""
        from inference_framework.experiment.runner import ExperimentRunner
        
        runner = ExperimentRunner()
        assert runner is not None
        assert len(runner.experiments) == 0
    
    def test_experiment_runner_run_experiment(self):
        """Test that run_experiment works correctly."""
        from inference_framework.experiment.runner import ExperimentRunner
        
        runner = ExperimentRunner()
        result = runner.run_experiment("test_exp")
        
        assert "name" in result
        assert "status" in result
        assert result["name"] == "test_exp"
        assert result["status"] == "success"
    
    def test_experiment_runner_no_experiments(self):
        """Test that run_experiment raises error when no experiments defined."""
        from inference_framework.experiment.runner import ExperimentRunner
        
        runner = ExperimentRunner()
        with pytest.raises(RuntimeError, match="No experiments defined"):
            runner.run_experiment("nonexistent_exp")


class TestInferencePipeline:
    """Tests for the complete inference pipeline."""
    
    def test_load_qrng_bridge(self):
        """Test loading QRNG bridge module."""
        from inference_framework.qrng_bridge import QRNGBridge, generate_random
        
        bridge = QRNGBridge()
        assert bridge is not None
    
    def test_generate_random_output_shape(self):
        """Test that generate_random returns correct output shape."""
        from inference_framework.qrng_bridge import generate_random
        
        # Create mock QRNG
        class MockQRNG:
            def __init__(self):
                self._counter = 0
            
            def random(self) -> float:
                self._counter += 1
                return float(self._counter) / (self._counter + 1)
        
        from inference_framework import qrng_bridge
        original_qrng = qrng_bridge.QRNGBridge.__init__
        
        def mock_init(self):
            self.qrng = MockQRNG()
        
        qrng_bridge.QRNGBridge.__init__ = mock_init
        
        try:
            samples = generate_random(10_000)
            assert len(samples) == 10_000
            assert all(isinstance(s, float) for s in samples)
        finally:
            qrng_bridge.QRNGBridge.__init__ = original_qrng
    
    def test_experiment_runner_output_structure(self):
        """Test that experiment runner returns correct output structure."""
        from inference_framework.experiment.runner import ExperimentRunner
        
        runner = ExperimentRunner()
        result = runner.run_experiment("test_exp")
        
        assert isinstance(result, dict)
        assert "name" in result
        assert "status" in result
        assert "results" in result
    
    def test_load_anomaly_scope(self):
        """Test loading anomaly scope module."""
        from helios_anomaly_scope import HelioAnomalyScope
        
        scope = HelioAnomalyScope()
        assert scope is not None
    
    def test_load_models(self):
        """Test loading model modules."""
        from models import Model
        
        model = Model()
        assert model is not None
    
    def test_load_dataset(self):
        """Test loading QRNG dataset."""
        from inference_framework.qrng_bridge import generate_random
        
        data = generate_random(10_000)
        assert len(data) == 10_000
        assert all(isinstance(x, float) for x in data)


class TestInferencePipelineWithAnomalyDetection:
    """Tests for inference pipeline with anomaly detection."""
    
    def test_anomaly_detection_output_shape(self):
        """Test that anomaly detection returns correct output shape."""
        from helios_anomaly_scope import HelioAnomalyScope
        
        scope = HelioAnomalyScope()
        
        # Create sample data
        import numpy as np
        data = np.random.randn(1000)
        metadata = {}
        
        anomalies, scopes = scope.detect(data, metadata)
        
        assert isinstance(anomalies, list)
        assert isinstance(scopes, dict)
    
    def test_anomaly_detection_with_normal_data(self):
        """Test anomaly detection with normal data produces few anomalies."""
        from helios_anomaly_scope import HelioAnomalyScope
        
        scope = HelioAnomalyScope()
        
        # Create normal Gaussian data
        import numpy as np
        data = np.random.randn(1000)
        metadata = {}
        
        anomalies, scopes = scope.detect(data, metadata)
        
        # Should have few or no anomalies in normal data
        assert len(anomalies) < 50
    
    def test_anomaly_detection_with_injected_anomalies(self):
        """Test anomaly detection with injected anomalies."""
        from helios_anomaly_scope import HelioAnomalyScope
        
        scope = HelioAnomalyScope()
        
        # Create data with injected anomalies
        import numpy as np
        data = np.random.randn(1000)
        anomaly_indices = [50, 150, 250, 350, 450]
        for idx in anomaly_indices:
            data[idx] = data[idx] + 5.0  # Inject large deviation
        
        metadata = {}
        anomalies, scopes = scope.detect(data, metadata)
        
        # Should detect some anomalies
        assert len(anomalies) > 0
    
    def test_anomaly_detection_scopes(self):
        """Test that anomaly scopes contain expected fields."""
        from helios_anomaly_scope import HelioAnomalyScope
        
        scope = HelioAnomalyScope()
        
        import numpy as np
        data = np.random.randn(1000)
        metadata = {}
        
        anomalies, scopes = scope.detect(data, metadata)
        
        if len(anomalies) > 0:
            # Check that scopes contain expected fields
            for anomaly in anomalies[:min(3, len(anomalies))]:
                assert "timestamp" in anomaly
                assert "score" in anomaly
                assert "type" in anomaly


class TestModelLoading:
    """Tests for model loading."""
    
    def test_model_initialization(self):
        """Test that Model initializes correctly."""
        from models import Model
        
        model = Model()
        assert model is not None
    
    def test_model_predict(self):
        """Test that model can make predictions."""
        from models import Model
        
        model = Model()
        
        # Create sample input
        import numpy as np
        input_data = np.random.randn(10, 5)
        
        # Make prediction
        prediction = model.predict(input_data)
        
        assert isinstance(prediction, np.ndarray)
        assert len(prediction.shape) == 1


class TestInferenceFrameworkIntegration:
    """Tests for integration between different modules."""
    
    def test_qrng_bridge_to_experiment_runner(self):
        """Test that QRNG bridge can feed into experiment runner."""
        from inference_framework.qrng_bridge import QRNGBridge, generate_random
        from inference_framework.experiment.runner import ExperimentRunner
        
        # Create QRNG bridge with mock
        class MockQRNG:
            def __init__(self):
                self._counter = 0
            
            def random(self) -> float:
                self._counter += 1
                return float(self._counter) / (self._counter + 1)
        
        from inference_framework import qrng_bridge, experiment
        original_qrng_init = qrng_bridge.QRNGBridge.__init__
        original_runner_init = experiment.runner.ExperimentRunner.__init__
        
        def mock_qrng_init(self):
            self.qrng = MockQRNG()
        
        def mock_runner_init(self):
            self.experiments = []
        
        qrng_bridge.QRNGBridge.__init__ = mock_qrng_init
        experiment.runner.ExperimentRunner.__init__ = mock_runner_init
        
        try:
            # Generate random data
            samples = generate_random(100)
            assert len(samples) == 100
            
            # Run experiment
            runner = ExperimentRunner()
            result = runner.run_experiment("integration_test")
            
            assert "name" in result
        finally:
            qrng_bridge.QRNGBridge.__init__ = original_qrng_init
            experiment.runner.ExperimentRunner.__init__ = original_runner_init
    
    def test_full_pipeline_with_anomaly_detection(self):
        """Test full pipeline with anomaly detection."""
        from inference_framework.qrng_bridge import QRNGBridge, generate_random
        from helios_anomaly_scope import HelioAnomalyScope
        
        # Generate random data
        samples = generate_random(1000)
        assert len(samples) == 1000
        
        # Run anomaly detection
        scope = HelioAnomalyScope()
        anomalies, scopes = scope.detect(samples, {})
        
        assert isinstance(anomalies, list)
        assert isinstance(scopes, dict)