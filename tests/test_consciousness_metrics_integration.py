"""
Test suite for consciousness metrics module.

This module provides unit tests for the advanced entropy-based metrics used in
consciousness analysis within the helios-trajectory-analysis framework.
"""

import unittest
import numpy as np
from typing import Dict, Any


class TestConsciousnessMetrics(unittest.TestCase):
    """Test cases for ConsciousnessMetrics class."""
    
    def setUp(self):
        self.metrics = ConsciousnessMetrics()
    
    def test_compute_phi_entropy_ratio_normal(self):
        """Test ΦER computation with normal values."""
        phi = 1.0
        h = 0.5
        expected = phi / h
        result = self.metrics.compute_phi_entropy_ratio(phi, h)
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_compute_phi_entropy_ratio_zero_h(self):
        """Test ΦER when H is zero."""
        phi = 1.0
        h = 0.0
        result = self.metrics.compute_phi_entropy_ratio(phi, h)
        self.assertTrue(np.isinf(result))
    
    def test_compute_phi_entropy_ratio_zero_both(self):
        """Test ΦER when both Phi and H are zero."""
        phi = 0.0
        h = 0.0
        result = self.metrics.compute_phi_entropy_ratio(phi, h)
        self.assertEqual(result, 0.0)
    
    def test_compute_causality_complexity_ratio_normal(self):
        """Test CCR computation with normal values."""
        i = 1.5
        c = 0.5
        expected = i / c
        result = self.metrics.compute_causality_complexity_ratio(i, c)
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_compute_causality_complexity_ratio_zero_c(self):
        """Test CCR when C is zero."""
        i = 1.0
        c = 0.0
        result = self.metrics.compute_causality_complexity_ratio(i, c)
        self.assertTrue(np.isinf(result))
    
    def test_compute_all_metrics(self):
        """Test compute_all_metrics method."""
        phi = 2.5
        h = 1.0
        i = 1.5
        c = 0.5
        result = self.metrics.compute_all_metrics(phi, h, i, c)
        
        self.assertIn('phi', result)
        self.assertIn('h', result)
        self.assertIn('i', result)
        self.assertIn('c', result)
        self.assertIn('phi_entropy_ratio', result)
        self.assertIn('causality_complexity_ratio', result)
    
    def test_analyze_consciousness_state(self):
        """Test analyze_consciousness_state method."""
        phi = 2.5
        h = 1.0
        i = 1.5
        c = 0.5
        result = self.metrics.analyze_consciousness_state(phi, h, i, c)
        
        self.assertIn('metrics', result)
        self.assertIn('consciousness_indicators', result)
        self.assertIn('interpretation', result)
    
    def test_interpret_metrics_strong(self):
        """Test interpretation for strong consciousness indicators."""
        metrics = {
            'phi_entropy_ratio': 1.5,
            'causality_complexity_ratio': 0.6
        }
        result = self.metrics._interpret_metrics(metrics)
        self.assertIn("Strong evidence", result)
    
    def test_interpret_metrics_moderate(self):
        """Test interpretation for moderate consciousness indicators."""
        metrics = {
            'phi_entropy_ratio': 0.7,
            'causality_complexity_ratio': 0.4
        }
        result = self.metrics._interpret_metrics(metrics)
        self.assertIn("Moderate", result)
    
    def test_interpret_metrics_low(self):
        """Test interpretation for low consciousness indicators."""
        metrics = {
            'phi_entropy_ratio': 0.1,
            'causality_complexity_ratio': 0.05
        }
        result = self.metrics._interpret_metrics(metrics)
        self.assertIn("Low", result)
    
    def test_interpret_metrics_ambiguous(self):
        """Test interpretation for ambiguous consciousness state."""
        metrics = {
            'phi_entropy_ratio': 0.3,
            'causality_complexity_ratio': 0.2
        }
        result = self.metrics._interpret_metrics(metrics)
        self.assertIn("Ambiguous", result)


class TestStandaloneFunctions(unittest.TestCase):
    """Test cases for standalone consciousness metric functions."""
    
    def test_compute_phi_entropy_ratio_standalone(self):
        """Test standalone ΦER computation."""
        from consciousness_metrics import compute_phi_entropy_ratio
        phi = 2.0
        h = 1.0
        expected = phi / h
        result = compute_phi_entropy_ratio(phi, h)
        self.assertAlmostEqual(result, expected, places=6)
    
    def test_compute_causality_complexity_ratio_standalone(self):
        """Test standalone CCR computation."""
        from consciousness_metrics import compute_causality_complexity_ratio
        i = 2.0
        c = 1.0
        expected = i / c
        result = compute_causality_complexity_ratio(i, c)
        self.assertAlmostEqual(result, expected, places=6)


class TestEntropyEstimatorIntegration(unittest.TestCase):
    """Integration tests for entropy estimators."""
    
    def test_entropy_estimator_creation(self):
        """Test EntropyEstimator instantiation."""
        from entropy_estimators import EntropyEstimator
        estimator = EntropyEstimator()
        self.assertIsNotNone(estimator)
    
    def test_entropy_computation_numpy(self):
        """Test entropy computation with NumPy arrays."""
        from entropy_estimators import EntropyEstimator
        import numpy as np
        
        data = np.array([0.1, 0.2, 0.3, 0.4, 0.5])
        estimator = EntropyEstimator()
        result = estimator._compute_entropy(data)
        self.assertIsInstance(result, float)
    
    def test_empty_array_entropy(self):
        """Test entropy computation with empty array."""
        from entropy_estimators import EntropyEstimator
        import numpy as np
        
        data = np.array([])
        estimator = EntropyEstimator()
        result = estimator._compute_entropy(data)
        self.assertEqual(result, 0.0)


class TestQRNGBridge(unittest.TestCase):
    """Test cases for QRNG bridge functions."""
    
    def test_generate_entropy(self):
        """Test generate_entropy function."""
        from qrng_bridge import generate_entropy
        data = [b'\x00\x01\x02', b'\xff\xfe\xfd']
        result = generate_entropy(data)
        self.assertIsInstance(result, int)
    
    def test_compute_entropy_from_bytes(self):
        """Test compute_entropy_from_bytes function."""
        from qrng_bridge import compute_entropy_from_bytes
        data = b'\x00\x01\x02\x03\x04'
        result = compute_entropy_from_bytes(data)
        self.assertIsInstance(result, float)
    
    def test_batch_entropy_estimation(self):
        """Test batch_entropy_estimation function."""
        from qrng_bridge import batch_entropy_estimation
        data_list = [b'\x00\x01', b'\x02\x03', b'\x04\x05']
        results = batch_entropy_estimation(data_list)
        self.assertIsInstance(results, list)
        self.assertEqual(len(results), 3)


class TestCuQuantumAccelerator(unittest.TestCase):
    """Test cases for CuQuantum accelerator functions."""
    
    def test_process_quantum_data(self):
        """Test process_quantum_data function."""
        from cuquantum_accelerator.core import process_quantum_data
        input_data = {
            'state': np.array([0.1, 0.2, 0.3]),
            'backend': 'CUDA',
            'options': {}
        }
        result = process_quantum_data(input_data)
        self.assertIn('entropy', result)
        self.assertIn('status', result)
    
    def test_initialize_quantum_backend(self):
        """Test initialize_quantum_backend function."""
        from cuquantum_accelerator.core import initialize_quantum_backend
        backend = initialize_quantum_backend('cuda')
        self.assertIsNotNone(backend)


if __name__ == '__main__':
    unittest.main()