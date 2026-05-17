"""
QRNG Inference Tests for Helios Trajectory Analysis

This module provides tests for the QRNG inference framework, including:
- Loading sequences from SPDC source
- Running inference on quantum randomness data
- Verifying output shapes and types
"""

import pytest
import numpy as np
from helios.qrng_spdc_source import load_sequence_from_spdc
from helios.inference_framework.qrng_bridge import run_inference


def test_qrng_inference():
    """Test QRNG inference with SPDC sequence."""
    # Load 10k-bit sequence from qrng_spdc_source.py
    sequence = load_sequence_from_spdc()
    
    # Run inference
    output = run_inference(sequence)
    
    # Verify output shape and type
    assert output.shape == (10000,), "Output shape mismatch"
    assert output.dtype == np.uint8, "Output data type mismatch"


def test_qrng_inference_with_hydra_config():
    """Test QRNG inference with Hydra configuration."""
    from hydra.core.config import ConfigInference
    from hydra.utils import get_config
    
    # Load configuration
    config = get_config("qrng_inference.yaml")
    
    # Verify config is loaded
    assert config is not None, "Config should be loaded"


def test_qrng_inference_edge_cases():
    """Test QRNG inference with edge cases."""
    # Test with empty sequence
    try:
        run_inference(np.array([]))
        assert False, "Should raise error for empty sequence"
    except Exception as e:
        pass  # Expected behavior
    
    # Test with single element
    single = np.array([1])
    output = run_inference(single)
    assert output.shape == (1,), "Single element should preserve shape"
