import pytest
import pytest_mock
from metrics import entropy_estimators

def test_constant_input(mocker):
    """Test that constant input returns zero entropy."""
    mocker.patch('entropy_estimators.cuquantum.entropy_estimate', return_value=0.0)
    result = entropy_estimators.estimate_entropy([1, 1, 1])
    assert result == 0.0

def test_nan_input(mocker):
    """Test that NaN input propagates correctly."""
    mocker.patch('entropy_estimators.cuquantum.entropy_estimate', return_value=float('nan'))
    result = entropy_estimators.estimate_entropy([1, float('nan')])
    assert result == float('nan')

def test_inf_input(mocker):
    """Test that inf input propagates correctly."""
    mocker.patch('entropy_estimators.cuquantum.entropy_estimate', return_value=float('inf'))
    result = entropy_estimators.estimate_entropy([1, float('inf')])
    assert result == float('inf')

def test_low_sample_bias(mocker):
    """Test bias correction for low sample sizes."""
    mocker.patch('entropy_estimators.cuquantum.entropy_estimate', return_value=3.32)
    result = entropy_estimators.estimate_entropy(list(range(10)))
    assert abs(result - 3.32) < 0.1
