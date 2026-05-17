import pytest
from hypothesis import given, settings
import numpy as np

def generate_logistic_data(params):
    """Generate logistic map data for testing."""
    r = params["r"]
    x0 = params["x0"]
    
    # Generate 100 iterations of logistic map
    x = [x0]
    for _ in range(99):
        x_next = r * x[-1] * (1 - x[-1])
        x.append(x_next)
    
    return x

# Assume epiplexity_estimator has functions: estimate_epiplexity, get_threshold
# Hypothesis is used for data generation

@pytest.fixture
def bernoulli_data():
    return [0.5 * (1 - 2 * x) for x in range(100)]  # i.i.d. Bernoulli(0.5)

@pytest.fixture
def logistic_map_params():
    return {"r": 3.5, "x0": 0.5}

@pytest.fixture
def window_sizes():
    return [1, 2, 3, 4, 5]

@given(bernoulli_data)
def test_epiplexity_bernoulli(bernoulli_data):
    result = estimate_epiplexity(bernoulli_data)
    assert abs(result) < 1e-6

@settings(max_examples=1000)
@given(logistic_map_params)
def test_epiplexity_logistic(logistic_map_params):
    data = generate_logistic_data(logistic_map_params)
    result = estimate_epiplexity(data)
    assert result > 0.1  # threshold

@settings(max_examples=1000)
def test_gradient_smoothness(window_sizes):
    values = [estimate_epiplexity(size) for size in window_sizes]
    delta = 1e-6
    for i in range(1, len(values)):
        d = (values[i] - values[i-1]) / delta
        assert abs(d) < 1e-3  # smoothness check
