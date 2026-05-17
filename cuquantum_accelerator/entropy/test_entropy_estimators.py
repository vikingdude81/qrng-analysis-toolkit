import numpy as np
import pytest
import cuquantum

def generate_data(n_samples=10000):
    return np.random.uniform(0, 1, n_samples)

@pytest.fixture
def data():
    return generate_data()

@pytest.fixture
def estimator(data):
    def estimate_entropy(data):
        # Placeholder for actual entropy estimation
        return np.log(len(data))
    return estimate_entropy

@ pytest.mark.skipif(not hasattr(cuquantum, 'cuquantum'), reason="cuquantum not installed")
def test_entropy_estimator(data, estimator):
    est = estimator(data)
    assert abs(est - np.log(10000)) <= 0.05 * np.log(10000)