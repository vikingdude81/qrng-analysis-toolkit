import numpy as np
import pytest
from consciousness_metrics import calculate_entropy, calculate_complexity

def logistic_map(r=4.0, x0=0.5, n=100):
    sequence = [x0]
    for _ in range(n-1):
        next_value = r * sequence[-1] * (1 - sequence[-1])
        sequence.append(next_value)
    return sequence

def henon_map(a=1.4, b=0.3, x0=0.5, y0=0.5, n=100):
    sequence = [(x0, y0)]
    for _ in range(n-1):
        next_x = a - b * sequence[-1][1]**2 + sequence[-1][0]
        next_y = sequence[-1][0]
        sequence.append((next_x, next_y))
    return [point[0] for point in sequence]

@pytest.fixture
def tmp_logistic_map(tmp_path):
    n_samples = 50
    sequence = logistic_map(n=n_samples)
    np.savetxt(tmp_path / 'logistic_map.txt', sequence)
    return tmp_path / 'logistic_map.txt'

@pytest.fixture
def tmp_henon_map(tmp_path):
    n_samples = 50
    sequence = henon_map(n=n_samples)
    np.savetxt(tmp_path / 'henon_map.txt', sequence)
    return tmp_path / 'henon_map.txt'

def test_calculate_entropy_short_sequence(tmp_logistic_map, tmp_henon_map):
    short_seq = logistic_map(n=10)
    assert calculate_entropy(short_seq) > 0

def test_calculate_complexity_non_ergodic(tmp_logistic_map, tmp_henon_map):
    non_ergodic_seq = [0.5] * 100
    assert calculate_complexity(non_ergodic_seq) == 0

def test_calculate_entropy_high_noise(tmp_logistic_map, tmp_henon_map):
    noise = np.random.normal(0, 0.1, 100)
    noisy_seq = [x + n for x, n in zip(logistic_map(), noise)]
    assert calculate_entropy(noisy_seq) > 0

def test_calculate_complexity_high_noise(tmp_logistic_map, tmp_henon_map):
    noise = np.random.normal(0, 1.0, 100)
    noisy_seq = [x + n for x, n in zip(henon_map(), noise)]
    assert calculate_complexity(noisy_seq) > 0
