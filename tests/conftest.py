"""
Pytest fixtures for HELIOS trajectory analysis tests.
"""

import pytest
import numpy as np
from typing import Tuple, List


@pytest.fixture
def random_seed():
    """Fixed seed for reproducible tests."""
    np.random.seed(42)
    return 42


@pytest.fixture
def pure_random_walk(random_seed) -> Tuple[List[float], List[float]]:
    """Generate a pure random walk (should show H≈0.5, α≈1, λ≈0)."""
    n_steps = 500
    x, y = [0.0], [0.0]
    for _ in range(n_steps - 1):
        angle = np.random.uniform(0, 2 * np.pi)
        step = 0.1
        x.append(x[-1] + step * np.cos(angle))
        y.append(y[-1] + step * np.sin(angle))
    return x, y


@pytest.fixture
def ballistic_trajectory() -> Tuple[List[float], List[float]]:
    """Generate ballistic motion (should show α≈2, directed drift)."""
    n_steps = 500
    velocity = (0.05, 0.03)  # Constant velocity
    noise = 0.01
    x, y = [0.0], [0.0]
    for i in range(n_steps - 1):
        x.append(x[-1] + velocity[0] + np.random.normal(0, noise))
        y.append(y[-1] + velocity[1] + np.random.normal(0, noise))
    return x, y


@pytest.fixture
def periodic_trajectory() -> Tuple[List[float], List[float]]:
    """Generate periodic/circular motion (should show periodicity)."""
    n_steps = 500
    t = np.linspace(0, 8 * np.pi, n_steps)
    noise = 0.02
    x = list(np.cos(t) + np.random.normal(0, noise, n_steps))
    y = list(np.sin(t) + np.random.normal(0, noise, n_steps))
    return x, y


@pytest.fixture
def lorenz_attractor() -> Tuple[List[float], List[float], List[float]]:
    """
    Generate Lorenz attractor trajectory (chaotic, λ > 0).
    Returns x, y, z coordinates.
    """
    # Lorenz parameters
    sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
    dt = 0.01
    n_steps = 5000
    
    x, y, z = [1.0], [1.0], [1.0]
    for _ in range(n_steps - 1):
        dx = sigma * (y[-1] - x[-1])
        dy = x[-1] * (rho - z[-1]) - y[-1]
        dz = x[-1] * y[-1] - beta * z[-1]
        x.append(x[-1] + dx * dt)
        y.append(y[-1] + dy * dt)
        z.append(z[-1] + dz * dt)
    
    # Scale to reasonable range
    x = [v / 20 for v in x]
    y = [v / 20 for v in y]
    z = [v / 20 for v in z]
    
    return x, y, z


@pytest.fixture
def attractor_collapse() -> Tuple[List[float], List[float]]:
    """
    Generate trajectory that collapses to a point attractor (λ < 0).
    Starts random, converges to origin.
    """
    n_steps = 500
    decay = 0.98
    x, y = [2.0], [2.0]
    for _ in range(n_steps - 1):
        # Decay toward origin with small noise
        x.append(x[-1] * decay + np.random.normal(0, 0.02))
        y.append(y[-1] * decay + np.random.normal(0, 0.02))
    return x, y


@pytest.fixture
def trending_series() -> List[float]:
    """Generate trending series with H > 0.5 (persistent)."""
    n = 500
    # Fractional Brownian motion approximation with H > 0.5
    increments = []
    for i in range(n):
        # Persistent: more likely to continue direction
        if i > 0 and len(increments) > 0:
            bias = 0.3 * np.sign(increments[-1])
        else:
            bias = 0
        increments.append(np.random.normal(bias, 1))
    
    series = np.cumsum(increments)
    return list(series)


@pytest.fixture
def mean_reverting_series() -> List[float]:
    """Generate mean-reverting series with H < 0.5 (anti-persistent)."""
    n = 500
    # Ornstein-Uhlenbeck process
    theta = 0.5  # Mean reversion speed
    mu = 0  # Long-term mean
    sigma = 0.3
    
    series = [0.0]
    for _ in range(n - 1):
        dx = theta * (mu - series[-1]) + sigma * np.random.normal()
        series.append(series[-1] + dx)
    
    return series


@pytest.fixture
def qrng_stream():
    """Get QRNG source for testing."""
    from qrng_spdc_source import SPDCQuantumSource
    return SPDCQuantumSource()


@pytest.fixture
def scope():
    """Get QRNGStreamScope instance."""
    from helios_anomaly_scope import QRNGStreamScope
    return QRNGStreamScope()


@pytest.fixture
def sample_trajectory():
    """Generate a sample random walk trajectory."""
    np.random.seed(42)
    n = 200
    x = np.cumsum(np.random.randn(n) * 0.1)
    y = np.cumsum(np.random.randn(n) * 0.1)
    return list(x), list(y)


@pytest.fixture
def biased_trajectory():
    """Generate a trajectory with systematic bias."""
    np.random.seed(42)
    n = 200
    t = np.arange(n)
    x = t * 0.1 + np.cumsum(np.random.randn(n) * 0.02)
    y = t * 0.15 + np.cumsum(np.random.randn(n) * 0.02)
    return list(x), list(y)


@pytest.fixture
def circular_trajectory():
    """Generate a circular trajectory."""
    t = np.linspace(0, 4 * np.pi, 200)
    x = np.cos(t) + np.random.randn(200) * 0.05
    y = np.sin(t) + np.random.randn(200) * 0.05
    return list(x), list(y)
