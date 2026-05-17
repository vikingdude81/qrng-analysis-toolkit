"""
Pytest configuration for helios-trajectory-analysis tests.

Sets up common fixtures and configurations for all test modules.
"""

import pytest
import numpy as np


def pytest_configure(config):
    """Configure pytest with additional markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m ""not slow""')"
    )
    config.addinivalue_line(
        "markers", "integration: marks integration tests"
    )


def pytest_collection(session):
    """Customize test collection."""
    session.config.option.verbose = 2


@pytest.fixture
def random_binary_sequence():
    """Generate a random binary sequence for testing."""
    np.random.seed(42)
    return np.random.randint(0, 2, size=1000)


@pytest.fixture
def uniform_distribution():
    """Generate a uniform distribution over values 0-3."""
    return [0, 1, 2, 3] * 100


@pytest.fixture
def deterministic_sequence():
    """Generate a deterministic (all zeros) sequence."""
    return np.zeros(1000, dtype=int)


@pytest.fixture
def alternating_sequence():
    """Generate an alternating binary sequence."""
    return [0, 1] * 500
