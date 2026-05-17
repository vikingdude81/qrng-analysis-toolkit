"""
Pytest configuration for helios-trajectory-analysis tests.

Sets up common fixtures and configurations for all test modules.
"""

import sys
from pathlib import Path

# Make the repo root and key subpackages importable as flat module names,
# matching the historical layout that many test modules still use.
_REPO_ROOT = Path(__file__).resolve().parent.parent
_FLAT_DIRS = [
    _REPO_ROOT,
    _REPO_ROOT / "metrics",
    _REPO_ROOT / "analysis",
    _REPO_ROOT / "collectors",
    _REPO_ROOT / "measures",
    _REPO_ROOT / "inference_framework",
    _REPO_ROOT / "utils",
    _REPO_ROOT / "cuquantum_accelerator",
    _REPO_ROOT / "meta_curiosity",
    _REPO_ROOT / "visualization",
]
for _p in _FLAT_DIRS:
    _s = str(_p)
    if _p.is_dir() and _s not in sys.path:
        sys.path.insert(0, _s)

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
