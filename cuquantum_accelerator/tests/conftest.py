# conftest.py
import pytest
from memory_profiler import profile

@pytest.fixture(scope="module")
def entropy():
    from cuquantum_accelerator.core import von_neumann_entropy
    return von_neumann_entropy

@pytest.fixture(scope="module")
def tensor_contraction():
    from cuquantum_accelerator.entropy import tensor_contraction as tc
    return tc
