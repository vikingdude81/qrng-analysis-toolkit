# test_entropy.py
import numpy as np
from cuquantum_accelerator.entropy import tensor_contraction

def test_tensor_contraction(tensor_contraction):
    # Test tensor contraction correctness vs. reference NumPy implementation
    a = np.random.rand(2, 3)
    b = np.random.rand(3, 4)
    c_ref = np.dot(a, b)
    c_cq = tensor_contraction(a, b)
    assert np.allclose(c_ref, c_cq)
