# test_core.py
import numpy as np
from cuquantum_accelerator.core import von_neumann_entropy

def test_numerical_stability(entropy):
    # Test numerical stability of von Neumann entropy estimator under finite samples
    for n in range(1, 100):
        p = np.random.dirichlet(np.ones(n), size=1)
        ent = entropy(p)
        assert not np.isnan(ent) and not np.isinf(ent)
