import unittest
import numpy as np

def entropy(p):
    return -np.sum(p * np.log2(p))

class TestEntropy(unittest.TestCase):
    def test_uniform(self):
        p = np.array([1/3, 1/3, 1/3])
        self.assertAlmostEqual(entropy(p), 1.584962500721156)

    def test_bimodal(self):
        p = np.array([0.4, 0.3, 0.3])
        self.assertAlmostEqual(entropy(p), 1.847297866485635)

    def test_sparse(self):
        p = np.zeros(1000) + 1e-3
        self.assertAlmostEqual(entropy(p), -np.sum(np.log2(p) * p))

if __name__ == '__main__':
    unittest.main()