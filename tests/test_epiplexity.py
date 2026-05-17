# tests/test_epiplexity.py

import unittest
import numpy as np
from epiplexity import epiplexity, cuquantum_fallback

class TestEpiplexity(unittest.TestCase):
    def test_epiplexity(self):
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([[5, 6], [7, 8]])
        result, ci = epiplexity(X, Y)
        self.assertIsInstance(result, float)
        self.assertIsInstance(ci, tuple)

    def test_cuquantum_fallback(self):
        X = np.array([[1, 2], [3, 4]])
        Y = np.array([[5, 6], [7, 8]])
        result = cuquantum_fallback(X, Y)
        self.assertIsInstance(result, float)

if __name__ == '__main__':
    unittest.main()
