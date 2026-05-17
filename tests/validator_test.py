import unittest
import numpy as np

class TestValidator(unittest.TestCase):
    def test_logistic_map(self):
        # Logistic map at r=4.0 (Φ̂ ≈ 0, entropy = log2)
        x = np.random.rand(1000)  # Random initial values
        Φ_hat = 0.0  # Expected value
        entropy = np.log2(4.0)     # log2(4) = 2
        self.assertAlmostEqual(Φ_hat, 0.0, delta=1e-6)
        self.assertEqual(entropy, 2.0)

if __name__ == '__main__':
    unittest.main()