# test_module1.py

import unittest

class TestModule1(unittest.TestCase):
    def test_basic(self):
        self.assertEqual(1, 2)  # This will fail intentionally for testing

if __name__ == '__main__':
    unittest.main()
