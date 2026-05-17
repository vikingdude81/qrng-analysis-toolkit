"""
Test runner script for entropy_estimators module.

Run with: python run_entropy_tests.py
Or use pytest: pytest tests/test_entropy_estimators.py
"""

import sys
import traceback
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))


def run_test(test_func, name):
    """Run a single test function."""
    print(f"\nRunning {name}...")
    try:
        test_func()
        print(f"  ✓ {name} passed")
        return True
    except Exception as e:
        print(f"  ✗ {name} failed: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all entropy estimator tests."""
    print("="*60)
    print("Entropy Estimators Test Suite")
    print("="*60)
    
    # Import test functions
    from test_entropy_estimators import (
        test_shannon_entropy_binary,
        test_shannon_entropy_uniform,
        test_renyi_alpha_binary,
        test_renyi_alpha_uniform,
        test_kolmogorov_sinai_placeholder,
        test_empty_input,
        test_qrng_sequence_analysis,
    )
    
    tests = [
        (test_shannon_entropy_binary, "Shannon entropy (binary)"),
        (test_shannon_entropy_uniform, "Shannon entropy (uniform)"),
        (test_renyi_alpha_binary, "Rényi entropy (binary)"),
        (test_renyi_alpha_uniform, "Rényi entropy (uniform)"),
        (test_kolmogorov_sinai_placeholder, "Kolmogorov-Sinai placeholder"),
        (test_empty_input, "Empty input handling"),
        (test_qrng_sequence_analysis, "QRNG sequence analysis"),
    ]
    
    passed = 0
    failed = 0
    
    for test_func, name in tests:
        if run_test(test_func, name):
            passed += 1
        else:
            failed += 1
    
    print("\n" + "="*60)
    print(f"Results: {passed} passed, {failed} failed")
    print("="*60)
    
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
