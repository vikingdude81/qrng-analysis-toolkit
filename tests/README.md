# Tests Directory

## Overview

This directory contains the test suite for helios-trajectory-analysis.

## Test Modules

### `test_entropy_estimators.py`
Tests for the entropy_estimators module:
- Shannon entropy calculations
- Rényi entropy calculations
- Kolmogorov-Sinai entropy (placeholder)
- Edge cases (empty inputs, uniform distributions)

### `test_qrng_trace_metrics.py`
Tests for QRNG trace analysis:
- Chaos metric calculation (matches at lag 1)
- Consciousness metric (Φ) calculation
- Both Python list and numpy implementations

### `conftest.py`
Pytest configuration with fixtures:
- `random_binary_sequence`: Random binary sequence fixture
- `uniform_distribution`: Uniform distribution over values 0-3
- `deterministic_sequence`: All zeros sequence
- `alternating_sequence`: Alternating binary sequence

### `run_entropy_tests.py`
Standalone test runner script for entropy estimators.

## Running Tests

### Using pytest (recommended):
```bash
pytest tests/
pytest tests/test_entropy_estimators.py -v
pytest tests/test_qrng_trace_metrics.py -v
```

### Using standalone runner:
```bash
python tests/run_entropy_tests.py
```

### Run with coverage:
```bash
pytest tests/ --cov=entropy_estimators --cov-report=html
```

## Test Coverage Goals

- **Entropy estimators**: 90%+ coverage
- **Edge cases**: All boundary conditions tested
- **Integration tests**: QRNG sequence analysis verified

## Dependencies

Tests require:
- `pytest>=7.0.0`
- `numpy>=1.20.0`
- Optional: `scipy`, `nolds`, `pyunicorn` for advanced tests

## CI/CD Integration

Add to `.github/workflows/test.yml`:
```yaml
- name: Run tests
  run: |
    pytest tests/ -v --tb=short
```

## Contributing

When adding new tests:
1. Place in `tests/` directory
2. Use pytest fixtures from `conftest.py`
3. Follow PEP8 naming conventions
4. Include docstrings for test functions
