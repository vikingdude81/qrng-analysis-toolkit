# Quick Start Guide - Improved HELIOS Project

Welcome! This guide gets you running with all the new improvements.

## ⚡ Installation

```bash
# Clone repository
git clone <your-repo-url>
cd helios-trajectory-analysis

# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Linux/Mac)
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

## 🧪 Run Tests

```bash
# All tests
pytest

# With coverage
pytest --cov=. --cov-report=html

# Specific tests
pytest tests/test_metrics.py -v

# Fast tests only (skip slow)
pytest -m "not slow"
```

## 🚀 Run Analysis

```bash
# Basic run (2000 steps)
python run_qrng_analysis.py

# Custom configuration
python run_qrng_analysis.py --steps 5000 --ring-sections 8 --pump-power 25

# Without terminal visualization (faster)
python run_qrng_analysis.py --steps 10000 --no-viz

# Different walk modes
python run_qrng_analysis.py --walk-mode angle
python run_qrng_analysis.py --walk-mode xy_independent
```

## 📊 Check Results

Results are saved in `qrng_results/`:

- `run_YYYYMMDD_HHMMSS.json` - Full data
- `run_YYYYMMDD_HHMMSS.xlsx` - Excel spreadsheet
- `*_trajectory.png` - Phase space plot
- `*_metrics.png` - Time series metrics
- `*_randomness.png` - QRNG quality
- `*_phase_analysis.png` - Detailed analysis

## 🔍 What's New?

All improvements from the audit are implemented:

1. **Tests**: `tests/` directory with 50+ tests
2. **Validation**: Robust input checking with clear errors
3. **Logging**: `logger_config.py` for professional logging
4. **Safety**: Atomic file writes prevent corruption
5. **Dependencies**: Pinned versions for stability
6. **CI/CD**: `.github/workflows/` for automated testing
7. **Config**: `config.example.json` for customization

## 📖 Key Files

| File | Purpose |
|------|---------|
| `helios_anomaly_scope.py` | Main detection engine |
| `qrng_spdc_source.py` | Quantum RNG simulation |
| `run_qrng_analysis.py` | Analysis runner |
| `validation.py` | Input validation |
| `logger_config.py` | Logging setup |
| `file_utils.py` | Safe file operations |
| `tests/` | Test suite |

## 🐛 Troubleshooting

### Import errors
```bash
# Make sure you're in the project directory
pwd

# Check dependencies
pip list | grep numpy
```

### Tests fail
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Run with verbose output
pytest -vv
```

### Out of memory
```bash
# Reduce steps
python run_qrng_analysis.py --steps 500

# Or reduce history
python run_qrng_analysis.py --steps 1000
# (modify code: history_len=50)
```

## 📚 Documentation

- `README.md` - Project overview
- `IMPROVEMENTS.md` - Details on all improvements
- `TESTING.md` - Testing procedures
- `THEORY.md` - Theoretical background
- `CHANGELOG.md` - Version history

## 🎯 Next Steps

1. **Run verification tests**: `pytest tests/test_anomaly_scope.py::TestWarmupPeriod -v`
2. **Try example analysis**: `python examples/basic_qrng_analysis.py`
3. **Customize configuration**: Copy `config.example.json` to `config.json`
4. **Enable GitHub Actions**: Push to GitHub to trigger CI/CD

## 💡 Tips

- Use `--no-viz` for faster runs without visualization
- Check logs in `logs/` directory for debugging
- Use `pytest -k pattern` to run specific tests
- See `IMPROVEMENTS.md` for migration guide

---

**Everything is ready to go!** 🎉

Start with: `pytest && python run_qrng_analysis.py --steps 1000`
