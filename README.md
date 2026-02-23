# QRNG Analysis Toolkit

**A comprehensive open-source toolkit for collecting, analyzing, and visualizing Quantum Random Number Generator (QRNG) data from multiple hardware sources.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

---

## What Is This?

This toolkit lets you:

- **Collect** true quantum random numbers from 5+ hardware sources (IBM Quantum, Outshift SPDC, ANU vacuum fluctuation, Cipherstone photonic, CPU RDRAND) plus software PRNG controls
- **Analyze** randomness quality using NIST SP 800-22 statistical tests, chaos theory metrics, phase space reconstruction, and information-theoretic measures
- **Visualize** distributions, phase space trajectories, recurrence plots, spectral density, temporal drift, and cross-source comparisons
- **Compare** quantum vs classical randomness with rigorous statistical methods

**Included: 509,332 pre-collected samples** from 8 different sources (4 quantum hardware, 2 classical controls, 2 experimental) ready for immediate analysis.

---

## Quick Start

### 1. Clone and install

```bash
git clone https://github.com/vikingdude81/qrng-analysis-toolkit.git
cd qrng-analysis-toolkit
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt
```

### 2. Analyze the included data (no API keys needed)

```bash
# Run quality analysis on all 509K pre-collected samples
python analysis/analyze_qrng_quality.py

# Run the 7-test NIST battery on quantum hardware keys
python analysis/run_qhw_test_battery.py

# Generate phase space visualizations
python visualization/visualize_qhw_keys.py
```

### 3. Collect fresh data (API keys required)

```bash
# Copy the template and add your keys
cp .env.example .env
# Edit .env with your API keys

# Collect from all available sources
python collectors/collect_all_sources.py -n 1000
```

---

## Data Sources

| Source | Type | Mechanism | Samples Included | API Key? |
|--------|------|-----------|------------------|----------|
| **Quantum Hardware 256-bit Keys** | Quantum | Physical quantum device | 386,232 | Pre-collected |
| **Outshift QRNG** | Quantum | SPDC photon pair detection | 26,100 | Required |
| **ANU QRNG** | Quantum | Vacuum fluctuation shot noise | 26,000 | Required |
| **IBM Quantum** | Quantum | Superconducting transmon qubit collapse | 15,000 | Required |
| **Cipherstone Qbert** | Quantum | Photonic chip | 7,000 | Required |
| **CPU RDRAND** | Classical | Thermal noise (hardware RNG) | 22,000 | None |
| **PRNG (Mersenne Twister)** | Classical | Deterministic algorithm | 22,000 | None |
| **Cipherstone M2 (Raw)** | Experimental | Unconditioned photonic | 5,000 | Required |

### Getting API Keys

- **Outshift QRNG**: Sign up at [random.outshift.com](https://random.outshift.com/)
- **ANU QRNG**: Register at [quantumnumbers.anu.edu.au](https://quantumnumbers.anu.edu.au/)
- **IBM Quantum**: Create account at [quantum.ibm.com](https://quantum.ibm.com/), then save credentials:
  ```python
  from qiskit_ibm_runtime import QiskitRuntimeService
  QiskitRuntimeService.save_account(token="YOUR_TOKEN", instance="YOUR_CRN")
  ```
- **Cipherstone**: Contact [cipherstone.com](https://cipherstone.com/)

---

## Repository Structure

```
qrng-analysis-toolkit/
├── collectors/               # Data collection from QRNG sources
│   ├── collect_all_sources.py       # Collect from ALL sources at once
│   ├── qrng_outshift_client.py      # Outshift SPDC API client
│   ├── ibm_quantum_qrng.py          # IBM Quantum hardware collector
│   ├── cpu_hwrng.py                 # CPU hardware RNG (RDRAND/BCrypt)
│   ├── qrng_spdc_source.py          # SPDC quantum simulation source
│   ├── qrng_daily_collector.py      # Automated daily collection
│   ├── collect_cipherstone_stream.py # Cipherstone Qbert collector
│   ├── import_qrng_keys.py          # Import base64-encoded QHW keys
│   └── test_qrng_live.py            # Quick API smoke test
│
├── analysis/                 # Statistical analysis scripts
│   ├── analyze_qrng_quality.py      # Full quality analysis (Chi-sq, K-S, entropy)
│   ├── run_qhw_test_battery.py      # 7-test NIST battery (monobit, runs, etc.)
│   ├── analyze_bientropy.py         # BiEntropy analysis (Croll 2013)
│   ├── comprehensive_qrng_analysis.py # All metrics on saved streams
│   ├── chaos_analysis.py            # Chaos theory: Lyapunov, Hurst, RQA
│   ├── deep_pattern_analysis.py     # Emergent patterns, cross-source FFT
│   ├── influence_detection.py       # External influence & burst detection
│   ├── analyze_source_stability.py  # Source stability comparison
│   ├── compare_qrng_sources.py      # A/B source comparison
│   ├── qrng_deep_dive.py            # Hurst stability, rolling windows
│   ├── qrng_deep_dive_v2.py         # V2 with proper methodology
│   └── analyze_inference_statistics.py # QRNG vs PRNG inference stats
│
├── metrics/                  # Core computation modules
│   ├── helios_anomaly_scope.py      # Phase space, MSD, coherence, influence
│   ├── bientropy_metrics.py         # Binary derivative entropy
│   ├── chaos_detector.py            # Lyapunov, criticality, bifurcation
│   ├── consciousness_metrics.py     # Mode entropy, coherence, criticality
│   ├── epiplexity_estimator.py      # Structural info estimation
│   └── validation.py                # Input validation utilities
│
├── visualization/            # Plotting and dashboard scripts
│   ├── visualize_qhw_keys.py       # 10-panel QHW key analysis
│   ├── visualize_qrng_data.py      # Distribution + autocorrelation dashboard
│   ├── phase_space_visualizer.py    # Animated phase space trajectories
│   ├── qrng_dashboard.py           # Full multi-source dashboard
│   ├── generate_full_report.py     # PDF-ready visual report
│   └── visualize_inference_results.py # Inference experiment plots
│
├── inference_framework/      # QRNG vs PRNG inference experiments
│   ├── architectures.py            # Strange Attractor, Code Duality, Tensegrity
│   ├── classifier.py               # Reasoning mode classifier
│   ├── experiment.py               # Experimental framework
│   └── qrng_bridge.py              # QRNG → inference provider bridge
│
├── cuquantum_accelerator/    # GPU-accelerated analysis (optional)
│   ├── core.py                     # GPU init and management
│   ├── entropy.py                  # GPU Shannon, BiEntropy, sample entropy
│   ├── tensor_analysis.py          # GPU Lyapunov, correlation dimension
│   ├── quantum_simulator.py        # Quantum state simulation
│   └── benchmarks.py               # CPU vs GPU benchmarks
│
├── tests/                    # Test suite
│   ├── nist_tests.py               # NIST SP 800-22 full 15-test suite
│   ├── test_metrics.py             # Unit tests for core math
│   ├── test_anomaly_scope.py       # Integration tests
│   ├── test_chaos_detector.py      # Chaos metric tests
│   └── ...                         # 15 test files total
│
├── examples/                 # Getting started examples
│   ├── basic_qrng_analysis.py      # Basic analysis walkthrough
│   └── helios_integration.py       # Neural network integration
│
├── data/                     # Pre-collected data (509K+ samples)
│   ├── qrng_streams/               # 471 JSON stream files
│   ├── analysis_results/           # Past analysis JSON outputs
│   ├── visualizations/             # Generated plot PNGs
│   └── inference_results/          # QRNG vs PRNG experiment results
│
├── docs/                     # Documentation
│   ├── THEORY.md                   # Mathematical foundations
│   ├── THRESHOLDS.md               # Test threshold reference
│   ├── GLOSSARY.md                 # Term definitions
│   ├── RESEARCH_PROTOCOL.md        # Experimental methodology
│   └── ...
│
├── utils/                    # Shared utilities
│   ├── file_utils.py               # Atomic JSON I/O
│   ├── logger_config.py            # Logging setup
│   └── data_stream_loader.py       # Generic data loading
│
├── .env.example              # API key template
├── config.example.json       # Configuration template
├── requirements.txt          # Python dependencies
├── pyproject.toml            # Project metadata
└── pytest.ini                # Test configuration
```

---

## Analysis Capabilities

### Statistical Tests

| Test | What It Checks | Implementation |
|------|---------------|----------------|
| **Monobit Frequency** | Equal count of 0s and 1s | NIST SP 800-22 §2.1 |
| **Block Frequency** | Uniform 1s density across blocks | NIST SP 800-22 §2.2 |
| **Runs Test** | No unexpected bit streaks | NIST SP 800-22 §2.3 |
| **Chi-Square (Bytes)** | All 256 byte values equally frequent | Pearson χ² |
| **Shannon Entropy** | Information density per byte (ideal: 8.0) | Shannon 1948 |
| **Monte Carlo Pi** | Geometric randomness via π estimation | Hit-or-miss |
| **Compression Ratio** | Incompressibility (Kolmogorov proxy) | zlib level 9 |
| **Kolmogorov-Smirnov** | Distribution matches Uniform[0,1] | K-S test |
| **Serial Correlation** | No autocorrelation at any lag | ACF analysis |
| **BiEntropy** | Binary derivative entropy | Croll 2013 |
| **Full NIST SP 800-22** | All 15 NIST randomness tests | Custom implementation |

### Chaos Theory & Phase Space

| Metric | What It Measures | Ideal Value |
|--------|-----------------|-------------|
| **Hurst Exponent** | Long-range memory via R/S analysis | 0.50 (memoryless) |
| **Lyapunov Exponent** | Sensitivity to initial conditions | < 0 (stable) |
| **Correlation Dimension** | Attractor complexity | High (no structure) |
| **Recurrence Quantification** | Deterministic patterns in phase space | Low recurrence rate |
| **Spectral Entropy** | Frequency distribution flatness | ~1.0 (white noise) |

### Information Theory

| Metric | Description |
|--------|-------------|
| **Shannon Entropy** | Average information per symbol |
| **Min-Entropy** | Worst-case unpredictability |
| **BiEntropy (TBiEn)** | Weighted entropy of binary derivatives |
| **Epiplexity** | Structural information extractable by bounded observer |
| **Spectral Entropy** | Normalized entropy of power spectrum |

---

## Pre-Collected Results

### Quality Analysis Summary (509,332 samples)

| Source | Tests | Tier | Rating |
|--------|-------|------|--------|
| Outshift SPDC | 5/5 | PRODUCTION | EXCELLENT |
| ANU Vacuum | 5/5 | PRODUCTION | EXCELLENT |
| CPU RDRAND | 5/5 | CONTROL | EXCELLENT |
| PRNG (MT) | 5/5 | CONTROL | EXCELLENT |
| Cipherstone M1 | 5/5 | PRODUCTION | EXCELLENT |
| Quantum Hardware 256bit | 4/5 | — | GOOD |
| IBM Quantum | 3/5 | — | MARGINAL |
| Cipherstone M2 (Raw) | 0/5 | EXPERIMENTAL | FAILING* |

*\*Cipherstone M2 is unconditioned raw output — failure is expected and informative.*

### NIST Battery Results (Quantum Hardware 256-bit Keys)

| Test | Result | Verdict |
|------|--------|---------|
| Monobit Frequency | P = 0.3200 | PASS |
| Block Frequency | P = 0.8583 | PASS |
| Runs Test | P = 0.3782 | PASS |
| Chi-Square (Bytes) | P = 0.9876 | PASS |
| Shannon Entropy | 7.9996 bits | PASS |
| Monte Carlo Pi | 3.1420 (0.0% error) | PASS |
| Compression Ratio | 1.0003 | PASS |

### Key Findings

- **Hurst Exponent**: 0.5268 (full 386K dataset) — near-ideal 0.50, no long-range memory
- **Autocorrelation**: All lags < 0.003 (within 95% CI)
- **Phase Space**: Uniform fill, no attractors, bands, or lattice structure
- **Shannon Entropy**: 7.9996 / 8.0 bits per byte (99.995% efficiency)
- **Spectral Density**: Flat (white noise), no periodic components

---

## Adding Your Own Source

To add a new QRNG source:

### 1. Create a collector

Create a file in `collectors/` that outputs the standard stream format:

```python
# collectors/my_new_source.py
import json
from datetime import datetime
from pathlib import Path

def collect_my_source(count: int, output_dir: Path):
    """Collect from your custom QRNG source."""
    
    # ... your collection logic here ...
    values = [0.123, 0.456, ...]  # list of floats in [0, 1)
    
    stream = {
        "source": "my_custom_source",
        "timestamp": datetime.now().isoformat(),
        "count": len(values),
        "floats": values,
        "stats": {
            "mean": sum(values) / len(values),
            # ... other stats
        }
    }
    
    filepath = output_dir / f"my_source_{datetime.now():%Y%m%d_%H%M%S}.json"
    filepath.write_text(json.dumps(stream, indent=2))
    return stream
```

### 2. Register it in `collect_all_sources.py`

Add your collector function to `collectors/collect_all_sources.py`:

```python
def collect_my_source(count, output_dir):
    from my_new_source import collect_my_source as _collect
    result = _collect(count, output_dir)
    console.print(f"  [green]✓ {result['count']} samples[/]")
    return result
```

### 3. Run analysis

All analysis scripts will automatically pick up your new source from `data/qrng_streams/`:

```bash
python analysis/analyze_qrng_quality.py
python visualization/visualize_qrng_data.py
```

### Stream JSON Format

Every source must output JSON files with this structure:

```json
{
  "source": "source_identifier_string",
  "timestamp": "2026-02-23T12:30:00",
  "count": 1000,
  "floats": [0.1234, 0.5678, ...],
  "stats": {
    "mean": 0.4998,
    "std": 0.2887,
    "min": 0.0001,
    "max": 0.9999
  }
}
```

---

## Importing External Random Data

### From CSV of base64-encoded keys

```bash
# Set the path to your key file
set QRNG_KEYS_CSV=path/to/your/saved_keys.csv
python collectors/import_qrng_keys.py
```

### From raw binary files

```python
from utils.data_stream_loader import DataStreamLoader

loader = DataStreamLoader()
floats = loader.load_binary("my_random_bytes.bin", dtype="uint32")
```

### From a URL / live API

See `collectors/qrng_outshift_client.py` as a template for building API clients.

---

## Running Tests

```bash
# Run unit tests
pytest tests/test_metrics.py -v

# Run NIST tests on generated data
pytest tests/test_nist.py -v

# Run full integration tests (needs stream data in data/qrng_streams/)
pytest tests/test_qrng_integration.py -v

# Run all tests
pytest tests/ -v
```

---

## GPU Acceleration (Optional)

If you have an NVIDIA GPU with CUDA:

```bash
pip install torch  # with CUDA support
python -c "from cuquantum_accelerator import CuQuantumCore; print(CuQuantumCore().get_info())"
```

The `cuquantum_accelerator/` module provides GPU-accelerated versions of:
- Shannon entropy, BiEntropy, sample entropy, permutation entropy
- Correlation dimension, Lyapunov exponent
- Recurrence quantification analysis
- Multi-scale entropy

Typical speedup: 10-50x on large datasets.

---

## Detailed Documentation

| Document | Description |
|----------|-------------|
| [docs/THEORY.md](docs/THEORY.md) | Mathematical foundations for all metrics |
| [docs/THRESHOLDS.md](docs/THRESHOLDS.md) | Test threshold values and their justification |
| [docs/GLOSSARY.md](docs/GLOSSARY.md) | Definitions of all terms used |
| [docs/RESEARCH_PROTOCOL.md](docs/RESEARCH_PROTOCOL.md) | Experimental methodology and protocols |
| [docs/TESTING.md](docs/TESTING.md) | Testing strategy and coverage |
| [docs/SCRIPTS.md](docs/SCRIPTS.md) | Script-by-script reference |
| [docs/NEW_PAPERS_ANALYSIS.md](docs/NEW_PAPERS_ANALYSIS.md) | Recent research papers and their relevance |

---

## Contributing

1. Fork this repo
2. Add your QRNG source or analysis method
3. Include test coverage in `tests/`
4. Submit a PR with:
   - Description of the source/method
   - Sample data (~1000 values minimum)
   - Quality analysis results showing the output

**Never commit API keys** — use `.env` for all secrets.

---

## References

- NIST SP 800-22: [A Statistical Test Suite for Random and Pseudorandom Number Generators](https://csrc.nist.gov/publications/detail/sp/800-22/rev-1a/final)
- Croll, G.J. (2013): [BiEntropy — The Approximate Entropy of a Finite Binary String](https://arxiv.org/abs/1305.0954)
- Hurst, H.E. (1951): Long-term storage capacity of reservoirs
- Shannon, C.E. (1948): A Mathematical Theory of Communication
- Wolf, A. et al. (1985): Determining Lyapunov exponents from a time series

---

## Citing This Work

If you use this toolkit in your research or projects, please cite it:

```bibtex
@software{vikingdude81_qrng_toolkit_2026,
  author       = {vikingdude81},
  title        = {QRNG Analysis Toolkit},
  year         = {2026},
  url          = {https://github.com/vikingdude81/qrng-analysis-toolkit},
  version      = {1.0.0},
  license      = {MIT}
}
```

Or in prose:

> vikingdude81 (2026). *QRNG Analysis Toolkit* (v1.0.0). GitHub. https://github.com/vikingdude81/qrng-analysis-toolkit

GitHub also provides a built-in **"Cite this repository"** button in the sidebar (powered by [CITATION.cff](CITATION.cff)).

---

## License

MIT License — see [LICENSE](LICENSE) for details.
