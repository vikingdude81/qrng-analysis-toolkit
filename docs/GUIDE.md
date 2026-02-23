# QRNG Analysis Toolkit — Reference Guide

A complete how-to reference for every tool in this toolkit, with worked examples and interpretation guidance.

---

## Table of Contents

1. [Setup & Configuration](#setup--configuration)
2. [Data Collection](#data-collection)
3. [Analysis Scripts](#analysis-scripts)
4. [Visualization Tools](#visualization-tools)
5. [Test Battery](#test-battery)
6. [Inference Framework](#inference-framework)
7. [Interpreting Results](#interpreting-results)
8. [Troubleshooting](#troubleshooting)

---

## Setup & Configuration

### Python Environment

Requires Python 3.10+. CUDA-capable GPU optional (for `cuquantum_accelerator/`).

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# Linux/Mac
source .venv/bin/activate
pip install -r requirements.txt
```

### API Key Configuration

Copy `.env.example` to `.env` and fill in your keys:

```bash
cp .env.example .env
```

| Variable | Service | Where to Get It |
|----------|---------|----------------|
| `OUTSHIFT_API_KEY` | Outshift SPDC QRNG | [random.outshift.com](https://random.outshift.com/) |
| `ANU_API_KEY` | ANU Vacuum QRNG | [quantumnumbers.anu.edu.au](https://quantumnumbers.anu.edu.au/) |
| `IBM_QUANTUM_TOKEN` | IBM Quantum | [quantum.ibm.com](https://quantum.ibm.com/) |
| `IBM_QUANTUM_INSTANCE` | IBM Quantum CRN | Listed on your IBM Quantum dashboard |
| `CIPHERSTONE_API_KEY` | Cipherstone Qbert | [cipherstone.com](https://cipherstone.com/) |

### IBM Quantum Credential Setup

IBM Quantum credentials are saved locally via qiskit:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

# First time only — saves to ~/.qiskit/qiskit-ibm.json
QiskitRuntimeService.save_account(
    channel="ibm_quantum",
    token="YOUR_IBM_QUANTUM_TOKEN",
    instance="crn:v1:bluemix:public:quantum-computing:us-east:a/XXXX:YYYY::"
)

# Verify
service = QiskitRuntimeService()
print("Available backends:", [b.name for b in service.backends()])
```

### Configuration File

`config.example.json` provides runtime settings. Copy to `config.json` and customize:

```json
{
  "output_dir": "data/qrng_streams",
  "default_count": 1000,
  "sources": {
    "outshift": {"enabled": true, "priority": 1},
    "anu": {"enabled": true, "priority": 2},
    "ibm_quantum": {"enabled": true, "priority": 3},
    "cipherstone": {"enabled": false, "priority": 4},
    "cpu_hwrng": {"enabled": true, "priority": 5},
    "prng": {"enabled": true, "priority": 6}
  }
}
```

---

## Data Collection

### `collectors/collect_all_sources.py` — Multi-Source Collector

The primary collection script. Iterates through all configured sources and collects quantum random numbers.

```bash
# Collect 1000 samples from each source
python collectors/collect_all_sources.py -n 1000

# Collect 10000 samples (may hit rate limits on some APIs)
python collectors/collect_all_sources.py -n 10000
```

**Output**: JSON files in `data/qrng_streams/`, one per source per run.

**Rate Limits**:
- Outshift: ~2,100 per batch before daily cap
- ANU: 1,024 per request, no hard daily limit
- IBM Quantum: 8,192 shots per circuit execution
- CPU/PRNG: Unlimited

### `collectors/ibm_quantum_qrng.py` — IBM Quantum Hardware

Collects random numbers from real IBM superconducting quantum processors.

**Mechanism**: Creates a circuit with N Hadamard gates, measures all qubits. Each shot produces N random bits from genuine wavefunction collapse on a physical qubit.

```python
from collectors.ibm_quantum_qrng import collect_ibm_quantum, collect_ibm_quantum_batch

# Single batch (up to 8192 shots)
result = collect_ibm_quantum(count=1000)

# Multi-batch for larger collections
result = collect_ibm_quantum_batch(count=50000, batch_size=8000)
```

**Available Backends** (as of Feb 2026):
| Backend | Qubits | Queue | Notes |
|---------|--------|-------|-------|
| ibm_fez | 156 | Short | Best availability |
| ibm_marrakesh | 156 | Medium | Good alternative |
| ibm_torino | 133 | Variable | Eagle r3 processor |

### `collectors/qrng_outshift_client.py` — Outshift SPDC

Collects from Outshift's Spontaneous Parametric Down-Conversion quantum source.

```python
from collectors.qrng_outshift_client import OutshiftQRNGClient

client = OutshiftQRNGClient(api_key="your_key")
numbers = client.get_random_floats(count=1000)
```

### `collectors/cpu_hwrng.py` — CPU Hardware RNG

Collects from the CPU's built-in hardware random number generator (Intel RDRAND, or OS cryptographic API).

```python
from collectors.cpu_hwrng import collect_cpu_hwrng

result = collect_cpu_hwrng(count=10000)
# Uses RDRAND on Intel/AMD, BCryptGenRandom on Windows, /dev/urandom on Linux
```

**No API key required.** This is the classical control source.

### `collectors/import_qrng_keys.py` — Import Existing Keys

Import base64-encoded quantum random keys from a CSV file:

```bash
# Set environment variable to your CSV file path
set QRNG_KEYS_CSV=C:\path\to\your\saved_keys.csv
python collectors/import_qrng_keys.py
```

CSV format expected: one base64-encoded key per row.

---

## Analysis Scripts

### `analysis/analyze_qrng_quality.py` — Quality Analysis

The main quality assessment script. Runs 5 core tests on every source:

1. **Chi-Square Goodness of Fit** — byte frequency uniformity
2. **Kolmogorov-Smirnov** — distribution match to Uniform[0,1]
3. **Serial Correlation** — autocorrelation at lag 1
4. **Shannon Entropy** — information content per byte
5. **Runs Test** — sequential independence

```bash
python analysis/analyze_qrng_quality.py
```

**Output**: JSON to `data/analysis_results/` plus console summary table.

**Interpreting Results**:
- **PASS** threshold: p-value > 0.01 for statistical tests
- **Shannon Entropy**: > 7.99 bits/byte = excellent, > 7.95 = good
- **Serial Correlation**: < 0.01 = no detectable correlation

### `analysis/run_qhw_test_battery.py` — 7-Test NIST Battery

Focused battery for quantum hardware keys. Runs all 7 tests:

```bash
python analysis/run_qhw_test_battery.py
```

Each test produces a p-value (or equivalent metric). All p-values > 0.01 = PASS.

### `analysis/analyze_bientropy.py` — Binary Entropy Analysis

BiEntropy measures the entropy of a binary string via its successive derivatives. Based on Croll (2013).

```bash
python analysis/analyze_bientropy.py
```

**Key Metric**: TBiEn (Total BiEntropy)
- 1.0 = maximum complexity (truly random)
- 0.0 = completely ordered

### `analysis/chaos_analysis.py` — Chaos Theory Metrics

Phase space reconstruction and dynamical systems analysis:

```bash
python analysis/chaos_analysis.py
```

**Metrics computed**:

| Metric | What It Means | Good QRNG Value |
|--------|--------------|-----------------|
| Hurst Exponent (H) | Long-range memory. H=0.5 = memoryless, H>0.5 = persistent, H<0.5 = anti-persistent | 0.48–0.52 |
| Lyapunov Exponent (λ) | Exponential divergence rate. λ>0 = chaotic, λ<0 = convergent | Small positive |
| Correlation Dimension | Fractal dimension of phase space attractor | High (no low-D attractor) |
| Recurrence Rate | Fraction of recurrent points in phase space | Low, ~1/N |
| Determinism (DET) | Fraction of recurrent points forming diagonal lines | Low |
| Laminarity (LAM) | Fraction of recurrent points forming vertical lines | Low |

### `analysis/deep_pattern_analysis.py` — Emergent Pattern Detection

Scans for hidden structure across time scales:

```bash
python analysis/deep_pattern_analysis.py
```

Looks for:
- Spectral peaks above noise floor
- Temporal clustering of anomalies
- Cross-source correlations
- Non-stationarity

### `analysis/influence_detection.py` — External Influence Detection

Tests whether external factors (time of day, solar activity, device temperature) correlate with QRNG output:

```bash
python analysis/influence_detection.py
```

### `analysis/compare_qrng_sources.py` — Source Comparison

Statistical comparison between any two sources:

```bash
python analysis/compare_qrng_sources.py
```

Uses Kolmogorov-Smirnov 2-sample test, Mann-Whitney U, and distribution overlap metrics.

---

## Visualization Tools

### `visualization/visualize_qhw_keys.py` — 10-Panel Analysis

Generates 10 diagnostic plots for quantum hardware key data:

```bash
python visualization/visualize_qhw_keys.py
```

**Plots generated**:

| # | Plot | What It Shows |
|---|------|--------------|
| 1 | Distribution Histogram | Frequency distribution vs. ideal uniform |
| 2 | 2D Phase Space (Scatter) | x(n) vs x(n+1) lag plot |
| 3 | 2D Phase Space (Heatmap) | Density in phase space |
| 4 | 2D Phase Space (Lag-2) | x(n) vs x(n+2) |
| 5 | 3D Phase Space Trajectory | [x(n), x(n+1), x(n+2)] embedding |
| 6 | Recurrence Plot | Distance matrix in phase space |
| 7 | Hurst R/S Scaling | Log-log R/S analysis with H estimate |
| 8 | Autocorrelation Function | Correlation at lags 1–100 |
| 9 | Power Spectral Density | FFT frequency spectrum |
| 10 | Source Comparison | Side-by-side source statistics |

Output: PNG files in `data/visualizations/`.

### `visualization/visualize_qrng_data.py` — Quick Dashboard

Fast overview with distribution + autocorrelation for all sources:

```bash
python visualization/visualize_qrng_data.py
```

### `visualization/phase_space_visualizer.py` — Animated Trajectories

Creates animated 3D phase space trajectories:

```python
from visualization.phase_space_visualizer import PhaseSpaceVisualizer

viz = PhaseSpaceVisualizer()
viz.animate_trajectory(data, output="trajectory.mp4")
```

### `visualization/qrng_dashboard.py` — Full Dashboard

Multi-panel dashboard with all sources overlaid:

```bash
python visualization/qrng_dashboard.py
```

---

## Test Battery

### `tests/nist_tests.py` — Full NIST SP 800-22 Suite

Implementation of all 15 NIST statistical tests:

1. Frequency (Monobit)
2. Block Frequency
3. Runs
4. Longest Run of Ones
5. Binary Matrix Rank
6. Discrete Fourier Transform
7. Non-overlapping Template Matching
8. Overlapping Template Matching
9. Maurer's Universal Statistical
10. Linear Complexity
11. Serial
12. Approximate Entropy
13. Cumulative Sums
14. Random Excursions
15. Random Excursions Variant

```bash
pytest tests/nist_tests.py -v
```

### Running Custom Data Through the Battery

```python
import numpy as np
from tests.nist_tests import NISTTests

# Your data as numpy array of floats in [0, 1)
data = np.array([0.123, 0.456, ...])

# Convert to bits
bits = (data * 256).astype(np.uint8)
bitstring = ''.join(format(b, '08b') for b in bits)

# Run tests
nist = NISTTests(bitstring)
results = nist.run_all()

for name, pvalue, passed in results:
    print(f"{name}: P={pvalue:.4f} {'PASS' if passed else 'FAIL'}")
```

---

## Inference Framework

The inference framework tests whether LLM reasoning quality differs when seeded with quantum random numbers vs classical PRNG.

### Architecture Overview

Three experimental architectures in `inference_framework/architectures.py`:

1. **Strange Attractor Architecture** — Uses phase space structure of QRNG streams to modulate attention
2. **Code Duality Architecture** — Dual-path processing with quantum/classical channels
3. **Tensegrity Architecture** — Tension-compression network structure

### Running an Experiment

```bash
python run_qrng_inference_pilot.py
```

Or programmatically:

```python
from inference_framework.experiment import QRNGInferenceExperiment

exp = QRNGInferenceExperiment(
    architecture="strange_attractor",
    num_trials=100,
    qrng_source="outshift"
)
results = exp.run()
print(f"QRNG accuracy: {results['qrng_accuracy']:.3f}")
print(f"PRNG accuracy: {results['prng_accuracy']:.3f}")
print(f"P-value: {results['comparison_pvalue']:.4f}")
```

### Included Experiment Results

8 pilot experiments are included in `data/inference_results/`. Summary statistics in `data/inference_results/statistical_analysis.json`.

---

## Interpreting Results

### What Makes Good QRNG Data?

| Property | Ideal Value | Why |
|----------|------------|-----|
| Mean | 0.500 | Symmetric distribution |
| Std Dev | 0.2887 | = 1/√12 for uniform [0,1) |
| Shannon Entropy | 8.000 bits/byte | Maximum information density |
| Hurst Exponent | 0.500 | No long-range memory |
| Chi-Square P-value | > 0.01 | Cannot reject uniformity |
| K-S P-value | > 0.01 | Cannot reject Uniform[0,1] |
| Serial Correlation | < 0.01 | No autocorrelation |
| Compression Ratio | ≥ 1.0 | Incompressible |
| Runs P-value | > 0.01 | No sequential patterns |
| BiEntropy (TBiEn) | ~1.0 | High derivative complexity |

### Red Flags

- **P-value < 0.001**: Strong evidence of non-randomness
- **Hurst > 0.55 or < 0.45**: Long-range correlations or anti-correlations
- **Compression ratio < 0.95**: Data has exploitable structure
- **Autocorrelation spikes**: Periodic patterns in the source
- **Phase space bands**: Quantization artifacts or limited resolution
- **Spectral peaks**: Periodic contamination (check for 50/60 Hz power line)

### Quantum vs Classical: What to Look For

True quantum random numbers and well-seeded classical PRNGs both pass standard statistical tests. The differences are:

1. **Algorithmic origin**: PRNG output is deterministic given the seed; QRNG is fundamentally non-deterministic
2. **Predictability**: PRNG state can theoretically be reconstructed; quantum measurement outcomes cannot
3. **Long sequences**: PRNGs have periods (Mersenne Twister: 2^19937-1); QRNG does not
4. **Device artifacts**: QRNG may show slight biases from hardware imperfections; PRNG is mathematically clean

The analysis tools help verify your quantum source is performing properly and doesn't have device-specific biases that would compromise its quantum advantage.

---

## Troubleshooting

### Common Issues

**"No module named 'qiskit'"**
- Ensure your venv is activated: `.venv\Scripts\activate` (Windows)
- Install: `pip install qiskit qiskit-ibm-runtime`

**"403 Forbidden" from Cipherstone**
- API key may be expired. Contact Cipherstone for a new key.

**"Rate limit exceeded" from Outshift**
- Daily limit is ~2,100 samples. Wait 24 hours or reduce `-n` count.

**"No backends available" from IBM Quantum**
- Verify credentials: `QiskitRuntimeService().backends()`
- Check [quantum.ibm.com/services](https://quantum.ibm.com/services) for maintenance windows

**Analysis script can't find data**
- Ensure stream files are in `data/qrng_streams/`
- Check that JSON files have the `"floats"` key

**Plots not displaying**
- Scripts save to `data/visualizations/` by default
- For interactive display: `import matplotlib; matplotlib.use('TkAgg')` before running

**GPU acceleration not working**
- Verify CUDA: `python -c "import torch; print(torch.cuda.is_available())"`
- CPU fallback is automatic — analysis will still run

### Data Format Issues

If you see `KeyError: 'floats'` or `KeyError: 'source'`, your stream JSON may use a different format. The toolkit expects:

```json
{
  "source": "string_identifier",
  "floats": [0.0, ..., 1.0],
  "count": 1000,
  "timestamp": "ISO-8601"
}
```

Some older files use `"values"` instead of `"floats"`. The loader handles both, but custom scripts may not.

---

## File Size Notes

The included dataset is approximately 32 MB:
- `data/qrng_streams/`: 471 files (19 MB) — raw collected data
- `data/visualizations/`: 34 PNGs (10 MB) — generated plots
- `data/analysis_results/`: 6 files (2.7 MB) — analysis outputs
- `data/inference_results/`: 8 files (0.9 MB) — experiment results

All data is JSON (text) or PNG (images) — no binary blobs, no LFS needed.
