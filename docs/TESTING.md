# Testing Guide

This document describes how to run and verify the HELIOS trajectory analysis system.

## Quick Start

```bash
# Activate environment
source .venv/bin/activate

# Run a basic analysis (400 steps)
python run_qrng_analysis.py --steps 400

# Run without visualization (faster)
python run_qrng_analysis.py --steps 1000 --no-viz

# Run with SHA-256 whitening (extra randomness hardening)
python run_qrng_analysis.py --steps 1000 --no-viz
# Note: Enable in code with use_sha256_whitening=True
```

## Command Line Options

```
--steps N          Number of QRNG samples (default: 2000)
--output-dir DIR   Output directory (default: qrng_results)
--ring-sections N  SPDC ring sections: 4, 8, 16 (default: 4)
--pump-power MW    Pump laser power in mW (default: 17.0)
--walk-mode MODE   Random walk mode: angle, xy_independent, takens (default: angle)
--no-viz           Disable terminal visualization
```

## Verification Tests

### 1. Warmup Artifact Test

Verifies that no detections occur during warmup period (steps ≤60):

```bash
python3 << 'EOF'
from helios_anomaly_scope import QRNGStreamScope
from qrng_spdc_source import get_quantum_random
import numpy as np

first_steps = []
for i in range(100):
    scope = QRNGStreamScope()
    for step in range(200):
        scope.update_from_stream(get_quantum_random())
    if scope.events:
        first_steps.append(scope.events[0].step)
    if (i+1) % 20 == 0:
        print(f"Progress: {i+1}/100")

print(f"\nResults:")
print(f"  Min first step: {min(first_steps)}")
print(f"  In warmup (≤60): {sum(1 for s in first_steps if s <= 60)}/100")
print(f"  Expected: 0 in warmup")
EOF
```

**Expected**: All first detections at step 61 or later.

### 2. QRNG Autocorrelation Test

Verifies QRNG output has no significant autocorrelation:

```bash
python3 << 'EOF'
from qrng_spdc_source import SPDCQuantumSource
import numpy as np

for whitening in [False, True]:
    mode = "SHA-256" if whitening else "Standard"
    source = SPDCQuantumSource(use_sha256_whitening=whitening)
    vals = np.array([source.get_random() for _ in range(2000)])
    vals_c = vals - vals.mean()
    var = np.sum(vals_c**2)
    ci = 1.96/np.sqrt(len(vals))
    
    print(f"\n{mode} mode:")
    significant = 0
    for lag in [1, 5, 10, 20, 30, 50]:
        acf = np.sum(vals_c[:-lag] * vals_c[lag:]) / var
        if abs(acf) > ci:
            significant += 1
            print(f"  Lag {lag:2d}: {acf:+.4f} ⚠")
        else:
            print(f"  Lag {lag:2d}: {acf:+.4f} ✓")
    
    print(f"  95% CI: ±{ci:.4f}")
    print(f"  Significant lags: {significant}/6 (expect ≤1 by chance)")
EOF
```

**Expected**: 0-1 significant lags (5% false positive rate).

### 3. Statistical Quality Test

Verifies QRNG output matches expected uniform distribution:

```bash
python3 << 'EOF'
from qrng_spdc_source import SPDCQuantumSource
import numpy as np

source = SPDCQuantumSource()
vals = np.array([source.get_random() for _ in range(10000)])

print("QRNG Statistical Quality:")
print(f"  Mean: {vals.mean():.4f} (expected: 0.5000)")
print(f"  Std:  {vals.std():.4f} (expected: 0.2887)")
print(f"  Min:  {vals.min():.4f} (expected: ~0)")
print(f"  Max:  {vals.max():.4f} (expected: ~1)")

# Chi-square test for uniformity
bins = 10
observed, _ = np.histogram(vals, bins=bins)
expected = len(vals) / bins
chi2 = np.sum((observed - expected)**2 / expected)
print(f"  Chi²: {chi2:.2f} (expect < 16.9 for p>0.05)")
EOF
```

**Expected**: Mean ≈ 0.5, Std ≈ 0.289, Chi² < 16.9.

### 4. Regime Detection Test

Verifies detector can identify known patterns:

```bash
python3 << 'EOF'
from helios_anomaly_scope import QRNGStreamScope
import numpy as np

# Test with biased input (should trigger drift)
scope = QRNGStreamScope()

# Normal random for warmup
for i in range(100):
    scope.update_from_stream(np.random.random())

# Inject drift (biased values)
for i in range(100):
    scope.update_from_stream(0.7 + 0.1 * np.random.random())

events = scope.events
drift_count = sum(1 for e in events if e.event_type == 'drift')
print(f"Drift events detected: {drift_count}")
print(f"Expected: >0 (biased input should trigger drift)")
EOF
```

**Expected**: Multiple drift events after step 100.

## Batch Testing

Run multiple analyses and summarize:

```bash
# Run 10 analyses
for i in $(seq 1 10); do
    echo "Run $i/10..."
    python run_qrng_analysis.py --steps 500 --no-viz
done

# Summarize results
python3 << 'EOF'
import json
import glob
import os

files = sorted(glob.glob('qrng_results/run_*.json'), key=os.path.getmtime)[-10:]
print(f"Last 10 runs:")
print(f"{'Run ID':<20} {'Events':>7} {'First@':>7} {'Hurst':>7}")
print("-"*50)

for f in files:
    run = json.load(open(f))
    events = run.get('events', [])
    current = run.get('summary', {}).get('current_state', {})
    first = events[0]['step'] if events else 'N/A'
    hurst = current.get('hurst', 0)
    print(f"{run['run_id']:<20} {len(events):>7} {str(first):>7} {hurst:>7.3f}")
EOF
```

## Output Files

Each run produces:

| File | Description |
|------|-------------|
| `run_YYYYMMDD_HHMMSS.json` | Full results with trajectory, metrics, events |
| `run_YYYYMMDD_HHMMSS.xlsx` | Excel spreadsheet for analysis |
| `*_trajectory.png` | Phase space trajectory plot |
| `*_metrics.png` | Time series of metrics |
| `*_randomness.png` | QRNG quality diagnostics |
| `*_phase_analysis.png` | MSD and diffusion analysis |

## Interpreting Results

### Event Types

| Event | Meaning | Threshold |
|-------|---------|-----------|
| `drift` | Systematic bias in trajectory | MSD trend > 0.05 |
| `trending_behavior` | Persistent motion (Hurst > 0.6) | H > 0.6 for 10+ steps |
| `attractor_lock` | Trajectory confined to region | Velocity < baseline/3 |
| `coherence_spike` | High coherence detected | Coherence > 0.8 |
| `ballistic_motion` | Linear motion (not diffusive) | α > 1.5 |
| `chaotic_sensitivity` | Positive Lyapunov | λ > 0.1 |
| `convergent_attractor` | Strongly negative Lyapunov | λ < -0.2 |

### Key Metrics

| Metric | Normal Range | Anomalous |
|--------|--------------|-----------|
| Hurst exponent | 0.4-0.6 | >0.7 (persistent) or <0.3 (anti-persistent) |
| Lyapunov exponent | -0.1 to +0.05 | >0.1 (chaotic) or <-0.2 (attracting) |
| Diffusion α | 0.9-1.1 | >1.5 (ballistic) or <0.5 (sub-diffusive) |
| MSD trend | < 0.05 | > 0.1 (directed motion) |

### Signal Classification

- **noise**: Random fluctuations, no pattern
- **anomalous**: Some metrics outside normal, unverified
- **verified**: Multiple independent metrics confirm pattern
- **emergence**: Verified pattern with high significance
