# QRNG Inference Setup

## Overview

This document describes the setup and usage of the QRNG inference framework in Helios Trajectory Analysis.

## Configuration

The inference framework uses Hydra configuration files. The main config file is `qrng_inference.yaml`:

```yaml
# qrng_inference.yaml
model: ${oc.env:MODEL, "default"}
dataset: ${oc.env:DATASET, "spdc"}
batch_size: 1024
device: ${oc.env:DEVICE, "cuda"}
```

## Usage

### Loading SPDC Sequence

```python
from helios.qrng_spdc_source import load_sequence_from_spdc

sequence = load_sequence_from_spdc()
print(f"Loaded sequence of shape: {sequence.shape}")
```

### Running Inference

```python
from helios.inference_framework.qrng_bridge import run_inference

output = run_inference(sequence)
print(f"Inference output shape: {output.shape}")
```

## Testing

Run the inference tests:

```bash
cd tests
pytest test_qrng_inference.py -v
```

## Notes

- Replace `load_sequence_from_spdc()` with actual implementation from `qrng_spdc_source.py`.
- Ensure Hydra config (`qrng_inference.yaml`) is correctly referenced.
- Verify that the bridge module (`qrng_bridge.py`) is properly imported and adapted.
