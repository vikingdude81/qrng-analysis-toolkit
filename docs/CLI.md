# Helios Benchmark CLI

## Usage

```bash
helios-benchmark --model=spdc_perturbed --noise_level=0.15 --n_samples=1e7
```

## GitHub Actions Workflow Example

```yaml
name: Helios Benchmark CI

on:
  push:
    branches:
      - main
  pull_request:
    branches:
      - main

jobs:
  run-benchmark:
    runs-on: ubuntu-latest
    steps:
      - name: Checkout code
        uses: actions/checkout@v2
      - name: Set up Python
        uses: actions/setup-python@v3
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -e .
      - name: Run benchmark
        run: |
          helios-benchmark --model=spdc_perturbed --noise_level=0.15 --n_samples=1e7
        outputs:
          results-json: ${{ steps.run-benchmark.outputs.results-json }}
```

## Description

This setup adds the CLI, integrates into GitHub Actions, and captures JSON output with required metrics.