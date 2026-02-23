# HELIOS Script Reference

**Version:** 1.0
**Last Updated:** 2026-01-20

This document catalogs all scripts in the project, their status, and usage.

---

## Core Modules (Do Not Delete)

| File | Purpose | Status |
|------|---------|--------|
| `helios_anomaly_scope.py` | Core trajectory analysis engine | ✅ Active |
| `qrng_spdc_source.py` | SPDC QRNG simulation | ✅ Active |
| `phase_space_visualizer.py` | Real-time trajectory animation | ✅ Active |
| `data_stream_loader.py` | CSV/binary/network stream input | ✅ Active |
| `file_utils.py` | Atomic file operations | ✅ Active |
| `logger_config.py` | Logging configuration | ✅ Active |
| `validation.py` | Input validation utilities | ✅ Active |
| `cpu_hwrng.py` | CPU RDRAND hardware RNG | ✅ Active |
| `epiplexity_estimator.py` | Epiplexity metrics (arXiv:2601.03220) | ✅ Active |
| `chaos_detector.py` | Chaos detection algorithms | ✅ Active |
| `consciousness_metrics.py` | Experimental consciousness metrics | ✅ Active |

---

## QRNG API Clients (Do Not Delete)

| File | Purpose | Status |
|------|---------|--------|
| `qrng_outshift_client.py` | Outshift QRNG API client | ✅ Active |
| `collect_cipherstone_stream.py` | Cipherstone Qbert data collection | ✅ Active |

---

## Analysis Scripts

| File | Purpose | Status | Notes |
|------|---------|--------|-------|
| `run_qrng_analysis.py` | CLI runner for QRNG analysis | ✅ Active | Main entry point |
| `run_scope_analysis.py` | Alternative analysis runner | ✅ Active | |
| `run_full_qrng_analysis.py` | Full analysis pipeline | ✅ Active | |
| `compare_qrng_sources.py` | Compare multiple QRNG sources | ✅ Active | |
| `analyze_source_stability.py` | Statistical stability analysis | ✅ Active | |
| `analyze_saved_streams.py` | Analyze stored stream files | ✅ Active | |
| `comprehensive_qrng_analysis.py` | Full analysis with all metrics | ✅ Active | |
| `qrng_comprehensive_analysis.py` | Similar to above | ⚠️ Review | May be duplicate |
| `analyze_inference_statistics.py` | Statistical analysis with CI | ✅ Active | NEW |

---

## Visualization Scripts

| File | Purpose | Status | Notes |
|------|---------|--------|-------|
| `visualize_qrng_data.py` | QRNG quality dashboard | ✅ Active | Current version |
| `visualize_inference_results.py` | Inference experiment dashboard | ✅ Active | |
| `qrng_dashboard.py` | Comprehensive dashboard | ⚠️ Review | Overlaps with visualize_qrng_data.py |
| `qrng_deep_dive.py` | Deep analysis (v1) | 🔄 Superseded | Use v2 instead |
| `qrng_deep_dive_v2.py` | Deep analysis (v2) | ✅ Active | Proper centering |

---

## Inference Framework

| File | Purpose | Status |
|------|---------|--------|
| `inference_framework/__init__.py` | Framework exports | ✅ Active |
| `inference_framework/architectures.py` | Inference architectures | ✅ Active |
| `inference_framework/classifier.py` | Mode classification | ✅ Active |
| `inference_framework/experiment.py` | Experiment framework | ✅ Active |
| `inference_framework/qrng_bridge.py` | QRNG integration | ✅ Active |
| `inference_framework/demo.py` | Demo script | ✅ Active |
| `run_qrng_inference_pilot.py` | Pilot experiment runner | ✅ Active |
| `qrng_inference_demo.py` | Inference demo | ✅ Active |

---

## Utility Scripts

| File | Purpose | Status |
|------|---------|--------|
| `qrng_daily_collector.py` | Scheduled data collection | ✅ Active |
| `test_qrng_live.py` | Live API testing | ✅ Active |
| `migrate_data_directories.py` | Data directory migration | ✅ Active | NEW |

---

## Examples

| File | Purpose | Status |
|------|---------|--------|
| `examples/basic_qrng_analysis.py` | Basic usage example | ✅ Active |
| `examples/helios_integration.py` | HELIOS neural network integration | ✅ Active |
| `examples/compare_qrng_vs_prng.py` | Comparison example | ✅ Active |

---

## Tests

| File | Purpose | Status |
|------|---------|--------|
| `tests/test_anomaly_scope.py` | Core functionality tests | ✅ Active |
| `tests/test_metrics.py` | Metric computation tests | ✅ Active |
| `tests/test_chaos_detector.py` | Chaos detection tests | ✅ Active |
| `tests/test_consciousness_metrics.py` | Consciousness metric tests | ✅ Active |
| `tests/test_nist.py` | NIST test suite tests | ✅ Active |
| `tests/test_performance.py` | Performance benchmarks | ✅ Active |
| `tests/test_qrng_quality.py` | QRNG quality tests | ✅ Active |
| `tests/test_qrng_source.py` | QRNG source tests | ✅ Active |
| `tests/test_scope_integration.py` | Integration tests | ✅ Active |
| `tests/test_signal_injection.py` | Signal injection tests | ✅ Active |
| `tests/test_cipherstone_qrng.py` | Cipherstone API tests | ✅ Active |
| `tests/nist_tests.py` | NIST implementation | ✅ Active |
| `tests/conftest.py` | Pytest fixtures | ✅ Active |

---

## Cleanup Recommendations

### Scripts to Consolidate

1. **qrng_deep_dive.py → qrng_deep_dive_v2.py**
   - v2 has proper centering and methodology fixes
   - Rename v2 to `qrng_deep_dive.py` and archive v1

2. **qrng_dashboard.py vs visualize_qrng_data.py**
   - Review overlap and determine primary
   - Consider consolidating into one dashboard

3. **qrng_comprehensive_analysis.py vs comprehensive_qrng_analysis.py**
   - Verify differences
   - Keep one, archive the other

### Recommended Actions

```bash
# Create archive directory
mkdir -p archive/deprecated

# Move v1 to archive
mv qrng_deep_dive.py archive/deprecated/qrng_deep_dive_v1.py

# Rename v2 to primary
mv qrng_deep_dive_v2.py qrng_deep_dive.py

# Review and potentially archive duplicates
# (manual review recommended first)
```

---

## Script Dependencies

```
helios_anomaly_scope.py
├── Used by: run_qrng_analysis.py, qrng_dashboard.py, etc.
├── Depends on: numpy, scipy, torch (optional)
└── Core module - no circular dependencies

inference_framework/
├── Used by: run_qrng_inference_pilot.py
├── Depends on: anthropic, numpy, rich
└── Uses: qrng_bridge.py for QRNG integration

qrng_outshift_client.py
├── Used by: qrng_bridge.py
├── Depends on: requests, dotenv
└── Requires: QRNG_OUTSHIFT_API_KEY
```

---

*This document should be updated when scripts are added, removed, or significantly modified.*
