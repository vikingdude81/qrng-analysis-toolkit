# Class Compatibility Reference

## Helios Trajectory Analysis Classes

| Class Name | Source Repo | Key Parameters | Compatibility Notes |
|------------|-------------|----------------|--------------------|
| EntropyCalculator | helios-trajectory-analysis | data, base | Requires numpy, supports base=2 or base=e |
| ChaosMeasure | helios-trajectory-analysis | trajectory, window_size | Requires scipy, window_size must be positive integer |
| ConsciousnessIndex | helios-trajectory-analysis | trajectory, time_window | Requires pandas, time_window must be in seconds |

## QRNG Analysis Toolkit Classes

| Class Name | Source Repo | Key Parameters | Compatibility Notes |
|------------|-------------|----------------|--------------------|
| QRNGEntropy | qrng-analysis-toolkit | sequence, window_size | Requires numpy, window_size must be at least 1 |
| ChaosScore | qrng-analysis-toolkit | sequence, threshold | Requires pandas, threshold must be a float between 0 and 1 |
| ConsciousnessMetric | qrng-analysis-toolkit | sequence, model | Requires sklearn, model must be a string (e.g., 'linear') |

## Usage Guidelines

### Entropy Calculations
- Use `EntropyCalculator` for helios-specific entropy measures with configurable bases
- Use `QRNGEntropy` for standard QRNG entropy analysis

### Chaos Analysis
- Use `ChaosMeasure` for trajectory-based chaos metrics (helios)
- Use `ChaosScore` for sequence-based chaos scoring (toolkit)

### Consciousness Metrics
- Use `ConsciousnessIndex` for helios trajectory consciousness measures
- Use `ConsciousnessMetric` for toolkit-based consciousness modeling

## Dependencies Summary

| Dependency | Used By |
|------------|---------|
| numpy | EntropyCalculator, QRNGEntropy |
| scipy | ChaosMeasure |
| pandas | ConsciousnessIndex, ChaosScore |
| sklearn | ConsciousnessMetric |
