# Helios Analysis Gaps and Novel Analyses

## Current Helios Methods
- Shannon entropy
- Rényi entropy
- Permutation entropy
- Lempel-Ziv complexity

## Identified Gaps
1. **No multiscale entropy** - Missing analysis across multiple time scales for trajectory data
2. **Outdated Lempel-Ziv implementation** - Uses generic string compression instead of time-series optimized methods
3. **No quantum methods** - Missing quantum mutual information and other quantum-specific metrics

## Novel Analyses to Implement

### 1. Multiscale Entropy (MSE)
- Analyze trajectory data across multiple temporal scales
- Provides insight into complexity at different observation windows
- Essential for understanding hierarchical structure in consciousness-related signals

### 2. Quantum Mutual Information
- Measure correlations between detector channels using quantum formalism
- Captures non-classical correlations beyond standard mutual information
- Critical for SPDC source analysis and entanglement detection

### 3. Causal Entropy Estimation
- Estimate entropy conditioned on causal relationships
- Helps distinguish between correlation and causation in trajectory data
- Important for influence detection modules

## Implementation Priority
1. **High**: Multiscale entropy - directly applicable to existing trajectory analysis pipelines
2. **High**: Quantum mutual information - essential for quantum randomness verification
3. **Medium**: Causal entropy estimation - complements existing influence detection
4. **Low**: Lempel-Ziv update - can be addressed after core quantum methods are in place