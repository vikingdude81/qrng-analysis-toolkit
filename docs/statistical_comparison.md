# Statistical Soundness Comparison

## KDE vs k-NN Entropy vs Permutation Entropy

### KDE (Kernel Density Estimate)
- **Robust for**: Low-dimensional data
- **Sensitivity**: Sensitive to noise and high dimensionality due to kernel overlap
- **Risk**: May overestimate complexity in sparse datasets

### k-NN Entropy
- **Robust for**: Simple systems with clear distance metrics
- **Sensitivity**: Prone to bias from small k or noisy data
- **Risk**: Performance depends heavily on distance metric choice

### Permutation Entropy
- **Robust for**: Detecting complexity in time series
- **Sensitivity**: May fail in non-stationary or multi-variable systems
- **Risk**: Limited interpretability for complex dynamics

## Proposed Metrics (Next Iteration)

### 1. Multivariate Sample Entropy (MSE)
- **Purpose**: Extends permutation entropy to multiple variables
- **Benefit**: Improves robustness for complex systems
- **Use Case**: Multi-variable consciousness metrics

### 2. Conditional Entropy from Transfer Operators
- **Purpose**: Uses transfer operator to model system dynamics
- **Benefit**: Captures non-linear dependencies and temporal structure
- **Use Case**: Interpretable dynamical systems analysis

## Rationale
Both proposed metrics build on recent advances in multivariate analysis and nonlinear dynamics. MSE addresses multi-variable complexity, while transfer operators enhance interpretability of dynamical systems.