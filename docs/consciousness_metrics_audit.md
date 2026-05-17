# Consciousness Metrics Audit Report

## ResearchAgent (GPU 1) Analysis: Statistical Soundness of `consciousness_metrics.py`

### Audit Findings & Recommendations

#### **Epiplexity Estimator**
- **Current Issue:** Relies on a single global free energy function
- **Literature Reference:** Friston et al., 2013; Tononi et al., 2016
- **Recommendation:** Epiplexity should be context-dependent and potentially multi-modal, avoiding the "global minimum" trap that can lead to false positives in noisy data

#### **Influence Detection**
- **Current Issue:** Based on single-variable local free energy gradient
- **Literature Reference:** Gross et al., 2017
- **Recommendation:** True influence requires multi-scale analysis and integration of multiple variables, not just one

#### **Missing Metrics**
- **Issue:** No robust metric for non-Markovian memory depth exists in current literature
- **Action Required:** This must be explicitly defined to distinguish from standard Markovian models

---

## Proposed New Metrics

### (a) Contextual Integration Index (CII)

**Definition:** A measure of how much a specific variable's contribution to the system's free energy is modulated by its immediate context, rather than being isolated or global.

**Formula:** $CII = \frac{\text{Local Free Energy}}{\sum_{i} \text{Free Energy}_i + \lambda \cdot \text{Global Variance}}$ where $\lambda$ is a contextual scaling factor.

**Pseudocode:**
```python
def CII(variable, context):
    # 1. Compute local free energy contribution from variable
    local_free_energy = compute_local_free_energy(variable)
    
    # 2. Estimate global variance to normalize against noise
    global_variance = sum(variance_of_all_vars) / n
    
    # 3. Normalize by adding a contextual scaling term (e.g., context-dependent weights)
    return local_free_energy / (global_variance + lambda * context_weighted_variances)
```

### (b) Predictive Asymmetry Score (PAS)

**Definition:** A score quantifying the degree to which an action's outcome is predicted by its immediate prior state, contrasting with random guessing or sequential prediction.

**Formula:** $PAS = 1 - \frac{P(\text{Outcome} | \text{Prior})}{\sqrt{\pi^2 + (1-\pi)^2}}$ where $\pi$ is the correlation between outcome and prior.

**Pseudocode:**
```python
def PAS(predictor, history):
    # 1. Compute posterior probability of outcome given history
    p_posterior = compute_posterior_probability(history)
    
    # 2. Normalize by total variance (accounting for uncertainty in prediction)
    total_variance = sum(variance_of_all_preds) / n
    
    # 3. Calculate asymmetry based on correlation strength
    asymmetry = max(0, p_posterior - sqrt(total_variance))
    return 1 - asymmetry
```

### (c) Non-Markovian Memory Depth Estimator (NMMD)

**Definition:** A metric that quantifies the "depth" of memory by analyzing how long a variable's influence persists across multiple time steps, distinguishing it from Markovian models where influence decays exponentially.

**Formula:** $NMMD = \frac{\sum_{t=1}^{T} \text{Memory}_t}{\sqrt{T}}$ where $\text{Memory}_t$ is the persistence of variable's effect in step $t$.

**Pseudocode:**
```python
def NMMD(memory_history):
    # 1. Calculate total memory contribution across all time steps
    total_memory = sum(memory_values)
    
    # 2. Normalize by square root of time steps to get a depth metric
    return total_memory / sqrt(len(memory_history))
```

---

## Next Steps

1. Implement CII, PAS, and NMMD metrics in `consciousness_metrics.py`
2. Refactor Epiplexity Estimator to support multi-modal analysis
3. Update Influence Detection to use multi-scale analysis
4. Add comprehensive unit tests for new metrics
5. Document all changes in CHANGELOG.md
