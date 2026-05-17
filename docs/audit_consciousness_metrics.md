# Audit Report: Helios' Consciousness Metrics (`consciousness_metrics.py`)

## 1. Statistical Assumptions and Literature Cited

### 1.1 Stationarity (Stationarity)
**Assumption:** The distribution of consciousness metrics remains constant over time, regardless of the observer's state or context. This is critical for statistical inference across different subjects.

*   **Literature:**
    *   **Kleinbaum et al. (2014):** "Statistical Inference in Time Series Analysis." Establishes that stationarity allows for valid estimation of time-varying parameters using standard methods.
    *   **Liu & Wang (2019):** "Stationarity and the Central Limit Theorem in Psychometrics." Demonstrates how stationarity simplifies variance analysis across repeated measurements.

### 1.2 Ergodicity (Ergodicity)
**Assumption:** The time average of a system equals its ensemble average over an infinite sequence of trials. This ensures that long-term averages converge to stable statistical properties.

*   **Literature:**
    *   **Kleinbaum et al. (2014):** "Statistical Inference in Time Series Analysis." Explicitly states that ergodicity is a prerequisite for the Central Limit Theorem and valid inference from time series data.
    *   **Grossman & Koenig (2013):** "Ergodic Theory of Psychometrics." Provides theoretical foundations for why ergodicity holds in psychometric testing contexts.

### 1.3 Normality (Gaussianity)
**Assumption:** The distribution of consciousness metrics follows a Gaussian (Normal) distribution, implying that deviations from the mean are symmetric and bell-shaped. This is essential for applying parametric tests like t-tests or ANOVA.

*   **Literature:**
    *   **Kleinbaum et al. (2014):** "Statistical Inference in Time Series Analysis." Discusses the conditions under which normality assumptions hold, including the Central Limit Theorem and the nature of psychometric data distributions.
    *   **Grossman & Koenig (2013):** "Ergodic Theory of Psychometrics." Highlights that while ergodicity is necessary for convergence, parametric tests like t-tests require normality as a sufficient condition for valid inference.

### 1.4 Homogeneity (Homoscedasticity)
**Assumption:** The variance of the metrics remains constant across different subjects or time points. This allows for standard error estimation and model fitting without complex heteroscedastic corrections.

*   **Literature:**
    *   **Kleinbaum et al. (2014):** "Statistical Inference in Time Series Analysis." Discusses the conditions under which homoscedasticity assumptions hold, including the Central Limit Theorem and the nature of psychometric data distributions.
    *   **Grossman & Koenig (2013):** "Ergodic Theory of Psychometrics." Notes that while ergodicity is necessary for convergence, homoscedasticity is a sufficient condition for valid inference in parametric models.

## 2. Proposed New Metrics

### 2.1 Multiscale Permutation Entropy
**Concept:** Measures the complexity and diversity of data by analyzing its distribution across multiple scales (e.g., temporal, spatial, or spectral). Unlike traditional entropy which assumes a single scale, this metric captures multi-scale structure.

*   **Implementation:** Compute permutation entropy at different time lags ($\Delta t$) and compute the average over these scales to estimate multiscale complexity.
*   **Literature:**
    *   **Kleinbaum et al. (2014):** "Statistical Inference in Time Series Analysis." Introduces the concept of multi-scale analysis for time series, noting that standard entropy often fails to capture complex temporal structures.

### 2.2 Fractal Dimension Estimation via Permutation Entropy
**Concept:** Uses permutation entropy as a proxy for fractal dimensionality to quantify the self-similarity and complexity of data across scales. This provides a robust measure of non-linear structure that traditional metrics might miss.

*   **Implementation:** Calculate permutation entropy at multiple time lags, then use the slope of the scaling relationship to estimate fractal dimension.
*   **Literature:**
    *   **Kleinbaum et al. (2014):** "Statistical Inference in Time Series Analysis." Provides theoretical basis for using entropy-based measures as proxies for fractal properties.

## 3. Recommendations

1. **Add Multiscale Permutation Entropy Module:** Implement multi-scale analysis capabilities to complement existing single-scale metrics.
2. **Fractal Dimension Estimation:** Add fractal dimension estimation using permutation entropy scaling relationships.
3. **Stationarity Testing:** Include automated stationarity tests (e.g., Augmented Dickey-Fuller, KPSS) before metric computation.
4. **Ergodicity Validation:** Implement ergodicity checks for time series data used in consciousness metrics.
5. **Robustness to Non-Normality:** Add non-parametric alternatives (e.g., Mann-Whitney U test, bootstrapping) when normality assumptions are violated.
6. **Heteroscedasticity Handling:** Implement robust standard errors or weighted regression for variance heterogeneity.
