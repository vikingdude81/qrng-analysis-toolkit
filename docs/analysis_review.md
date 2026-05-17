# Implementation Review

## Kolmogorov-Smirnov (KS)
- Correctly implemented for two-sample and one-sample tests
- Handles continuity correction and p-values

## Ljung-Box
- Properly computes autocorrelation test statistics and p-values
- Includes adjustments for small sample sizes

## Permutation Entropy
- Implemented using the standard algorithm (e.g., permutation-based entropy)
- Correct handling of embedding dimensions and scaling

## Novel Analysis (Multiscale Entropy - MSE)
**Prototype:** Use MSE to analyze QRNG time series by computing entropy at multiple scales (e.g., 1, 2, 4, 8 times the embedding dimension). This captures complex dynamics and nonlinearity, offering better resolution than traditional permutation entropy.

**Key Advantage:** MSE adapts to non-stationary data and quantifies complexity across scales, making it suitable for QRNG applications.
