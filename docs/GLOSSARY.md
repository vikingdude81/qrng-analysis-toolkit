# HELIOS Glossary of Terms

**Version:** 1.0
**Last Updated:** 2026-01-20

---

## Core Concepts

### Anomaly
A statistically significant deviation from expected random behavior in a trajectory or time series. In HELIOS, anomalies are detected through multiple metrics (Hurst, Lyapunov, MSD) exceeding their respective thresholds.

### Attractor
A set of numerical values toward which a system tends to evolve. In trajectory analysis:
- **Point attractor**: System converges to a single point
- **Limit cycle**: System oscillates in a repeating pattern
- **Strange attractor**: System exhibits deterministic chaos with fractal structure

### Convergence
The process by which iterative computations approach a stable value or state. In inference experiments, convergence occurs when confidence exceeds a threshold.

---

## Trajectory Analysis

### Diffusion
The spread of a random walker over time. Types:
- **Normal diffusion**: MSD ∝ t (α ≈ 1.0)
- **Subdiffusion**: MSD ∝ t^α where α < 1 (trapped motion)
- **Superdiffusion**: MSD ∝ t^α where α > 1 (directed motion)
- **Ballistic motion**: MSD ∝ t² (α = 2.0, straight-line motion)

### Diffusion Exponent (α)
The power-law exponent relating Mean Squared Displacement to time: MSD(t) ∝ t^α
- α < 1: Subdiffusion (confined, trapped)
- α ≈ 1: Normal diffusion (Brownian motion)
- α > 1: Superdiffusion (directed, persistent)
- α = 2: Ballistic (deterministic trajectory)

### Hurst Exponent (H)
A measure of long-range dependence in a time series, ranging from 0 to 1:
- H < 0.5: Anti-persistent (mean-reverting, oscillatory)
- H = 0.5: Random walk (no memory)
- H > 0.5: Persistent (trending, momentum)

Calculated using Rescaled Range (R/S) analysis.

### Lyapunov Exponent (λ)
Measures sensitivity to initial conditions (chaos):
- λ < 0: Convergent (attractor)
- λ ≈ 0: Neutral (random walk)
- λ > 0: Divergent (chaotic)

Positive λ indicates trajectories diverge exponentially; negative λ indicates convergence.

### Mean Squared Displacement (MSD)
Average squared distance traveled by a walker from its starting point:
```
MSD(τ) = ⟨|x(t+τ) - x(t)|²⟩
```
Used to characterize diffusion type.

### Phase Space
An abstract space where each possible state of a system is represented as a unique point. For trajectory analysis, we typically use 2D or 3D phase space reconstructed from 1D time series via time-delay embedding.

### Time-Delay Embedding (Takens' Theorem)
Method to reconstruct phase space from a single time series:
```
[x(t), x(t-τ), x(t-2τ), ...]
```
Preserves topological properties of the original attractor.

---

## Signal Classification

### NOISE
Pure random walk with no detectable structure:
- H ≈ 0.5
- λ ≈ 0
- α ≈ 1
- Random run distribution

### DRIFT
Gradual systematic bias:
- H > 0.5 (persistent)
- Linear trend in data
- Not necessarily anomalous

### ATTRACTOR
Convergent behavior toward a fixed point or region:
- λ < 0 (negative Lyapunov)
- Trajectory collapse in phase space
- Reduced variance over time

### PERIODIC
Cyclic or oscillatory patterns:
- Strong spectral peaks
- Characteristic frequency
- Repeating trajectory in phase space

### CHAOTIC
Deterministic chaos (sensitive to initial conditions):
- λ > 0 (positive Lyapunov)
- Bounded trajectory
- Strange attractor structure

### ANOMALOUS
Non-standard diffusion:
- α significantly different from 1
- Subdiffusion or superdiffusion
- May indicate external influence

### INFLUENCE (EMERGENCE)
Multi-metric agreement suggesting external influence:
- Multiple indicators simultaneously anomalous
- Cross-metric correlation
- Highest confidence detection

---

## QRNG Sources

### QRNG (Quantum Random Number Generator)
Hardware that generates random numbers from quantum mechanical processes, providing true randomness (not deterministic).

### PRNG (Pseudo-Random Number Generator)
Algorithm that generates sequences that appear random but are deterministic given the seed. Common: Mersenne Twister (MT19937).

### SPDC (Spontaneous Parametric Down-Conversion)
Quantum optical process where a photon splits into two lower-energy photons. Used in Outshift QRNG.

### Vacuum Fluctuation
Quantum mechanical phenomenon where particle-antiparticle pairs briefly appear in empty space. Used in ANU QRNG.

### Thermal Noise
Random electrical signals from thermal agitation of electrons. Used in CPU RDRAND.

### Min-Entropy
Information-theoretic measure of randomness; the minimum number of bits needed to describe the most likely outcome. For good QRNG: H_min > 0.95 (near-uniform).

---

## Inference Architecture

### Strange Attractor Architecture
Inference pattern that converges through iterative refinement, analogous to dynamics attracted to a fixed point. Uses randomness for exploration/exploitation.

### Code Duality Architecture
Inference pattern based on coupled complementary processes (encoder/decoder, question/answer).

### Tensegrity Architecture
Inference pattern maintaining balance between competing constraints, analogous to tensional integrity structures.

### Convergence Threshold
Confidence level at which iterative inference stops (default: 0.85).

### Exploration/Exploitation Tradeoff
Balance between trying new approaches (exploration) and refining current best (exploitation), modulated by randomness.

---

## Statistical Terms

### Effect Size
Standardized measure of the magnitude of an effect, independent of sample size.

### Cohen's d
Effect size for comparing two means:
```
d = (M₁ - M₂) / SD_pooled
```
- |d| < 0.2: Negligible
- 0.2 ≤ |d| < 0.5: Small
- 0.5 ≤ |d| < 0.8: Medium
- |d| ≥ 0.8: Large

### Confidence Interval (CI)
Range of values within which the true population parameter likely falls (typically 95% probability).

### Bonferroni Correction
Multiple comparison correction: α_adjusted = α / n_comparisons

### Power
Probability of detecting an effect if one exists. Target: ≥ 0.80.

### Type I Error (α)
False positive - rejecting null hypothesis when true. Default: α = 0.05.

### Type II Error (β)
False negative - failing to reject null hypothesis when false. Power = 1 - β.

---

## Quality Metrics

### Autocorrelation
Correlation of a signal with a delayed copy of itself. For good randomness: |ρ| < 0.01 at all lags.

### Spectral Entropy
Measure of the flatness of a power spectrum. Low entropy indicates periodic structure.

### Runs Test
Statistical test for randomness based on sequences of like values.

### g²(0) (Second-Order Correlation)
Measure of photon bunching/antibunching. For single photons: g²(0) < 1.

---

## Project-Specific Terms

### HELIOS
The trajectory analysis system and anomaly detector (Helios Anomaly Scope).

### Warmup Period
Initial steps excluded from analysis to allow metrics to stabilize (default: 60 steps).

### Influence Threshold
Sensitivity parameter for anomaly detection; lower values detect weaker signals.

### Ring State
Internal state representation in neural network integration (from LOCAL_Ai project).

### Epiplexity
Novel metric from recent research (arXiv:2601.03220) for measuring complexity/emergence.

---

## Abbreviations

| Abbrev | Full Form |
|--------|-----------|
| ANOVA | Analysis of Variance |
| ANU | Australian National University |
| CI | Confidence Interval |
| CPU | Central Processing Unit |
| LLM | Large Language Model |
| MSD | Mean Squared Displacement |
| NIST | National Institute of Standards and Technology |
| PPKTP | Periodically Poled Potassium Titanyl Phosphate |
| PRNG | Pseudo-Random Number Generator |
| QRNG | Quantum Random Number Generator |
| RDRAND | Intel Random Number Generator instruction |
| R/S | Rescaled Range (analysis) |
| SD | Standard Deviation |
| SEM | Standard Error of the Mean |
| SPDC | Spontaneous Parametric Down-Conversion |

---

*For theoretical background, see [THEORY.md](THEORY.md)*
