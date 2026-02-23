# HELIOS QRNG Research Protocol

**Version:** 1.0
**Last Updated:** 2026-01-20
**Status:** Active Research

---

## 1. Research Overview

### 1.1 Project Title
**Quantum Randomness Effects on Large Language Model Inference Dynamics**

### 1.2 Principal Investigator
Alex Bondar

### 1.3 Research Question
Does the source of randomness (quantum vs. pseudo-random) measurably affect the convergence behavior, confidence, or output characteristics of LLM inference when used in stochastic decision-making architectures?

### 1.4 Null Hypothesis (H₀)
There is no statistically significant difference in LLM inference metrics (iterations to convergence, final confidence, response consistency) between quantum random number generator (QRNG) sources and pseudo-random number generator (PRNG) sources.

### 1.5 Alternative Hypothesis (H₁)
QRNG-seeded inference produces measurably different convergence dynamics compared to PRNG-seeded inference, potentially due to true randomness properties affecting attractor basin exploration.

---

## 2. Randomness Sources

### 2.1 QRNG Sources (Quantum)

| Source | Provider | Physics | Samples | API |
|--------|----------|---------|---------|-----|
| **Outshift SPDC** | Outshift.com | Photon pair coincidence timing | 8,000 | REST |
| **ANU Vacuum** | Australian National University | Vacuum fluctuation shot noise | 5,000 | REST |
| **Cipherstone Qbert** | Cipherstone.co | Quantum noise (conditioned) | 5,000 | REST |

### 2.2 Control Sources

| Source | Type | Physics | Samples | Method |
|--------|------|---------|---------|--------|
| **CPU RDRAND** | Hardware RNG | Thermal noise in silicon | 1,000 | BCrypt/RDRAND instruction |
| **PRNG** | Software | Mersenne Twister (MT19937) | Unlimited | numpy.random.default_rng() |

### 2.3 Data Quality Requirements

All QRNG data must pass:
- **Min-entropy test:** H_min > 0.95 (near-uniform)
- **Autocorrelation test:** |ρ| < 0.01 at all lags
- **Bias test:** |mean - 0.5| < 0.01 for normalized floats
- **NIST 800-22 subset:** Frequency, runs, longest-run tests

---

## 3. Experimental Design

### 3.1 Design Type
Between-subjects factorial design with repeated measures.

### 3.2 Independent Variable
**Randomness Source** (5 levels):
1. Outshift SPDC QRNG
2. ANU Vacuum QRNG
3. Cipherstone Qbert QRNG
4. CPU RDRAND (hardware control)
5. Mersenne Twister PRNG (software control)

### 3.3 Dependent Variables

| Variable | Measurement | Expected Range |
|----------|-------------|----------------|
| **Iterations to convergence** | Count | 1-10 |
| **Final confidence** | Float | 0.0-1.0 |
| **Convergence time** | Milliseconds | 5,000-80,000 |
| **Tokens used** | Count | 300-3,500 |
| **Response consistency** | Unique hash ratio | 0.0-1.0 |

### 3.4 Control Variables
- LLM model: `claude-sonnet-4-20250514` (fixed)
- Temperature: 0.7 (fixed)
- Max tokens: 1024 (fixed)
- Prompt: Identical across conditions
- Time of day: Varied (not controlled)

### 3.5 Experimental Design Requirements

**Current Status: EXPLORATORY (Pilot)**

The current experiment design has limitations that should be addressed for confirmatory studies:

#### Known Confounds in Pilot
- Runs may not be fully interleaved (source blocks vs randomized)
- API latency variation between sources
- Potential caching effects
- Background system load variation

#### Required for Confirmatory Study
1. **Interleaved trials**: Randomize source selection per trial
2. **Paired design**: Same prompt set appears once per source
3. **Controlled timing**: Fixed inter-trial intervals
4. **Blinded analysis**: Analyze results without knowing source until complete
5. **Pre-registration**: Lock hypothesis and analysis plan before data collection

#### Recommended Protocol for Confirmatory Run
```python
# Pseudocode for proper interleaved experiment
prompts = load_prompt_set()  # Fixed set of N prompts
sources = ['OUTSHIFT', 'ANU', 'CIPHERSTONE', 'CPU_RDRAND', 'PRNG']

trials = []
for prompt in prompts:
    for source in sources:
        trials.append((prompt, source))

random.shuffle(trials)  # Fully randomize order

for prompt, source in trials:
    # Fixed delay between trials
    time.sleep(INTER_TRIAL_DELAY)
    result = run_inference(prompt, source)
    save_result(result)
```

### 3.6 Sample Size Justification

**Power Analysis (a priori):**
- Effect size of interest: d = 0.5 (medium)
- Alpha: 0.05
- Power: 0.80
- Groups: 5

**Required N per group:** ~26 trials

**Current study:** n=10 per group (pilot), n=50 total

**Note:** This is an exploratory pilot. Results will inform power analysis for confirmatory study.

---

## 4. Statistical Analysis Plan

### 4.1 Primary Analysis
**One-way ANOVA** comparing iterations to convergence across 5 randomness sources.

### 4.2 Multiple Comparison Correction
**Bonferroni correction** for 10 pairwise comparisons:
- Adjusted alpha: α = 0.05 / 10 = 0.005

### 4.3 Effect Size Reporting
- **Cohen's d** for pairwise comparisons
- **95% confidence intervals** on all effect sizes
- **Interpretation:** |d| < 0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large

### 4.4 Secondary Analyses
1. **Kruskal-Wallis test** (non-parametric alternative)
2. **Distribution comparison:** K-S test, Mann-Whitney U
3. **Variance comparison:** Levene's test

### 4.5 Exploratory Analyses
- Correlation between QRNG quality metrics and inference performance
- Time-series analysis of convergence patterns
- Response semantic similarity (future: embedding comparison)

---

## 5. Stopping Rules

### 5.1 Data Collection Stopping
Data collection stops when:
- **Minimum:** n=10 per condition (pilot)
- **Target:** n=30 per condition (confirmatory)
- **Maximum:** n=50 per condition (budget limit)

### 5.2 Analysis Stopping
Analysis proceeds regardless of interim results. No early stopping for significance.

### 5.3 Null Result Interpretation
If H₀ not rejected:
- Report 95% CI on effect size
- State maximum detectable effect given N
- Do NOT claim "no effect" - only "effect bounded by CI"

---

## 6. Data Management

### 6.1 Data Storage Structure

```
data/
├── raw/                    # Original QRNG streams
│   ├── outshift/
│   ├── anu/
│   ├── cipherstone/
│   └── cpu_hwrng/
├── processed/              # Validated, normalized data
├── inference/              # LLM experiment results
└── analysis/               # Statistical outputs
```

### 6.2 File Naming Convention
```
{source}_{type}_{YYYYMMDD}_{HHMMSS}.json
```

Example: `anu_stream_20260119_220832.json`

### 6.3 Metadata Requirements
Each data file must include:
- `source`: Provider identifier
- `timestamp`: ISO 8601 format
- `count`: Number of samples
- `quality_metrics`: Validation results (if applicable)

### 6.4 Version Control
- All code changes tracked in Git
- Data files tracked (not gitignored for reproducibility)
- Analysis notebooks version-controlled

---

## 7. Quality Assurance

### 7.1 Data Validation Pipeline
1. **Collection:** API response validation
2. **Storage:** Atomic writes, checksum verification
3. **Loading:** Schema validation before analysis
4. **Analysis:** Range checks on all metrics

### 7.2 Code Testing
- Unit tests: pytest suite (>80% coverage target)
- Integration tests: Real QRNG data fixtures
- CI/CD: GitHub Actions on every push

### 7.3 Reproducibility Checklist
- [ ] Fixed random seeds for PRNG baseline
- [ ] Pinned dependency versions
- [ ] Documented API versions
- [ ] Timestamped all outputs
- [ ] Code version tracked with results

---

## 8. Ethical Considerations

### 8.1 Data Sources
All QRNG data is publicly available via APIs. No human subjects involved.

### 8.2 LLM Usage
LLM responses are generated content, not personal data. API usage within terms of service.

### 8.3 Conflicts of Interest
None declared. No funding from QRNG providers.

---

## 9. Reporting Standards

### 9.1 Required Elements
- Sample sizes for all conditions
- Means, standard deviations, and confidence intervals
- Effect sizes with confidence intervals
- Exact p-values (not just "p < 0.05")
- Full statistical test details (df, test statistic)

### 9.2 Null Result Reporting
If no significant effect found:
- Report equivalence bounds
- Report Bayesian evidence for null (if applicable)
- Avoid language implying "proof of no effect"

### 9.3 Visualization Standards
- Error bars: 95% CI (not SEM)
- Individual data points shown where feasible
- Effect size visualizations alongside p-values

---

## 10. Timeline

| Phase | Dates | Status |
|-------|-------|--------|
| **Setup** | Jan 1-14, 2026 | Complete |
| **Pilot (n=10)** | Jan 14-20, 2026 | Complete |
| **Analysis** | Jan 20-25, 2026 | In Progress |
| **Confirmatory (n=30)** | Jan 25-Feb 5, 2026 | Planned |
| **Write-up** | Feb 5-15, 2026 | Planned |

---

## 11. Results Summary (Pilot)

### 11.1 Current Findings (n=10 per condition, Jan 20, 2026)

| Source | Iterations (M±SD) | Confidence (M±SD) |
|--------|-------------------|-------------------|
| Outshift SPDC | 2.9 ± 1.5 | 0.639 ± 0.14 |
| ANU Vacuum | 3.3 ± 1.5 | 0.678 ± 0.11 |
| Cipherstone Qbert | 2.2 ± 1.4 | 0.561 ± 0.16 |
| CPU RDRAND | 2.5 ± 1.8 | 0.555 ± 0.17 |
| PRNG | 2.4 ± 1.8 | 0.556 ± 0.18 |

### 11.2 Effect Size (QRNG vs PRNG)
- **Cohen's d (iterations):** 0.296
- **Interpretation:** Small effect, not significant at α=0.05
- **95% CI:** To be calculated with updated analysis

### 11.3 Preliminary Conclusion
No statistically significant difference detected between QRNG and PRNG sources in pilot study. Effect size is small (d ≈ 0.3), bounded by confidence interval. Larger sample size needed for definitive conclusion.

---

## 12. Amendments

| Date | Section | Change | Reason |
|------|---------|--------|--------|
| 2026-01-20 | Initial | Document created | Audit recommendation |

---

## References

1. Cohen, J. (1988). *Statistical Power Analysis for the Behavioral Sciences*
2. NIST SP 800-22 (2010). *A Statistical Test Suite for Random Number Generators*
3. Intuition Machine (2026). *Inference Architectures* [arXiv:2601.03220]

---

*This protocol follows CONSORT and APA reporting guidelines adapted for computational experiments.*
