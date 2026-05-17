---
title: Comparing seven quantum random sources on the same bench
summary: Notes from building the QRNG Analysis Toolkit — what it actually takes to put IBM Quantum, ANU, SPDC, and CPU HWRNG streams side by side.
status: draft
authors: [alexander-bone]
tags: [qrng, entropy, chaos, nist-sp-800-22, reproducibility, quantum]
publishedAt: 2026-05-17
---

Most papers about quantum random number generators (QRNGs) test one source at a
time. They run NIST SP 800-22, the source passes, and that is the end of the
story. The interesting question — *do two "good" QRNGs behave the same way?* —
gets very little airtime, because nobody wants to write the glue code to put
seven different APIs and one optics rig on the same bench.

This is a short note about doing exactly that.

## The bench

The [QRNG Analysis Toolkit](https://github.com/vikingdude81/qrng-analysis-toolkit)
is a Python pipeline that pulls bytes from seven sources — IBM Quantum
(superconducting QPU), ANU (vacuum fluctuations), Cipherstone, Outshift, an SPDC
optics rig, a CPU hardware RNG, and a Mersenne Twister baseline — and runs the
same battery of tests on each one:

- NIST SP 800-22 (the standard randomness suite).
- BiEntropy and Shannon / sample / approximate / permutation entropies.
- Chaos metrics: Hurst exponent, Lyapunov estimates, fractal dimensions.
- Phase-space reconstructions of the byte stream as a 1D time series.
- A small classifier that tries to tell sources apart from their statistics.

The repo ships with ~509,000 samples already collected so you can rerun every
figure without API keys.

## What surprised me

**1. NIST is not a tie-breaker.** Every QRNG source we tested passes NIST. So
does a well-seeded Mersenne Twister. That is the whole point of NIST passing
being necessary-but-not-sufficient — it just hits harder when you see it on
your own bench.

**2. Chaos signatures diverge.** Hurst exponents and Lyapunov estimates show
visible spread across "indistinguishable from uniform" sources. Whether that is
*real physics* or *artifact of the estimator at finite N* is the next thing to
nail down — and it is exactly the kind of question that needs a second pair of
eyes.

**3. Small-N entropy estimators disagree a lot.** Sample entropy, approximate
entropy, and permutation entropy give meaningfully different rankings of the
same sources when N is small (< 20 windows). Hybrid estimators that combine
symbolic dynamics with wavelet denoising are the partial answer the toolkit
ships today, but the literature here is still moving.

**4. The classifier is the honest test.** If you cannot train a model to
separate two QRNG streams that both pass NIST, that is the strongest empirical
statement of indistinguishability you are going to get. The pilot in
`inference_framework/` is small but useful — and it occasionally catches
sources that *should* be indistinguishable and are not.

## What I want from this community

- **Independent reruns** of the cross-source comparison on different hardware.
  The cuQuantum-accelerated path should match the CPU path bit-for-bit; I would
  like that confirmed on a non-CUDA machine.
- **Statistical review** of the BiEntropy implementation and the epiplexity
  estimator. These are the spiciest pieces and deserve adversarial reading.
- **New sources.** If you have access to a QRNG that is not on the list, the
  collector interface is small.

The repo is MIT, the dataset is in-tree, the citation file is filled in. PRs and
issues both welcome.

— Alex
