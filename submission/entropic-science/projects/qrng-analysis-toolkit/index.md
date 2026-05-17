---
title: QRNG Analysis Toolkit
summary: Open-source pipeline for collecting, comparing, and stress-testing quantum random number generators across hardware sources.
scope: community
status: draft
owners: [alexander-bone]
repoUrl: https://github.com/vikingdude81/qrng-analysis-toolkit
tags: [qrng, entropy, chaos, nist-sp-800-22, quantum, reproducibility]
publishedAt: 2026-05-17
---

## What it is

The **QRNG Analysis Toolkit** is a Python pipeline for collecting, analyzing,
and visualizing data from multiple Quantum Random Number Generator (QRNG)
hardware sources. It bundles a battery of randomness, chaos, and
consciousness-style metrics so that streams from very different hardware can be
compared on equal footing.

It ships with **509,000+ pre-collected samples** across seven sources and
roughly 300 MB of raw stream data, so reviewers can reproduce every figure
without re-running the hardware collectors.

## What it produces

- **NIST SP 800-22** test battery results per source.
- **BiEntropy** and standard Shannon / sample / permutation / approximate
  entropy estimates.
- **Chaos metrics**: Hurst exponent, Lyapunov estimates, fractal dimensions,
  recurrence-style summaries.
- **Phase-space reconstructions** and trajectory plots for visual auditing.
- **Cross-source comparison reports** — Outshift vs. ANU vs. IBM Quantum vs.
  Cipherstone vs. SPDC vs. CPU HWRNG vs. Mersenne Twister baseline.
- **Inference pilots** that train small classifiers to try to distinguish
  sources, as a sanity check on "indistinguishability from uniform."

## Sources currently wired up

| Source | Type | Collector |
|---|---|---|
| IBM Quantum (Qiskit) | Superconducting QPU | `collectors/ibm_quantum_qrng.py` |
| ANU QRNG | Vacuum-fluctuation API | `collectors/qrng_outshift_client.py`* |
| Cipherstone | Commercial QRNG API | `collectors/collect_cipherstone_stream.py` |
| Outshift | Commercial QRNG API | `collectors/qrng_outshift_client.py` |
| SPDC rig | Spontaneous parametric down-conversion | `collectors/qrng_spdc_source.py` |
| CPU HWRNG | Hardware RNG (RDRAND / `/dev/hwrng`) | `collectors/cpu_hwrng.py` |
| PRNG baseline | Mersenne Twister | reference baseline |

\* shared client; API keys per source.

## Why it exists

Most randomness audits stop at NIST SP 800-22. That tells you whether a stream
is *not obviously broken*, not whether two "good" sources behave the same way.
This toolkit treats QRNG streams as **time series** and asks chaos- and
information-theoretic questions on top of the standard tests:

- Are the chaos signatures consistent across sources that all pass NIST?
- Do entropy estimators agree at small sample sizes?
- Can a small classifier separate sources that should be indistinguishable?
- What does the trajectory of a 24-hour collection actually look like?

## What state it is in

- ~40 analysis modules under `analysis/`, `metrics/`, `measures/`, and
  `inference_framework/`.
- 35 unit tests passing; 19 known import-path errors in `tests/` that do not
  block the analysis pipeline.
- GPU acceleration path via `cuquantum_accelerator/` (CUDA 12 + cuQuantum).
- Reproducible dataset under `qrng_streams/` (296 MB, kept under GitHub's
  single-file limit).
- Citation file in `CITATION.cff`, MIT licensed.

## What help is wanted

- **Fresh API keys / new sources.** Cipherstone, ANU, Outshift, and IBM Quantum
  rotate keys; contributions for new public QRNG endpoints are welcome.
- **Statistical review** of the BiEntropy and epiplexity implementations.
- **Independent reruns** of the cross-source comparison on a different machine
  to confirm the cuQuantum-accelerated path matches the CPU path bit-for-bit.
- **Visualization polish** — the trajectory plots could use a designer.

## Links

- Repo: <https://github.com/vikingdude81/qrng-analysis-toolkit>
- Sister repo (trajectory analysis): <https://github.com/vikingdude81/helios-trajectory-analysis>
- Cite via `CITATION.cff` in the repo root.
