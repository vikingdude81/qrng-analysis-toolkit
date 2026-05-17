---
title: A Statistical Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications
summary: The NIST SP 800-22 Revision 1a test battery — the de-facto baseline for randomness testing of QRNGs and PRNGs.
authors: ["Bassham, L.", "Rukhin, A.", "Soto, J.", "Nechvatal, J.", "Smid, M.", "Barker, E.", "Leigh, S.", "Levenson, M.", "Vangel, M.", "Banks, D.", "Heckert, A.", "Dray, J.", "Vo, S."]
year: 2010
venue: NIST Special Publication 800-22 Revision 1a
pdfUrl: https://nvlpubs.nist.gov/nistpubs/Legacy/SP/nistspecialpublication800-22r1a.pdf
plainSummary: |
  NIST SP 800-22 is the standard randomness test battery used by basically
  every quantum random number generator (QRNG) paper. It defines 15 statistical
  tests — frequency, runs, longest-run, rank, DFT, template-matching,
  approximate entropy, cumulative sums, random-excursions, and more — each
  producing a p-value. A generator "passes" by clearing all tests at a
  pre-chosen significance level over a fixed sample size.

  The important caveat is that **passing NIST is necessary but not sufficient**.
  A well-seeded Mersenne Twister passes it. So does every commercial QRNG.
  NIST tells you a stream is not obviously broken; it does not tell you that
  two passing streams behave the same way under chaos- or information-theoretic
  scrutiny.
citation: "Bassham, L. et al. (2010). A Statistical Test Suite for Random and Pseudorandom Number Generators for Cryptographic Applications. NIST Special Publication 800-22 Revision 1a."
tags: [nist, randomness-testing, qrng, prng, cryptography]
publishedAt: 2026-05-17
---

NIST SP 800-22 Rev 1a is the baseline every QRNG audit starts from, and it is
the baseline the [QRNG Analysis Toolkit](https://github.com/vikingdude81/qrng-analysis-toolkit)
runs first on every source.

It matters to us for two reasons:

1. **It is the floor, not the ceiling.** Any QRNG that fails SP 800-22 is
   broken in a way you do not need a chaos-theoretic argument to see. Any QRNG
   that passes is *eligible* for the harder comparisons — BiEntropy, Lyapunov,
   classifier-based indistinguishability — that actually separate sources.
2. **It is the lingua franca.** When we publish cross-source comparisons, the
   first thing reviewers want to know is "did they all pass NIST?" Having the
   battery in the pipeline as a precondition makes every downstream claim
   easier to defend.

The toolkit's `tests/nist_tests.py` is a thin wrapper around the standard
implementation; reproducing the SP 800-22 baseline is part of the default
`run_full_qrng_analysis.py` pass.
