"""
Quantum Hardware Key — Standard Randomness Test Battery
========================================================

Runs the same 7 tests from the reference table against the
48,279 quantum hardware 256-bit keys (386,232 floats → 3,089,856 bits).

Tests:
  1. Monobit Frequency       (NIST SP 800-22 §2.1)
  2. Block Frequency          (NIST SP 800-22 §2.2)
  3. Runs Test                (NIST SP 800-22 §2.3)
  4. Chi-Square (Bytes)       (Byte-level uniformity)
  5. Shannon Entropy           (Information density per byte)
  6. Monte Carlo Pi            (Geometric randomness via π estimation)
  7. Compression Ratio         (Kolmogorov complexity proxy via zlib)
"""

import json
import math
import struct
import zlib
import numpy as np
from pathlib import Path
from scipy import special, stats

STREAM_DIR = Path("qrng_streams")


def load_qhw_floats():
    """Load all quantum hardware floats."""
    floats = []
    for f in sorted(STREAM_DIR.glob("qrng_stream_qhw_*.json")):
        d = json.loads(f.read_text())
        floats.extend(d["floats"])
    return np.array(floats)


def floats_to_bits(floats):
    """Convert floats → raw bytes → bit array."""
    # Pack each float as 8-byte double, extract all bits
    raw = b""
    for v in floats:
        raw += struct.pack(">d", v)
    bits = np.unpackbits(np.frombuffer(raw, dtype=np.uint8))
    return bits


def floats_to_bytes(floats):
    """Convert floats → byte stream (quantize to 0-255)."""
    return (floats * 256).astype(np.uint8)


# ════════════════════════════════════════════════════════════════
#  Test 1: Monobit Frequency  (NIST SP 800-22 §2.1)
# ════════════════════════════════════════════════════════════════
def test_monobit(bits):
    n = len(bits)
    s = np.sum(2 * bits.astype(np.int64) - 1)
    s_obs = abs(s) / math.sqrt(n)
    p = float(special.erfc(s_obs / math.sqrt(2)))
    return {
        "name": "Monobit Frequency",
        "result": f"P = {p:.4f}",
        "passed": p >= 0.01,
        "checks": "Checks if the number of 0s and 1s is equal. (Ideal P≥0.01)",
        "p": p,
    }


# ════════════════════════════════════════════════════════════════
#  Test 2: Block Frequency  (NIST SP 800-22 §2.2)
# ════════════════════════════════════════════════════════════════
def test_block_frequency(bits, M=128):
    n = len(bits)
    N = n // M
    chi2 = 0.0
    for i in range(N):
        block = bits[i * M : (i + 1) * M]
        pi = np.sum(block) / M
        chi2 += (pi - 0.5) ** 2
    chi2 *= 4 * M
    p = float(special.gammaincc(N / 2, chi2 / 2))
    return {
        "name": "Block Frequency",
        "result": f"P = {p:.4f}",
        "passed": p >= 0.01,
        "checks": "Checks if the density of 1s is consistent across local blocks.",
        "p": p,
    }


# ════════════════════════════════════════════════════════════════
#  Test 3: Runs Test  (NIST SP 800-22 §2.3)
# ════════════════════════════════════════════════════════════════
def test_runs(bits):
    n = len(bits)
    pi = np.sum(bits) / n

    # Pre-test
    tau = 2 / math.sqrt(n)
    if abs(pi - 0.5) >= tau:
        return {
            "name": "Runs Test",
            "result": "P = 0.0000 (pre-test fail)",
            "passed": False,
            "checks": "Checks for streaks of consecutive bits (e.g., 000 or 111).",
            "p": 0.0,
        }

    # Count runs
    runs = 1 + np.sum(bits[1:] != bits[:-1])

    num = abs(runs - 2 * n * pi * (1 - pi))
    den = 2 * math.sqrt(2 * n) * pi * (1 - pi)
    p = float(special.erfc(num / den))
    return {
        "name": "Runs Test",
        "result": f"P = {p:.4f}",
        "passed": p >= 0.01,
        "checks": "Checks for streaks of consecutive bits (e.g., 000 or 111).",
        "p": p,
    }


# ════════════════════════════════════════════════════════════════
#  Test 4: Chi-Square (Bytes)
# ════════════════════════════════════════════════════════════════
def test_chi_square_bytes(byte_data):
    observed = np.bincount(byte_data, minlength=256)
    expected = len(byte_data) / 256.0
    chi2 = np.sum((observed - expected) ** 2 / expected)
    p = float(1 - stats.chi2.cdf(chi2, df=255))
    return {
        "name": "Chi-Square (Bytes)",
        "result": f"P = {p:.4f}",
        "passed": p >= 0.01,
        "checks": "Verifies that all 256 byte values (0-255) appear with equal frequency.",
        "p": p,
    }


# ════════════════════════════════════════════════════════════════
#  Test 5: Shannon Entropy
# ════════════════════════════════════════════════════════════════
def test_shannon_entropy(byte_data):
    counts = np.bincount(byte_data, minlength=256)
    probs = counts / len(byte_data)
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return {
        "name": "Shannon Entropy",
        "result": f"{entropy:.4f} bits",
        "passed": entropy >= 7.9,
        "checks": "Measures information density per byte (Ideal is 8.0).",
        "p": entropy,
    }


# ════════════════════════════════════════════════════════════════
#  Test 6: Monte Carlo Pi
# ════════════════════════════════════════════════════════════════
def test_monte_carlo_pi(floats):
    n = len(floats) // 2
    x = floats[: 2 * n : 2]
    y = floats[1 : 2 * n : 2]
    inside = np.sum(x ** 2 + y ** 2 <= 1.0)
    pi_est = 4.0 * inside / n
    error_pct = abs(pi_est - math.pi) / math.pi * 100
    return {
        "name": "Monte Carlo Pi",
        "result": f"{pi_est:.4f}",
        "passed": error_pct < 1.0,
        "checks": f"Estimates π using random coordinate pairs (Error ≈{error_pct:.1f}%).",
        "p": pi_est,
    }


# ════════════════════════════════════════════════════════════════
#  Test 7: Compression Ratio
# ════════════════════════════════════════════════════════════════
def test_compression_ratio(byte_data):
    raw = byte_data.tobytes()
    compressed = zlib.compress(raw, 9)
    ratio = len(compressed) / len(raw)
    return {
        "name": "Compression Ratio",
        "result": f"{ratio:.4f}",
        "passed": ratio >= 0.99,
        "checks": "Attempts to compress the data; failure to compress proves high entropy.",
        "p": ratio,
    }


# ════════════════════════════════════════════════════════════════
#  Main
# ════════════════════════════════════════════════════════════════
def main():
    print("=" * 72)
    print("  Quantum Hardware 256-bit Keys — Standard Randomness Battery")
    print("=" * 72)

    print("\nLoading quantum hardware data...")
    floats = load_qhw_floats()
    print(f"  {len(floats):,} float values")

    # Convert to bits (use quantized bytes → bits for NIST tests)
    byte_data = floats_to_bytes(floats)
    bits = np.unpackbits(byte_data)
    print(f"  {len(byte_data):,} bytes → {len(bits):,} bits")

    print("\nRunning tests...\n")

    results = [
        test_monobit(bits),
        test_block_frequency(bits),
        test_runs(bits),
        test_chi_square_bytes(byte_data),
        test_shannon_entropy(byte_data),
        test_monte_carlo_pi(floats),
        test_compression_ratio(byte_data),
    ]

    # ── Pretty table ──
    header = f"{'Test Name':<24} {'Result / P-Value':<20} {'Verdict':<8} {'What it Checks'}"
    print(header)
    print("-" * len(header))
    for r in results:
        verdict = "PASS" if r["passed"] else "FAIL"
        print(f"{r['name']:<24} {r['result']:<20} {verdict:<8} {r['checks']}")

    passed = sum(1 for r in results if r["passed"])
    total = len(results)
    print("-" * len(header))
    print(f"\nResult:  {passed}/{total} tests passed")

    if passed == total:
        print("★  All tests PASS — data is consistent with true randomness.")
    else:
        failed = [r["name"] for r in results if not r["passed"]]
        print(f"⚠  Failed: {', '.join(failed)}")

    # Save JSON
    out = {
        "source": "quantum_hardware_256bit",
        "n_floats": int(len(floats)),
        "n_bytes": int(len(byte_data)),
        "n_bits": int(len(bits)),
        "tests": [
            {"name": r["name"], "result": r["result"], "passed": bool(r["passed"]), "checks": r["checks"]}
            for r in results
        ],
        "passed": passed,
        "total": total,
    }
    out_path = Path("qrng_analysis_results") / "qhw_test_battery.json"
    out_path.parent.mkdir(exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2))
    print(f"\nResults saved → {out_path}")


if __name__ == "__main__":
    main()
