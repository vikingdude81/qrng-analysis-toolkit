#!/usr/bin/env python3
"""Comprehensive QRNG quality analysis with detailed statistical tests."""

import json
import numpy as np
from pathlib import Path
from scipy import stats
from datetime import datetime

# Try to import BiEntropy (optional)
try:
    from bientropy_metrics import bientropy_analysis
    BIENTROPY_AVAILABLE = True
except ImportError:
    BIENTROPY_AVAILABLE = False

# Friendly names
SOURCE_NAMES = {
    'outshift_qrng_api': 'Outshift SPDC',
    'anu_qrng_vacuum_fluctuation': 'ANU Vacuum',
    'cipherstone_qbert_conditioned': 'Cipherstone M1',
    'cipherstone_qbert_raw': 'Cipherstone M2 (Raw)',
    'cpu_hwrng_bcrypt': 'CPU RDRAND',
    'prng_mersenne_twister': 'PRNG (MT)'
}

# Quality tiers
SOURCE_TIERS = {
    'outshift_qrng_api': 'PRODUCTION',
    'anu_qrng_vacuum_fluctuation': 'PRODUCTION', 
    'cipherstone_qbert_conditioned': 'PRODUCTION',
    'cipherstone_qbert_raw': 'EXPERIMENTAL',
    'cpu_hwrng_bcrypt': 'CONTROL',
    'prng_mersenne_twister': 'CONTROL'
}

def load_streams():
    """Load all QRNG streams from files."""
    streams = {}
    streams_dir = Path(__file__).parent / 'qrng_streams'
    
    for f in sorted(streams_dir.glob('*.json')):
        try:
            with open(f) as fp:
                d = json.load(fp)
            source = d.get('source', 'unknown')
            if source == 'unknown' or source is None:
                continue
            floats = d.get('floats', [])
            if not floats:
                continue
            if source not in streams:
                streams[source] = []
            streams[source].extend(floats)
        except Exception:
            pass
    return streams

def calculate_autocorrelation(arr, lag=1):
    """Calculate autocorrelation at given lag."""
    arr_centered = arr - np.mean(arr)
    if len(arr) <= lag:
        return 0
    return np.corrcoef(arr_centered[:-lag], arr_centered[lag:])[0, 1]

def runs_test(arr):
    """Perform Wald-Wolfowitz runs test."""
    median = np.median(arr)
    binary = (arr > median).astype(int)
    runs = 1 + np.sum(binary[1:] != binary[:-1])
    n1 = np.sum(binary)
    n0 = len(binary) - n1
    expected_runs = 1 + 2*n1*n0 / (n1 + n0)
    var_runs = 2*n1*n0*(2*n1*n0 - n1 - n0) / ((n1+n0)**2 * (n1+n0-1))
    z = (runs - expected_runs) / np.sqrt(var_runs)
    p = 2 * (1 - stats.norm.cdf(abs(z)))
    return runs, expected_runs, z, p

def entropy_analysis(arr):
    """Calculate Shannon entropy of 8-bit discretization."""
    bins = np.floor(arr * 256).astype(int)
    bins = np.clip(bins, 0, 255)
    counts = np.bincount(bins, minlength=256)
    probs = counts / counts.sum()
    probs = probs[probs > 0]
    entropy = -np.sum(probs * np.log2(probs))
    return entropy, entropy / 8.0 * 100  # efficiency %

def chi_square_uniformity(arr, bins=20):
    """Chi-square test for uniformity."""
    observed, _ = np.histogram(arr, bins=bins, range=(0, 1))
    chi2, p = stats.chisquare(observed)
    return chi2, p

def main():
    print("=" * 80)
    print(f"DETAILED QRNG QUALITY ANALYSIS - {datetime.now().strftime('%B %d, %Y')}")
    print("=" * 80)
    
    streams = load_streams()
    
    if not streams:
        print("No QRNG streams found!")
        return
    
    total_samples = sum(len(s) for s in streams.values())
    print(f"\nLoaded {total_samples:,} samples from {len(streams)} sources\n")
    
    # Basic statistics
    print("SAMPLE COUNTS BY SOURCE")
    print("-" * 60)
    for source, samples in sorted(streams.items(), key=lambda x: -len(x[1])):
        name = SOURCE_NAMES.get(source, source)
        tier = SOURCE_TIERS.get(source, '?')
        print(f"  {name:<30} {len(samples):>8,}  [{tier}]")
    print(f"  {'TOTAL':<30} {total_samples:>8,}")
    
    print("\n" + "=" * 80)
    print("DESCRIPTIVE STATISTICS")
    print("=" * 80)
    header = f"{'Source':<25} {'N':>8} {'Mean':>8} {'Std':>8} {'Skew':>8} {'Kurt':>8} {'Tier':<12}"
    print(header)
    print("-" * 85)
    
    for source, samples in sorted(streams.items(), key=lambda x: -len(x[1])):
        arr = np.array(samples)
        name = SOURCE_NAMES.get(source, source)[:24]
        tier = SOURCE_TIERS.get(source, '?')
        print(f"{name:<25} {len(arr):>8,} {np.mean(arr):>8.4f} {np.std(arr):>8.4f} {stats.skew(arr):>8.4f} {stats.kurtosis(arr):>8.4f} {tier:<12}")
    
    # Expected values for uniform distribution
    print("\n  Expected (Uniform[0,1]): Mean=0.5000, Std=0.2887, Skew=0.0000, Kurt=-1.2000")
    
    # Uniformity test
    print("\n" + "=" * 80)
    print("UNIFORMITY TESTS (Chi-Square, 20 bins)")
    print("=" * 80)
    print(f"\n{'Source':<25} {'Chi-sq':>10} {'p-value':>10} {'Status':>12}")
    print("-" * 60)
    
    for source, samples in sorted(streams.items(), key=lambda x: -len(x[1])):
        arr = np.array(samples)
        name = SOURCE_NAMES.get(source, source)[:24]
        chi2, p = chi_square_uniformity(arr)
        status = "PASS" if p > 0.01 else "FAIL"
        print(f"{name:<25} {chi2:>10.2f} {p:>10.4f} {status:>12}")
    
    # Kolmogorov-Smirnov test
    print("\n" + "=" * 80)
    print("KOLMOGOROV-SMIRNOV TEST (vs Uniform[0,1])")
    print("=" * 80)
    print(f"\n{'Source':<25} {'KS stat':>10} {'p-value':>10} {'Status':>12}")
    print("-" * 60)
    
    for source, samples in sorted(streams.items(), key=lambda x: -len(x[1])):
        arr = np.array(samples)
        name = SOURCE_NAMES.get(source, source)[:24]
        ks_stat, p = stats.kstest(arr, 'uniform')
        status = "PASS" if p > 0.01 else "FAIL"
        print(f"{name:<25} {ks_stat:>10.4f} {p:>10.4f} {status:>12}")
    
    # Autocorrelation
    print("\n" + "=" * 80)
    print("SERIAL CORRELATION (Autocorrelation at Lags 1-5)")
    print("=" * 80)
    print(f"\n{'Source':<25} {'Lag-1':>8} {'Lag-2':>8} {'Lag-3':>8} {'Lag-4':>8} {'Lag-5':>8}")
    print("-" * 75)
    
    for source, samples in sorted(streams.items(), key=lambda x: -len(x[1])):
        arr = np.array(samples)
        name = SOURCE_NAMES.get(source, source)[:24]
        acfs = [calculate_autocorrelation(arr, lag) for lag in range(1, 6)]
        print(f"{name:<25} {acfs[0]:>8.4f} {acfs[1]:>8.4f} {acfs[2]:>8.4f} {acfs[3]:>8.4f} {acfs[4]:>8.4f}")
    
    # Critical value for 95% confidence
    print("\n  Critical value |r| < 0.02 for n=10000 (95% CI)")
    
    # Entropy analysis
    print("\n" + "=" * 80)
    print("BIT ENTROPY (Shannon entropy of 8-bit discretization)")
    print("=" * 80)
    print(f"\n{'Source':<25} {'H(bits)':>10} {'Max H':>10} {'Efficiency':>12}")
    print("-" * 60)
    
    for source, samples in sorted(streams.items(), key=lambda x: -len(x[1])):
        arr = np.array(samples)
        name = SOURCE_NAMES.get(source, source)[:24]
        entropy, efficiency = entropy_analysis(arr)
        status = "OK" if efficiency > 99 else "LOW" if efficiency > 95 else "BAD"
        print(f"{name:<25} {entropy:>10.4f} {8.0:>10.4f} {efficiency:>10.2f}% {status}")
    
    # Runs test
    print("\n" + "=" * 80)
    print("RUNS TEST (Wald-Wolfowitz)")
    print("=" * 80)
    print(f"\n{'Source':<25} {'Runs':>8} {'Expected':>10} {'Z-score':>10} {'p-value':>10} {'Status':>8}")
    print("-" * 80)
    
    for source, samples in sorted(streams.items(), key=lambda x: -len(x[1])):
        arr = np.array(samples)
        name = SOURCE_NAMES.get(source, source)[:24]
        runs, expected, z, p = runs_test(arr)
        status = "PASS" if p > 0.01 else "FAIL"
        print(f"{name:<25} {runs:>8,} {expected:>10.1f} {z:>10.3f} {p:>10.4f} {status:>8}")
    
    # Spectral analysis (simple)
    print("\n" + "=" * 80)
    print("SPECTRAL ANALYSIS (FFT Peak Detection)")
    print("=" * 80)
    print(f"\n{'Source':<25} {'DC Comp':>10} {'Max Peak':>10} {'Peak Freq':>10} {'Quality':>10}")
    print("-" * 75)
    
    for source, samples in sorted(streams.items(), key=lambda x: -len(x[1])):
        arr = np.array(samples[:4096])  # Use power of 2
        name = SOURCE_NAMES.get(source, source)[:24]
        arr_centered = arr - np.mean(arr)
        fft = np.abs(np.fft.fft(arr_centered))
        dc = fft[0] / len(arr)
        fft_no_dc = fft[1:len(arr)//2]
        max_peak = np.max(fft_no_dc) / len(arr)
        peak_freq = np.argmax(fft_no_dc) + 1
        # Good random should have flat spectrum (low peaks)
        quality = "Flat" if max_peak < 0.03 else "Peaks" if max_peak < 0.1 else "Strong"
        print(f"{name:<25} {dc:>10.4f} {max_peak:>10.4f} {peak_freq:>10} {quality:>10}")
    
    # BiEntropy analysis (if available)
    if BIENTROPY_AVAILABLE:
        print("\n" + "=" * 80)
        print("BIENTROPY ANALYSIS (Croll 2013 - Binary Derivative Entropy)")
        print("=" * 80)
        print(f"\n{'Source':<25} {'TBiEn':>10} {'BiEn':>10} {'Deriv H':>10} {'Rating':>12}")
        print("-" * 70)
        
        for source, samples in sorted(streams.items(), key=lambda x: -len(x[1])):
            arr = np.array(samples)
            name = SOURCE_NAMES.get(source, source)[:24]
            analysis = bientropy_analysis(arr)
            tbien = analysis['global_tbien_mean']
            
            if tbien >= 0.97:
                rating = "EXCELLENT"
            elif tbien >= 0.95:
                rating = "VERY GOOD"
            elif tbien >= 0.90:
                rating = "GOOD"
            elif tbien >= 0.50:
                rating = "FAIR"
            else:
                rating = "POOR"
            
            print(f"{name:<25} {tbien:>10.4f} {analysis['global_bien_mean']:>10.4f} {analysis['derivative_entropy_mean']:>10.4f} {rating:>12}")
        
        print("\n  TBiEn > 0.95 indicates true random behavior")
        print("  Deriv H ~1.0 means binary derivatives appear random")
    
    # Summary
    print("\n" + "=" * 80)
    print("QUALITY TIER SUMMARY")
    print("=" * 80)
    
    results = {}
    for source, samples in streams.items():
        arr = np.array(samples)
        chi2, chi_p = chi_square_uniformity(arr)
        _, _, runs_z, runs_p = runs_test(arr)
        acf1 = abs(calculate_autocorrelation(arr, 1))
        ks_stat, ks_p = stats.kstest(arr, 'uniform')
        entropy, eff = entropy_analysis(arr)
        
        # Score: number of tests passed
        passed = sum([
            chi_p > 0.01,
            runs_p > 0.01,
            acf1 < 0.02,
            ks_p > 0.01,
            eff > 99
        ])
        results[source] = {
            'samples': len(samples),
            'passed': passed,
            'chi_p': chi_p,
            'runs_p': runs_p,
            'acf1': acf1,
            'ks_p': ks_p,
            'entropy_eff': eff
        }
    
    print(f"\n{'Source':<25} {'Tests':>8} {'Tier':>15} {'Rating':>12}")
    print("-" * 65)
    
    ratings = {5: 'EXCELLENT', 4: 'GOOD', 3: 'MARGINAL', 2: 'POOR', 1: 'FAILING', 0: 'FAILING'}
    
    for source, r in sorted(results.items(), key=lambda x: (-x[1]['passed'], -x[1]['samples'])):
        name = SOURCE_NAMES.get(source, source)[:24]
        tier = SOURCE_TIERS.get(source, '?')
        rating = ratings[r['passed']]
        print(f"{name:<25} {r['passed']}/5{' ':>5} {tier:>15} {rating:>12}")
    
    print("\n" + "-" * 65)
    print("Tests: Chi-Square, Runs, ACF<0.02, K-S, Entropy>99%")
    print("Note: Cipherstone M2 (Raw) is unconditioned - failure expected")
    
    # Final statistics
    print("\n" + "=" * 80)
    print("COLLECTION SUMMARY")
    print("=" * 80)
    for source, samples in sorted(streams.items(), key=lambda x: -len(x[1])):
        name = SOURCE_NAMES.get(source, source)
        tier = SOURCE_TIERS.get(source, '?')
        print(f"  {name:<30} {len(samples):>8,}  [{tier}]")
    print(f"\n  {'TOTAL':<30} {total_samples:>8,} samples across {len(streams)} sources")


if __name__ == "__main__":
    main()
