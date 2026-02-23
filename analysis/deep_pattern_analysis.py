#!/usr/bin/env python3
"""
Deep Pattern Analysis for QRNG Data
Analyzes 43K+ samples looking for emergent patterns, temporal correlations,
cross-source relationships, and anomalies.
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import defaultdict
from scipy import stats, signal
from scipy.fft import fft, fftfreq
import warnings
warnings.filterwarnings('ignore')

# Try imports for visualization
try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False


def load_all_streams():
    """Load all QRNG streams with timestamps."""
    streams_dir = Path("qrng_streams")
    sources = defaultdict(list)
    
    for f in sorted(streams_dir.glob("*.json")):
        with open(f) as fp:
            data = json.load(fp)
        
        source = data.get("source", "unknown")
        # Try multiple keys for sample data
        samples = data.get("floats", data.get("samples", data.get("values", [])))
        timestamp = data.get("timestamp", f.stem[-15:])
        
        # Parse timestamp
        try:
            ts = datetime.strptime(timestamp[:15], "%Y%m%d_%H%M%S")
        except:
            ts = datetime.now()
        
        sources[source].append({
            "samples": np.array(samples),
            "timestamp": ts,
            "file": f.name,
            "n": len(samples)
        })
    
    return sources


def temporal_drift_analysis(sources):
    """Analyze if source statistics drift over time."""
    print("\n" + "="*80)
    print("TEMPORAL DRIFT ANALYSIS")
    print("="*80)
    print("Looking for statistical changes across collection sessions...\n")
    
    for source, runs in sources.items():
        if len(runs) < 2:
            continue
        
        # Sort by timestamp
        runs_sorted = sorted(runs, key=lambda x: x["timestamp"])
        
        means = [r["samples"].mean() for r in runs_sorted]
        stds = [r["samples"].std() for r in runs_sorted]
        timestamps = [r["timestamp"] for r in runs_sorted]
        
        # Linear regression on means over time
        if len(means) >= 3:
            x = np.arange(len(means))
            slope, intercept, r_value, p_value, std_err = stats.linregress(x, means)
            
            drift_status = "STABLE" if abs(slope) < 0.001 else ("DRIFTING UP" if slope > 0 else "DRIFTING DOWN")
            
            print(f"{source}:")
            print(f"  Sessions: {len(runs_sorted)}")
            print(f"  Mean range: {min(means):.4f} - {max(means):.4f}")
            print(f"  Trend slope: {slope:.6f} (p={p_value:.4f})")
            print(f"  Status: {drift_status}")
            print()


def cross_source_correlation(sources):
    """Analyze correlations between different QRNG sources."""
    print("\n" + "="*80)
    print("CROSS-SOURCE CORRELATION ANALYSIS")
    print("="*80)
    print("Testing if different quantum sources show any correlation...\n")
    
    # Combine all samples per source, filtering out empty sources
    combined = {}
    for source, runs in sources.items():
        valid_samples = [r["samples"] for r in runs if len(r["samples"]) > 0]
        if valid_samples:
            all_samples = np.concatenate(valid_samples)
            if len(all_samples) > 0:
                combined[source] = all_samples
    
    source_names = list(combined.keys())
    n_sources = len(source_names)
    
    if n_sources < 2:
        print("Insufficient sources for correlation analysis")
        return
    
    # Correlation matrix (using equal-length segments)
    min_len = min(len(combined[s]) for s in source_names)
    segment_len = min(min_len, 5000)  # Use up to 5000 samples
    
    if segment_len < 10:
        print("Insufficient samples for correlation analysis")
        return
    
    print(f"Using {segment_len} samples per source for correlation\n")
    
    corr_matrix = np.zeros((n_sources, n_sources))
    
    for i, s1 in enumerate(source_names):
        for j, s2 in enumerate(source_names):
            if i <= j:
                seg1 = combined[s1][:segment_len]
                seg2 = combined[s2][:segment_len]
                corr, _ = stats.pearsonr(seg1, seg2)
                corr_matrix[i, j] = corr
                corr_matrix[j, i] = corr
    
    # Print correlation matrix
    print("Cross-Source Pearson Correlation Matrix:")
    print("-" * 70)
    
    # Header
    header = "Source".ljust(25)
    for s in source_names:
        header += s[:8].rjust(10)
    print(header)
    print("-" * 70)
    
    for i, s1 in enumerate(source_names):
        row = s1[:24].ljust(25)
        for j, s2 in enumerate(source_names):
            corr = corr_matrix[i, j]
            if i == j:
                row += "   1.0000"
            else:
                row += f"{corr:10.4f}"
        print(row)
    
    # Flag significant correlations
    print("\nSignificant correlations (|r| > 0.05, excluding self):")
    found_significant = False
    for i in range(n_sources):
        for j in range(i+1, n_sources):
            if abs(corr_matrix[i, j]) > 0.05:
                print(f"  ⚠️  {source_names[i]} <-> {source_names[j]}: r = {corr_matrix[i, j]:.4f}")
                found_significant = True
    
    if not found_significant:
        print("  ✓ No significant cross-source correlations detected")
        print("    (Sources appear statistically independent)")


def long_range_autocorrelation(sources):
    """Analyze autocorrelation at longer lags."""
    print("\n" + "="*80)
    print("LONG-RANGE AUTOCORRELATION ANALYSIS")
    print("="*80)
    print("Testing for hidden patterns at extended lag distances...\n")
    
    lags = [1, 10, 50, 100, 500, 1000, 2000]
    
    for source, runs in sources.items():
        all_samples = np.concatenate([r["samples"] for r in runs])
        n = len(all_samples)
        
        if n < 3000:
            continue
        
        print(f"{source} (n={n:,}):")
        
        # Compute autocorrelation at various lags
        mean = all_samples.mean()
        var = all_samples.var()
        
        acf_values = []
        for lag in lags:
            if lag >= n:
                acf_values.append(np.nan)
                continue
            
            cov = np.mean((all_samples[:-lag] - mean) * (all_samples[lag:] - mean))
            acf = cov / var if var > 0 else 0
            acf_values.append(acf)
        
        # Print lag table
        print("  Lag:    ", end="")
        for lag in lags:
            print(f"{lag:8}", end="")
        print()
        
        print("  ACF:    ", end="")
        for acf in acf_values:
            if np.isnan(acf):
                print("     N/A", end="")
            else:
                status = "*" if abs(acf) > 2/np.sqrt(n) else " "
                print(f"{acf:7.4f}{status}", end="")
        print()
        
        # Check for any significant long-range correlation
        critical = 2 / np.sqrt(n)
        significant = [l for l, a in zip(lags, acf_values) if not np.isnan(a) and abs(a) > critical]
        
        if significant:
            print(f"  ⚠️  Significant ACF at lags: {significant}")
        else:
            print(f"  ✓ No significant long-range autocorrelation (critical |r| < {critical:.4f})")
        print()


def spectral_periodicity_search(sources):
    """Deep FFT analysis looking for hidden periodicities."""
    print("\n" + "="*80)
    print("SPECTRAL PERIODICITY SEARCH")
    print("="*80)
    print("Deep FFT analysis for hidden cycles or periodicities...\n")
    
    for source, runs in sources.items():
        all_samples = np.concatenate([r["samples"] for r in runs])
        n = len(all_samples)
        
        if n < 1000:
            continue
        
        # Use power of 2 for efficient FFT
        n_fft = 2 ** int(np.log2(n))
        samples = all_samples[:n_fft]
        
        # Remove mean (DC component)
        samples_centered = samples - samples.mean()
        
        # Compute FFT
        fft_vals = np.abs(fft(samples_centered))[:n_fft//2]
        freqs = fftfreq(n_fft, 1)[:n_fft//2]
        
        # Normalize
        fft_normalized = fft_vals / n_fft
        
        # Find peaks above noise floor
        noise_floor = np.median(fft_normalized) * 3
        peaks_idx = signal.find_peaks(fft_normalized, height=noise_floor, distance=10)[0]
        
        # Top 5 peaks (excluding DC)
        if len(peaks_idx) > 0:
            peaks_sorted = sorted(peaks_idx, key=lambda i: fft_normalized[i], reverse=True)[:5]
            
            print(f"{source} (n={n:,}, FFT size={n_fft:,}):")
            print(f"  Noise floor: {noise_floor:.6f}")
            print(f"  Top spectral peaks:")
            
            for idx in peaks_sorted:
                freq = freqs[idx]
                period = 1/freq if freq > 0 else np.inf
                power = fft_normalized[idx]
                snr = power / noise_floor
                print(f"    Freq {freq:.4f} (period ~{period:.0f} samples): power={power:.4f}, SNR={snr:.1f}x")
            
            if max(fft_normalized[p] for p in peaks_sorted) > noise_floor * 5:
                print(f"  ⚠️  Strong spectral component detected!")
            else:
                print(f"  ✓ No dominant periodicities (flat spectrum)")
        else:
            print(f"{source}: ✓ Flat spectrum, no peaks detected")
        
        print()


def runs_length_distribution(sources):
    """Analyze distribution of runs (consecutive above/below median)."""
    print("\n" + "="*80)
    print("RUNS LENGTH DISTRIBUTION")
    print("="*80)
    print("Analyzing consecutive run lengths for pattern detection...\n")
    
    for source, runs_data in sources.items():
        all_samples = np.concatenate([r["samples"] for r in runs_data])
        n = len(all_samples)
        
        if n < 1000:
            continue
        
        # Convert to binary (above/below median)
        median = np.median(all_samples)
        binary = (all_samples > median).astype(int)
        
        # Find run lengths
        run_lengths = []
        current_run = 1
        for i in range(1, len(binary)):
            if binary[i] == binary[i-1]:
                current_run += 1
            else:
                run_lengths.append(current_run)
                current_run = 1
        run_lengths.append(current_run)
        
        run_lengths = np.array(run_lengths)
        
        # Expected for random: geometric distribution with p=0.5
        # E[run length] = 2, Var = 2
        mean_run = run_lengths.mean()
        max_run = run_lengths.max()
        
        # Distribution of run lengths
        unique, counts = np.unique(run_lengths, return_counts=True)
        
        print(f"{source} (n={n:,}):")
        print(f"  Total runs: {len(run_lengths):,}")
        print(f"  Mean run length: {mean_run:.3f} (expected: 2.0)")
        print(f"  Max run length: {max_run}")
        
        # Expected max run for n samples: ~log2(n)
        expected_max = np.log2(n) + 2
        if max_run > expected_max * 1.5:
            print(f"  ⚠️  Unusually long run detected (expected max ~{expected_max:.0f})")
        else:
            print(f"  ✓ Run lengths within expected bounds")
        
        # Distribution summary
        print(f"  Run length distribution:")
        for length in [1, 2, 3, 4, 5, 6, 7, 8]:
            count = counts[unique == length][0] if length in unique else 0
            pct = 100 * count / len(run_lengths)
            expected_pct = 100 * (0.5 ** length)
            diff = pct - expected_pct
            marker = "⚠️" if abs(diff) > 3 else "  "
            print(f"    Length {length}: {pct:5.1f}% (expected {expected_pct:5.1f}%) {marker}")
        
        print()


def byte_pattern_analysis(sources):
    """Look for recurring byte patterns in the data."""
    print("\n" + "="*80)
    print("BYTE PATTERN ANALYSIS")
    print("="*80)
    print("Searching for recurring patterns in 8-bit discretization...\n")
    
    for source, runs in sources.items():
        all_samples = np.concatenate([r["samples"] for r in runs])
        n = len(all_samples)
        
        if n < 1000 or "Raw" in source:
            continue
        
        # Convert to bytes (0-255)
        bytes_data = (all_samples * 255).astype(np.uint8)
        
        # Frequency analysis
        unique, counts = np.unique(bytes_data, return_counts=True)
        freq = counts / n
        
        # Chi-square for uniformity
        expected = n / 256
        chi2 = np.sum((counts - expected)**2 / expected)
        chi2_pval = 1 - stats.chi2.cdf(chi2, df=255)
        
        # Entropy
        entropy = -np.sum(freq * np.log2(freq + 1e-10))
        
        # Most/least common bytes
        sorted_idx = np.argsort(counts)[::-1]
        
        print(f"{source} (n={n:,}):")
        print(f"  Byte entropy: {entropy:.4f} bits (max 8.0)")
        print(f"  Chi-square: {chi2:.1f} (p={chi2_pval:.4f})")
        
        print(f"  Most common bytes: ", end="")
        for i in sorted_idx[:5]:
            print(f"0x{unique[i]:02X}({counts[i]})", end=" ")
        print()
        
        print(f"  Least common bytes: ", end="")
        for i in sorted_idx[-5:]:
            print(f"0x{unique[i]:02X}({counts[i]})", end=" ")
        print()
        
        # Check for suspicious patterns
        max_freq = counts.max() / n
        min_freq = counts.min() / n
        expected_freq = 1/256
        
        if max_freq > expected_freq * 1.5 or min_freq < expected_freq * 0.5:
            print(f"  ⚠️  Non-uniform byte distribution")
        else:
            print(f"  ✓ Byte distribution uniform")
        
        print()


def gap_test(sources):
    """Gap test - analyze gaps between occurrences of specific ranges."""
    print("\n" + "="*80)
    print("GAP TEST ANALYSIS")
    print("="*80)
    print("Analyzing gaps between occurrences in specific ranges...\n")
    
    for source, runs in sources.items():
        all_samples = np.concatenate([r["samples"] for r in runs])
        n = len(all_samples)
        
        if n < 2000 or "Raw" in source:
            continue
        
        # Test gaps for values in [0, 0.1)
        alpha, beta = 0.0, 0.1
        in_range = (all_samples >= alpha) & (all_samples < beta)
        
        # Find gaps
        gaps = []
        current_gap = 0
        for in_r in in_range:
            if in_r:
                gaps.append(current_gap)
                current_gap = 0
            else:
                current_gap += 1
        
        if len(gaps) < 10:
            continue
        
        gaps = np.array(gaps[1:])  # Skip first incomplete gap
        
        # Expected: geometric distribution with p = beta - alpha = 0.1
        p = beta - alpha
        expected_mean = (1 - p) / p  # = 9 for p=0.1
        
        actual_mean = gaps.mean()
        actual_std = gaps.std()
        
        # Chi-square test for geometric distribution
        max_gap = min(50, gaps.max())
        observed = np.array([np.sum(gaps == g) for g in range(max_gap + 1)])
        expected_dist = len(gaps) * stats.geom.pmf(np.arange(max_gap + 1) + 1, p)
        
        # Combine small expected values
        observed_combined = observed[observed > 0]
        expected_combined = expected_dist[observed > 0]
        
        if len(observed_combined) > 1 and len(expected_combined) > 1:
            # Normalize expected to match observed sum
            expected_combined = expected_combined * (observed_combined.sum() / expected_combined.sum())
            chi2, pval = stats.chisquare(observed_combined, expected_combined)
        else:
            chi2, pval = 0, 1.0
        
        print(f"{source}:")
        print(f"  Testing gaps for values in [{alpha}, {beta})")
        print(f"  Number of gaps: {len(gaps):,}")
        print(f"  Mean gap: {actual_mean:.2f} (expected: {expected_mean:.2f})")
        print(f"  Std gap: {actual_std:.2f}")
        print(f"  Chi-square: {chi2:.1f} (p={pval:.4f})")
        
        if pval < 0.01:
            print(f"  ⚠️  Gap distribution deviates from expected")
        else:
            print(f"  ✓ Gap distribution matches geometric (p={p})")
        
        print()


def inter_session_stability(sources):
    """Analyze stability between collection sessions."""
    print("\n" + "="*80)
    print("INTER-SESSION STABILITY ANALYSIS")
    print("="*80)
    print("Testing if quality metrics are stable across collection sessions...\n")
    
    for source, runs in sources.items():
        if len(runs) < 3:
            continue
        
        runs_sorted = sorted(runs, key=lambda x: x["timestamp"])
        
        metrics_per_session = []
        for run in runs_sorted:
            samples = run["samples"]
            if len(samples) < 100:
                continue
            
            # Compute quality metrics
            mean = samples.mean()
            std = samples.std()
            
            # Autocorrelation lag-1
            acf1 = np.corrcoef(samples[:-1], samples[1:])[0, 1]
            
            # Entropy
            hist, _ = np.histogram(samples, bins=256, range=(0, 1))
            freq = hist / hist.sum()
            entropy = -np.sum(freq * np.log2(freq + 1e-10))
            
            metrics_per_session.append({
                "timestamp": run["timestamp"],
                "n": len(samples),
                "mean": mean,
                "std": std,
                "acf1": acf1,
                "entropy": entropy
            })
        
        if len(metrics_per_session) < 3:
            continue
        
        print(f"{source} ({len(metrics_per_session)} sessions):")
        
        # Compute variation in metrics
        means = [m["mean"] for m in metrics_per_session]
        stds = [m["std"] for m in metrics_per_session]
        acfs = [m["acf1"] for m in metrics_per_session]
        entropies = [m["entropy"] for m in metrics_per_session]
        
        print(f"  Mean:    {np.mean(means):.4f} ± {np.std(means):.4f} (CV: {100*np.std(means)/np.mean(means):.2f}%)")
        print(f"  Std:     {np.mean(stds):.4f} ± {np.std(stds):.4f}")
        print(f"  ACF(1):  {np.mean(acfs):.4f} ± {np.std(acfs):.4f}")
        print(f"  Entropy: {np.mean(entropies):.4f} ± {np.std(entropies):.4f}")
        
        # Flag high variation
        cv_mean = np.std(means) / np.mean(means) if np.mean(means) > 0 else 0
        if cv_mean > 0.02:
            print(f"  ⚠️  High variation in mean across sessions")
        else:
            print(f"  ✓ Stable across sessions")
        
        print()


def quantum_vs_classical_comparison(sources):
    """Statistical comparison of quantum vs classical sources."""
    print("\n" + "="*80)
    print("QUANTUM VS CLASSICAL COMPARISON")
    print("="*80)
    print("Comparing quantum sources against classical controls...\n")
    
    quantum_sources = ["outshift_qrng_api", "anu_qrng_vacuum_fluctuation", 
                       "cipherstone_qbert_m1_conditioned"]
    classical_sources = ["cpu_hwrng_bcrypt", "prng_mersenne_twister"]
    
    quantum_data = []
    classical_data = []
    
    for source, runs in sources.items():
        all_samples = np.concatenate([r["samples"] for r in runs])
        
        if any(q in source.lower().replace(" ", "_") for q in ["outshift", "anu", "cipherstone"]):
            if "raw" not in source.lower():
                quantum_data.append((source, all_samples))
        elif any(c in source.lower().replace(" ", "_") for c in ["cpu", "prng", "mersenne"]):
            classical_data.append((source, all_samples))
    
    if not quantum_data or not classical_data:
        print("Insufficient data for comparison")
        return
    
    # Combine quantum and classical
    all_quantum = np.concatenate([d[1] for d in quantum_data])
    all_classical = np.concatenate([d[1] for d in classical_data])
    
    print(f"Quantum pool: {len(all_quantum):,} samples from {len(quantum_data)} sources")
    print(f"Classical pool: {len(all_classical):,} samples from {len(classical_data)} sources")
    print()
    
    # Statistical comparison
    print("Metric Comparison:")
    print("-" * 60)
    
    metrics = [
        ("Mean", np.mean(all_quantum), np.mean(all_classical)),
        ("Std Dev", np.std(all_quantum), np.std(all_classical)),
        ("Skewness", stats.skew(all_quantum), stats.skew(all_classical)),
        ("Kurtosis", stats.kurtosis(all_quantum), stats.kurtosis(all_classical)),
    ]
    
    print(f"{'Metric':<15} {'Quantum':>12} {'Classical':>12} {'Diff':>12}")
    print("-" * 60)
    for name, q_val, c_val in metrics:
        diff = q_val - c_val
        print(f"{name:<15} {q_val:>12.6f} {c_val:>12.6f} {diff:>+12.6f}")
    
    # Two-sample tests
    print()
    print("Two-Sample Statistical Tests:")
    print("-" * 60)
    
    # Use subsamples for efficiency
    n_test = min(10000, len(all_quantum), len(all_classical))
    q_sample = np.random.choice(all_quantum, n_test, replace=False)
    c_sample = np.random.choice(all_classical, n_test, replace=False)
    
    # K-S test
    ks_stat, ks_pval = stats.ks_2samp(q_sample, c_sample)
    print(f"Kolmogorov-Smirnov: D={ks_stat:.4f}, p={ks_pval:.4f}")
    
    # Mann-Whitney U test
    mw_stat, mw_pval = stats.mannwhitneyu(q_sample, c_sample)
    print(f"Mann-Whitney U:     U={mw_stat:.0f}, p={mw_pval:.4f}")
    
    # Anderson-Darling
    ad_stat, _, ad_pval = stats.anderson_ksamp([q_sample[:5000], c_sample[:5000]])
    print(f"Anderson-Darling:   A²={ad_stat:.4f}, p≈{ad_pval:.4f}")
    
    print()
    if ks_pval > 0.05 and mw_pval > 0.05:
        print("✓ No significant difference between quantum and classical sources")
        print("  Both produce statistically indistinguishable random distributions")
    else:
        print("⚠️  Detectable difference between quantum and classical sources")


def main():
    print("="*80)
    print("DEEP PATTERN ANALYSIS - QRNG DATA")
    print("="*80)
    print(f"Analysis Date: {datetime.now().strftime('%B %d, %Y %H:%M')}")
    print()
    
    # Load all data
    sources = load_all_streams()
    
    total = sum(sum(r["n"] for r in runs) for runs in sources.values())
    print(f"Loaded {total:,} samples from {len(sources)} sources")
    
    for source, runs in sorted(sources.items()):
        n = sum(r["n"] for r in runs)
        sessions = len(runs)
        print(f"  {source}: {n:,} samples ({sessions} sessions)")
    
    # Run all analyses
    temporal_drift_analysis(sources)
    cross_source_correlation(sources)
    long_range_autocorrelation(sources)
    spectral_periodicity_search(sources)
    runs_length_distribution(sources)
    byte_pattern_analysis(sources)
    gap_test(sources)
    inter_session_stability(sources)
    quantum_vs_classical_comparison(sources)
    
    print("\n" + "="*80)
    print("ANALYSIS COMPLETE")
    print("="*80)


if __name__ == "__main__":
    main()
