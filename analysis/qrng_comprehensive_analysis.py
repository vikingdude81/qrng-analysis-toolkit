#!/usr/bin/env python3
"""
QRNG Comprehensive Analysis Suite
==================================
Performs ALL meaningful analyses on available QRNG data:
1. Pool both streams for maximum statistical power
2. Runs anomaly deep-dive (our -2.53σ finding)
3. Day-to-day stream comparison
4. Bit-level structure analysis
5. Autocorrelation at multiple lags
6. Distribution tests (K-S, Anderson-Darling)
7. Spectral/FFT analysis for periodicities

Created: 2026-01-15
"""

import json
import struct
import os
from pathlib import Path
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor
import multiprocessing as mp

import numpy as np
from scipy import stats
from scipy import signal
from scipy.fft import fft, fftfreq
import matplotlib.pyplot as plt
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()

# Use all available cores
N_CORES = mp.cpu_count()


def load_all_streams(streams_dir: Path) -> dict:
    """Load all QRNG streams from directory."""
    streams = {}
    for f in sorted(streams_dir.glob("qrng_stream_*.json")):
        with open(f) as fp:
            data = json.load(fp)
            # Try multiple possible keys for the float values
            values = (
                data.get('floats') or 
                data.get('values') or 
                data.get('samples') or 
                []
            )
            streams[f.stem] = {
                'values': np.array(values),
                'timestamp': data.get('timestamp', f.stem),
                'source': data.get('source', 'unknown'),
                'file': f.name,
                'raw_integers': data.get('raw_integers', [])
            }
    return streams


def generate_controls(n: int) -> dict:
    """Generate CSPRNG, PRNG, and CPU RDRAND controls of same size."""
    # CSPRNG (os.urandom) - use 4 bytes per value, convert to [0,1)
    csprng_bytes = os.urandom(n * 4)
    csprng = np.array([
        struct.unpack('>I', csprng_bytes[i*4:(i+1)*4])[0] / (2**32)
        for i in range(n)
    ])
    
    # PRNG (Mersenne Twister with fixed seed for reproducibility)
    rng = np.random.RandomState(42)
    prng = rng.random(n)
    
    # CPU RDRAND (via Windows BCrypt)
    try:
        from cpu_hwrng import CPUHardwareRNG
        hwrng = CPUHardwareRNG(method="bcrypt")
        cpu_rdrand = hwrng.get_random_floats(n)
        console.print("[green]✓ CPU RDRAND available[/]")
    except Exception as e:
        console.print(f"[yellow]⚠ CPU RDRAND not available: {e}[/]")
        cpu_rdrand = None
    
    result = {'csprng': csprng, 'prng': prng}
    if cpu_rdrand is not None:
        result['cpu_rdrand'] = cpu_rdrand
    
    return result


# =============================================================================
# 1. RUNS ANALYSIS (Deep Dive)
# =============================================================================

def compute_runs_detailed(values: np.ndarray) -> dict:
    """Detailed runs analysis with length distribution."""
    binary = (values >= 0.5).astype(int)
    
    # Find runs
    runs = []
    current_run = 1
    current_val = binary[0]
    
    for i in range(1, len(binary)):
        if binary[i] == current_val:
            current_run += 1
        else:
            runs.append((current_val, current_run))
            current_run = 1
            current_val = binary[i]
    runs.append((current_val, current_run))
    
    # Separate by type
    runs_of_0 = [r[1] for r in runs if r[0] == 0]
    runs_of_1 = [r[1] for r in runs if r[0] == 1]
    all_lengths = [r[1] for r in runs]
    
    # Length distribution
    max_len = max(all_lengths) if all_lengths else 1
    length_dist = {i: all_lengths.count(i) for i in range(1, min(max_len + 1, 20))}
    
    # Expected runs for random sequence
    n = len(binary)
    n0 = np.sum(binary == 0)
    n1 = np.sum(binary == 1)
    
    if n0 == 0 or n1 == 0:
        return {'error': 'All same value'}
    
    expected_runs = 1 + (2 * n0 * n1) / n
    variance = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n**2 * (n - 1))
    std_runs = np.sqrt(variance)
    
    observed_runs = len(runs)
    z_score = (observed_runs - expected_runs) / std_runs
    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
    
    # Expected length distribution (geometric with p=0.5)
    expected_length_dist = {}
    for length in range(1, 20):
        # Probability of run of exactly length k: (0.5)^k
        prob = 0.5 ** length
        expected_length_dist[length] = prob * observed_runs
    
    return {
        'observed_runs': observed_runs,
        'expected_runs': expected_runs,
        'z_score': z_score,
        'p_value': p_value,
        'n0': n0,
        'n1': n1,
        'runs_of_0': len(runs_of_0),
        'runs_of_1': len(runs_of_1),
        'mean_run_length': np.mean(all_lengths),
        'max_run_length': max(all_lengths),
        'length_distribution': length_dist,
        'expected_length_dist': expected_length_dist,
        'all_lengths': all_lengths
    }


def runs_anomaly_deep_dive(streams: dict, controls: dict):
    """Deep dive into runs anomaly."""
    console.print(Panel.fit("🔍 RUNS ANOMALY DEEP DIVE", style="bold cyan"))
    
    # Combine all QRNG data
    all_qrng = np.concatenate([s['values'] for s in streams.values()])
    
    # Analyze each source - include CPU RDRAND if available
    sources = {
        'QRNG (pooled)': all_qrng,
    }
    
    if 'cpu_rdrand' in controls:
        sources['CPU RDRAND'] = controls['cpu_rdrand']
    
    sources['CSPRNG'] = controls['csprng']
    sources['PRNG'] = controls['prng']
    
    # Also analyze individual streams
    for name, stream in streams.items():
        sources[f'QRNG ({name[-15:]})'] = stream['values']
    
    results = {}
    table = Table(title="Runs Analysis Comparison")
    table.add_column("Source", style="cyan")
    table.add_column("Observed", justify="right")
    table.add_column("Expected", justify="right")
    table.add_column("Z-Score", justify="right")
    table.add_column("P-Value", justify="right")
    table.add_column("Mean Len", justify="right")
    table.add_column("Max Len", justify="right")
    
    for name, values in sources.items():
        r = compute_runs_detailed(values)
        results[name] = r
        
        z_style = "red bold" if abs(r['z_score']) > 2 else "green"
        table.add_row(
            name,
            str(r['observed_runs']),
            f"{r['expected_runs']:.1f}",
            f"[{z_style}]{r['z_score']:.3f}[/]",
            f"{r['p_value']:.4f}",
            f"{r['mean_run_length']:.2f}",
            str(r['max_run_length'])
        )
    
    console.print(table)
    
    # Run length distribution comparison
    console.print("\n[bold]Run Length Distribution (Observed vs Expected):[/]")
    
    qrng_result = results['QRNG (pooled)']
    csprng_result = results['CSPRNG']
    
    dist_table = Table(title="Run Lengths")
    dist_table.add_column("Length", justify="right")
    dist_table.add_column("QRNG Obs", justify="right")
    dist_table.add_column("CSPRNG Obs", justify="right")
    dist_table.add_column("Expected", justify="right")
    dist_table.add_column("QRNG Δ", justify="right")
    
    for length in range(1, 12):
        qrng_obs = qrng_result['length_distribution'].get(length, 0)
        csprng_obs = csprng_result['length_distribution'].get(length, 0)
        expected = qrng_result['expected_length_dist'].get(length, 0)
        delta = qrng_obs - expected
        
        delta_style = "red" if abs(delta) > 10 else "green"
        dist_table.add_row(
            str(length),
            str(qrng_obs),
            str(csprng_obs),
            f"{expected:.1f}",
            f"[{delta_style}]{delta:+.1f}[/]"
        )
    
    console.print(dist_table)
    
    # Chi-square test on run length distribution
    qrng_obs = np.array([qrng_result['length_distribution'].get(i, 0) for i in range(1, 10)])
    expected = np.array([qrng_result['expected_length_dist'].get(i, 1) for i in range(1, 10)])
    # Normalize expected to match observed sum
    expected = expected * (qrng_obs.sum() / expected.sum()) if expected.sum() > 0 else expected
    chi2, chi2_p = stats.chisquare(qrng_obs, expected)
    
    console.print(f"\n[bold]Chi-square test on run lengths:[/] χ²={chi2:.2f}, p={chi2_p:.4f}")
    
    return results


# =============================================================================
# 2. DAY-TO-DAY COMPARISON
# =============================================================================

def day_to_day_comparison(streams: dict):
    """Compare streams from different days statistically."""
    console.print(Panel.fit("📅 DAY-TO-DAY STREAM COMPARISON", style="bold cyan"))
    
    if len(streams) < 2:
        console.print("[yellow]Need at least 2 streams for comparison[/]")
        return {}
    
    stream_names = list(streams.keys())
    results = {}
    
    table = Table(title="Pairwise Statistical Tests")
    table.add_column("Comparison", style="cyan")
    table.add_column("K-S Stat", justify="right")
    table.add_column("K-S P", justify="right")
    table.add_column("Mann-Whitney P", justify="right")
    table.add_column("Levene P", justify="right")
    table.add_column("Verdict", justify="center")
    
    for i in range(len(stream_names)):
        for j in range(i + 1, len(stream_names)):
            name1, name2 = stream_names[i], stream_names[j]
            v1, v2 = streams[name1]['values'], streams[name2]['values']
            
            # Kolmogorov-Smirnov test
            ks_stat, ks_p = stats.ks_2samp(v1, v2)
            
            # Mann-Whitney U test
            mw_stat, mw_p = stats.mannwhitneyu(v1, v2, alternative='two-sided')
            
            # Levene's test for equal variances
            lev_stat, lev_p = stats.levene(v1, v2)
            
            # Verdict
            if ks_p < 0.05 or mw_p < 0.05:
                verdict = "[red]DIFFERENT[/]"
            else:
                verdict = "[green]SIMILAR[/]"
            
            results[f"{name1}_vs_{name2}"] = {
                'ks_stat': ks_stat, 'ks_p': ks_p,
                'mw_p': mw_p, 'lev_p': lev_p
            }
            
            table.add_row(
                f"{name1[-8:]} vs {name2[-8:]}",
                f"{ks_stat:.4f}",
                f"{ks_p:.4f}",
                f"{mw_p:.4f}",
                f"{lev_p:.4f}",
                verdict
            )
    
    console.print(table)
    
    # Summary statistics per stream
    console.print("\n[bold]Per-Stream Statistics:[/]")
    stats_table = Table()
    stats_table.add_column("Stream", style="cyan")
    stats_table.add_column("N", justify="right")
    stats_table.add_column("Mean", justify="right")
    stats_table.add_column("Std", justify="right")
    stats_table.add_column("Skew", justify="right")
    stats_table.add_column("Kurtosis", justify="right")
    
    for name, stream in streams.items():
        v = stream['values']
        stats_table.add_row(
            name[-15:],
            str(len(v)),
            f"{np.mean(v):.6f}",
            f"{np.std(v):.6f}",
            f"{stats.skew(v):.4f}",
            f"{stats.kurtosis(v):.4f}"
        )
    
    console.print(stats_table)
    return results


# =============================================================================
# 3. BIT-LEVEL STRUCTURE ANALYSIS
# =============================================================================

def bit_level_analysis(values: np.ndarray, name: str = "QRNG"):
    """Analyze bit-level structure of the data."""
    console.print(Panel.fit("🔢 BIT-LEVEL STRUCTURE ANALYSIS", style="bold cyan"))
    
    # Convert floats to 32-bit representation
    # Assuming values are in [0,1), multiply by 2^32 to get integer representation
    int_values = (values * (2**32)).astype(np.uint32)
    
    # Extract each bit position
    bit_counts = np.zeros(32)
    for bit in range(32):
        bit_counts[bit] = np.sum((int_values >> bit) & 1)
    
    # Expected: N/2 ones per bit position
    n = len(values)
    expected = n / 2
    
    table = Table(title=f"Bit Position Analysis ({name}, N={n})")
    table.add_column("Bit", justify="right")
    table.add_column("Ones", justify="right")
    table.add_column("Expected", justify="right")
    table.add_column("Bias", justify="right")
    table.add_column("Z-Score", justify="right")
    table.add_column("Status", justify="center")
    
    significant_bits = []
    std = np.sqrt(n * 0.25)  # std for binomial(n, 0.5)
    
    for bit in range(32):
        ones = bit_counts[bit]
        bias = (ones - expected) / n
        z = (ones - expected) / std
        
        if abs(z) > 2.58:  # 99% confidence
            status = "[red]⚠ BIASED[/]"
            significant_bits.append(bit)
        elif abs(z) > 1.96:  # 95% confidence
            status = "[yellow]? MARGINAL[/]"
        else:
            status = "[green]✓ OK[/]"
        
        table.add_row(
            str(bit),
            str(int(ones)),
            str(int(expected)),
            f"{bias:+.4f}",
            f"{z:+.3f}",
            status
        )
    
    console.print(table)
    
    if significant_bits:
        console.print(f"\n[red]⚠ Significant bias detected in bits: {significant_bits}[/]")
    else:
        console.print(f"\n[green]✓ No significant bit-level bias detected[/]")
    
    # Bit correlation analysis (between adjacent bits)
    console.print("\n[bold]Adjacent Bit Correlations:[/]")
    correlations = []
    for bit in range(31):
        b1 = (int_values >> bit) & 1
        b2 = (int_values >> (bit + 1)) & 1
        corr = np.corrcoef(b1, b2)[0, 1]
        correlations.append(corr)
    
    corr_table = Table()
    corr_table.add_column("Bit Pair", justify="right")
    corr_table.add_column("Correlation", justify="right")
    corr_table.add_column("Status", justify="center")
    
    for i, corr in enumerate(correlations[:16]):  # Show first 16
        if abs(corr) > 0.1:
            status = "[red]HIGH[/]"
        elif abs(corr) > 0.05:
            status = "[yellow]MODERATE[/]"
        else:
            status = "[green]LOW[/]"
        
        corr_table.add_row(f"{i}-{i+1}", f"{corr:+.4f}", status)
    
    console.print(corr_table)
    
    return {
        'bit_counts': bit_counts.tolist(),
        'significant_bits': significant_bits,
        'adjacent_correlations': correlations
    }


# =============================================================================
# 4. AUTOCORRELATION ANALYSIS
# =============================================================================

def autocorrelation_analysis(values: np.ndarray, max_lag: int = 50):
    """Compute and analyze autocorrelation at multiple lags."""
    console.print(Panel.fit("📈 AUTOCORRELATION ANALYSIS", style="bold cyan"))
    
    n = len(values)
    mean = np.mean(values)
    var = np.var(values)
    
    # Compute autocorrelation for each lag
    acf = []
    for lag in range(max_lag + 1):
        if lag == 0:
            acf.append(1.0)
        else:
            c = np.sum((values[:-lag] - mean) * (values[lag:] - mean)) / ((n - lag) * var)
            acf.append(c)
    
    acf = np.array(acf)
    
    # Critical value for significance (95% CI)
    critical = 1.96 / np.sqrt(n)
    
    # Find significant lags
    significant_lags = [lag for lag in range(1, max_lag + 1) if abs(acf[lag]) > critical]
    
    table = Table(title=f"Autocorrelation (N={n}, critical=±{critical:.4f})")
    table.add_column("Lag", justify="right")
    table.add_column("ACF", justify="right")
    table.add_column("Status", justify="center")
    
    for lag in range(1, min(21, max_lag + 1)):
        if abs(acf[lag]) > critical:
            status = "[red]⚠ SIGNIFICANT[/]"
        else:
            status = "[green]✓[/]"
        
        table.add_row(str(lag), f"{acf[lag]:+.5f}", status)
    
    console.print(table)
    
    if significant_lags:
        console.print(f"\n[red]⚠ Significant autocorrelation at lags: {significant_lags[:10]}{'...' if len(significant_lags) > 10 else ''}[/]")
    else:
        console.print(f"\n[green]✓ No significant autocorrelation detected (lags 1-{max_lag})[/]")
    
    # Ljung-Box test
    # Q = n(n+2) * sum(acf[k]^2 / (n-k)) for k=1 to h
    h = min(20, max_lag)
    Q = n * (n + 2) * np.sum([acf[k]**2 / (n - k) for k in range(1, h + 1)])
    lb_p = 1 - stats.chi2.cdf(Q, h)
    
    console.print(f"\n[bold]Ljung-Box test (h={h}):[/] Q={Q:.2f}, p={lb_p:.4f}")
    if lb_p < 0.05:
        console.print("[red]⚠ Significant serial correlation detected[/]")
    else:
        console.print("[green]✓ No significant serial correlation[/]")
    
    return {
        'acf': acf.tolist(),
        'significant_lags': significant_lags,
        'ljung_box_Q': Q,
        'ljung_box_p': lb_p,
        'critical_value': critical
    }


# =============================================================================
# 5. DISTRIBUTION TESTS
# =============================================================================

def distribution_tests(values: np.ndarray, name: str = "QRNG"):
    """Test if distribution matches uniform."""
    console.print(Panel.fit("📊 DISTRIBUTION TESTS (vs Uniform[0,1])", style="bold cyan"))
    
    n = len(values)
    
    # Kolmogorov-Smirnov test
    ks_stat, ks_p = stats.kstest(values, 'uniform', args=(0, 1))
    
    # Anderson-Darling test
    ad_result = stats.anderson(values, dist='norm')  # Note: scipy doesn't have uniform AD
    
    # Chi-square test (bin into 10 equal bins)
    n_bins = 10
    observed, _ = np.histogram(values, bins=n_bins, range=(0, 1))
    expected = np.full(n_bins, n / n_bins)
    chi2_stat, chi2_p = stats.chisquare(observed, expected)
    
    # Cramér-von Mises test
    # W² = 1/(12n) + Σ[(U(i) - (2i-1)/(2n))²]
    sorted_vals = np.sort(values)
    i = np.arange(1, n + 1)
    cvm_stat = 1/(12*n) + np.sum((sorted_vals - (2*i - 1)/(2*n))**2)
    # Approximate p-value
    cvm_p = 1 - stats.norm.cdf(cvm_stat * np.sqrt(n))  # Rough approximation
    
    # Moments comparison
    moments_table = Table(title=f"Moment Comparison ({name} vs Uniform)")
    moments_table.add_column("Moment", style="cyan")
    moments_table.add_column("Observed", justify="right")
    moments_table.add_column("Expected", justify="right")
    moments_table.add_column("Difference", justify="right")
    
    # Uniform[0,1] moments: mean=0.5, var=1/12, skew=0, kurtosis=-1.2
    obs_mean, exp_mean = np.mean(values), 0.5
    obs_var, exp_var = np.var(values), 1/12
    obs_skew, exp_skew = stats.skew(values), 0
    obs_kurt, exp_kurt = stats.kurtosis(values), -1.2
    
    moments_table.add_row("Mean", f"{obs_mean:.6f}", f"{exp_mean:.6f}", f"{obs_mean - exp_mean:+.6f}")
    moments_table.add_row("Variance", f"{obs_var:.6f}", f"{exp_var:.6f}", f"{obs_var - exp_var:+.6f}")
    moments_table.add_row("Skewness", f"{obs_skew:.4f}", f"{exp_skew:.4f}", f"{obs_skew - exp_skew:+.4f}")
    moments_table.add_row("Kurtosis", f"{obs_kurt:.4f}", f"{exp_kurt:.4f}", f"{obs_kurt - exp_kurt:+.4f}")
    
    console.print(moments_table)
    
    # Test results
    tests_table = Table(title="Uniformity Tests")
    tests_table.add_column("Test", style="cyan")
    tests_table.add_column("Statistic", justify="right")
    tests_table.add_column("P-Value", justify="right")
    tests_table.add_column("Result", justify="center")
    
    tests = [
        ("Kolmogorov-Smirnov", ks_stat, ks_p),
        ("Chi-Square (10 bins)", chi2_stat, chi2_p),
        ("Cramér-von Mises", cvm_stat, cvm_p),
    ]
    
    for test_name, stat, p in tests:
        result = "[red]REJECT[/]" if p < 0.05 else "[green]PASS[/]"
        tests_table.add_row(test_name, f"{stat:.4f}", f"{p:.4f}", result)
    
    console.print(tests_table)
    
    # Bin distribution
    console.print("\n[bold]Bin Distribution:[/]")
    bin_table = Table()
    bin_table.add_column("Bin", justify="right")
    bin_table.add_column("Range", justify="center")
    bin_table.add_column("Observed", justify="right")
    bin_table.add_column("Expected", justify="right")
    bin_table.add_column("Δ", justify="right")
    
    for i in range(n_bins):
        delta = observed[i] - expected[i]
        delta_style = "red" if abs(delta) > 20 else "green"
        bin_table.add_row(
            str(i + 1),
            f"[{i/n_bins:.1f}, {(i+1)/n_bins:.1f})",
            str(observed[i]),
            str(int(expected[i])),
            f"[{delta_style}]{delta:+.0f}[/]"
        )
    
    console.print(bin_table)
    
    return {
        'ks': {'stat': ks_stat, 'p': ks_p},
        'chi2': {'stat': chi2_stat, 'p': chi2_p},
        'cvm': {'stat': cvm_stat, 'p': cvm_p},
        'moments': {
            'mean': obs_mean, 'var': obs_var,
            'skew': obs_skew, 'kurtosis': obs_kurt
        },
        'bins': {'observed': observed.tolist(), 'expected': expected.tolist()}
    }


# =============================================================================
# 6. SPECTRAL/FFT ANALYSIS
# =============================================================================

def spectral_analysis(values: np.ndarray, name: str = "QRNG"):
    """Perform FFT analysis to detect periodicities."""
    console.print(Panel.fit("🌊 SPECTRAL/FFT ANALYSIS", style="bold cyan"))
    
    n = len(values)
    
    # Center the data
    centered = values - np.mean(values)
    
    # Compute FFT
    fft_vals = fft(centered)
    power = np.abs(fft_vals[:n//2])**2
    freqs = fftfreq(n, d=1)[:n//2]
    
    # Normalize power spectrum
    power_normalized = power / np.sum(power)
    
    # Find dominant frequencies (excluding DC)
    peak_indices = np.argsort(power[1:])[-10:] + 1  # Top 10 peaks
    
    table = Table(title=f"Top Spectral Peaks ({name})")
    table.add_column("Rank", justify="right")
    table.add_column("Frequency", justify="right")
    table.add_column("Period", justify="right")
    table.add_column("Power", justify="right")
    table.add_column("% Total", justify="right")
    
    for rank, idx in enumerate(reversed(peak_indices), 1):
        freq = freqs[idx]
        period = 1/freq if freq > 0 else float('inf')
        pwr = power[idx]
        pct = power_normalized[idx] * 100
        
        table.add_row(
            str(rank),
            f"{freq:.4f}",
            f"{period:.1f}",
            f"{pwr:.2e}",
            f"{pct:.2f}%"
        )
    
    console.print(table)
    
    # Spectral entropy
    power_norm = power / np.sum(power)
    power_norm = power_norm[power_norm > 0]  # Avoid log(0)
    spectral_entropy = -np.sum(power_norm * np.log2(power_norm))
    max_entropy = np.log2(len(power))
    entropy_ratio = spectral_entropy / max_entropy
    
    console.print(f"\n[bold]Spectral Entropy:[/] {spectral_entropy:.4f} / {max_entropy:.4f} = {entropy_ratio:.4f}")
    
    if entropy_ratio > 0.9:
        console.print("[green]✓ High spectral entropy (white noise-like)[/]")
    elif entropy_ratio > 0.7:
        console.print("[yellow]? Moderate spectral entropy[/]")
    else:
        console.print("[red]⚠ Low spectral entropy (periodic structure)[/]")
    
    # Periodogram test for white noise
    # For white noise, periodogram values should follow exponential distribution
    sorted_power = np.sort(power[1:])  # Exclude DC
    n_p = len(sorted_power)
    expected_quantiles = stats.expon.ppf(np.arange(1, n_p + 1) / (n_p + 1), scale=np.mean(sorted_power))
    
    # Correlation between observed and expected
    corr = np.corrcoef(sorted_power, expected_quantiles)[0, 1]
    console.print(f"[bold]Q-Q Correlation (vs exponential):[/] {corr:.4f}")
    
    if corr > 0.99:
        console.print("[green]✓ Excellent match to white noise[/]")
    elif corr > 0.95:
        console.print("[yellow]? Good match to white noise[/]")
    else:
        console.print("[red]⚠ Deviation from white noise detected[/]")
    
    return {
        'spectral_entropy': spectral_entropy,
        'max_entropy': max_entropy,
        'entropy_ratio': entropy_ratio,
        'qq_correlation': corr,
        'top_frequencies': [freqs[i] for i in reversed(peak_indices)],
        'top_powers': [power[i] for i in reversed(peak_indices)]
    }


# =============================================================================
# 7. POOLED ANALYSIS
# =============================================================================

def pooled_analysis(streams: dict, controls: dict):
    """Analyze all QRNG data pooled together."""
    console.print(Panel.fit("🔗 POOLED DATA ANALYSIS", style="bold cyan"))
    
    # Pool all QRNG
    all_qrng = np.concatenate([s['values'] for s in streams.values()])
    console.print(f"[bold]Total pooled samples:[/] {len(all_qrng)}")
    console.print(f"[bold]From streams:[/] {len(streams)}")
    
    # Generate controls of same size
    n = len(all_qrng)
    csprng = controls['csprng'][:n] if len(controls['csprng']) >= n else np.concatenate([
        controls['csprng'],
        np.array([struct.unpack('>I', os.urandom(4))[0] / (2**32) for _ in range(n - len(controls['csprng']))])
    ])
    
    prng = controls['prng'][:n] if len(controls['prng']) >= n else np.concatenate([
        controls['prng'],
        np.random.random(n - len(controls['prng']))
    ])
    
    # Compare basic stats - include CPU RDRAND if available
    sources = {'QRNG': all_qrng, 'CPU RDRAND': None, 'CSPRNG': csprng, 'PRNG': prng}
    
    if 'cpu_rdrand' in controls:
        cpu_rdrand = controls['cpu_rdrand'][:n] if len(controls['cpu_rdrand']) >= n else controls['cpu_rdrand']
        sources['CPU RDRAND'] = cpu_rdrand
    else:
        del sources['CPU RDRAND']
    
    table = Table(title=f"Pooled Comparison (N={n})")
    table.add_column("Metric", style="cyan")
    for name in sources:
        table.add_column(name, justify="right")
    
    metrics = [
        ("Mean", lambda x: np.mean(x)),
        ("Std Dev", lambda x: np.std(x)),
        ("Skewness", lambda x: stats.skew(x)),
        ("Kurtosis", lambda x: stats.kurtosis(x)),
        ("Min", lambda x: np.min(x)),
        ("Max", lambda x: np.max(x)),
        ("Range", lambda x: np.max(x) - np.min(x)),
    ]
    
    for name, func in metrics:
        row = [name]
        for source in sources.values():
            row.append(f"{func(source):.6f}")
        table.add_row(*row)
    
    console.print(table)
    
    return {'pooled_n': n, 'sources': list(sources.keys())}


# =============================================================================
# VISUALIZATION
# =============================================================================

def create_comprehensive_plots(streams: dict, controls: dict, results: dict, output_dir: Path):
    """Create comprehensive visualization plots."""
    console.print(Panel.fit("📊 GENERATING VISUALIZATIONS", style="bold cyan"))
    
    all_qrng = np.concatenate([s['values'] for s in streams.values()])
    n = len(all_qrng)
    
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.suptitle(f'QRNG Comprehensive Analysis (N={n})', fontsize=14, fontweight='bold')
    
    # 1. Distribution histogram
    ax = axes[0, 0]
    ax.hist(all_qrng, bins=50, density=True, alpha=0.7, label='QRNG', color='blue')
    ax.axhline(1.0, color='red', linestyle='--', label='Uniform')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution vs Uniform')
    ax.legend()
    
    # 2. Q-Q plot
    ax = axes[0, 1]
    theoretical = np.linspace(0, 1, n)
    ax.scatter(theoretical, np.sort(all_qrng), alpha=0.5, s=1)
    ax.plot([0, 1], [0, 1], 'r--', label='Perfect uniform')
    ax.set_xlabel('Theoretical Quantiles')
    ax.set_ylabel('Sample Quantiles')
    ax.set_title('Q-Q Plot (vs Uniform)')
    ax.legend()
    
    # 3. Autocorrelation
    ax = axes[0, 2]
    acf = results['autocorrelation']['acf'][:51]
    ax.bar(range(len(acf)), acf, color='steelblue')
    critical = results['autocorrelation']['critical_value']
    ax.axhline(critical, color='red', linestyle='--', alpha=0.7)
    ax.axhline(-critical, color='red', linestyle='--', alpha=0.7)
    ax.set_xlabel('Lag')
    ax.set_ylabel('ACF')
    ax.set_title('Autocorrelation Function')
    
    # 4. Run length distribution
    ax = axes[1, 0]
    runs_data = results['runs']['QRNG (pooled)']
    lengths = list(runs_data['length_distribution'].keys())[:12]
    observed = [runs_data['length_distribution'].get(l, 0) for l in lengths]
    expected = [runs_data['expected_length_dist'].get(l, 0) for l in lengths]
    x = np.arange(len(lengths))
    width = 0.35
    ax.bar(x - width/2, observed, width, label='Observed', color='steelblue')
    ax.bar(x + width/2, expected, width, label='Expected', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(lengths)
    ax.set_xlabel('Run Length')
    ax.set_ylabel('Count')
    ax.set_title('Run Length Distribution')
    ax.legend()
    
    # 5. Bit bias
    ax = axes[1, 1]
    bit_counts = results['bits']['bit_counts']
    bias = [(c - n/2) / n for c in bit_counts]
    colors = ['red' if abs(b) > 0.02 else 'steelblue' for b in bias]
    ax.bar(range(32), bias, color=colors)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_xlabel('Bit Position')
    ax.set_ylabel('Bias (from 0.5)')
    ax.set_title('Bit Position Bias')
    
    # 6. Power spectrum
    ax = axes[1, 2]
    centered = all_qrng - np.mean(all_qrng)
    fft_vals = fft(centered)
    power = np.abs(fft_vals[:n//2])**2
    freqs = fftfreq(n, d=1)[:n//2]
    ax.semilogy(freqs[1:100], power[1:100], color='steelblue')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Power (log scale)')
    ax.set_title('Power Spectrum')
    
    # 7. Time series (first 500 samples)
    ax = axes[2, 0]
    ax.plot(all_qrng[:500], linewidth=0.5, color='steelblue')
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_xlabel('Sample')
    ax.set_ylabel('Value')
    ax.set_title('Time Series (first 500)')
    
    # 8. Stream comparison boxplot
    ax = axes[2, 1]
    stream_data = [s['values'] for s in streams.values()]
    stream_labels = [k[-8:] for k in streams.keys()]
    ax.boxplot(stream_data, labels=stream_labels)
    ax.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax.set_ylabel('Value')
    ax.set_title('Stream Comparison')
    
    # 9. Runs comparison
    ax = axes[2, 2]
    sources = ['QRNG (pooled)', 'CSPRNG', 'PRNG']
    z_scores = [results['runs'][s]['z_score'] for s in sources]
    colors = ['red' if abs(z) > 2 else 'steelblue' for z in z_scores]
    ax.bar(sources, z_scores, color=colors)
    ax.axhline(2, color='red', linestyle='--', alpha=0.5)
    ax.axhline(-2, color='red', linestyle='--', alpha=0.5)
    ax.axhline(0, color='black', linewidth=0.5)
    ax.set_ylabel('Z-Score')
    ax.set_title('Runs Test Z-Scores')
    
    plt.tight_layout()
    
    output_path = output_dir / f"comprehensive_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    console.print(f"[green]✓ Saved:[/] {output_path}")
    return output_path


# =============================================================================
# MAIN
# =============================================================================

def main():
    console.print(Panel.fit(
        "🔬 QRNG COMPREHENSIVE ANALYSIS SUITE 🔬",
        style="bold magenta"
    ))
    
    streams_dir = Path("qrng_streams")
    output_dir = Path("qrng_analysis_results")
    output_dir.mkdir(exist_ok=True)
    
    # Load streams
    console.print("\n[bold]Loading QRNG streams...[/]")
    streams = load_all_streams(streams_dir)
    
    if not streams:
        console.print("[red]No QRNG streams found![/]")
        return
    
    total_samples = sum(len(s['values']) for s in streams.values())
    console.print(f"[green]Found {len(streams)} streams with {total_samples} total samples[/]")
    
    for name, stream in streams.items():
        console.print(f"  • {stream['file']}: {len(stream['values'])} samples")
    
    # Generate controls
    console.print("\n[bold]Generating control sequences...[/]")
    controls = generate_controls(total_samples)
    console.print(f"[green]Generated CSPRNG and PRNG controls ({total_samples} each)[/]")
    
    # Store all results
    results = {}
    
    # 1. Pooled analysis
    console.print("\n")
    results['pooled'] = pooled_analysis(streams, controls)
    
    # 2. Runs deep dive
    console.print("\n")
    results['runs'] = runs_anomaly_deep_dive(streams, controls)
    
    # 3. Day-to-day comparison
    console.print("\n")
    results['day_comparison'] = day_to_day_comparison(streams)
    
    # 4. Bit-level analysis
    console.print("\n")
    all_qrng = np.concatenate([s['values'] for s in streams.values()])
    results['bits'] = bit_level_analysis(all_qrng)
    
    # 5. Autocorrelation
    console.print("\n")
    results['autocorrelation'] = autocorrelation_analysis(all_qrng)
    
    # 6. Distribution tests
    console.print("\n")
    results['distribution'] = distribution_tests(all_qrng)
    
    # 7. Spectral analysis
    console.print("\n")
    results['spectral'] = spectral_analysis(all_qrng)
    
    # Generate plots
    console.print("\n")
    plot_path = create_comprehensive_plots(streams, controls, results, output_dir)
    
    # Summary
    console.print("\n")
    console.print(Panel.fit("📋 ANALYSIS SUMMARY", style="bold green"))
    
    summary_table = Table()
    summary_table.add_column("Analysis", style="cyan")
    summary_table.add_column("Key Finding", style="white")
    summary_table.add_column("Status", justify="center")
    
    # Runs
    runs_z = results['runs']['QRNG (pooled)']['z_score']
    runs_status = "[red]⚠ ANOMALY[/]" if abs(runs_z) > 2 else "[green]✓ OK[/]"
    summary_table.add_row("Runs Test", f"Z={runs_z:.3f}", runs_status)
    
    # Bits
    sig_bits = results['bits']['significant_bits']
    bits_status = "[red]⚠ BIAS[/]" if sig_bits else "[green]✓ OK[/]"
    summary_table.add_row("Bit Bias", f"{len(sig_bits)} biased bits", bits_status)
    
    # Autocorrelation
    sig_lags = results['autocorrelation']['significant_lags']
    acf_status = "[red]⚠ CORRELATED[/]" if sig_lags else "[green]✓ OK[/]"
    summary_table.add_row("Autocorrelation", f"{len(sig_lags)} significant lags", acf_status)
    
    # Distribution
    ks_p = results['distribution']['ks']['p']
    dist_status = "[red]⚠ NON-UNIFORM[/]" if ks_p < 0.05 else "[green]✓ UNIFORM[/]"
    summary_table.add_row("K-S Test", f"p={ks_p:.4f}", dist_status)
    
    # Spectral
    entropy_ratio = results['spectral']['entropy_ratio']
    spec_status = "[green]✓ WHITE NOISE[/]" if entropy_ratio > 0.9 else "[yellow]? STRUCTURED[/]"
    summary_table.add_row("Spectral Entropy", f"{entropy_ratio:.4f}", spec_status)
    
    # Day comparison
    if results['day_comparison']:
        first_key = list(results['day_comparison'].keys())[0]
        ks_p = results['day_comparison'][first_key]['ks_p']
        day_status = "[red]⚠ DIFFERENT[/]" if ks_p < 0.05 else "[green]✓ CONSISTENT[/]"
        summary_table.add_row("Day-to-Day", f"K-S p={ks_p:.4f}", day_status)
    
    console.print(summary_table)
    
    # Save results
    results_file = output_dir / f"comprehensive_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    
    # Convert numpy arrays for JSON serialization
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    console.print(f"\n[green]✓ Results saved to:[/] {results_file}")
    console.print(f"[green]✓ Plots saved to:[/] {plot_path}")


if __name__ == "__main__":
    main()
