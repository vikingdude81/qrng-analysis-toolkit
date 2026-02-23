#!/usr/bin/env python3
"""
QRNG Deep Dive Analysis v2 - With Proper Statistical Methodology

Fixes from v1 based on expert feedback:
1. CENTER data before Hurst computation (subtract 0.5 for uniform [0,1])
2. Fix null model to use same centering
3. Add CSPRNG control comparison (os.urandom)
4. Add permutation control (shuffle to test temporal vs distributional)
5. Clarify data format in output
6. Add recommended metrics: bit bias, min-entropy estimate, change-point detection

Key insight: Computing Hurst on uniform [0,1] values directly creates
artificial persistence because of positive mean. Must use zero-mean increments.
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import scipy.stats as stats
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os
import secrets

# Get CPU count for parallel processing
N_WORKERS = min(multiprocessing.cpu_count(), 64)

# Import our analysis modules
from helios_anomaly_scope import (
    compute_hurst_exponent,
    compute_lyapunov_exponent,
    compute_msd_from_trajectory,
    compute_runs_test,
    compute_spectral_entropy
)


def load_qrng_stream(filepath: str) -> Tuple[np.ndarray, dict]:
    """Load QRNG stream and return raw data with metadata."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if 'raw_integers' in data:
        raw = np.array(data['raw_integers'], dtype=np.uint32)
        # Store both raw and normalized
        data['_raw_uint32'] = raw
        data['_floats'] = raw.astype(np.float64) / (2**32)
        data['_format'] = 'uint32'
        data['_bits_per_sample'] = 32
        values = data['_floats']
    elif 'floats' in data:
        values = np.array(data['floats'])
        data['_floats'] = values
        data['_format'] = 'float64'
        data['_bits_per_sample'] = 64
    else:
        raise ValueError("Unknown data format")
    
    return values, data


def center_for_hurst(values: np.ndarray) -> np.ndarray:
    """
    Convert uniform [0,1] values to zero-mean increments for proper Hurst analysis.
    
    For uniform [0,1]: subtract 0.5 to get mean ~0
    This prevents artificial persistence from positive-mean accumulation.
    """
    return values - 0.5


def generate_csprng_control(n_samples: int) -> np.ndarray:
    """Generate control stream from OS CSPRNG (os.urandom via secrets)."""
    # Generate 32-bit integers from CSPRNG
    raw = np.array([
        int.from_bytes(secrets.token_bytes(4), 'little') 
        for _ in range(n_samples)
    ], dtype=np.uint32)
    return raw.astype(np.float64) / (2**32)


def generate_permutation_control(values: np.ndarray) -> np.ndarray:
    """Shuffle the QRNG values to destroy temporal structure while preserving distribution."""
    shuffled = values.copy()
    np.random.shuffle(shuffled)
    return shuffled


def _single_bootstrap_hurst_centered(args):
    """Single bootstrap iteration with CENTERED data."""
    series_centered, block_size, seed = args
    np.random.seed(seed)
    n = len(series_centered)
    n_blocks = n // block_size
    
    if n_blocks < 2:
        return None
    
    block_indices = np.random.randint(0, n_blocks, size=n_blocks)
    resampled = np.concatenate([
        series_centered[i*block_size:(i+1)*block_size] 
        for i in block_indices
    ])
    
    h = compute_hurst_exponent(resampled)
    return h if 0 < h < 1 else None


def _single_null_hurst_centered(args):
    """Single null hypothesis with CENTERED random data."""
    n_samples, seed = args
    np.random.seed(seed)
    # Generate uniform [0,1] then CENTER it
    null_series = np.random.random(n_samples) - 0.5  # CENTERED!
    h = compute_hurst_exponent(null_series)
    return h if 0 < h < 1 else None


def bootstrap_hurst_parallel(series_centered: np.ndarray, n_bootstrap: int = 1000, 
                              block_size: int = 50) -> Tuple[float, float, float, np.ndarray]:
    """
    Bootstrap confidence interval for Hurst exponent.
    EXPECTS ALREADY CENTERED DATA.
    """
    args = [(series_centered, block_size, i) for i in range(n_bootstrap)]
    
    hurst_samples = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(_single_bootstrap_hurst_centered, arg) for arg in args]
        for future in as_completed(futures):
            h = future.result()
            if h is not None:
                hurst_samples.append(h)
    
    hurst_samples = np.array(hurst_samples)
    
    if len(hurst_samples) == 0:
        return 0.5, 0.5, 0.5, np.array([0.5])
    
    mean_h = np.mean(hurst_samples)
    ci_low = np.percentile(hurst_samples, 2.5)
    ci_high = np.percentile(hurst_samples, 97.5)
    
    return mean_h, ci_low, ci_high, hurst_samples


def test_hurst_significance_parallel(h_observed: float, n_samples: int, 
                                      n_simulations: int = 500) -> Tuple[float, np.ndarray]:
    """
    Test if observed Hurst differs from properly-centered random null.
    """
    args = [(n_samples, i + 10000) for i in range(n_simulations)]
    
    null_hursts = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(_single_null_hurst_centered, arg) for arg in args]
        for future in as_completed(futures):
            h = future.result()
            if h is not None:
                null_hursts.append(h)
    
    null_hursts = np.array(null_hursts)
    
    if len(null_hursts) == 0:
        return 1.0, np.array([0.5])
    
    # Two-tailed p-value
    if h_observed > 0.5:
        p_value = np.mean(null_hursts >= h_observed) * 2
    else:
        p_value = np.mean(null_hursts <= h_observed) * 2
    
    return min(p_value, 1.0), null_hursts


def compute_bit_bias(values: np.ndarray) -> Dict[str, float]:
    """
    Compute bit-level bias metrics.
    For uint32 data, analyze individual bit positions.
    """
    # Convert floats back to bits if needed
    if values.dtype == np.float64:
        # Assume [0,1) floats from uint32
        uint_vals = (values * (2**32)).astype(np.uint64)
    else:
        uint_vals = values.astype(np.uint64)
    
    n_samples = len(uint_vals)
    
    # Overall bit bias (proportion of 1s)
    total_bits = 0
    total_ones = 0
    
    bit_biases = []
    for bit_pos in range(32):
        bit_vals = (uint_vals >> bit_pos) & 1
        p1 = np.mean(bit_vals)
        bit_biases.append(p1)
        total_ones += np.sum(bit_vals)
        total_bits += n_samples
    
    overall_p1 = total_ones / total_bits
    
    # Worst bit bias (furthest from 0.5)
    worst_bias = max(abs(p - 0.5) for p in bit_biases)
    worst_bit = np.argmax([abs(p - 0.5) for p in bit_biases])
    
    return {
        'overall_p1': overall_p1,
        'bias_from_half': abs(overall_p1 - 0.5),
        'worst_bit_bias': worst_bias,
        'worst_bit_position': int(worst_bit),
        'bit_biases': bit_biases
    }


def compute_min_entropy_estimate(values: np.ndarray, n_bins: int = 256) -> float:
    """
    Estimate min-entropy (worst-case entropy).
    H_min = -log2(p_max) where p_max is the highest probability in the distribution.
    
    For perfectly uniform: H_min = log2(n_bins)
    """
    # Bin the data
    hist, _ = np.histogram(values, bins=n_bins, range=(0, 1))
    probs = hist / np.sum(hist)
    
    # Max probability
    p_max = np.max(probs)
    
    if p_max > 0:
        h_min = -np.log2(p_max)
    else:
        h_min = 0
    
    # Expected for uniform: log2(n_bins)
    h_max = np.log2(n_bins)
    
    return h_min, h_max, h_min / h_max  # raw, expected, ratio


def detect_change_points(values: np.ndarray, window_size: int = 100) -> List[int]:
    """
    Simple change-point detection using rolling mean shifts.
    Returns indices where significant mean shifts occur.
    """
    if len(values) < window_size * 2:
        return []
    
    change_points = []
    threshold = 3.0  # z-score threshold
    
    for i in range(window_size, len(values) - window_size):
        left = values[i-window_size:i]
        right = values[i:i+window_size]
        
        # Two-sample t-test statistic
        mean_diff = abs(np.mean(left) - np.mean(right))
        pooled_std = np.sqrt((np.var(left) + np.var(right)) / 2)
        
        if pooled_std > 0:
            z = mean_diff / (pooled_std / np.sqrt(window_size))
            if z > threshold:
                change_points.append(i)
    
    # Cluster nearby change points
    if len(change_points) > 1:
        clustered = [change_points[0]]
        for cp in change_points[1:]:
            if cp - clustered[-1] > window_size:
                clustered.append(cp)
        return clustered
    
    return change_points


def analyze_with_controls(values: np.ndarray, metadata: dict) -> Dict:
    """
    Full analysis with proper controls:
    1. QRNG data (centered)
    2. CSPRNG control
    3. Permutation control
    """
    n_samples = len(values)
    
    print("=" * 70)
    print("QRNG DEEP DIVE ANALYSIS v2 - With Proper Statistical Methodology")
    print("=" * 70)
    
    # Data format clarification
    print("\n📋 DATA FORMAT")
    print("-" * 50)
    print(f"Source: {metadata.get('source', 'unknown')}")
    print(f"Format: {metadata.get('_format', 'unknown')}")
    print(f"Bits per sample: {metadata.get('_bits_per_sample', 'unknown')}")
    print(f"Total samples: {n_samples}")
    print(f"Total bits: {n_samples * metadata.get('_bits_per_sample', 32)}")
    print(f"Value range: [{values.min():.6f}, {values.max():.6f}]")
    print(f"Mean: {values.mean():.6f} (expect 0.5 for uniform)")
    print(f"Std: {values.std():.6f} (expect {1/np.sqrt(12):.6f} for uniform)")
    
    # Generate controls
    print("\n⚙️ GENERATING CONTROLS")
    print("-" * 50)
    print(f"Using {N_WORKERS} CPU cores")
    
    print("Generating CSPRNG control (os.urandom)...")
    csprng_control = generate_csprng_control(n_samples)
    
    print("Generating permutation control (shuffled QRNG)...")
    perm_control = generate_permutation_control(values)
    
    # CENTER all data for Hurst analysis
    qrng_centered = center_for_hurst(values)
    csprng_centered = center_for_hurst(csprng_control)
    perm_centered = center_for_hurst(perm_control)
    
    print("✓ Data centered (subtracted 0.5) for Hurst analysis")
    
    results = {
        'n_samples': n_samples,
        'data_format': metadata.get('_format', 'unknown'),
    }
    
    # =========================================
    # BIT-LEVEL ANALYSIS
    # =========================================
    print("\n📊 BIT-LEVEL ANALYSIS")
    print("-" * 50)
    
    bit_stats = compute_bit_bias(values)
    print(f"Overall P(1): {bit_stats['overall_p1']:.6f} (expect 0.5)")
    print(f"Bias from 0.5: {bit_stats['bias_from_half']:.6f}")
    print(f"Worst bit position: {bit_stats['worst_bit_position']} (bias={bit_stats['worst_bit_bias']:.6f})")
    
    results['bit_bias'] = bit_stats
    
    # Min-entropy
    h_min, h_max, h_ratio = compute_min_entropy_estimate(values)
    print(f"\nMin-entropy: {h_min:.4f} bits (max={h_max:.4f}, ratio={h_ratio:.4f})")
    if h_ratio > 0.95:
        print("  ✓ Good: min-entropy close to maximum")
    else:
        print("  ⚠ Warning: min-entropy below 95% of maximum")
    
    results['min_entropy'] = {'h_min': h_min, 'h_max': h_max, 'ratio': h_ratio}
    
    # =========================================
    # HURST EXPONENT - CORRECTED
    # =========================================
    print("\n📊 HURST EXPONENT ANALYSIS (CORRECTED)")
    print("-" * 50)
    print("Note: Using CENTERED data (values - 0.5) to avoid artificial persistence")
    
    # Compute Hurst for all three streams
    h_qrng = compute_hurst_exponent(qrng_centered)
    h_csprng = compute_hurst_exponent(csprng_centered)
    h_perm = compute_hurst_exponent(perm_centered)
    
    print(f"\nHurst exponents:")
    print(f"  QRNG:       H = {h_qrng:.4f}")
    print(f"  CSPRNG:     H = {h_csprng:.4f} (control)")
    print(f"  Permuted:   H = {h_perm:.4f} (shuffled QRNG)")
    
    # Bootstrap for QRNG
    print("\nBootstrap confidence interval (1000 iterations)...")
    h_mean, h_ci_low, h_ci_high, h_samples = bootstrap_hurst_parallel(
        qrng_centered, n_bootstrap=1000
    )
    print(f"  Bootstrap mean: {h_mean:.4f}")
    print(f"  95% CI: [{h_ci_low:.4f}, {h_ci_high:.4f}]")
    
    # Significance test with CORRECTED null
    print("\nSignificance test (500 simulations with centered null)...")
    p_value, null_dist = test_hurst_significance_parallel(h_qrng, n_samples, n_simulations=500)
    print(f"  Null distribution mean: {np.mean(null_dist):.4f} (should be ~0.5)")
    print(f"  P-value (H ≠ 0.5): {p_value:.4f}")
    
    if p_value < 0.05:
        print("  ⚠ SIGNIFICANT: Hurst differs from random at α=0.05")
    else:
        print("  ✓ Not significant: consistent with random walk")
    
    # Compare QRNG to CSPRNG
    print(f"\nComparison:")
    h_diff = abs(h_qrng - h_csprng)
    print(f"  |H_QRNG - H_CSPRNG| = {h_diff:.4f}")
    if h_diff < 0.1:
        print("  ✓ QRNG and CSPRNG have similar Hurst (good)")
    else:
        print("  ⚠ Notable difference between QRNG and CSPRNG")
    
    # Compare to permuted (tests temporal structure)
    h_diff_perm = abs(h_qrng - h_perm)
    print(f"  |H_QRNG - H_permuted| = {h_diff_perm:.4f}")
    if h_diff_perm < 0.1:
        print("  ✓ Shuffling doesn't change Hurst (no temporal structure)")
    else:
        print("  ⚠ Shuffling changes Hurst (possible temporal structure)")
    
    results['hurst'] = {
        'qrng': h_qrng,
        'csprng': h_csprng,
        'permuted': h_perm,
        'bootstrap_mean': h_mean,
        'ci_low': h_ci_low,
        'ci_high': h_ci_high,
        'p_value': p_value,
        'null_mean': float(np.mean(null_dist))
    }
    
    # =========================================
    # RUNS TEST
    # =========================================
    print("\n📊 RUNS TEST")
    print("-" * 50)
    
    z_qrng, random_qrng = compute_runs_test(values)
    z_csprng, random_csprng = compute_runs_test(csprng_control)
    z_perm, random_perm = compute_runs_test(perm_control)
    
    print(f"QRNG:     z = {z_qrng:+.4f} {'✓' if random_qrng else '⚠'}")
    print(f"CSPRNG:   z = {z_csprng:+.4f} {'✓' if random_csprng else '⚠'}")
    print(f"Permuted: z = {z_perm:+.4f} {'✓' if random_perm else '⚠'}")
    
    results['runs_test'] = {
        'qrng_z': z_qrng,
        'csprng_z': z_csprng,
        'permuted_z': z_perm
    }
    
    # =========================================
    # CHANGE-POINT DETECTION
    # =========================================
    print("\n📊 CHANGE-POINT DETECTION")
    print("-" * 50)
    
    change_points = detect_change_points(values, window_size=100)
    if change_points:
        print(f"⚠ Detected {len(change_points)} potential drift point(s): {change_points}")
    else:
        print("✓ No significant drift detected")
    
    results['change_points'] = change_points
    
    # =========================================
    # SPECTRAL ENTROPY
    # =========================================
    print("\n📊 SPECTRAL ENTROPY")
    print("-" * 50)
    
    se_qrng = compute_spectral_entropy(qrng_centered)
    se_csprng = compute_spectral_entropy(csprng_centered)
    
    print(f"QRNG:   {se_qrng:.4f} (1.0 = white noise)")
    print(f"CSPRNG: {se_csprng:.4f}")
    
    results['spectral_entropy'] = {'qrng': se_qrng, 'csprng': se_csprng}
    
    return results, h_samples, null_dist, csprng_control, perm_control


def create_visualizations_v2(values: np.ndarray, results: Dict, 
                              h_samples: np.ndarray, null_dist: np.ndarray,
                              csprng_control: np.ndarray, perm_control: np.ndarray,
                              output_dir: str = "qrng_visualizations"):
    """Create corrected visualizations."""
    
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    plt.style.use('dark_background')
    
    # =========================================
    # Figure 1: Hurst Analysis (Corrected)
    # =========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    
    # Bootstrap distribution
    ax1 = axes[0, 0]
    ax1.hist(h_samples, bins=40, alpha=0.7, color='cyan', edgecolor='white', label='Bootstrap')
    ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Random (H=0.5)')
    ax1.axvline(results['hurst']['qrng'], color='yellow', linewidth=2, 
                label=f'QRNG (H={results["hurst"]["qrng"]:.3f})')
    ax1.set_xlabel('Hurst Exponent')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Bootstrap Distribution (CENTERED data)')
    ax1.legend()
    
    # Null distribution comparison
    ax2 = axes[0, 1]
    ax2.hist(null_dist, bins=40, alpha=0.6, color='gray', label=f'Null (mean={np.mean(null_dist):.3f})', 
             edgecolor='white')
    ax2.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Expected (0.5)')
    ax2.axvline(results['hurst']['qrng'], color='yellow', linewidth=2, label='QRNG')
    ax2.axvline(results['hurst']['csprng'], color='lime', linewidth=2, label='CSPRNG')
    ax2.set_xlabel('Hurst Exponent')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Null Distribution (p={results["hurst"]["p_value"]:.3f})')
    ax2.legend()
    
    # Three-way comparison
    ax3 = axes[1, 0]
    labels = ['QRNG', 'CSPRNG', 'Permuted']
    h_vals = [results['hurst']['qrng'], results['hurst']['csprng'], results['hurst']['permuted']]
    colors = ['cyan', 'lime', 'orange']
    bars = ax3.bar(labels, h_vals, color=colors, alpha=0.7, edgecolor='white')
    ax3.axhline(0.5, color='red', linestyle='--', linewidth=2, label='Random (H=0.5)')
    ax3.set_ylabel('Hurst Exponent')
    ax3.set_title('Three-Way Comparison')
    ax3.set_ylim(0.3, 0.7)
    ax3.legend()
    for bar, val in zip(bars, h_vals):
        ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{val:.3f}', ha='center', va='bottom', fontsize=10)
    
    # Bit bias by position
    ax4 = axes[1, 1]
    bit_biases = results['bit_bias']['bit_biases']
    ax4.bar(range(32), [b - 0.5 for b in bit_biases], color='cyan', alpha=0.7)
    ax4.axhline(0, color='red', linestyle='--')
    ax4.axhline(0.01, color='orange', linestyle=':', alpha=0.5, label='±1% threshold')
    ax4.axhline(-0.01, color='orange', linestyle=':', alpha=0.5)
    ax4.set_xlabel('Bit Position')
    ax4.set_ylabel('Bias from 0.5')
    ax4.set_title('Bit-Level Bias Analysis')
    ax4.legend()
    
    plt.tight_layout()
    hurst_file = f"{output_dir}/hurst_corrected_{timestamp}.png"
    plt.savefig(hurst_file, facecolor='black')
    print(f"\n📈 Saved: {hurst_file}")
    plt.close()
    
    # =========================================
    # Figure 2: Distribution Comparison
    # =========================================
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), dpi=150)
    
    # Histograms
    ax1 = axes[0, 0]
    ax1.hist(values, bins=50, alpha=0.5, color='cyan', label='QRNG', density=True, edgecolor='white')
    ax1.hist(csprng_control, bins=50, alpha=0.5, color='lime', label='CSPRNG', density=True, edgecolor='white')
    ax1.axhline(1.0, color='red', linestyle='--', label='Uniform')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution Comparison')
    ax1.legend()
    
    # QQ plot
    ax2 = axes[0, 1]
    sorted_qrng = np.sort(values)
    sorted_csprng = np.sort(csprng_control)
    expected = np.linspace(0, 1, len(values))
    ax2.scatter(expected, sorted_qrng, s=2, alpha=0.5, c='cyan', label='QRNG')
    ax2.scatter(expected, sorted_csprng, s=2, alpha=0.5, c='lime', label='CSPRNG')
    ax2.plot([0, 1], [0, 1], 'r--', label='Perfect uniform')
    ax2.set_xlabel('Expected (Uniform)')
    ax2.set_ylabel('Observed')
    ax2.set_title('Q-Q Plot')
    ax2.legend()
    
    # Autocorrelation comparison
    def compute_acf(x, max_lag=50):
        n = len(x)
        mean = np.mean(x)
        var = np.var(x)
        acf = np.zeros(max_lag)
        for lag in range(max_lag):
            if var > 0:
                acf[lag] = np.mean((x[:n-lag] - mean) * (x[lag:] - mean)) / var
        return acf
    
    ax3 = axes[1, 0]
    acf_qrng = compute_acf(values)
    acf_csprng = compute_acf(csprng_control)
    ax3.bar(np.arange(50) - 0.2, acf_qrng, width=0.4, alpha=0.7, color='cyan', label='QRNG')
    ax3.bar(np.arange(50) + 0.2, acf_csprng, width=0.4, alpha=0.7, color='lime', label='CSPRNG')
    sig_bound = 1.96 / np.sqrt(len(values))
    ax3.axhline(sig_bound, color='red', linestyle='--', alpha=0.7)
    ax3.axhline(-sig_bound, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('Autocorrelation')
    ax3.set_title('ACF Comparison')
    ax3.legend()
    
    # Time series
    ax4 = axes[1, 1]
    ax4.plot(values[:300], 'c-', alpha=0.7, linewidth=0.5, label='QRNG')
    ax4.plot(csprng_control[:300], 'g-', alpha=0.7, linewidth=0.5, label='CSPRNG')
    ax4.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Sample')
    ax4.set_ylabel('Value')
    ax4.set_title('Time Series (first 300)')
    ax4.legend()
    
    plt.tight_layout()
    dist_file = f"{output_dir}/distribution_v2_{timestamp}.png"
    plt.savefig(dist_file, facecolor='black')
    print(f"📈 Saved: {dist_file}")
    plt.close()
    
    return hurst_file, dist_file


def main():
    # Find latest QRNG stream
    stream_dir = Path("qrng_streams")
    
    if not stream_dir.exists():
        print("No qrng_streams directory found")
        return
    
    streams = sorted(stream_dir.glob("qrng_stream_*.json"))
    
    if not streams:
        print("No QRNG stream files found")
        return
    
    latest = streams[-1]
    print(f"Loading: {latest}")
    
    values, metadata = load_qrng_stream(str(latest))
    
    # Run analysis with controls
    results, h_samples, null_dist, csprng, perm = analyze_with_controls(values, metadata)
    
    # Create visualizations
    create_visualizations_v2(values, results, h_samples, null_dist, csprng, perm)
    
    # Save results
    results_file = Path("qrng_visualizations") / "deep_dive_v2_results.json"
    
    # Convert numpy types for JSON
    def convert_for_json(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.floating, np.integer)):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_for_json(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_for_json(v) for v in obj]
        return obj
    
    with open(results_file, 'w') as f:
        json.dump(convert_for_json(results), f, indent=2)
    
    print(f"\n📊 Results saved to: {results_file}")
    
    # =========================================
    # SUMMARY
    # =========================================
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    print(f"""
Key Findings:
  • Hurst (QRNG):    {results['hurst']['qrng']:.4f} [CI: {results['hurst']['ci_low']:.3f}-{results['hurst']['ci_high']:.3f}]
  • Hurst (CSPRNG):  {results['hurst']['csprng']:.4f} (control)
  • Hurst (Null):    {results['hurst']['null_mean']:.4f} (should be ~0.5)
  • P-value:         {results['hurst']['p_value']:.4f}
  • Bit bias:        {results['bit_bias']['bias_from_half']:.6f} from 0.5
  • Min-entropy:     {results['min_entropy']['ratio']:.4f} of maximum
  
Methodology:
  ✓ Data CENTERED (subtracted 0.5) before Hurst analysis
  ✓ CSPRNG control generated for comparison
  ✓ Permutation control to test temporal vs distributional
  ✓ Null model uses same centering as data
    """)


if __name__ == "__main__":
    main()
