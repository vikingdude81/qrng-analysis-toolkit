#!/usr/bin/env python3
"""
QRNG Deep Dive Analysis with Visualizations

Investigates:
1. Hurst exponent stability and confidence intervals
2. Rolling window analysis to see temporal patterns
3. Phase space trajectory visualization
4. Distribution analysis
5. Autocorrelation structure
6. Bootstrap significance testing
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Tuple, Optional
import scipy.stats as stats
from datetime import datetime
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import os

# Get CPU count for parallel processing
N_WORKERS = min(multiprocessing.cpu_count(), 64)  # Cap at 64 for 5995WX

# Import our analysis modules
from helios_anomaly_scope import (
    compute_hurst_exponent,
    compute_lyapunov_exponent,
    compute_msd_from_trajectory,
    compute_runs_test,
    compute_spectral_entropy
)

try:
    from chaos_detector import (
        compute_lyapunov as rosenstein_lyapunov,
        compute_correlation_dimension,
        detect_phase_transition
    )
    CHAOS_AVAILABLE = True
except ImportError:
    CHAOS_AVAILABLE = False

try:
    from consciousness_metrics import ConsciousnessMetrics
    CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_AVAILABLE = False


def load_qrng_stream(filepath: str) -> Tuple[np.ndarray, dict]:
    """Load QRNG stream and return normalized floats."""
    with open(filepath, 'r') as f:
        data = json.load(f)
    
    if 'floats' in data:
        values = np.array(data['floats'])
    elif 'raw_integers' in data:
        # Normalize integers to [0, 1)
        raw = np.array(data['raw_integers'], dtype=np.float64)
        values = raw / (2**32)
    else:
        raise ValueError("Unknown data format")
    
    return values, data


def _single_bootstrap_hurst(args):
    """Single bootstrap iteration for parallel processing."""
    series, block_size, seed = args
    np.random.seed(seed)
    n = len(series)
    n_blocks = n // block_size
    
    block_indices = np.random.randint(0, n_blocks, size=n_blocks)
    resampled = np.concatenate([
        series[i*block_size:(i+1)*block_size] 
        for i in block_indices
    ])
    
    h = compute_hurst_exponent(resampled)
    return h if 0 < h < 1 else None


def _single_null_hurst(args):
    """Single null hypothesis simulation for parallel processing."""
    n_samples, seed = args
    np.random.seed(seed)
    null_series = np.random.random(n_samples)
    h = compute_hurst_exponent(null_series)
    return h if 0 < h < 1 else None


def bootstrap_hurst(series: np.ndarray, n_bootstrap: int = 1000, 
                    block_size: int = 50) -> Tuple[float, float, float, np.ndarray]:
    """
    Bootstrap confidence interval for Hurst exponent.
    Uses block bootstrap to preserve temporal structure.
    PARALLEL version using ProcessPoolExecutor.
    """
    # Prepare arguments with unique seeds
    args = [(series, block_size, i) for i in range(n_bootstrap)]
    
    hurst_samples = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(_single_bootstrap_hurst, arg) for arg in args]
        for future in as_completed(futures):
            h = future.result()
            if h is not None:
                hurst_samples.append(h)
    
    hurst_samples = np.array(hurst_samples)
    
    mean_h = np.mean(hurst_samples)
    ci_low = np.percentile(hurst_samples, 2.5)
    ci_high = np.percentile(hurst_samples, 97.5)
    
    return mean_h, ci_low, ci_high, hurst_samples


def rolling_hurst(series: np.ndarray, window_size: int = 200, 
                  step: int = 20) -> Tuple[np.ndarray, np.ndarray]:
    """Compute Hurst exponent over rolling windows."""
    hursts = []
    positions = []
    
    for i in range(0, len(series) - window_size, step):
        window = series[i:i+window_size]
        h = compute_hurst_exponent(window)
        hursts.append(h)
        positions.append(i + window_size // 2)
    
    return np.array(positions), np.array(hursts)


def build_phase_space_trajectory(values: np.ndarray, 
                                  mode: str = 'angle') -> Tuple[np.ndarray, np.ndarray]:
    """
    Convert 1D stream to 2D phase space trajectory.
    
    Modes:
    - 'angle': Use value as angle, fixed step (polar walk)
    - 'xy': Use consecutive pairs as (x, y) displacements
    - 'takens': Time-delay embedding (x_t, x_{t+1})
    """
    if mode == 'angle':
        # Polar random walk
        angles = values * 2 * np.pi
        dx = np.cos(angles) * 0.1
        dy = np.sin(angles) * 0.1
        x = np.cumsum(dx)
        y = np.cumsum(dy)
    elif mode == 'xy':
        # Use pairs as displacements
        n = len(values) // 2 * 2
        dx = (values[:n:2] - 0.5) * 0.2
        dy = (values[1:n:2] - 0.5) * 0.2
        x = np.cumsum(dx)
        y = np.cumsum(dy)
    elif mode == 'takens':
        # Time-delay embedding
        x = values[:-1]
        y = values[1:]
    else:
        raise ValueError(f"Unknown mode: {mode}")
    
    return x, y


def analyze_autocorrelation(values: np.ndarray, max_lag: int = 50) -> np.ndarray:
    """Compute autocorrelation function."""
    n = len(values)
    mean = np.mean(values)
    var = np.var(values)
    
    acf = np.zeros(max_lag)
    for lag in range(max_lag):
        if var > 0:
            acf[lag] = np.mean((values[:n-lag] - mean) * (values[lag:] - mean)) / var
        else:
            acf[lag] = 0
    
    return acf


def test_hurst_significance(h_observed: float, n_samples: int, 
                            n_simulations: int = 1000) -> Tuple[float, np.ndarray]:
    """
    Test if observed Hurst is significantly different from 0.5.
    PARALLEL version using ProcessPoolExecutor.
    
    Returns p-value and null distribution.
    """
    # Prepare arguments with unique seeds
    args = [(n_samples, i + 10000) for i in range(n_simulations)]  # Offset seeds from bootstrap
    
    null_hursts = []
    with ProcessPoolExecutor(max_workers=N_WORKERS) as executor:
        futures = [executor.submit(_single_null_hurst, arg) for arg in args]
        for future in as_completed(futures):
            h = future.result()
            if h is not None:
                null_hursts.append(h)
    
    null_hursts = np.array(null_hursts)
    
    # Two-tailed p-value
    if h_observed > 0.5:
        p_value = np.mean(null_hursts >= h_observed) * 2
    else:
        p_value = np.mean(null_hursts <= h_observed) * 2
    
    p_value = min(p_value, 1.0)
    
    return p_value, null_hursts


def create_visualizations(values: np.ndarray, output_dir: str = "qrng_visualizations"):
    """Create comprehensive visualizations."""
    
    Path(output_dir).mkdir(exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Set up style
    plt.style.use('dark_background')
    fig_params = {'figsize': (14, 10), 'dpi': 150}
    
    print("=" * 60)
    print("QRNG DEEP DIVE ANALYSIS")
    print("=" * 60)
    print(f"Samples: {len(values)}")
    print(f"Source: Outshift QRNG API (quantum vacuum fluctuations)")
    print()
    
    # =========================================
    # 1. HURST EXPONENT ANALYSIS
    # =========================================
    print("📊 HURST EXPONENT ANALYSIS")
    print("-" * 40)
    print(f"Using {N_WORKERS} CPU cores for parallel processing")
    
    # Overall Hurst
    h_overall = compute_hurst_exponent(values)
    print(f"Overall Hurst: {h_overall:.4f}")
    
    # Bootstrap confidence interval
    print("Running bootstrap (1000 iterations)...")
    h_mean, h_ci_low, h_ci_high, h_samples = bootstrap_hurst(values, n_bootstrap=1000)
    print(f"Bootstrap mean: {h_mean:.4f}")
    print(f"95% CI: [{h_ci_low:.4f}, {h_ci_high:.4f}]")
    
    # Significance test
    print("Running significance test (500 simulations)...")
    p_value, null_dist = test_hurst_significance(h_overall, len(values), n_simulations=500)
    print(f"P-value (H ≠ 0.5): {p_value:.4f}")
    
    if p_value < 0.05:
        print("✓ SIGNIFICANT: Hurst differs from random walk at α=0.05")
    else:
        print("○ Not significant at α=0.05 (could be random)")
    
    # Interpretation
    if h_ci_low > 0.5:
        print(f"→ Interpretation: PERSISTENT (trending) behavior")
        print(f"  The series shows 'memory' - high values tend to follow high values")
    elif h_ci_high < 0.5:
        print(f"→ Interpretation: ANTI-PERSISTENT (mean-reverting) behavior")
    else:
        print(f"→ Interpretation: CI includes 0.5, may be random")
    
    # Rolling Hurst
    positions, rolling_h = rolling_hurst(values, window_size=200, step=20)
    
    # Plot Hurst analysis
    fig, axes = plt.subplots(2, 2, **fig_params)
    
    # Bootstrap distribution
    ax1 = axes[0, 0]
    ax1.hist(h_samples, bins=40, alpha=0.7, color='cyan', edgecolor='white')
    ax1.axvline(0.5, color='red', linestyle='--', linewidth=2, label='Random (H=0.5)')
    ax1.axvline(h_overall, color='yellow', linewidth=2, label=f'Observed (H={h_overall:.3f})')
    ax1.axvline(h_ci_low, color='green', linestyle=':', label=f'95% CI')
    ax1.axvline(h_ci_high, color='green', linestyle=':')
    ax1.set_xlabel('Hurst Exponent')
    ax1.set_ylabel('Frequency')
    ax1.set_title('Bootstrap Distribution of Hurst Exponent')
    ax1.legend()
    
    # Null distribution comparison
    ax2 = axes[0, 1]
    ax2.hist(null_dist, bins=40, alpha=0.5, color='gray', label='Null (random)', edgecolor='white')
    ax2.hist(h_samples, bins=40, alpha=0.5, color='cyan', label='Bootstrap', edgecolor='white')
    ax2.axvline(h_overall, color='yellow', linewidth=2, label=f'Observed')
    ax2.set_xlabel('Hurst Exponent')
    ax2.set_ylabel('Frequency')
    ax2.set_title(f'Null vs Bootstrap (p={p_value:.3f})')
    ax2.legend()
    
    # Rolling Hurst over time
    ax3 = axes[1, 0]
    ax3.plot(positions, rolling_h, 'c-', linewidth=1.5)
    ax3.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Random')
    ax3.axhline(0.65, color='orange', linestyle=':', alpha=0.7, label='Trending threshold')
    ax3.fill_between(positions, 0.45, 0.55, alpha=0.2, color='gray', label='Random zone')
    ax3.set_xlabel('Sample Position')
    ax3.set_ylabel('Hurst Exponent')
    ax3.set_title('Rolling Hurst (window=200)')
    ax3.legend()
    ax3.set_ylim(0.3, 0.9)
    
    # Hurst stability over different window sizes
    ax4 = axes[1, 1]
    window_sizes = [50, 100, 150, 200, 300, 400, 500, 700, 900]
    hursts_by_window = [compute_hurst_exponent(values[:w]) for w in window_sizes if w <= len(values)]
    valid_windows = [w for w in window_sizes if w <= len(values)]
    ax4.plot(valid_windows, hursts_by_window, 'co-', markersize=8)
    ax4.axhline(0.5, color='red', linestyle='--', alpha=0.7)
    ax4.set_xlabel('Window Size (samples)')
    ax4.set_ylabel('Hurst Exponent')
    ax4.set_title('Hurst Stability vs Sample Size')
    ax4.set_ylim(0.3, 0.9)
    
    plt.tight_layout()
    hurst_file = f"{output_dir}/hurst_analysis_{timestamp}.png"
    plt.savefig(hurst_file, facecolor='black')
    print(f"\n📈 Saved: {hurst_file}")
    plt.close()
    
    # =========================================
    # 2. PHASE SPACE TRAJECTORY
    # =========================================
    print("\n📊 PHASE SPACE ANALYSIS")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, **fig_params)
    
    # Angle walk trajectory
    x_angle, y_angle = build_phase_space_trajectory(values, mode='angle')
    ax1 = axes[0, 0]
    colors = np.linspace(0, 1, len(x_angle))
    scatter = ax1.scatter(x_angle, y_angle, c=colors, cmap='viridis', s=1, alpha=0.5)
    ax1.plot(x_angle[0], y_angle[0], 'go', markersize=10, label='Start')
    ax1.plot(x_angle[-1], y_angle[-1], 'ro', markersize=10, label='End')
    ax1.set_xlabel('X')
    ax1.set_ylabel('Y')
    ax1.set_title('Angle Walk Trajectory')
    ax1.legend()
    ax1.set_aspect('equal')
    plt.colorbar(scatter, ax=ax1, label='Time')
    
    # Displacement from origin over time
    r = np.sqrt(x_angle**2 + y_angle**2)
    ax2 = axes[0, 1]
    ax2.plot(r, 'c-', alpha=0.7)
    ax2.set_xlabel('Step')
    ax2.set_ylabel('Distance from Origin')
    ax2.set_title('Radial Distance Over Time')
    
    # Fit: R ~ t^alpha
    t = np.arange(1, len(r) + 1)
    log_t = np.log(t[r > 0])
    log_r = np.log(r[r > 0])
    if len(log_t) > 10:
        slope, intercept = np.polyfit(log_t, log_r, 1)
        ax2.set_title(f'Radial Distance (α = {slope:.3f}, expect 0.5 for random)')
    
    # MSD analysis
    lags, msd, alpha = compute_msd_from_trajectory(x_angle, y_angle)
    ax3 = axes[1, 0]
    ax3.loglog(lags, msd, 'co-', markersize=4)
    ax3.loglog(lags, lags * msd[0], 'r--', alpha=0.5, label='α=1 (diffusive)')
    ax3.loglog(lags, lags**2 * msd[0]/lags[0], 'g--', alpha=0.5, label='α=2 (ballistic)')
    ax3.set_xlabel('Lag (τ)')
    ax3.set_ylabel('MSD(τ)')
    ax3.set_title(f'Mean Square Displacement (α = {alpha:.3f})')
    ax3.legend()
    print(f"MSD exponent α = {alpha:.4f} (1.0 = diffusive, 2.0 = ballistic)")
    
    # Takens embedding (return map)
    ax4 = axes[1, 1]
    ax4.scatter(values[:-1], values[1:], s=2, alpha=0.5, c='cyan')
    ax4.plot([0, 1], [0, 1], 'r--', alpha=0.3, label='x[t+1] = x[t]')
    ax4.set_xlabel('x[t]')
    ax4.set_ylabel('x[t+1]')
    ax4.set_title('Return Map (Takens Embedding)')
    ax4.set_xlim(0, 1)
    ax4.set_ylim(0, 1)
    ax4.set_aspect('equal')
    
    plt.tight_layout()
    phase_file = f"{output_dir}/phase_space_{timestamp}.png"
    plt.savefig(phase_file, facecolor='black')
    print(f"📈 Saved: {phase_file}")
    plt.close()
    
    # =========================================
    # 3. DISTRIBUTION & RANDOMNESS ANALYSIS
    # =========================================
    print("\n📊 DISTRIBUTION ANALYSIS")
    print("-" * 40)
    
    fig, axes = plt.subplots(2, 2, **fig_params)
    
    # Histogram
    ax1 = axes[0, 0]
    ax1.hist(values, bins=50, alpha=0.7, color='cyan', edgecolor='white', density=True)
    ax1.axhline(1.0, color='red', linestyle='--', label='Uniform (expected)')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution of QRNG Values')
    ax1.legend()
    
    # QQ plot against uniform
    ax2 = axes[0, 1]
    sorted_vals = np.sort(values)
    expected = np.linspace(0, 1, len(values))
    ax2.scatter(expected, sorted_vals, s=2, alpha=0.5, c='cyan')
    ax2.plot([0, 1], [0, 1], 'r--', label='Perfect uniform')
    ax2.set_xlabel('Expected (Uniform)')
    ax2.set_ylabel('Observed')
    ax2.set_title('Q-Q Plot vs Uniform')
    ax2.legend()
    
    # Autocorrelation
    acf = analyze_autocorrelation(values, max_lag=50)
    ax3 = axes[1, 0]
    ax3.bar(range(len(acf)), acf, color='cyan', alpha=0.7)
    # Significance bounds (approximate)
    sig_bound = 1.96 / np.sqrt(len(values))
    ax3.axhline(sig_bound, color='red', linestyle='--', alpha=0.7, label='95% significance')
    ax3.axhline(-sig_bound, color='red', linestyle='--', alpha=0.7)
    ax3.set_xlabel('Lag')
    ax3.set_ylabel('Autocorrelation')
    ax3.set_title('Autocorrelation Function')
    ax3.legend()
    
    # Check for significant autocorrelation
    sig_lags = np.where(np.abs(acf[1:]) > sig_bound)[0] + 1  # Skip lag 0
    if len(sig_lags) > 0:
        print(f"⚠ Significant autocorrelation at lags: {sig_lags[:10]}")
    else:
        print("✓ No significant autocorrelation detected")
    
    # Time series plot
    ax4 = axes[1, 1]
    ax4.plot(values[:500], 'c-', alpha=0.7, linewidth=0.5)
    ax4.axhline(0.5, color='red', linestyle='--', alpha=0.5)
    ax4.set_xlabel('Sample')
    ax4.set_ylabel('Value')
    ax4.set_title('Raw QRNG Stream (first 500)')
    
    plt.tight_layout()
    dist_file = f"{output_dir}/distribution_{timestamp}.png"
    plt.savefig(dist_file, facecolor='black')
    print(f"📈 Saved: {dist_file}")
    plt.close()
    
    # =========================================
    # 4. ADVANCED METRICS
    # =========================================
    print("\n📊 ADVANCED METRICS")
    print("-" * 40)
    
    # Lyapunov exponent
    lyap = compute_lyapunov_exponent(x_angle, y_angle)
    print(f"Lyapunov exponent (trajectory): {lyap:.4f}")
    if lyap > 0:
        print("  → Positive: Chaotic/divergent behavior")
    elif lyap < -0.1:
        print("  → Negative: Convergent/stable behavior")
    else:
        print("  → Near zero: Neutral stability")
    
    # Spectral entropy
    s_ent = compute_spectral_entropy(values)
    print(f"Spectral entropy: {s_ent:.4f} (1.0 = white noise)")
    
    # Runs test
    z_runs, runs_random = compute_runs_test(values)
    print(f"Runs test z-score: {z_runs:.4f}")
    if runs_random:
        print("  ✓ Consistent with random sequence")
    else:
        print("  ⚠ Deviates from random (too few/many runs)")
    
    if CHAOS_AVAILABLE:
        print("\n📊 CHAOS THEORY METRICS")
        print("-" * 40)
        
        lyap_r = rosenstein_lyapunov(values)
        print(f"Lyapunov (Rosenstein): {lyap_r:.4f}")
        
        corr_dim = compute_correlation_dimension(values)
        print(f"Correlation dimension: {corr_dim:.4f}")
        
        transitions = detect_phase_transition(values)
        print(f"Phase transitions detected: {len(transitions)}")
    
    if CONSCIOUSNESS_AVAILABLE:
        print("\n📊 CONSCIOUSNESS METRICS")
        print("-" * 40)
        
        cm = ConsciousnessMetrics()
        # Build logits history from QRNG values as pseudo-logits
        # Interpret each pair of values as a probability distribution
        logits_history = []
        for v in values[:200]:  # Use subset for speed
            logits = np.array([v, 1-v])  # Binary logits
            logits_history.append(logits)
        
        # Compute consciousness metrics on the full history
        result = cm.compute(logits_history)
        
        for key, val in result.items():
            if isinstance(val, float):
                print(f"{key}: {val:.4f}")
            else:
                print(f"{key}: {val}")
    
    # =========================================
    # 5. SUMMARY
    # =========================================
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    
    findings = []
    
    if h_ci_low > 0.5:
        findings.append(f"⚠ HURST={h_overall:.3f} [{h_ci_low:.3f}-{h_ci_high:.3f}]: Persistent trending (p={p_value:.3f})")
    elif p_value < 0.05:
        findings.append(f"⚠ HURST={h_overall:.3f}: Significant deviation from random")
    else:
        findings.append(f"✓ HURST={h_overall:.3f}: Consistent with random walk")
    
    if not runs_random:
        findings.append(f"⚠ RUNS TEST z={z_runs:.2f}: Non-random run structure")
    else:
        findings.append(f"✓ RUNS TEST z={z_runs:.2f}: Random run structure")
    
    if s_ent > 0.9:
        findings.append(f"✓ SPECTRAL ENTROPY={s_ent:.3f}: White noise-like spectrum")
    else:
        findings.append(f"⚠ SPECTRAL ENTROPY={s_ent:.3f}: Structured spectrum")
    
    for f in findings:
        print(f)
    
    print("\n📁 Visualizations saved to:", output_dir)
    
    return {
        'hurst': h_overall,
        'hurst_ci': (h_ci_low, h_ci_high),
        'hurst_pvalue': p_value,
        'lyapunov': lyap,
        'spectral_entropy': s_ent,
        'runs_z': z_runs,
        'runs_random': runs_random,
        'msd_alpha': alpha
    }


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
    
    results = create_visualizations(values)
    
    # Save results
    results_file = Path("qrng_visualizations") / "deep_dive_results.json"
    with open(results_file, 'w') as f:
        # Convert numpy types
        serializable = {}
        for k, v in results.items():
            if isinstance(v, np.floating):
                serializable[k] = float(v)
            elif isinstance(v, tuple):
                serializable[k] = [float(x) for x in v]
            else:
                serializable[k] = v
        json.dump(serializable, f, indent=2)
    
    print(f"\n📊 Results saved to: {results_file}")


if __name__ == "__main__":
    main()
