#!/usr/bin/env python3
"""
HELIOS Full Visual Report Generator
====================================

Generates comprehensive visualization reports for:
1. Each QRNG source individually (detailed analysis)
2. Cross-source comparison dashboard
3. Inference experiment results
4. Summary PDF-ready report

Usage:
    python generate_full_report.py              # Generate all reports
    python generate_full_report.py --source anu # Single source
    python generate_full_report.py --inference  # Inference only
"""

import json
import argparse
from datetime import datetime
from pathlib import Path
from typing import Optional
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
from scipy import stats

# Force UTF-8 for Windows
import sys
import io
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console(force_terminal=True)

# ============================================================================
# Configuration
# ============================================================================

SOURCE_CONFIG = {
    # TIER 1: Trusted Production Sources (expected to pass all tests)
    'outshift_qrng_api': {
        'label': 'Outshift SPDC',
        'color': '#FF6B6B',
        'physics': 'Photon Pair Coincidence',
        'description': 'Quantum random numbers from spontaneous parametric down-conversion',
        'tier': 'production',
        'expected_quality': 'PASS',
    },
    'anu_qrng_vacuum_fluctuation': {
        'label': 'ANU Vacuum',
        'color': '#4ECDC4',
        'physics': 'Vacuum Fluctuation',
        'description': 'Shot noise from quantum vacuum fluctuations',
        'tier': 'production',
        'expected_quality': 'PASS',
    },
    'cipherstone_qbert_conditioned': {
        'label': 'Cipherstone Qbert',
        'color': '#45B7D1',
        'physics': 'Quantum Noise (Conditioned)',
        'description': 'Quantum noise processed through conditioning',
        'tier': 'production',
        'expected_quality': 'PASS',
    },
    # TIER 2: Experimental/Raw Sources (may show artifacts, expected)
    'cipherstone_qbert_raw': {
        'label': 'Cipherstone Raw',
        'color': '#96CEB4',
        'physics': 'Quantum Noise (Raw)',
        'description': 'Raw quantum noise WITHOUT conditioning - artifacts expected',
        'tier': 'experimental',
        'expected_quality': 'WARN',  # Raw sources often show memory/bias
    },
    # TIER 3: Control Sources (hardware/software baselines)
    'cpu_hwrng_bcrypt': {
        'label': 'CPU RDRAND',
        'color': '#FFEAA7',
        'physics': 'Thermal Noise',
        'description': 'Hardware RNG from CPU thermal noise',
        'tier': 'control',
        'expected_quality': 'PASS',
    },
}

STREAMS_DIR = Path(__file__).parent / "qrng_streams"
INFERENCE_DIR = Path(__file__).parent / "inference_results"
OUTPUT_DIR = Path(__file__).parent / "reports"


# ============================================================================
# Data Loading
# ============================================================================

def load_streams_by_source() -> dict:
    """Load all QRNG streams grouped by source."""
    sources = {}

    if not STREAMS_DIR.exists():
        console.print(f"[yellow]Warning: {STREAMS_DIR} not found[/]")
        return sources

    for filepath in sorted(STREAMS_DIR.glob("*.json")):
        try:
            with open(filepath) as f:
                data = json.load(f)

            source = data.get('source', 'unknown')
            if source == 'unknown':
                # Try to infer from filename
                name = filepath.name.lower()
                if 'anu' in name:
                    source = 'anu_qrng_vacuum_fluctuation'
                elif 'cipherstone' in name:
                    source = 'cipherstone_qbert_conditioned'
                elif 'cpu' in name or 'hwrng' in name:
                    source = 'cpu_hwrng_bcrypt'
                else:
                    source = 'outshift_qrng_api'

            floats = data.get('floats', data.get('values', []))
            if not floats:
                continue

            if source not in sources:
                sources[source] = {
                    'values': [],
                    'files': [],
                    'timestamps': [],
                    'metadata': []
                }

            sources[source]['values'].extend(floats)
            sources[source]['files'].append(filepath.name)
            sources[source]['timestamps'].append(data.get('timestamp', ''))
            sources[source]['metadata'].append(data)

        except Exception as e:
            console.print(f"[yellow]Could not load {filepath.name}: {e}[/]")

    # Convert to numpy
    for source in sources:
        sources[source]['values'] = np.array(sources[source]['values'])

    return sources


def load_inference_results() -> list:
    """Load all inference experiment results."""
    results = []

    if not INFERENCE_DIR.exists():
        return results

    for filepath in sorted(INFERENCE_DIR.glob("pilot_experiment_*.json")):
        try:
            with open(filepath) as f:
                data = json.load(f)
            data['filepath'] = filepath
            results.append(data)
        except Exception as e:
            console.print(f"[yellow]Could not load {filepath.name}: {e}[/]")

    return results


# ============================================================================
# Metrics Computation
# ============================================================================

def compute_comprehensive_metrics(values: np.ndarray) -> dict:
    """Compute all quality and trajectory metrics."""
    n = len(values)
    if n < 10:
        return {'n': n, 'error': 'Insufficient samples'}

    # Basic statistics
    mean = np.mean(values)
    std = np.std(values)
    skew = stats.skew(values)
    kurtosis = stats.kurtosis(values)

    # Uniformity test (chi-squared)
    observed, _ = np.histogram(values, bins=10, range=(0, 1))
    expected = n / 10
    chi2_stat, chi2_p = stats.chisquare(observed, f_exp=[expected]*10)

    # Runs test
    median = np.median(values)
    binary = (values > median).astype(int)
    runs = 1 + np.sum(np.diff(binary) != 0)
    n1 = np.sum(binary == 1)
    n0 = np.sum(binary == 0)
    if n0 > 0 and n1 > 0:
        expected_runs = (2 * n0 * n1) / (n0 + n1) + 1
        var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / ((n0 + n1)**2 * (n0 + n1 - 1))
        runs_z = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
        runs_p = 2 * (1 - stats.norm.cdf(abs(runs_z)))
    else:
        runs_z, runs_p = 0, 1.0

    # Autocorrelation (multiple lags)
    autocorr_lags = {}
    for lag in [1, 5, 10, 20]:
        if n > lag:
            autocorr_lags[lag] = np.corrcoef(values[:-lag], values[lag:])[0, 1]

    # Shannon entropy (8-bit resolution)
    hist, _ = np.histogram(values, bins=256, range=(0, 1))
    probs = hist / n
    probs = probs[probs > 0]
    shannon_h = -np.sum(probs * np.log2(probs))
    max_entropy = np.log2(256)

    # Hurst exponent (on increments)
    if n >= 50:
        increments = np.diff(values)
        hurst = compute_hurst_rs(increments)
    else:
        hurst = 0.5

    # Bias test (one-sample t-test against 0.5)
    t_stat, bias_p = stats.ttest_1samp(values, 0.5)

    return {
        'n': n,
        'mean': mean,
        'std': std,
        'bias': mean - 0.5,
        'bias_p': bias_p,
        'skewness': skew,
        'kurtosis': kurtosis,
        'chi2_stat': chi2_stat,
        'chi2_p': chi2_p,
        'runs': runs,
        'runs_z': runs_z,
        'runs_p': runs_p,
        'autocorr': autocorr_lags,
        'shannon_entropy': shannon_h,
        'entropy_ratio': shannon_h / max_entropy,
        'hurst': hurst,
    }


def compute_hurst_rs(data: np.ndarray, min_window: int = 10) -> float:
    """Compute Hurst exponent using R/S analysis."""
    n = len(data)
    if n < min_window * 2:
        return 0.5

    # Range of window sizes
    window_sizes = []
    rs_values = []

    size = min_window
    while size <= n // 2:
        window_sizes.append(size)

        # Compute R/S for this window size
        rs_list = []
        for start in range(0, n - size + 1, size):
            window = data[start:start + size]
            mean = np.mean(window)
            cumsum = np.cumsum(window - mean)
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(window, ddof=1)
            if S > 0:
                rs_list.append(R / S)

        if rs_list:
            rs_values.append(np.mean(rs_list))

        size = int(size * 1.5)

    if len(window_sizes) < 2:
        return 0.5

    # Linear regression in log-log space
    log_sizes = np.log(window_sizes)
    log_rs = np.log(rs_values)
    slope, _ = np.polyfit(log_sizes, log_rs, 1)

    return np.clip(slope, 0, 1)


# ============================================================================
# Individual Source Report
# ============================================================================

def generate_source_report(source: str, data: dict, output_dir: Path):
    """Generate comprehensive visual report for a single QRNG source."""
    values = data['values']
    config = SOURCE_CONFIG.get(source, {
        'label': source,
        'color': '#888888',
        'physics': 'Unknown',
        'description': ''
    })

    metrics = compute_comprehensive_metrics(values)

    # Create figure
    fig = plt.figure(figsize=(16, 20))

    # Title
    fig.suptitle(f'{config["label"]} QRNG Analysis Report\n{config["physics"]}',
                 fontsize=16, fontweight='bold', y=0.98)

    gs = GridSpec(5, 3, figure=fig, hspace=0.4, wspace=0.3,
                  height_ratios=[1, 1, 1, 1, 0.5])

    # Row 1: Distribution analysis
    # 1a. Histogram
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.hist(values, bins=100, density=True, color=config['color'], alpha=0.7, edgecolor='white')
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Uniform')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title(f'Distribution (n={metrics["n"]:,})')
    ax1.set_xlim(0, 1)
    ax1.legend()

    # 1b. QQ plot
    ax2 = fig.add_subplot(gs[0, 1])
    uniform_quantiles = np.linspace(0, 1, 100)
    sample_quantiles = np.percentile(values, np.linspace(0, 100, 100))
    ax2.scatter(uniform_quantiles, sample_quantiles, s=10, c=config['color'], alpha=0.7)
    ax2.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax2.set_xlabel('Uniform Quantiles')
    ax2.set_ylabel('Sample Quantiles')
    ax2.set_title(f'Q-Q Plot (chi2 p={metrics["chi2_p"]:.4f})')

    # 1c. Time series (first 1000)
    ax3 = fig.add_subplot(gs[0, 2])
    plot_n = min(1000, len(values))
    ax3.plot(values[:plot_n], linewidth=0.5, color=config['color'], alpha=0.7)
    ax3.axhline(y=0.5, color='black', linestyle='--', alpha=0.3)
    ax3.set_xlabel('Sample Index')
    ax3.set_ylabel('Value')
    ax3.set_title(f'Time Series (first {plot_n})')
    ax3.set_ylim(0, 1)

    # Row 2: Randomness tests
    # 2a. Autocorrelation
    ax4 = fig.add_subplot(gs[1, 0])
    max_lag = min(50, len(values) // 10)
    acf = []
    for lag in range(max_lag + 1):
        if len(values) > lag:
            acf.append(np.corrcoef(values[:-lag-1], values[lag+1:])[0, 1] if lag < len(values)-1 else 0)
        else:
            acf.append(0)
    ax4.bar(range(max_lag + 1), acf, color=config['color'], alpha=0.7)
    ax4.axhline(y=1.96/np.sqrt(len(values)), color='red', linestyle='--', alpha=0.5)
    ax4.axhline(y=-1.96/np.sqrt(len(values)), color='red', linestyle='--', alpha=0.5)
    ax4.axhline(y=0, color='black', alpha=0.3)
    ax4.set_xlabel('Lag')
    ax4.set_ylabel('ACF')
    ax4.set_title(f'Autocorrelation (ACF[1]={metrics["autocorr"].get(1, 0):.4f})')

    # 2b. Runs visualization
    ax5 = fig.add_subplot(gs[1, 1])
    binary = (values[:500] > 0.5).astype(int)
    colors_runs = [config['color'] if b else '#CCCCCC' for b in binary]
    ax5.scatter(range(len(binary)), binary, c=colors_runs, s=10, alpha=0.7)
    ax5.set_xlabel('Sample Index')
    ax5.set_ylabel('Above/Below Median')
    ax5.set_title(f'Runs Test (Z={metrics["runs_z"]:.3f}, p={metrics["runs_p"]:.4f})')
    ax5.set_yticks([0, 1])
    ax5.set_yticklabels(['Below', 'Above'])

    # 2c. Spectral density
    ax6 = fig.add_subplot(gs[1, 2])
    fft_vals = np.abs(np.fft.fft(values[:1024]))[:512]
    freqs = np.fft.fftfreq(1024)[:512]
    ax6.semilogy(freqs[1:], fft_vals[1:], color=config['color'], alpha=0.7, linewidth=0.5)
    ax6.set_xlabel('Frequency')
    ax6.set_ylabel('Power (log)')
    ax6.set_title('Power Spectrum')

    # Row 3: Trajectory analysis
    # 3a. 2D Phase space
    ax7 = fig.add_subplot(gs[2, 0])
    x = np.cumsum(values[:1000] - 0.5)
    y = np.cumsum(np.roll(values[:1000], 1) - 0.5)
    ax7.plot(x[1:], y[1:], linewidth=0.5, color=config['color'], alpha=0.5)
    ax7.scatter(x[1], y[1], c='green', s=50, zorder=5, label='Start')
    ax7.scatter(x[-1], y[-1], c='red', s=50, zorder=5, label='End')
    ax7.set_xlabel('X displacement')
    ax7.set_ylabel('Y displacement')
    ax7.set_title(f'Phase Space (Hurst={metrics["hurst"]:.3f})')
    ax7.legend()
    ax7.axis('equal')

    # 3b. MSD plot
    ax8 = fig.add_subplot(gs[2, 1])
    trajectory = np.column_stack([x[1:], y[1:]])
    lags = np.arange(1, min(100, len(trajectory)//2))
    msd = []
    for lag in lags:
        displacements = trajectory[lag:] - trajectory[:-lag]
        msd.append(np.mean(np.sum(displacements**2, axis=1)))
    ax8.loglog(lags, msd, 'o-', color=config['color'], markersize=3, alpha=0.7)
    # Fit line
    if len(lags) > 2:
        slope, intercept = np.polyfit(np.log(lags), np.log(msd), 1)
        ax8.loglog(lags, np.exp(intercept) * lags**slope, 'k--', alpha=0.5,
                   label=f'alpha={slope:.2f}')
        ax8.legend()
    ax8.set_xlabel('Lag')
    ax8.set_ylabel('MSD')
    ax8.set_title('Mean Squared Displacement')

    # 3c. Cumulative sum
    ax9 = fig.add_subplot(gs[2, 2])
    cumsum = np.cumsum(values - 0.5)
    ax9.plot(cumsum, color=config['color'], linewidth=0.5, alpha=0.7)
    ax9.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax9.fill_between(range(len(cumsum)), cumsum, alpha=0.3, color=config['color'])
    ax9.set_xlabel('Sample Index')
    ax9.set_ylabel('Cumulative Sum')
    ax9.set_title('Random Walk (CUSUM)')

    # Row 4: Bit-level analysis
    # 4a. Bit distribution (8 bins)
    ax10 = fig.add_subplot(gs[3, 0])
    bins_8 = np.floor(values * 8).astype(int)
    bins_8 = np.clip(bins_8, 0, 7)
    bin_counts = np.bincount(bins_8, minlength=8)
    expected_count = len(values) / 8
    ax10.bar(range(8), bin_counts, color=config['color'], alpha=0.7, edgecolor='white')
    ax10.axhline(y=expected_count, color='black', linestyle='--', alpha=0.5)
    ax10.set_xlabel('Bin (3-bit)')
    ax10.set_ylabel('Count')
    ax10.set_title('3-Bit Distribution')

    # 4b. Entropy by window
    ax11 = fig.add_subplot(gs[3, 1])
    window_size = 1000
    entropies = []
    for i in range(0, len(values) - window_size, window_size // 2):
        window = values[i:i + window_size]
        hist, _ = np.histogram(window, bins=256, range=(0, 1))
        probs = hist / window_size
        probs = probs[probs > 0]
        h = -np.sum(probs * np.log2(probs))
        entropies.append(h / np.log2(256))
    ax11.plot(entropies, color=config['color'], linewidth=1.5, alpha=0.7)
    ax11.axhline(y=1.0, color='black', linestyle='--', alpha=0.5)
    ax11.set_xlabel('Window Index')
    ax11.set_ylabel('Entropy Ratio')
    ax11.set_title(f'Rolling Entropy (mean={np.mean(entropies):.4f})')
    ax11.set_ylim(0.9, 1.01)

    # 4c. Lag scatter plot
    ax12 = fig.add_subplot(gs[3, 2])
    ax12.scatter(values[:-1][:2000], values[1:][:2000], s=1, c=config['color'], alpha=0.3)
    ax12.set_xlabel('x(t)')
    ax12.set_ylabel('x(t+1)')
    ax12.set_title('Lag-1 Scatter')
    ax12.set_xlim(0, 1)
    ax12.set_ylim(0, 1)

    # Row 5: Summary metrics table
    ax13 = fig.add_subplot(gs[4, :])
    ax13.axis('off')

    # Create metrics table
    table_data = [
        ['Samples', f'{metrics["n"]:,}', 'Mean', f'{metrics["mean"]:.6f}', 'Bias (p)', f'{metrics["bias_p"]:.4f}'],
        ['Std Dev', f'{metrics["std"]:.6f}', 'Skewness', f'{metrics["skewness"]:.4f}', 'Kurtosis', f'{metrics["kurtosis"]:.4f}'],
        ['Entropy Ratio', f'{metrics["entropy_ratio"]:.4f}', 'Hurst H', f'{metrics["hurst"]:.4f}', 'ACF(1)', f'{metrics["autocorr"].get(1, 0):.4f}'],
        ['Runs Z', f'{metrics["runs_z"]:.4f}', 'Runs p', f'{metrics["runs_p"]:.4f}', 'Chi2 p', f'{metrics["chi2_p"]:.4f}'],
    ]

    table = ax13.table(cellText=table_data, loc='center', cellLoc='center',
                       colWidths=[0.12, 0.12, 0.12, 0.12, 0.12, 0.12])
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)

    # Quality assessment - use RAW metrics, not scaled values
    # Thresholds based on THRESHOLDS.md
    quality_issues = []

    # Chi-squared uniformity test (p < 0.01 = non-uniform)
    if metrics['chi2_p'] < 0.01:
        quality_issues.append('Non-uniform')

    # Runs test - check BOTH p-value AND absolute Z-score
    # |Z| > 1.96 indicates non-random at 95% confidence
    if metrics['runs_p'] < 0.01 or abs(metrics['runs_z']) > 2.58:  # 99% threshold
        quality_issues.append(f'Non-random runs (Z={metrics["runs_z"]:.2f})')

    # Autocorrelation at lag 1 - threshold 0.05 for high-quality QRNG
    acf1 = abs(metrics['autocorr'].get(1, 0))
    if acf1 > 0.05:
        quality_issues.append(f'High ACF ({acf1:.4f})')

    # Bias test (mean significantly different from 0.5)
    if metrics['bias_p'] < 0.01:
        quality_issues.append(f'Biased (mean={metrics["mean"]:.4f})')

    # Entropy ratio should be > 0.99 for good QRNG
    if metrics['entropy_ratio'] < 0.95:
        quality_issues.append(f'Low entropy ({metrics["entropy_ratio"]:.3f})')

    # Determine quality tier
    if not quality_issues:
        quality = 'PASS - Production Ready'
        quality_color = 'green'
    elif len(quality_issues) == 1 and 'Low entropy' not in str(quality_issues):
        quality = f'MARGINAL: {quality_issues[0]}'
        quality_color = 'orange'
    else:
        quality = f'FAIL: {", ".join(quality_issues)}'
        quality_color = 'red'

    ax13.text(0.5, -0.1, f'Quality: {quality}', ha='center', va='top',
              fontsize=12, fontweight='bold', color=quality_color,
              transform=ax13.transAxes)

    # Timestamp
    fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             ha='right', fontsize=8, alpha=0.5)

    # Save
    output_path = output_dir / f'{config["label"].replace(" ", "_").lower()}_report.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path, metrics


# ============================================================================
# Comparison Dashboard
# ============================================================================

def generate_comparison_dashboard(sources: dict, output_dir: Path):
    """Generate cross-source comparison dashboard."""
    if len(sources) < 2:
        console.print("[yellow]Need at least 2 sources for comparison[/]")
        return None

    fig = plt.figure(figsize=(20, 16))
    fig.suptitle('QRNG Source Comparison Dashboard', fontsize=16, fontweight='bold', y=0.98)

    gs = GridSpec(4, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Collect metrics for all sources
    all_metrics = {}
    for source, data in sources.items():
        all_metrics[source] = compute_comprehensive_metrics(data['values'])

    # 1. Distribution overlay - use counts, normalize by max for visibility
    ax1 = fig.add_subplot(gs[0, 0])
    max_density = 0
    for source, data in sources.items():
        config = SOURCE_CONFIG.get(source, {'label': source, 'color': '#888888'})
        # Use fewer bins and step histogram for better visibility
        counts, bins, _ = ax1.hist(data['values'], bins=30, density=True, alpha=0.5,
                                   color=config['color'], label=config['label'],
                                   histtype='stepfilled', edgecolor=config['color'],
                                   linewidth=1.5)
        max_density = max(max_density, counts.max())
    ax1.axhline(y=1.0, color='black', linestyle='--', alpha=0.7, linewidth=2, label='Uniform')
    ax1.set_xlabel('Value')
    ax1.set_ylabel('Density')
    ax1.set_title('Distribution Comparison')
    ax1.legend(fontsize=7, loc='upper right')
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, max(1.5, max_density * 1.1))  # Ensure uniform line visible

    # 2. Mean comparison
    ax2 = fig.add_subplot(gs[0, 1])
    labels = [SOURCE_CONFIG.get(s, {'label': s})['label'] for s in all_metrics.keys()]
    means = [m['mean'] for m in all_metrics.values()]
    colors = [SOURCE_CONFIG.get(s, {'color': '#888888'})['color'] for s in all_metrics.keys()]
    bars = ax2.bar(range(len(labels)), means, color=colors, alpha=0.8)
    ax2.axhline(y=0.5, color='black', linestyle='--', alpha=0.5)
    ax2.set_xticks(range(len(labels)))
    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=8)
    ax2.set_ylabel('Mean')
    ax2.set_title('Mean Comparison (0.5 = unbiased)')
    ax2.set_ylim(0.48, 0.52)

    # 3. Sample counts pie
    ax3 = fig.add_subplot(gs[0, 2])
    sizes = [m['n'] for m in all_metrics.values()]
    pie_labels = [f'{SOURCE_CONFIG.get(s, {"label": s})["label"]}\n({m["n"]:,})'
                  for s, m in all_metrics.items()]
    ax3.pie(sizes, labels=pie_labels, colors=colors, autopct='%1.1f%%',
            startangle=90, textprops={'fontsize': 8})
    ax3.set_title(f'Sample Pool ({sum(sizes):,} total)')

    # 4. Quality metrics heatmap
    ax4 = fig.add_subplot(gs[1, :2])
    metric_names = ['Entropy Ratio', 'Hurst', '|ACF(1)|', '|Runs Z|', 'Chi2 p']
    source_names = [SOURCE_CONFIG.get(s, {'label': s})['label'] for s in all_metrics.keys()]

    heatmap_data = []
    for source, metrics in all_metrics.items():
        row = [
            metrics['entropy_ratio'],
            metrics['hurst'],
            abs(metrics['autocorr'].get(1, 0)),
            abs(metrics['runs_z']) / 3,  # Normalize
            metrics['chi2_p']
        ]
        heatmap_data.append(row)

    heatmap_data = np.array(heatmap_data)
    im = ax4.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
    ax4.set_xticks(range(len(metric_names)))
    ax4.set_xticklabels(metric_names)
    ax4.set_yticks(range(len(source_names)))
    ax4.set_yticklabels(source_names)
    ax4.set_title('Quality Metrics Heatmap (green=good)')
    plt.colorbar(im, ax=ax4, shrink=0.8)

    # Add values to heatmap
    for i in range(len(source_names)):
        for j in range(len(metric_names)):
            ax4.text(j, i, f'{heatmap_data[i, j]:.3f}', ha='center', va='center',
                     fontsize=8, color='black' if heatmap_data[i, j] > 0.5 else 'white')

    # 5. Autocorrelation comparison
    ax5 = fig.add_subplot(gs[1, 2])
    for source, data in sources.items():
        config = SOURCE_CONFIG.get(source, {'label': source, 'color': '#888888'})
        values = data['values']
        max_lag = min(30, len(values) // 10)
        acf = []
        for lag in range(max_lag + 1):
            if len(values) > lag + 1:
                acf.append(np.corrcoef(values[:-lag-1], values[lag+1:])[0, 1] if lag < len(values)-1 else 0)
        ax5.plot(range(len(acf)), acf, color=config['color'], label=config['label'], alpha=0.7)
    ax5.axhline(y=0, color='black', alpha=0.3)
    ax5.set_xlabel('Lag')
    ax5.set_ylabel('ACF')
    ax5.set_title('Autocorrelation Comparison')
    ax5.legend(fontsize=7)

    # 6. Phase space comparison (2x2 grid)
    phase_axes = [fig.add_subplot(gs[2, i]) for i in range(3)]
    for idx, (source, data) in enumerate(list(sources.items())[:3]):
        config = SOURCE_CONFIG.get(source, {'label': source, 'color': '#888888'})
        values = data['values'][:1000]
        x = np.cumsum(values - 0.5)
        y = np.cumsum(np.roll(values, 1) - 0.5)
        phase_axes[idx].plot(x[1:], y[1:], linewidth=0.3, color=config['color'], alpha=0.5)
        phase_axes[idx].set_title(f'{config["label"]}', fontsize=10)
        phase_axes[idx].axis('equal')
        phase_axes[idx].set_xlabel('X')
        phase_axes[idx].set_ylabel('Y')

    # 7. Rolling Hurst comparison
    ax9 = fig.add_subplot(gs[3, :2])
    for source, data in sources.items():
        config = SOURCE_CONFIG.get(source, {'label': source, 'color': '#888888'})
        values = data['values']
        window_size = 500
        hursts = []
        for i in range(0, len(values) - window_size, window_size // 2):
            window = values[i:i + window_size]
            increments = np.diff(window)
            h = compute_hurst_rs(increments)
            hursts.append(h)
        if hursts:
            ax9.plot(hursts, color=config['color'], label=config['label'], alpha=0.7)
    ax9.axhline(y=0.5, color='black', linestyle='--', alpha=0.5, label='Random Walk')
    ax9.set_xlabel('Window Index')
    ax9.set_ylabel('Hurst Exponent')
    ax9.set_title('Rolling Hurst Exponent')
    ax9.legend(fontsize=8)
    ax9.set_ylim(0.3, 0.7)

    # 8. Summary statistics table with consistent quality assessment
    ax10 = fig.add_subplot(gs[3, 2])
    ax10.axis('off')

    table_data = [['Source', 'N', 'Mean', 'ACF(1)', 'Runs Z', 'Quality']]
    for source, metrics in all_metrics.items():
        label = SOURCE_CONFIG.get(source, {'label': source})['label'][:12]

        # Consistent quality assessment using RAW metrics
        issues = []
        if metrics['chi2_p'] < 0.01:
            issues.append('dist')
        if metrics['runs_p'] < 0.01 or abs(metrics['runs_z']) > 2.58:
            issues.append('runs')
        if abs(metrics['autocorr'].get(1, 0)) > 0.05:
            issues.append('acf')

        if not issues:
            quality = 'PASS'
        elif len(issues) == 1:
            quality = 'WARN'
        else:
            quality = 'FAIL'

        table_data.append([
            label,
            f'{metrics["n"]:,}',
            f'{metrics["mean"]:.4f}',
            f'{abs(metrics["autocorr"].get(1, 0)):.4f}',
            f'{metrics["runs_z"]:.2f}',
            quality
        ])

    table = ax10.table(cellText=table_data, loc='center', cellLoc='center',
                       colWidths=[0.2, 0.15, 0.15, 0.15, 0.15, 0.1])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Timestamp
    fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             ha='right', fontsize=8, alpha=0.5)

    output_path = output_dir / 'source_comparison_dashboard.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


# ============================================================================
# Inference Results Visualization
# ============================================================================

def generate_inference_report(results: list, output_dir: Path):
    """Generate visualization of inference experiment results."""
    if not results:
        console.print("[yellow]No inference results to visualize[/]")
        return None

    # Use latest results
    latest = results[-1]

    fig = plt.figure(figsize=(18, 14))
    fig.suptitle(f'QRNG vs PRNG Inference Experiment Results\n{latest.get("timestamp", "")}',
                 fontsize=14, fontweight='bold', y=0.98)

    gs = GridSpec(3, 3, figure=fig, hspace=0.35, wspace=0.3)

    # Extract data by source
    by_source = {}
    for result in latest.get('results', []):
        source = result.get('source_type', 'UNKNOWN')
        if source not in by_source:
            by_source[source] = {
                'iterations': [],
                'confidence': [],
                'time_ms': [],
                'tokens': []
            }
        by_source[source]['iterations'].append(result.get('iterations', 0))
        by_source[source]['confidence'].append(result.get('final_confidence', 0))
        by_source[source]['time_ms'].append(result.get('convergence_time_ms', 0))
        by_source[source]['tokens'].append(result.get('tokens_used', 0))

    source_colors = {
        'OUTSHIFT_STREAM': '#FF6B6B',
        'ANU_QRNG': '#4ECDC4',
        'CIPHERSTONE_QRNG': '#45B7D1',
        'CPU_RDRAND': '#FFEAA7',
        'PRNG': '#888888'
    }

    # 1. Iterations boxplot
    ax1 = fig.add_subplot(gs[0, 0])
    data_iters = [by_source[s]['iterations'] for s in by_source.keys()]
    bp1 = ax1.boxplot(data_iters, labels=[s[:10] for s in by_source.keys()], patch_artist=True)
    for patch, source in zip(bp1['boxes'], by_source.keys()):
        patch.set_facecolor(source_colors.get(source, '#888888'))
        patch.set_alpha(0.7)
    ax1.set_ylabel('Iterations')
    ax1.set_title('Iterations to Convergence')
    ax1.tick_params(axis='x', rotation=45)

    # 2. Confidence boxplot
    ax2 = fig.add_subplot(gs[0, 1])
    data_conf = [by_source[s]['confidence'] for s in by_source.keys()]
    bp2 = ax2.boxplot(data_conf, labels=[s[:10] for s in by_source.keys()], patch_artist=True)
    for patch, source in zip(bp2['boxes'], by_source.keys()):
        patch.set_facecolor(source_colors.get(source, '#888888'))
        patch.set_alpha(0.7)
    ax2.set_ylabel('Final Confidence')
    ax2.set_title('Convergence Confidence')
    ax2.tick_params(axis='x', rotation=45)

    # 3. Scatter: iterations vs confidence
    ax3 = fig.add_subplot(gs[0, 2])
    for source, data in by_source.items():
        ax3.scatter(data['iterations'], data['confidence'],
                   c=source_colors.get(source, '#888888'), label=source[:10],
                   alpha=0.7, s=50)
    ax3.set_xlabel('Iterations')
    ax3.set_ylabel('Confidence')
    ax3.set_title('Iterations vs Confidence')
    ax3.legend(fontsize=8)

    # 4. Mean iterations bar chart with error bars
    ax4 = fig.add_subplot(gs[1, 0])
    sources_list = list(by_source.keys())
    means = [np.mean(by_source[s]['iterations']) for s in sources_list]
    stds = [np.std(by_source[s]['iterations']) for s in sources_list]
    colors = [source_colors.get(s, '#888888') for s in sources_list]
    x = np.arange(len(sources_list))
    bars = ax4.bar(x, means, yerr=stds, capsize=5, color=colors, alpha=0.8)
    ax4.set_xticks(x)
    ax4.set_xticklabels([s[:10] for s in sources_list], rotation=45, ha='right')
    ax4.set_ylabel('Mean Iterations')
    ax4.set_title('Mean Iterations by Source')

    # 5. Effect size visualization (vs PRNG)
    ax5 = fig.add_subplot(gs[1, 1])
    if 'PRNG' in by_source:
        prng_iters = by_source['PRNG']['iterations']
        effect_sizes = []
        ci_lower = []
        ci_upper = []
        labels = []

        for source in sources_list:
            if source != 'PRNG':
                qrng_iters = by_source[source]['iterations']
                if len(qrng_iters) > 1 and len(prng_iters) > 1:
                    # Cohen's d
                    pooled_std = np.sqrt((np.var(qrng_iters) + np.var(prng_iters)) / 2)
                    d = (np.mean(qrng_iters) - np.mean(prng_iters)) / pooled_std if pooled_std > 0 else 0
                    # CI approximation
                    se = np.sqrt(2/len(qrng_iters) + d**2 / (2*len(qrng_iters)))
                    effect_sizes.append(d)
                    ci_lower.append(d - 1.96*se)
                    ci_upper.append(d + 1.96*se)
                    labels.append(source[:10])

        x = np.arange(len(labels))
        ax5.barh(x, effect_sizes, color=[source_colors.get(l[:10], '#888888') for l in labels], alpha=0.8)
        ax5.errorbar(effect_sizes, x, xerr=[np.array(effect_sizes)-np.array(ci_lower),
                                             np.array(ci_upper)-np.array(effect_sizes)],
                     fmt='none', color='black', capsize=3)
        ax5.axvline(x=0, color='black', linestyle='--', alpha=0.5)
        ax5.axvline(x=0.2, color='gray', linestyle=':', alpha=0.3)
        ax5.axvline(x=-0.2, color='gray', linestyle=':', alpha=0.3)
        ax5.set_yticks(x)
        ax5.set_yticklabels(labels)
        ax5.set_xlabel("Cohen's d (vs PRNG)")
        ax5.set_title('Effect Sizes with 95% CI')

    # 6. Time distribution
    ax6 = fig.add_subplot(gs[1, 2])
    for source, data in by_source.items():
        times_sec = np.array(data['time_ms']) / 1000
        ax6.hist(times_sec, bins=10, alpha=0.5, label=source[:10],
                 color=source_colors.get(source, '#888888'))
    ax6.set_xlabel('Time (seconds)')
    ax6.set_ylabel('Count')
    ax6.set_title('Response Time Distribution')
    ax6.legend(fontsize=8)

    # 7. Individual trial results
    ax7 = fig.add_subplot(gs[2, :2])
    trial_idx = 0
    for source, data in by_source.items():
        for i, iters in enumerate(data['iterations']):
            ax7.scatter(trial_idx, iters, c=source_colors.get(source, '#888888'),
                       s=30, alpha=0.7)
            trial_idx += 1
    ax7.set_xlabel('Trial Index')
    ax7.set_ylabel('Iterations')
    ax7.set_title('All Trials (colored by source)')

    # Create legend
    legend_elements = [Patch(facecolor=source_colors.get(s, '#888888'), label=s[:15])
                       for s in by_source.keys()]
    ax7.legend(handles=legend_elements, loc='upper right', fontsize=8)

    # 8. Summary table
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.axis('off')

    table_data = [['Source', 'N', 'Iters (M+/-SD)', 'Conf', 'd vs PRNG']]

    prng_mean = np.mean(by_source.get('PRNG', {'iterations': [0]})['iterations'])
    prng_std = np.std(by_source.get('PRNG', {'iterations': [0]})['iterations'])

    for source, data in by_source.items():
        iters = data['iterations']
        conf = data['confidence']
        mean_i = np.mean(iters)
        std_i = np.std(iters)

        if source != 'PRNG' and 'PRNG' in by_source and len(iters) > 1:
            pooled = np.sqrt((std_i**2 + prng_std**2) / 2)
            d = (mean_i - prng_mean) / pooled if pooled > 0 else 0
            d_str = f'{d:+.3f}'
        else:
            d_str = 'baseline' if source == 'PRNG' else 'N/A'

        table_data.append([
            source[:12],
            str(len(iters)),
            f'{mean_i:.1f} +/- {std_i:.1f}',
            f'{np.mean(conf):.3f}',
            d_str
        ])

    table = ax8.table(cellText=table_data, loc='center', cellLoc='center',
                      colWidths=[0.25, 0.1, 0.25, 0.15, 0.2])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)

    # Timestamp
    fig.text(0.99, 0.01, f'Generated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}',
             ha='right', fontsize=8, alpha=0.5)

    output_path = output_dir / 'inference_results_report.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Generate HELIOS visual reports')
    parser.add_argument('--source', type=str, help='Generate report for specific source only')
    parser.add_argument('--inference', action='store_true', help='Generate inference report only')
    parser.add_argument('--comparison', action='store_true', help='Generate comparison only')
    parser.add_argument('--output', type=Path, default=OUTPUT_DIR, help='Output directory')

    args = parser.parse_args()

    console.print(Panel.fit('HELIOS Full Report Generator', style='bold blue'))

    # Ensure output directory exists
    args.output.mkdir(parents=True, exist_ok=True)

    # Load data
    console.print('\n[bold]Loading data...[/]')
    sources = load_streams_by_source()
    inference_results = load_inference_results()

    console.print(f'  Found {len(sources)} QRNG sources')
    console.print(f'  Found {len(inference_results)} inference experiments')

    total_samples = sum(len(d['values']) for d in sources.values())
    console.print(f'  Total QRNG samples: {total_samples:,}')

    generated = []

    # Generate reports
    with Progress(SpinnerColumn(), TextColumn('[progress.description]{task.description}'),
                  console=console) as progress:

        if not args.inference and not args.comparison:
            # Generate individual source reports
            for source, data in sources.items():
                if args.source and args.source.lower() not in source.lower():
                    continue

                config = SOURCE_CONFIG.get(source, {'label': source})
                task = progress.add_task(f'Generating {config["label"]} report...')

                try:
                    path, metrics = generate_source_report(source, data, args.output)
                    generated.append(path)
                    console.print(f'  [green]OK[/] {path.name}')
                except Exception as e:
                    console.print(f'  [red]FAIL[/] {source}: {e}')

                progress.remove_task(task)

        if not args.source and not args.inference:
            # Generate comparison dashboard
            task = progress.add_task('Generating comparison dashboard...')
            try:
                path = generate_comparison_dashboard(sources, args.output)
                if path:
                    generated.append(path)
                    console.print(f'  [green]OK[/] {path.name}')
            except Exception as e:
                console.print(f'  [red]FAIL[/] Comparison: {e}')
            progress.remove_task(task)

        if not args.source and (args.inference or not args.comparison):
            # Generate inference report
            if inference_results:
                task = progress.add_task('Generating inference report...')
                try:
                    path = generate_inference_report(inference_results, args.output)
                    if path:
                        generated.append(path)
                        console.print(f'  [green]OK[/] {path.name}')
                except Exception as e:
                    console.print(f'  [red]FAIL[/] Inference: {e}')
                progress.remove_task(task)

    # Summary
    console.print(f'\n[bold green]Generated {len(generated)} reports in {args.output}[/]')
    for path in generated:
        console.print(f'  - {path.name}')


if __name__ == '__main__':
    main()
