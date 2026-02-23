#!/usr/bin/env python3
"""
QRNG Data Visualization Dashboard

Creates comprehensive visualizations for all collected QRNG streams:
- Distribution histograms by source
- Autocorrelation plots  
- Runs test visualization
- Source comparison overlays
- Quality metrics summary
"""

import json
from datetime import datetime
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from rich.console import Console

console = Console()

# Color scheme for sources
SOURCE_COLORS = {
    'outshift_qrng_api': '#FF6B6B',           # Red - SPDC photons
    'anu_qrng_vacuum_fluctuation': '#4ECDC4',  # Teal - Vacuum fluctuation
    'cipherstone_qbert_conditioned': '#45B7D1', # Blue - Qbert conditioned
    'cipherstone_qbert_raw': '#96CEB4',        # Green - Qbert raw
    'cpu_hwrng_bcrypt': '#FFEAA7',             # Yellow - CPU thermal
}

SOURCE_LABELS = {
    'outshift_qrng_api': 'Outshift SPDC',
    'anu_qrng_vacuum_fluctuation': 'ANU Vacuum',
    'cipherstone_qbert_conditioned': 'Cipherstone (Conditioned)',
    'cipherstone_qbert_raw': 'Cipherstone (Raw)',
    'cpu_hwrng_bcrypt': 'CPU HWRNG',
}


def load_all_streams(streams_dir: Path = None) -> dict:
    """Load all QRNG streams grouped by source."""
    if streams_dir is None:
        streams_dir = Path(__file__).parent / "qrng_streams"
    
    sources = {}
    
    for f in sorted(streams_dir.glob("*.json")):
        try:
            with open(f) as file:
                data = json.load(file)
            
            source = data.get('source', 'unknown')
            if source == 'unknown' or source is None:
                continue
                
            floats = data.get('floats', [])
            if not floats:
                continue
            
            if source not in sources:
                sources[source] = {
                    'values': [],
                    'files': [],
                    'timestamps': []
                }
            
            sources[source]['values'].extend(floats)
            sources[source]['files'].append(f.name)
            sources[source]['timestamps'].append(data.get('timestamp', ''))
            
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load {f.name}: {e}[/]")
    
    # Convert to numpy arrays
    for source in sources:
        sources[source]['values'] = np.array(sources[source]['values'])
    
    return sources


def compute_quality_metrics(values: np.ndarray) -> dict:
    """Compute quality metrics for a QRNG stream."""
    n = len(values)
    
    # Basic stats
    mean = np.mean(values)
    std = np.std(values)
    
    # Runs test
    median = np.median(values)
    binary = (values > median).astype(int)
    runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i-1]:
            runs += 1
    n1 = np.sum(binary == 1)
    n0 = np.sum(binary == 0)
    if n0 > 0 and n1 > 0:
        expected_runs = (2 * n0 * n1) / (n0 + n1) + 1
        var_runs = (2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / ((n0 + n1)**2 * (n0 + n1 - 1))
        runs_z = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
    else:
        runs_z = 0
    
    # Autocorrelation lag-1
    if n > 1:
        autocorr = np.corrcoef(values[:-1], values[1:])[0, 1]
    else:
        autocorr = 0
    
    # Chi-squared uniformity test (10 bins)
    observed, _ = np.histogram(values, bins=10, range=(0, 1))
    expected = n / 10
    chi2_stat = np.sum((observed - expected)**2 / expected)
    chi2_p = 1 - stats.chi2.cdf(chi2_stat, df=9)
    
    # Shannon entropy
    hist, _ = np.histogram(values, bins=256, range=(0, 1))
    probs = hist / n
    probs = probs[probs > 0]
    shannon_h = -np.sum(probs * np.log2(probs))
    max_h = np.log2(256)
    
    return {
        'n': n,
        'mean': mean,
        'std': std,
        'runs_z': runs_z,
        'autocorr': autocorr,
        'chi2_stat': chi2_stat,
        'chi2_p': chi2_p,
        'shannon_h': shannon_h,
        'entropy_ratio': shannon_h / max_h
    }


def plot_distribution_comparison(sources: dict, ax):
    """Plot overlaid histograms for all sources."""
    for source, data in sources.items():
        if len(data['values']) < 10:
            continue
        color = SOURCE_COLORS.get(source, '#888888')
        label = SOURCE_LABELS.get(source, source)
        ax.hist(data['values'], bins=50, alpha=0.5, density=True,
                color=color, label=f"{label} (n={len(data['values']):,})")
    
    # Reference uniform
    ax.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Uniform')
    ax.set_xlabel('Value')
    ax.set_ylabel('Density')
    ax.set_title('Distribution Comparison (All QRNG Sources)')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, 1)


def plot_autocorrelation(sources: dict, ax):
    """Plot autocorrelation for each source."""
    max_lag = 50
    
    for source, data in sources.items():
        values = data['values']
        if len(values) < max_lag + 10:
            continue
        
        color = SOURCE_COLORS.get(source, '#888888')
        label = SOURCE_LABELS.get(source, source)
        
        # Compute autocorrelation
        n = len(values)
        mean = np.mean(values)
        var = np.var(values)
        acf = []
        for lag in range(max_lag + 1):
            if var > 0:
                c = np.sum((values[:n-lag] - mean) * (values[lag:] - mean)) / (n * var)
            else:
                c = 0
            acf.append(c)
        
        ax.plot(range(max_lag + 1), acf, color=color, label=label, alpha=0.8)
    
    # 95% CI bounds
    ax.axhline(y=1.96/np.sqrt(1000), color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=-1.96/np.sqrt(1000), color='gray', linestyle='--', alpha=0.5)
    ax.axhline(y=0, color='black', alpha=0.3)
    
    ax.set_xlabel('Lag')
    ax.set_ylabel('Autocorrelation')
    ax.set_title('Autocorrelation by Source')
    ax.legend(loc='upper right', fontsize=8)
    ax.set_xlim(0, max_lag)


def plot_quality_metrics(sources: dict, ax):
    """Bar chart of quality metrics by source."""
    metrics_data = {}
    
    for source, data in sources.items():
        if len(data['values']) < 100:
            continue
        metrics = compute_quality_metrics(data['values'])
        metrics_data[source] = metrics
    
    if not metrics_data:
        ax.text(0.5, 0.5, 'Insufficient data', ha='center', va='center')
        return
    
    # Create grouped bar chart
    sources_list = list(metrics_data.keys())
    x = np.arange(len(sources_list))
    width = 0.2
    
    # Normalize metrics for comparison
    runs_z = [abs(metrics_data[s]['runs_z']) for s in sources_list]
    autocorr = [abs(metrics_data[s]['autocorr']) * 10 for s in sources_list]  # Scale up
    entropy = [metrics_data[s]['entropy_ratio'] for s in sources_list]
    chi2_p = [metrics_data[s]['chi2_p'] for s in sources_list]
    
    ax.bar(x - 1.5*width, runs_z, width, label='|Runs Z|', color='#FF6B6B', alpha=0.8)
    ax.bar(x - 0.5*width, autocorr, width, label='|ACF(1)|×10', color='#4ECDC4', alpha=0.8)
    ax.bar(x + 0.5*width, entropy, width, label='Entropy Ratio', color='#45B7D1', alpha=0.8)
    ax.bar(x + 1.5*width, chi2_p, width, label='χ² p-value', color='#96CEB4', alpha=0.8)
    
    ax.set_xticks(x)
    ax.set_xticklabels([SOURCE_LABELS.get(s, s)[:12] for s in sources_list], rotation=45, ha='right')
    ax.set_ylabel('Score')
    ax.set_title('Quality Metrics by Source')
    ax.legend(loc='upper right', fontsize=8)
    ax.axhline(y=1.96, color='red', linestyle='--', alpha=0.3, label='Runs Z threshold')


def plot_sample_counts(sources: dict, ax):
    """Pie chart of sample counts by source."""
    labels = []
    sizes = []
    colors = []
    
    for source, data in sources.items():
        if len(data['values']) < 10:
            continue
        labels.append(f"{SOURCE_LABELS.get(source, source)}\n({len(data['values']):,})")
        sizes.append(len(data['values']))
        colors.append(SOURCE_COLORS.get(source, '#888888'))
    
    if sizes:
        ax.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%',
               startangle=90, textprops={'fontsize': 8})
        ax.set_title(f"QRNG Pool: {sum(sizes):,} Total Samples")


def plot_sequential_scatter(sources: dict, ax):
    """Sequential scatter plot showing sample index vs value."""
    offset = 0
    
    for source, data in sources.items():
        values = data['values']
        if len(values) < 10:
            continue
        
        color = SOURCE_COLORS.get(source, '#888888')
        label = SOURCE_LABELS.get(source, source)
        
        # Subsample if too large
        if len(values) > 2000:
            indices = np.linspace(0, len(values)-1, 2000, dtype=int)
            plot_vals = values[indices]
        else:
            indices = np.arange(len(values))
            plot_vals = values
        
        ax.scatter(indices + offset, plot_vals, s=1, c=color, alpha=0.3, label=label)
        offset += len(values)
    
    ax.set_xlabel('Sample Index (cumulative)')
    ax.set_ylabel('Value')
    ax.set_title('Sequential Values by Source')
    ax.legend(loc='upper right', fontsize=8, markerscale=5)
    ax.set_ylim(0, 1)


def create_dashboard(output_path: Path = None):
    """Create comprehensive QRNG visualization dashboard."""
    console.print("\n[bold cyan]╭─────────────────────────────────────╮[/]")
    console.print("[bold cyan]│  QRNG DATA VISUALIZATION DASHBOARD  │[/]")
    console.print("[bold cyan]╰─────────────────────────────────────╯[/]\n")
    
    # Load data
    console.print("Loading QRNG streams...")
    sources = load_all_streams()
    
    total_samples = sum(len(d['values']) for d in sources.values())
    console.print(f"Loaded [cyan]{total_samples:,}[/] samples from [cyan]{len(sources)}[/] sources\n")
    
    for source, data in sources.items():
        label = SOURCE_LABELS.get(source, source)
        console.print(f"  {label}: {len(data['values']):,} samples")
    
    # Create figure
    fig = plt.figure(figsize=(16, 12))
    fig.suptitle('QRNG Data Quality Dashboard', fontsize=14, fontweight='bold')
    
    gs = GridSpec(3, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Distribution comparison (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_distribution_comparison(sources, ax1)
    
    # Autocorrelation (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_autocorrelation(sources, ax2)
    
    # Quality metrics (middle left)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_quality_metrics(sources, ax3)
    
    # Sample counts pie (middle right)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_sample_counts(sources, ax4)
    
    # Sequential scatter (bottom, full width)
    ax5 = fig.add_subplot(gs[2, :])
    plot_sequential_scatter(sources, ax5)
    
    # Add timestamp
    fig.text(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             ha='right', fontsize=8, alpha=0.5)
    
    # Save
    if output_path is None:
        output_path = Path(__file__).parent / "qrng_results" / f"dashboard_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    console.print(f"\n[green]✓ Dashboard saved to:[/] {output_path}")
    
    # Also save metrics table
    console.print("\n[bold]Quality Metrics Summary:[/]")
    console.print("-" * 80)
    console.print(f"{'Source':<30} {'N':>8} {'Mean':>8} {'|ACF|':>8} {'|Runs Z|':>8} {'χ² p':>8}")
    console.print("-" * 80)
    
    for source, data in sources.items():
        if len(data['values']) < 100:
            continue
        m = compute_quality_metrics(data['values'])
        label = SOURCE_LABELS.get(source, source)[:28]
        console.print(f"{label:<30} {m['n']:>8,} {m['mean']:>8.4f} {abs(m['autocorr']):>8.4f} {abs(m['runs_z']):>8.3f} {m['chi2_p']:>8.4f}")
    
    console.print("-" * 80)
    
    plt.show()
    
    return output_path


if __name__ == "__main__":
    create_dashboard()
