#!/usr/bin/env python3
"""
Inference Experiment Visualization

Creates visualizations for the QRNG vs PRNG inference experiments:
- Iterations to convergence by source
- Confidence distributions
- Effect size visualization
- Source comparison violin plots
"""

import json
from datetime import datetime
from pathlib import Path
from collections import defaultdict
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from scipy import stats
from rich.console import Console

console = Console()

# Color scheme
SOURCE_COLORS = {
    'OUTSHIFT_STREAM': '#FF6B6B',
    'ANU_QRNG': '#4ECDC4',
    'CIPHERSTONE_QRNG': '#45B7D1',
    'CPU_RDRAND': '#FFEAA7',
    'PRNG': '#888888',
}


def load_experiment_results(results_dir: Path = None) -> list:
    """Load all inference experiment results."""
    if results_dir is None:
        results_dir = Path(__file__).parent / "inference_results"
    
    all_results = []
    
    for f in sorted(results_dir.glob("pilot_experiment_*.json")):
        try:
            with open(f) as file:
                data = json.load(file)
            data['filename'] = f.name
            all_results.append(data)
        except Exception as e:
            console.print(f"[yellow]Warning: Could not load {f.name}: {e}[/]")
    
    return all_results


def group_trials_by_source(results: list) -> dict:
    """Group all trials by source across all experiments."""
    grouped = defaultdict(list)
    
    for exp in results:
        for trial in exp.get('results', []):
            source = trial.get('source_type', 'unknown')
            grouped[source].append({
                'iterations': trial.get('iterations', 0),
                'confidence': trial.get('final_confidence', 0),
                'time_ms': trial.get('convergence_time_ms', 0),
                'rng_calls': trial.get('rng_calls', 0),
                'experiment': exp.get('filename', '')
            })
    
    return dict(grouped)


def plot_iterations_violin(grouped: dict, ax):
    """Violin plot of iterations by source."""
    sources = sorted(grouped.keys())
    data = [np.array([t['iterations'] for t in grouped[s]]) for s in sources]
    
    positions = range(len(sources))
    colors = [SOURCE_COLORS.get(s, '#888888') for s in sources]
    
    parts = ax.violinplot(data, positions=positions, showmeans=True, showmedians=True)
    
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(colors[i])
        pc.set_alpha(0.7)
    
    # Add scatter points
    for i, (source, vals) in enumerate(zip(sources, data)):
        jitter = np.random.normal(0, 0.05, len(vals))
        ax.scatter(np.full_like(vals, i) + jitter, vals, 
                   c=colors[i], alpha=0.3, s=20)
    
    ax.set_xticks(positions)
    ax.set_xticklabels(sources, rotation=45, ha='right')
    ax.set_ylabel('Iterations to Convergence')
    ax.set_title('Iterations Distribution by Source')
    
    # Add sample sizes
    for i, source in enumerate(sources):
        n = len(grouped[source])
        ax.text(i, ax.get_ylim()[1], f'n={n}', ha='center', va='bottom', fontsize=8)


def plot_confidence_distributions(grouped: dict, ax):
    """Overlaid histograms of final confidence by source."""
    for source, trials in grouped.items():
        if not trials:
            continue
        confidences = [t['confidence'] for t in trials]
        color = SOURCE_COLORS.get(source, '#888888')
        ax.hist(confidences, bins=20, alpha=0.5, density=True,
                color=color, label=f"{source} (n={len(trials)})")
    
    ax.set_xlabel('Final Confidence')
    ax.set_ylabel('Density')
    ax.set_title('Confidence Score Distribution')
    ax.legend(loc='upper left', fontsize=8)


def plot_effect_sizes(grouped: dict, ax):
    """Bar chart of effect sizes (Cohen's d) vs PRNG."""
    if 'PRNG' not in grouped:
        ax.text(0.5, 0.5, 'No PRNG baseline', ha='center', va='center')
        return
    
    prng_iters = np.array([t['iterations'] for t in grouped['PRNG']])
    prng_mean = np.mean(prng_iters)
    prng_std = np.std(prng_iters, ddof=1)
    
    sources = [s for s in grouped.keys() if s != 'PRNG']
    effect_sizes = []
    colors = []
    
    for source in sources:
        iters = np.array([t['iterations'] for t in grouped[source]])
        source_mean = np.mean(iters)
        source_std = np.std(iters, ddof=1)
        
        # Pooled std
        pooled_std = np.sqrt((prng_std**2 + source_std**2) / 2)
        if pooled_std > 0:
            d = (source_mean - prng_mean) / pooled_std
        else:
            d = 0
        
        effect_sizes.append(d)
        colors.append(SOURCE_COLORS.get(source, '#888888'))
    
    x = range(len(sources))
    bars = ax.bar(x, effect_sizes, color=colors, alpha=0.8)
    
    # Add effect size interpretation lines
    ax.axhline(y=0.2, color='green', linestyle='--', alpha=0.3, label='Small (0.2)')
    ax.axhline(y=-0.2, color='green', linestyle='--', alpha=0.3)
    ax.axhline(y=0.5, color='orange', linestyle='--', alpha=0.3, label='Medium (0.5)')
    ax.axhline(y=-0.5, color='orange', linestyle='--', alpha=0.3)
    ax.axhline(y=0.8, color='red', linestyle='--', alpha=0.3, label='Large (0.8)')
    ax.axhline(y=-0.8, color='red', linestyle='--', alpha=0.3)
    ax.axhline(y=0, color='black', alpha=0.5)
    
    ax.set_xticks(x)
    ax.set_xticklabels(sources, rotation=45, ha='right')
    ax.set_ylabel("Cohen's d (vs PRNG)")
    ax.set_title('Effect Size: Source vs PRNG Baseline')
    ax.legend(loc='upper right', fontsize=8)
    
    # Add value labels
    for bar, d in zip(bars, effect_sizes):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height(),
                f'{d:.3f}', ha='center', va='bottom', fontsize=9)


def plot_summary_stats(grouped: dict, ax):
    """Summary statistics table."""
    ax.axis('off')
    
    headers = ['Source', 'N', 'Mean Iters', 'Std', 'Mean Conf', 'Mean Time (ms)']
    rows = []
    
    for source in sorted(grouped.keys()):
        trials = grouped[source]
        if not trials:
            continue
        
        iters = np.array([t['iterations'] for t in trials])
        confs = np.array([t['confidence'] for t in trials])
        times = np.array([t['time_ms'] for t in trials])
        
        rows.append([
            source,
            str(len(trials)),
            f'{np.mean(iters):.2f}',
            f'{np.std(iters, ddof=1):.2f}',
            f'{np.mean(confs):.3f}',
            f'{np.mean(times):.0f}'
        ])
    
    table = ax.table(
        cellText=rows,
        colLabels=headers,
        cellLoc='center',
        loc='center',
        colColours=['#E8E8E8'] * len(headers)
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.2, 1.5)
    ax.set_title('Summary Statistics', pad=20)


def create_inference_dashboard(output_path: Path = None):
    """Create comprehensive inference experiment visualization."""
    console.print("\n[bold cyan]╭──────────────────────────────────────────╮[/]")
    console.print("[bold cyan]│  INFERENCE EXPERIMENT VISUALIZATION      │[/]")
    console.print("[bold cyan]╰──────────────────────────────────────────╯[/]\n")
    
    # Load data
    console.print("Loading experiment results...")
    results = load_experiment_results()
    
    if not results:
        console.print("[red]No experiment results found![/]")
        return None
    
    console.print(f"Loaded [cyan]{len(results)}[/] experiments")
    
    # Group trials
    grouped = group_trials_by_source(results)
    
    total_trials = sum(len(t) for t in grouped.values())
    console.print(f"Total trials: [cyan]{total_trials}[/]")
    
    for source, trials in grouped.items():
        console.print(f"  {source}: {len(trials)} trials")
    
    # Create figure
    fig = plt.figure(figsize=(14, 10))
    fig.suptitle('QRNG vs PRNG Inference Experiment Results', fontsize=14, fontweight='bold')
    
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.25)
    
    # Violin plot (top left)
    ax1 = fig.add_subplot(gs[0, 0])
    plot_iterations_violin(grouped, ax1)
    
    # Confidence distributions (top right)
    ax2 = fig.add_subplot(gs[0, 1])
    plot_confidence_distributions(grouped, ax2)
    
    # Effect sizes (bottom left)
    ax3 = fig.add_subplot(gs[1, 0])
    plot_effect_sizes(grouped, ax3)
    
    # Summary table (bottom right)
    ax4 = fig.add_subplot(gs[1, 1])
    plot_summary_stats(grouped, ax4)
    
    # Add timestamp
    fig.text(0.99, 0.01, f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
             ha='right', fontsize=8, alpha=0.5)
    
    # Save
    if output_path is None:
        output_path = Path(__file__).parent / "inference_results" / f"visualization_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
    
    output_path.parent.mkdir(exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    console.print(f"\n[green]✓ Dashboard saved to:[/] {output_path}")
    
    plt.show()
    
    return output_path


if __name__ == "__main__":
    create_inference_dashboard()
