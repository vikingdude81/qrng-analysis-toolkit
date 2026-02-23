#!/usr/bin/env python3
"""
Analyze stability differences between QRNG sources (Outshift vs ANU).

Investigates:
1. Variance in iterations to convergence
2. Confidence distribution shapes
3. Serial correlation (autocorrelation) in results
4. Min-entropy estimation from iteration patterns
"""

import json
from pathlib import Path
import numpy as np
from scipy import stats
from rich.console import Console
from rich.table import Table

console = Console()

def load_latest_results():
    """Load the most recent experiment results."""
    results_dir = Path("inference_results")
    files = sorted(results_dir.glob("pilot_experiment_*.json"))
    if not files:
        raise FileNotFoundError("No experiment results found")
    
    latest = files[-1]
    console.print(f"Loading: [cyan]{latest.name}[/]")
    
    with open(latest) as f:
        return json.load(f)


def analyze_source(name: str, trials: list) -> dict:
    """Compute detailed statistics for a single source."""
    iterations = [t["iterations"] for t in trials]
    confidences = [t["final_confidence"] for t in trials]
    
    iters = np.array(iterations)
    confs = np.array(confidences)
    
    # Basic stats
    mean_iter = np.mean(iters)
    std_iter = np.std(iters, ddof=1)
    cv_iter = std_iter / mean_iter if mean_iter > 0 else 0  # coefficient of variation
    
    # Distribution shape
    skew_iter = stats.skew(iters)
    kurtosis_iter = stats.kurtosis(iters)
    
    # Entropy estimation (from iteration frequencies)
    unique, counts = np.unique(iters, return_counts=True)
    probs = counts / len(iters)
    shannon_entropy = -np.sum(probs * np.log2(probs + 1e-10))
    max_entropy = np.log2(len(unique)) if len(unique) > 1 else 0
    efficiency = shannon_entropy / max_entropy if max_entropy > 0 else 0
    
    # Runs test for randomness
    median = np.median(iters)
    binary = (iters > median).astype(int)
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
    
    # Lag-1 autocorrelation
    if len(iters) > 1:
        autocorr = np.corrcoef(iters[:-1], iters[1:])[0, 1]
    else:
        autocorr = 0
    
    return {
        "name": name,
        "n": len(trials),
        "mean_iter": mean_iter,
        "std_iter": std_iter,
        "cv": cv_iter,
        "skewness": skew_iter,
        "kurtosis": kurtosis_iter,
        "shannon_entropy": shannon_entropy,
        "entropy_efficiency": efficiency,
        "runs_z": runs_z,
        "autocorr": autocorr,
        "mean_conf": np.mean(confs),
        "std_conf": np.std(confs, ddof=1),
        "iterations": iters.tolist(),
        "confidences": confs.tolist()
    }


def compare_distributions(src1: dict, src2: dict):
    """Statistical comparison of two sources."""
    iter1 = np.array(src1["iterations"])
    iter2 = np.array(src2["iterations"])
    
    # Mann-Whitney U (non-parametric)
    u_stat, u_p = stats.mannwhitneyu(iter1, iter2, alternative='two-sided')
    
    # Levene's test for variance equality
    lev_stat, lev_p = stats.levene(iter1, iter2)
    
    # K-S test for distribution shape
    ks_stat, ks_p = stats.ks_2samp(iter1, iter2)
    
    return {
        "mannwhitney_p": u_p,
        "levene_p": lev_p,
        "ks_p": ks_p
    }


def main():
    console.print("\n[bold cyan]╭────────────────────────────────────╮[/]")
    console.print("[bold cyan]│  QRNG SOURCE STABILITY ANALYSIS    │[/]")
    console.print("[bold cyan]╰────────────────────────────────────╯[/]\n")
    
    data = load_latest_results()
    
    # Group trials by source
    from collections import defaultdict
    grouped = defaultdict(list)
    for trial in data["results"]:
        grouped[trial["source_type"]].append(trial)
    
    # Analyze each source
    sources = {}
    for source_name, trials in grouped.items():
        analysis = analyze_source(source_name, trials)
        sources[source_name] = analysis
    
    # Display stats table
    table = Table(title="Stability Metrics by Source")
    table.add_column("Metric", style="cyan")
    for name in sources:
        table.add_column(name[:12], justify="right")
    
    metrics = [
        ("N trials", "n", "d"),
        ("Mean Iters", "mean_iter", ".2f"),
        ("Std Dev", "std_iter", ".2f"),
        ("CV (σ/μ)", "cv", ".3f"),
        ("Skewness", "skewness", ".3f"),
        ("Kurtosis", "kurtosis", ".3f"),
        ("Shannon H", "shannon_entropy", ".3f"),
        ("H Efficiency", "entropy_efficiency", ".3f"),
        ("Runs Z", "runs_z", ".3f"),
        ("Autocorr(1)", "autocorr", ".3f"),
        ("Mean Conf", "mean_conf", ".3f"),
    ]
    
    for label, key, fmt in metrics:
        row = [label]
        for src in sources.values():
            row.append(f"{src[key]:{fmt}}")
        table.add_row(*row)
    
    console.print(table)
    
    # Pairwise comparisons
    console.print("\n[bold]Pairwise Distribution Tests:[/]")
    
    if "OUTSHIFT_STREAM" in sources and "ANU_QRNG" in sources:
        comp = compare_distributions(sources["OUTSHIFT_STREAM"], sources["ANU_QRNG"])
        console.print(f"\n  [cyan]Outshift vs ANU:[/]")
        console.print(f"    Mann-Whitney p = {comp['mannwhitney_p']:.4f}")
        console.print(f"    Levene (variance) p = {comp['levene_p']:.4f}")
        console.print(f"    K-S (shape) p = {comp['ks_p']:.4f}")
    
    if "OUTSHIFT_STREAM" in sources and "PRNG" in sources:
        comp = compare_distributions(sources["OUTSHIFT_STREAM"], sources["PRNG"])
        console.print(f"\n  [cyan]Outshift vs PRNG:[/]")
        console.print(f"    Mann-Whitney p = {comp['mannwhitney_p']:.4f}")
        console.print(f"    Levene (variance) p = {comp['levene_p']:.4f}")
        console.print(f"    K-S (shape) p = {comp['ks_p']:.4f}")
    
    if "ANU_QRNG" in sources and "PRNG" in sources:
        comp = compare_distributions(sources["ANU_QRNG"], sources["PRNG"])
        console.print(f"\n  [cyan]ANU vs PRNG:[/]")
        console.print(f"    Mann-Whitney p = {comp['mannwhitneyu_p']:.4f}" if 'mannwhitneyu_p' in comp else f"    Mann-Whitney p = {comp['mannwhitney_p']:.4f}")
        console.print(f"    Levene (variance) p = {comp['levene_p']:.4f}")
        console.print(f"    K-S (shape) p = {comp['ks_p']:.4f}")
    
    # Key findings
    console.print("\n[bold]━━━ Key Findings ━━━[/]")
    
    # CV comparison
    cv_sorted = sorted(sources.items(), key=lambda x: x[1]["cv"])
    console.print(f"\n  📊 [green]Most stable (lowest CV):[/] {cv_sorted[0][0]} (CV={cv_sorted[0][1]['cv']:.3f})")
    console.print(f"  📊 [yellow]Least stable (highest CV):[/] {cv_sorted[-1][0]} (CV={cv_sorted[-1][1]['cv']:.3f})")
    
    # Entropy
    ent_sorted = sorted(sources.items(), key=lambda x: x[1]["shannon_entropy"], reverse=True)
    console.print(f"\n  🎲 [green]Highest entropy:[/] {ent_sorted[0][0]} (H={ent_sorted[0][1]['shannon_entropy']:.3f} bits)")
    console.print(f"  🎲 [yellow]Lowest entropy:[/] {ent_sorted[-1][0]} (H={ent_sorted[-1][1]['shannon_entropy']:.3f} bits)")
    
    # Autocorrelation
    acf_sorted = sorted(sources.items(), key=lambda x: abs(x[1]["autocorr"]))
    console.print(f"\n  🔗 [green]Most random (lowest |ACF|):[/] {acf_sorted[0][0]} (r={acf_sorted[0][1]['autocorr']:.3f})")
    console.print(f"  🔗 [yellow]Most serial (highest |ACF|):[/] {acf_sorted[-1][0]} (r={acf_sorted[-1][1]['autocorr']:.3f})")
    
    console.print("\n✓ Analysis complete!")


if __name__ == "__main__":
    main()
