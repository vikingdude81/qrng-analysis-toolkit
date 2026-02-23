#!/usr/bin/env python3
"""Compare ANU vs Outshift QRNG sources."""

import json
import numpy as np
from scipy import stats
from pathlib import Path
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()

def runs_test(x):
    """Compute runs test for randomness."""
    median = np.median(x)
    runs = 1
    above = x[0] > median
    for v in x[1:]:
        if (v > median) != above:
            runs += 1
            above = v > median
    n1 = np.sum(x > median)
    n2 = len(x) - n1
    exp_runs = (2*n1*n2)/(n1+n2) + 1
    std_runs = np.sqrt((2*n1*n2*(2*n1*n2-n1-n2))/((n1+n2)**2*(n1+n2-1)))
    z = (runs - exp_runs) / std_runs
    return z, runs

def main():
    console.print(Panel.fit("🔬 ANU vs Outshift QRNG Comparison", style="bold magenta"))
    
    # Load ANU streams
    anu_files = list(Path('qrng_streams').glob('anu_stream_*.json'))
    console.print(f"\nFound {len(anu_files)} ANU stream(s)")
    
    all_anu = []
    for f in sorted(anu_files):
        with open(f) as fp:
            data = json.load(fp)
        vals = data['floats']
        all_anu.extend(vals)
        console.print(f"  {f.name}: {len(vals)} values, mean={np.mean(vals):.4f}")
    
    # Load Outshift streams
    outshift_files = list(Path('qrng_streams').glob('qrng_stream_*.json'))
    console.print(f"\nFound {len(outshift_files)} Outshift stream(s)")
    
    all_outshift = []
    for f in sorted(outshift_files):
        with open(f) as fp:
            data = json.load(fp)
        vals = data.get('floats', data.get('values', []))
        if vals:
            all_outshift.extend(vals)
            console.print(f"  {f.name}: {len(vals)} values")
    
    anu = np.array(all_anu)
    outshift = np.array(all_outshift)
    
    # Comparison table
    console.print("\n")
    table = Table(title="QRNG Source Comparison")
    table.add_column("Metric", style="cyan")
    table.add_column("ANU (vacuum)", justify="right")
    table.add_column("Outshift (SPDC)", justify="right")
    
    table.add_row("Samples", f"{len(anu):,}", f"{len(outshift):,}")
    table.add_row("Mean", f"{anu.mean():.6f}", f"{outshift.mean():.6f}")
    table.add_row("Std Dev", f"{anu.std():.6f}", f"{outshift.std():.6f}")
    table.add_row("Min", f"{anu.min():.6f}", f"{outshift.min():.6f}")
    table.add_row("Max", f"{anu.max():.6f}", f"{outshift.max():.6f}")
    
    # K-S test vs uniform
    ks_anu = stats.kstest(anu, 'uniform')
    ks_out = stats.kstest(outshift, 'uniform')
    table.add_row("K-S vs Uniform (p)", f"{ks_anu.pvalue:.4f}", f"{ks_out.pvalue:.4f}")
    
    # Runs test
    z_anu, runs_anu = runs_test(anu)
    z_out, runs_out = runs_test(outshift)
    table.add_row("Runs Test (Z)", f"{z_anu:+.2f}", f"{z_out:+.2f}")
    
    console.print(table)
    
    # Compare the two sources directly
    console.print("\n")
    ks_stat, ks_p = stats.ks_2samp(anu, outshift)
    
    console.print(Panel.fit(
        f"[bold]ANU vs Outshift Two-Sample K-S Test[/]\n\n"
        f"K-S Statistic: {ks_stat:.4f}\n"
        f"p-value: {ks_p:.4f}\n\n"
        f"Result: {'[red]DIFFERENT distributions[/]' if ks_p < 0.05 else '[green]SAME distribution[/]'}",
        style="cyan"
    ))
    
    # Total QRNG pool
    total = len(anu) + len(outshift)
    console.print(f"\n[bold]Total QRNG Pool:[/] {total:,} samples ({total/10000*100:.0f}% to 10k target)")
    
    console.print("\n[bold]Physics:[/]")
    console.print("  • ANU: Vacuum fluctuation shot noise (continuous field)")
    console.print("  • Outshift: SPDC photon pair detection (discrete events)")

if __name__ == "__main__":
    main()
