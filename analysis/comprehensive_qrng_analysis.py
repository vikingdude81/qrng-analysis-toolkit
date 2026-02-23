#!/usr/bin/env python3
"""
Comprehensive QRNG Stream Analysis

Runs all available analysis modules on saved QRNG streams:
- Trajectory analysis (Hurst, Lyapunov, MSD)
- Chaos detection (phase transitions, correlation dimension)
- Consciousness metrics (entropy, coherence, criticality)
"""

import json
import numpy as np
from pathlib import Path
from datetime import datetime

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn, TaskProgressColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich import box

console = Console()


def analyze_qrng_stream(stream_path: str):
    """Run comprehensive analysis on a QRNG stream."""
    
    console.print(Panel.fit(
        f"[bold cyan]COMPREHENSIVE QRNG STREAM ANALYSIS[/]\n"
        f"[dim]Stream: {stream_path}[/]\n"
        f"[dim]Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}[/]",
        border_style="cyan"
    ))
    
    # Load data with progress
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        transient=True,
    ) as progress:
        task = progress.add_task("[cyan]Loading QRNG stream...", total=None)
        
        with open(stream_path) as f:
            data = json.load(f)
        
        floats = data.get('floats', data.get('samples', []))
        if not floats:
            console.print("[red]ERROR: No data in stream file[/]")
            return
        
        if isinstance(floats[0], dict):
            floats = [s['decimal'] for s in floats]
        
        sequence = np.array(floats)
    
    console.print(f"[green]✓[/] Loaded [bold]{len(sequence)}[/] samples from [cyan]{data.get('source', 'unknown')}[/]")
    
    # Basic statistics table
    stats_table = Table(title="📊 Basic Statistics", box=box.ROUNDED)
    stats_table.add_column("Metric", style="cyan")
    stats_table.add_column("Value", style="white")
    stats_table.add_column("Expected", style="dim")
    stats_table.add_column("Status", style="bold")
    
    mean_val = np.mean(sequence)
    std_val = np.std(sequence)
    
    stats_table.add_row("Mean", f"{mean_val:.4f}", "0.5000", 
                        "[green]✓[/]" if abs(mean_val - 0.5) < 0.02 else "[yellow]⚠[/]")
    stats_table.add_row("Std Dev", f"{std_val:.4f}", "~0.289",
                        "[green]✓[/]" if abs(std_val - 0.289) < 0.02 else "[yellow]⚠[/]")
    stats_table.add_row("Min", f"{np.min(sequence):.4f}", "~0", "")
    stats_table.add_row("Max", f"{np.max(sequence):.4f}", "~1", "")
    
    console.print(stats_table)
    
    # Analysis with visual progress
    results = {}
    
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TaskProgressColumn(),
        console=console,
    ) as progress:
        
        # Main progress
        main_task = progress.add_task("[bold cyan]Running Analysis...", total=100)
        
        # 1. TRAJECTORY ANALYSIS (0-35%)
        progress.update(main_task, description="[cyan]📈 Trajectory Analysis...")
        
        from helios_anomaly_scope import (
            compute_hurst_exponent, 
            compute_lyapunov_exponent,
            compute_msd_from_trajectory,
            compute_runs_test,
            compute_autocorrelation,
            compute_spectral_entropy
        )
        
        x = sequence[::2][:len(sequence)//2]
        y = sequence[1::2][:len(sequence)//2]
        
        progress.update(main_task, advance=5, description="[cyan]📈 Computing Hurst exponent...")
        hurst = compute_hurst_exponent(sequence)
        
        progress.update(main_task, advance=10, description="[cyan]📈 Computing Lyapunov exponent...")
        lyap = compute_lyapunov_exponent(list(x), list(y))
        
        progress.update(main_task, advance=10, description="[cyan]📈 Computing MSD...")
        lags, msd, alpha = compute_msd_from_trajectory(list(x), list(y))
        
        progress.update(main_task, advance=5, description="[cyan]📈 Running statistical tests...")
        z_score, is_random = compute_runs_test(sequence)
        autocorr = compute_autocorrelation(sequence, max_lag=10)
        spectral_ent = compute_spectral_entropy(sequence)
        
        progress.update(main_task, advance=5, description="[cyan]📈 Trajectory analysis complete!")
        
        # 2. CHAOS DETECTION (35-70%)
        progress.update(main_task, description="[magenta]🌀 Chaos Detection...")
        
        from chaos_detector import ChaosDetector
        chaos = ChaosDetector()
        
        progress.update(main_task, advance=10, description="[magenta]🌀 Computing Rosenstein Lyapunov...")
        lyap_chaos = chaos.compute_lyapunov(sequence)
        
        progress.update(main_task, advance=15, description="[magenta]🌀 Computing correlation dimension...")
        corr_dim = chaos.compute_correlation_dimension(sequence, embedding_dim=5)
        
        progress.update(main_task, advance=10, description="[magenta]🌀 Detecting phase transitions...")
        phase_trans = chaos.detect_phase_transition(sequence)
        
        # 3. CONSCIOUSNESS METRICS (70-100%)
        progress.update(main_task, description="[yellow]🧠 Consciousness Metrics...")
        
        from consciousness_metrics import ConsciousnessMetrics
        consciousness = ConsciousnessMetrics()
        
        window_size = 50
        logits_history = []
        for i in range(0, len(sequence) - window_size, 5):
            window = sequence[i:i + window_size]
            logits = (window - 0.5) * 10
            logits_history.append(logits)
        
        progress.update(main_task, advance=15, description="[yellow]🧠 Computing consciousness functional...")
        
        if len(logits_history) >= 20:
            consciousness_result = consciousness.compute(logits_history)
        else:
            consciousness_result = {}
        
        progress.update(main_task, advance=15, description="[green]✓ Analysis complete!")
    
    # Display Results Tables
    console.print()
    
    # Trajectory Results
    traj_table = Table(title="📈 Trajectory Analysis", box=box.ROUNDED)
    traj_table.add_column("Metric", style="cyan")
    traj_table.add_column("Value", style="white")
    traj_table.add_column("Interpretation", style="dim")
    traj_table.add_column("Status", style="bold")
    
    traj_table.add_row(
        "Hurst (H)", f"{hurst:.4f}",
        "H=0.5: random, >0.5: trending, <0.5: reverting",
        "[green]✓[/]" if 0.4 < hurst < 0.6 else "[yellow]⚠ ANOMALY[/]"
    )
    traj_table.add_row(
        "Lyapunov (λ)", f"{lyap:.4f}",
        "λ>0: chaotic, λ<0: stable, λ≈0: neutral",
        "[green]✓[/]" if abs(lyap) < 0.1 else "[yellow]⚠[/]"
    )
    traj_table.add_row(
        "Diffusion (α)", f"{alpha:.4f}",
        "α=1: normal, α>1: super, α<1: sub",
        "[green]✓[/]" if 0.8 < alpha < 1.2 else "[yellow]⚠[/]"
    )
    traj_table.add_row(
        "Spectral Entropy", f"{spectral_ent:.4f}",
        "~1: white noise, low: structure",
        "[green]✓[/]" if spectral_ent > 0.8 else "[yellow]⚠[/]"
    )
    traj_table.add_row(
        "Runs Test", f"z={z_score:.2f}",
        "Tests randomness of sequence",
        "[green]✓ RANDOM[/]" if is_random else "[red]✗ NON-RANDOM[/]"
    )
    traj_table.add_row(
        "Autocorr (lag 1)", f"{autocorr[0]:.4f}",
        "Should be near 0",
        "[green]✓[/]" if abs(autocorr[0]) < 0.1 else "[yellow]⚠[/]"
    )
    
    console.print(traj_table)
    
    # Chaos Results
    chaos_table = Table(title="🌀 Chaos Detection", box=box.ROUNDED)
    chaos_table.add_column("Metric", style="magenta")
    chaos_table.add_column("Value", style="white")
    chaos_table.add_column("Interpretation", style="dim")
    
    chaos_table.add_row("Lyapunov (Rosenstein)", f"{lyap_chaos:.4f}", "Alternative Lyapunov estimate")
    chaos_table.add_row("Correlation Dimension", f"{corr_dim:.2f}", "Attractor complexity")
    chaos_table.add_row("Phase Transitions", f"{phase_trans['n_transitions']}", 
                        f"Max change: {phase_trans['max_change']:.2f}" if phase_trans['transition_detected'] else "None detected")
    
    console.print(chaos_table)
    
    # Consciousness Results
    if consciousness_result:
        cons_table = Table(title="🧠 Consciousness Metrics", box=box.ROUNDED)
        cons_table.add_column("Metric", style="yellow")
        cons_table.add_column("Value", style="white")
        cons_table.add_column("Meaning", style="dim")
        
        cons_table.add_row("Mode Entropy (H)", f"{consciousness_result['h_mode']:.4f}", "Information content")
        cons_table.add_row("Participation (PR)", f"{consciousness_result['pr']:.4f}", "Active modes")
        cons_table.add_row("Coherence (R)", f"{consciousness_result['r']:.4f}", "Alignment")
        cons_table.add_row("Entropy Rate (ṡ)", f"{consciousness_result['s_dot']:.4f}", "Information generation")
        cons_table.add_row("Criticality (κ)", f"{consciousness_result['kappa']:.4f}", "Edge-of-chaos")
        cons_table.add_row("[bold]Consciousness (C)[/]", f"[bold]{consciousness_result['consciousness']:.4f}[/]", "Combined functional")
        cons_table.add_row("[bold]State[/]", f"[bold]{consciousness_result['state'].upper()}[/]", "Classification")
        
        console.print(cons_table)
    
    # Overall Assessment
    anomalies = []
    
    if hurst > 0.6:
        anomalies.append(f"Hurst={hurst:.3f} suggests persistent correlations")
    if hurst < 0.4:
        anomalies.append(f"Hurst={hurst:.3f} suggests anti-persistence")
    if not is_random:
        anomalies.append(f"Runs test failed (z={z_score:.2f})")
    if abs(autocorr[0]) > 0.1:
        anomalies.append(f"High lag-1 autocorrelation ({autocorr[0]:.3f})")
    if spectral_ent < 0.8:
        anomalies.append(f"Low spectral entropy ({spectral_ent:.3f})")
    
    console.print()
    
    if anomalies:
        anomaly_text = "\n".join([f"  • {a}" for a in anomalies])
        console.print(Panel(
            f"[yellow]⚠ ANOMALIES DETECTED[/]\n\n{anomaly_text}\n\n"
            "[dim]Possible explanations:[/]\n"
            "  1. Statistical fluctuations (need more samples)\n"
            "  2. Hardware correlations in QRNG\n"
            "  3. Interesting signal from QFC perspective 🔮",
            title="Assessment",
            border_style="yellow"
        ))
    else:
        console.print(Panel(
            "[green]✓ STREAM APPEARS RANDOM[/]\n\n"
            "All metrics within expected ranges for true quantum randomness.",
            title="Assessment",
            border_style="green"
        ))
    
    results = {
        'hurst': float(hurst),
        'lyapunov': float(lyap),
        'diffusion_alpha': float(alpha),
        'spectral_entropy': float(spectral_ent),
        'runs_test_random': bool(is_random),
        'autocorr_lag1': float(autocorr[0]),
        'correlation_dimension': float(corr_dim),
        'phase_transitions': int(phase_trans['n_transitions']),
        'consciousness': float(consciousness_result.get('consciousness', 0)) if consciousness_result else None,
        'consciousness_state': consciousness_result.get('state', None) if consciousness_result else None,
        'anomalies': anomalies,
    }
    
    return results


def main():
    stream_dir = Path("qrng_streams")
    
    if not stream_dir.exists():
        console.print("[red]No qrng_streams directory found[/]")
        return
    
    streams = sorted(stream_dir.glob("*.json"))
    
    if not streams:
        console.print("[red]No QRNG stream files found[/]")
        return
    
    # Show available streams
    console.print(f"\n[cyan]Found {len(streams)} stream(s)[/]")
    
    # Analyze latest stream
    latest = streams[-1]
    results = analyze_qrng_stream(str(latest))
    
    if results:
        # Save results
        results_file = stream_dir / f"analysis_{latest.stem}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        console.print(f"\n[green]📊 Analysis saved to:[/] {results_file}")


if __name__ == "__main__":
    main()
