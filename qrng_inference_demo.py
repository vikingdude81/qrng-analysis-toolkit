#!/usr/bin/env python3
"""
QRNG Inference Demo
===================
Demonstrates the integration between:
1. Helios QRNG data collection (Outshift API, CPU RDRAND)
2. Inference Architectures framework (Strange Attractor, Code Duality, Tensegrity)

This shows how different randomness sources might affect LLM inference dynamics.

Hypothesis: Strange Attractor and Tensegrity architectures (which use randomness
for convergence/exploration) may show different behavior with QRNG vs PRNG,
while Code Duality (deterministic) should show no difference.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
import numpy as np

from inference_framework import (
    Architecture,
    InferenceMode,
    QRNGSourceType,
    UnifiedRandomnessProvider,
    QRNGStreamProvider,
)

console = Console()


def demo_randomness_sources():
    """Demonstrate all available randomness sources."""
    console.print(Panel.fit(
        "🎲 RANDOMNESS SOURCES DEMO",
        style="bold magenta"
    ))
    
    provider = UnifiedRandomnessProvider()
    
    # Show stream stats
    stats = provider.stream_stats
    console.print(f"\n[bold]QRNG Stream Pool:[/]")
    console.print(f"  Total samples: {stats.count}")
    console.print(f"  Mean: {stats.mean:.6f}")
    console.print(f"  Std: {stats.std:.6f}")
    
    # Compare sources
    console.print(f"\n[bold]Comparing Randomness Sources:[/]")
    
    sources = [
        (QRNGSourceType.OUTSHIFT_STREAM, "Outshift QRNG (quantum photons)"),
        (QRNGSourceType.CPU_RDRAND, "CPU RDRAND (thermal noise)"),
        (QRNGSourceType.CSPRNG, "CSPRNG (os.urandom)"),
        (QRNGSourceType.PRNG, "PRNG (Mersenne Twister)"),
    ]
    
    table = Table(title="1000 Samples per Source")
    table.add_column("Source", style="cyan")
    table.add_column("Mean", justify="right")
    table.add_column("Std", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    
    for source_type, name in sources:
        try:
            source = provider.get_source(source_type, seed=42)
            values = [source() for _ in range(1000)]
            table.add_row(
                name,
                f"{np.mean(values):.4f}",
                f"{np.std(values):.4f}",
                f"{np.min(values):.4f}",
                f"{np.max(values):.4f}"
            )
        except Exception as e:
            table.add_row(name, f"[red]Error[/]", "-", "-", "-")
    
    console.print(table)


def demo_architecture_overview():
    """Show the three inference architectures."""
    console.print(Panel.fit(
        "🧠 INFERENCE ARCHITECTURES",
        style="bold cyan"
    ))
    
    architectures = [
        (
            "Strange Attractor",
            "Convergent, monadic inference",
            ["Abduction", "Induction", "Lateral/Analogical", "Counterfactual"],
            "Uses randomness for initial conditions and exploration",
            "HIGH"
        ),
        (
            "Code Duality",
            "Coupled, dyadic inference",
            ["Deduction", "Meta-reasoning", "Integrative"],
            "Deterministic transformations",
            "LOW"
        ),
        (
            "Tensegrity",
            "Balanced, tetradic inference",
            ["Critical Evaluation", "Collaborative", "Temporal", "Normative", "Error Correction"],
            "Uses randomness for force balancing",
            "MEDIUM"
        ),
    ]
    
    for name, desc, modes, rng_use, sensitivity in architectures:
        console.print(f"\n[bold cyan]{name}[/] - {desc}")
        console.print(f"  Modes: {', '.join(modes)}")
        console.print(f"  RNG Usage: {rng_use}")
        console.print(f"  Expected QRNG Sensitivity: [{'red' if sensitivity == 'HIGH' else 'yellow' if sensitivity == 'MEDIUM' else 'green'}]{sensitivity}[/]")


def demo_qrng_consumption_simulation():
    """Simulate how each architecture consumes randomness."""
    console.print(Panel.fit(
        "📊 RANDOMNESS CONSUMPTION SIMULATION",
        style="bold green"
    ))
    
    provider = UnifiedRandomnessProvider()
    
    # Simulate randomness consumption patterns
    simulations = [
        ("Strange Attractor - Abduction", 50, 0.3),   # Many calls, variable
        ("Strange Attractor - Induction", 30, 0.4),   # Medium calls
        ("Code Duality - Deduction", 0, 0.0),          # No randomness
        ("Code Duality - Meta-reasoning", 2, 0.1),     # Minimal
        ("Tensegrity - Critical Eval", 20, 0.2),       # Moderate
        ("Tensegrity - Collaborative", 40, 0.25),      # More calls
    ]
    
    table = Table(title="Simulated Randomness Consumption per Inference")
    table.add_column("Architecture + Mode", style="cyan")
    table.add_column("Avg RNG Calls", justify="right")
    table.add_column("Variance", justify="right")
    table.add_column("QRNG Impact?", justify="center")
    
    for name, avg_calls, variance in simulations:
        impact = "[red]YES[/]" if avg_calls > 20 else "[yellow]MAYBE[/]" if avg_calls > 5 else "[green]NO[/]"
        table.add_row(name, str(avg_calls), f"{variance:.2f}", impact)
    
    console.print(table)
    
    console.print("\n[bold]Hypothesis:[/]")
    console.print("  If QRNG has different statistical properties than PRNG,")
    console.print("  architectures with HIGH randomness consumption should show")
    console.print("  measurable differences in convergence behavior.")


def demo_experiment_preview():
    """Preview the experimental design."""
    console.print(Panel.fit(
        "🔬 EXPERIMENT DESIGN PREVIEW",
        style="bold yellow"
    ))
    
    console.print("""
[bold]Research Question:[/]
  Do LLM inference architectures show differential sensitivity to 
  quantum vs pseudo-random number sources?

[bold]Conditions:[/]
  1. QRNG (Outshift quantum photon source)
  2. CPU RDRAND (hardware thermal noise)
  3. CSPRNG (os.urandom)
  4. PRNG (Mersenne Twister, seeded)

[bold]Dependent Variables:[/]
  - Convergence time (iterations)
  - Final confidence score
  - Conclusion consistency (hash comparison)
  - Intermediate confidence trajectory

[bold]Analysis:[/]
  - Cohen's d effect sizes between conditions
  - Architecture × Source interaction
  - Jan 15 anomaly investigation (runs deficit)

[bold]Sample Size:[/]
  - Current QRNG pool: 5,000 samples
  - Target: 10,000+ for reliable statistics
  - Trials per condition: 10-30
""")
    
    # Show current data status
    provider = QRNGStreamProvider()
    count = provider.load_streams()
    
    console.print(f"\n[bold]Current QRNG Data Status:[/]")
    console.print(f"  Available samples: {count}")
    console.print(f"  Progress to target: {count/10000*100:.0f}%")
    
    progress_bar = "█" * int(count/10000 * 20) + "░" * (20 - int(count/10000 * 20))
    console.print(f"  [{progress_bar}]")


def main():
    console.print(Panel.fit(
        "🔮 QRNG + INFERENCE FRAMEWORK INTEGRATION 🔮",
        style="bold magenta"
    ))
    
    console.print("""
This demo shows how the helios-trajectory-analysis QRNG data
integrates with the inference architectures framework.

The goal: Investigate whether quantum randomness affects
LLM multi-modal reasoning differently than pseudo-randomness.
""")
    
    demo_randomness_sources()
    console.print("\n" + "="*60 + "\n")
    
    demo_architecture_overview()
    console.print("\n" + "="*60 + "\n")
    
    demo_qrng_consumption_simulation()
    console.print("\n" + "="*60 + "\n")
    
    demo_experiment_preview()
    
    console.print("\n[bold green]✓ Integration complete - ready for experiments[/]")
    console.print("\nNext steps:")
    console.print("  1. Continue daily QRNG collection (need 10k+ samples)")
    console.print("  2. Configure LLM API (Anthropic/OpenAI)")
    console.print("  3. Run pilot experiments with STANDARD_TASKS")
    console.print("  4. Analyze QRNG vs PRNG effect sizes per architecture")


if __name__ == "__main__":
    main()
