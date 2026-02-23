#!/usr/bin/env python3
"""
Collect QRNG data from ALL available sources.

Sources:
1. Cipherstone Qbert Mode 1 (conditioned)
2. Cipherstone Qbert Mode 2 (raw)
3. ANU QRNG (vacuum fluctuation)
4. Outshift QRNG (SPDC photons)
5. IBM Quantum (superconducting transmon qubits)
6. CPU RDRAND (hardware RNG control)
7. PRNG (Mersenne Twister software control)
"""

import json
import os
from datetime import datetime
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

console = Console()


def save_stream(source: str, values: list, output_dir: Path, metadata: dict = None):
    """Save a stream to JSON file."""
    timestamp = datetime.now()
    arr = np.array(values)
    
    stream_data = {
        "source": source,
        "timestamp": timestamp.isoformat(),
        "count": len(values),
        "floats": values,
        "stats": {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max())
        }
    }
    
    if metadata:
        stream_data.update(metadata)
    
    # Create filename based on source
    source_short = source.replace('_', '')[:15]
    filename = f"{source_short}_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(stream_data, f, indent=2)
    
    return filepath, stream_data


def collect_cipherstone(count: int, mode: int, output_dir: Path):
    """Collect from Cipherstone Qbert."""
    from inference_framework import CipherstoneQRNGProvider, CipherstoneQRNGMode
    
    mode_enum = CipherstoneQRNGMode.MODE_1_CONDITIONED if mode == 1 else CipherstoneQRNGMode.MODE_2_RAW
    mode_name = "conditioned" if mode == 1 else "raw"
    source_name = f"cipherstone_qbert_{mode_name}"
    
    console.print(f"\n[cyan]Cipherstone Mode {mode} ({mode_name}):[/]")
    
    provider = CipherstoneQRNGProvider(mode=mode_enum, cache_size=1024)
    
    if not provider.available:
        console.print("  [red]✗ Not available[/]")
        return None
    
    values = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), 
                  BarColumn(), TextColumn("{task.percentage:>3.0f}%"), console=console) as progress:
        task = progress.add_task("  Collecting...", total=count)
        
        batch_size = 100
        while len(values) < count:
            remaining = count - len(values)
            batch = min(batch_size, remaining)
            try:
                batch_values = provider.get_random_batch(batch)
                values.extend(batch_values.tolist())
                progress.update(task, completed=len(values))
            except Exception as e:
                console.print(f"  [red]Error: {e}[/]")
                break
    
    if values:
        filepath, data = save_stream(source_name, values, output_dir)
        console.print(f"  [green]✓ {len(values)} samples[/] (mean={data['stats']['mean']:.4f})")
        return data
    return None


def collect_anu(count: int, output_dir: Path):
    """Collect from ANU QRNG."""
    from inference_framework import ANUQRNGProvider
    
    console.print(f"\n[cyan]ANU QRNG (vacuum fluctuation):[/]")
    
    # Load API key from .env if not in environment
    if not os.environ.get("ANU_QRNG_API_KEY"):
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("ANU_QRNG_API_KEY="):
                        os.environ["ANU_QRNG_API_KEY"] = line.split("=", 1)[1].strip()
                        break
    
    provider = ANUQRNGProvider()
    
    if not provider.available:
        console.print("  [red]✗ Not available (check ANU_QRNG_API_KEY in .env)[/]")
        return None
    
    values = []
    with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), 
                  BarColumn(), TextColumn("{task.percentage:>3.0f}%"), console=console) as progress:
        task = progress.add_task("  Collecting...", total=count)
        
        batch_size = 100
        while len(values) < count:
            remaining = count - len(values)
            batch = min(batch_size, remaining)
            try:
                batch_values = [provider.get_random() for _ in range(batch)]
                values.extend(batch_values)
                progress.update(task, completed=len(values))
            except Exception as e:
                console.print(f"  [red]Error: {e}[/]")
                break
    
    if values:
        filepath, data = save_stream("anu_qrng_vacuum_fluctuation", values, output_dir)
        console.print(f"  [green]✓ {len(values)} samples[/] (mean={data['stats']['mean']:.4f})")
        return data
    return None


def collect_outshift(count: int, output_dir: Path):
    """Collect from Outshift QRNG."""
    console.print(f"\n[cyan]Outshift QRNG (SPDC photons):[/]")
    
    # Check if API key is available
    api_key = os.environ.get("QRNG_OUTSHIFT_API_KEY")
    if not api_key:
        # Try loading from .env
        env_file = Path(__file__).parent / ".env"
        if env_file.exists():
            with open(env_file) as f:
                for line in f:
                    if line.startswith("QRNG_OUTSHIFT_API_KEY="):
                        api_key = line.split("=", 1)[1].strip()
                        break
    
    if not api_key:
        console.print("  [red]✗ Not available (no API key)[/]")
        return None
    
    try:
        from qrng_outshift_client import OutshiftQRNGClient
        client = OutshiftQRNGClient()
        
        values = []
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), 
                      BarColumn(), TextColumn("{task.percentage:>3.0f}%"), console=console) as progress:
            task = progress.add_task("  Collecting...", total=count)
            
            batch_size = 100
            while len(values) < count:
                remaining = count - len(values)
                batch = min(batch_size, remaining)
                try:
                    batch_values = client.generate_normalized_floats(batch)
                    values.extend(batch_values)
                    progress.update(task, completed=len(values))
                except Exception as e:
                    if "daily limit" in str(e).lower() or "rate" in str(e).lower():
                        console.print(f"  [yellow]Rate limited after {len(values)} samples[/]")
                        break
                    console.print(f"  [red]Error: {e}[/]")
                    break
        
        if values:
            filepath, data = save_stream("outshift_qrng_api", values, output_dir)
            console.print(f"  [green]✓ {len(values)} samples[/] (mean={data['stats']['mean']:.4f})")
            return data
            
    except ImportError:
        console.print("  [red]✗ OutshiftQRNGClient not available[/]")
    except Exception as e:
        console.print(f"  [red]✗ Error: {e}[/]")
    
    return None


def collect_cpu_hwrng(count: int, output_dir: Path):
    """Collect from CPU hardware RNG."""
    console.print(f"\n[cyan]CPU RDRAND (thermal noise):[/]")
    
    try:
        from cpu_hwrng import CPUHardwareRNG
        
        rng = CPUHardwareRNG()
        
        with Progress(SpinnerColumn(), TextColumn("[progress.description]{task.description}"), 
                      BarColumn(), TextColumn("{task.percentage:>3.0f}%"), console=console) as progress:
            task = progress.add_task("  Collecting...", total=count)
            
            # Use get_random_floats for batch collection
            values = rng.get_random_floats(count).tolist()
            progress.update(task, completed=count)
        
        filepath, data = save_stream("cpu_hwrng_bcrypt", values, output_dir, 
                                     {"method": rng.method})
        console.print(f"  [green]✓ {len(values)} samples[/] (mean={data['stats']['mean']:.4f})")
        return data
        
    except Exception as e:
        console.print(f"  [red]✗ Error: {e}[/]")
        return None


def collect_prng(count: int, output_dir: Path):
    """Collect from software PRNG (Mersenne Twister)."""
    console.print(f"\n[cyan]PRNG (Mersenne Twister):[/]")
    
    import random
    
    # Use fresh seed for reproducibility documentation
    seed = int(datetime.now().timestamp())
    random.seed(seed)
    
    values = [random.random() for _ in range(count)]
    
    filepath, data = save_stream("prng_mersenne_twister", values, output_dir,
                                 {"seed": seed, "algorithm": "Mersenne Twister (MT19937)"})
    console.print(f"  [green]✓ {len(values)} samples[/] (mean={data['stats']['mean']:.4f})")
    return data


def collect_ibm_quantum_source(count: int, output_dir: Path):
    """Collect from IBM Quantum hardware (superconducting transmon qubits)."""
    console.print(f"\n[cyan]IBM Quantum (superconducting qubits):[/]")
    
    try:
        from ibm_quantum_qrng import collect_ibm_quantum, check_ibm_quantum_available
        
        if not check_ibm_quantum_available():
            return None
        
        result = collect_ibm_quantum(count=count, n_qubits=32, output_dir=output_dir)
        return result
        
    except ImportError:
        console.print("  [red]Error: ibm_quantum_qrng module not found[/]")
        return None
    except Exception as e:
        console.print(f"  [red]Error: {e}[/]")
        return None


def collect_all(count_per_source: int = 1000):
    """Collect from all available sources."""
    console.print("\n[bold cyan]╭───────────────────────────────────────╮[/]")
    console.print("[bold cyan]│  COLLECT FROM ALL QRNG SOURCES        │[/]")
    console.print("[bold cyan]╰───────────────────────────────────────╯[/]")
    
    console.print(f"\nTarget: [cyan]{count_per_source}[/] samples per source")
    
    output_dir = Path(__file__).parent / "qrng_streams"
    output_dir.mkdir(exist_ok=True)
    
    results = {}
    
    # Quantum sources
    console.print("\n[bold]━━━ Quantum Sources ━━━[/]")
    results['cipherstone_m1'] = collect_cipherstone(count_per_source, 1, output_dir)
    results['cipherstone_m2'] = collect_cipherstone(count_per_source, 2, output_dir)
    results['anu'] = collect_anu(count_per_source, output_dir)
    results['outshift'] = collect_outshift(count_per_source, output_dir)
    results['ibm_quantum'] = collect_ibm_quantum_source(count_per_source, output_dir)
    
    # Control sources
    console.print("\n[bold]━━━ Control Sources ━━━[/]")
    results['cpu_hwrng'] = collect_cpu_hwrng(count_per_source, output_dir)
    results['prng'] = collect_prng(count_per_source, output_dir)
    
    # Summary
    console.print("\n[bold]━━━ Collection Summary ━━━[/]")
    total = 0
    for name, data in results.items():
        if data:
            n = data['count']
            total += n
            console.print(f"  {name}: [green]{n:,}[/] samples")
        else:
            console.print(f"  {name}: [red]failed[/]")
    
    console.print(f"\n[bold green]Total collected: {total:,} samples[/]")
    
    return results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect from all QRNG sources")
    parser.add_argument("-n", "--count", type=int, default=1000, 
                       help="Samples per source (default: 1000)")
    
    args = parser.parse_args()
    collect_all(args.count)
