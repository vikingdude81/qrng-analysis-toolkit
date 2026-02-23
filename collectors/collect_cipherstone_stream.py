#!/usr/bin/env python3
"""
Collect QRNG data from Cipherstone Qbert and save to stream files.

Cipherstone Qbert uses a CloudFlare tunnel at https://qbert.cipherstone.co/
Two modes available:
- Mode 1: Raw with automatic noise conditioning based on live health tests
- Mode 2: Raw with no conditioning whatsoever
"""

import json
from datetime import datetime
from pathlib import Path
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, BarColumn

from inference_framework import CipherstoneQRNGProvider, CipherstoneQRNGMode

console = Console()


def collect_stream(
    count: int = 1000,
    mode: CipherstoneQRNGMode = CipherstoneQRNGMode.MODE_1_CONDITIONED,
    output_dir: Path = None
) -> dict:
    """
    Collect QRNG values from Cipherstone and save to stream file.
    
    Args:
        count: Number of values to collect
        mode: Operating mode (MODE_1_CONDITIONED or MODE_2_RAW)
        output_dir: Directory to save stream file
        
    Returns:
        Stream metadata dict
    """
    if output_dir is None:
        output_dir = Path(__file__).parent / "qrng_streams"
    output_dir.mkdir(exist_ok=True)
    
    console.print(f"\n[bold cyan]╭────────────────────────────────────╮[/]")
    console.print(f"[bold cyan]│  CIPHERSTONE QBERT DATA COLLECTION │[/]")
    console.print(f"[bold cyan]╰────────────────────────────────────╯[/]\n")
    
    mode_name = "conditioned" if mode == CipherstoneQRNGMode.MODE_1_CONDITIONED else "raw"
    console.print(f"Mode: [cyan]{mode.name}[/] ({mode_name})")
    console.print(f"Target: [cyan]{count}[/] samples")
    
    # Initialize provider
    provider = CipherstoneQRNGProvider(mode=mode, cache_size=1024)
    
    if not provider.available:
        console.print("[red]ERROR: Cipherstone API not available[/]")
        return None
    
    console.print(f"API: [green]connected[/]")
    console.print()
    
    # Collect values
    values = []
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console
    ) as progress:
        task = progress.add_task(f"Collecting from Qbert...", total=count)
        
        batch_size = 100
        while len(values) < count:
            remaining = count - len(values)
            batch = min(batch_size, remaining)
            
            try:
                batch_values = provider.get_random_batch(batch)
                values.extend(batch_values.tolist())
                progress.update(task, completed=len(values))
            except Exception as e:
                console.print(f"\n[red]Error: {e}[/]")
                if len(values) == 0:
                    return None
                console.print(f"[yellow]Continuing with {len(values)} values collected[/]")
                break
    
    # Create stream data
    timestamp = datetime.now()
    arr = np.array(values)
    
    stream_data = {
        "source": f"cipherstone_qbert_{mode_name}",
        "timestamp": timestamp.isoformat(),
        "count": len(values),
        "mode": mode.value,
        "mode_name": mode.name,
        "api_url": provider._api_url,
        "floats": values,
        "stats": {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max())
        }
    }
    
    # Save to file
    filename = f"cipherstone_stream_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
    filepath = output_dir / filename
    
    with open(filepath, 'w') as f:
        json.dump(stream_data, f, indent=2)
    
    console.print()
    console.print(f"[green]✓ Collected {len(values)} samples[/]")
    console.print(f"  Mean: {arr.mean():.4f}")
    console.print(f"  Std:  {arr.std():.4f}")
    console.print(f"  Range: [{arr.min():.4f}, {arr.max():.4f}]")
    console.print()
    console.print(f"Saved to: [cyan]{filepath}[/]")
    
    return stream_data


def collect_both_modes(count_per_mode: int = 1000):
    """Collect streams from both Mode 1 and Mode 2."""
    console.print("\n[bold]Collecting from both Cipherstone modes...[/]\n")
    
    # Mode 1 - Conditioned
    console.print("[bold]━━━ Mode 1 (Conditioned) ━━━[/]")
    stream1 = collect_stream(count=count_per_mode, mode=CipherstoneQRNGMode.MODE_1_CONDITIONED)
    
    # Mode 2 - Raw
    console.print("\n[bold]━━━ Mode 2 (Raw) ━━━[/]")
    stream2 = collect_stream(count=count_per_mode, mode=CipherstoneQRNGMode.MODE_2_RAW)
    
    return stream1, stream2


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Collect Cipherstone QRNG streams")
    parser.add_argument("-n", "--count", type=int, default=1000, help="Number of samples")
    parser.add_argument("-m", "--mode", choices=["1", "2", "both"], default="both",
                       help="Mode: 1=conditioned, 2=raw, both=collect from both")
    
    args = parser.parse_args()
    
    if args.mode == "both":
        collect_both_modes(args.count)
    elif args.mode == "1":
        collect_stream(args.count, CipherstoneQRNGMode.MODE_1_CONDITIONED)
    else:
        collect_stream(args.count, CipherstoneQRNGMode.MODE_2_RAW)
