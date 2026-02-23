"""
Full QRNG Analysis Pipeline

Fetches quantum random numbers from Outshift API,
runs trajectory analysis, and saves results.
"""
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from qrng_outshift_client import OutshiftQRNGClient
from helios_anomaly_scope import QRNGStreamScope, SignalClass


def run_full_qrng_analysis(n_samples: int = 2000, save_stream: bool = True):
    """
    Run complete QRNG analysis pipeline.
    
    Args:
        n_samples: Number of random samples to fetch
        save_stream: Whether to save raw stream data
    """
    print("=" * 60)
    print("HELIOS QRNG Analysis Pipeline")
    print("=" * 60)
    
    # Initialize QRNG client
    print("\n[1] Initializing Outshift QRNG Client...")
    client = OutshiftQRNGClient()
    print("    ✓ Connected to QRNG API")
    
    # Fetch quantum random numbers (API limit is 1000 per request)
    print(f"\n[2] Fetching {n_samples} quantum random numbers...")
    all_integers = []
    remaining = n_samples
    
    while remaining > 0:
        batch_size = min(remaining, 1000)
        result = client.generate(
            bits_per_block=32,
            number_of_blocks=batch_size,
            format='decimal'
        )
        raw_numbers = result.get('random_numbers', [])
        batch_ints = [int(n['decimal']) if isinstance(n, dict) else int(n) for n in raw_numbers]
        all_integers.extend(batch_ints)
        remaining -= batch_size
        print(f"    Fetched {len(all_integers)}/{n_samples}...")
    
    integers = all_integers
    floats = [n / (2**32) for n in integers]
    print(f"    ✓ Received {len(floats)} samples")
    
    # Save raw stream
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if save_stream:
        output_dir = Path("qrng_streams")
        output_dir.mkdir(exist_ok=True)
        stream_file = output_dir / f"analysis_stream_{timestamp}.json"
        
        with open(stream_file, 'w') as f:
            json.dump({
                "timestamp": timestamp,
                "source": "outshift_qrng_api",
                "count": len(integers),
                "raw_integers": integers,
                "floats": floats,
            }, f)
        print(f"    ✓ Raw stream saved to {stream_file}")
    
    # Initialize trajectory scope
    print("\n[3] Initializing QRNGStreamScope...")
    scope = QRNGStreamScope(
        history_len=500,
        warmup_samples=100,
        step_size=0.1,
        walk_mode='angle'
    )
    print("    ✓ Scope initialized")
    
    # Feed samples and analyze
    print(f"\n[4] Processing {len(floats)} samples...")
    events = []
    metrics_log = []
    
    for i, val in enumerate(floats):
        event = scope.update(val)
        if event:
            events.append({
                "step": i,
                "type": event.event_type,
                "message": event.message,
                "metrics": event.metrics
            })
        
        # Log metrics every 100 samples
        if i > 0 and i % 100 == 0:
            metrics = scope.get_current_metrics()
            metrics_log.append({
                "step": i,
                "hurst": metrics.get("hurst", 0.5),
                "lyapunov": metrics.get("lyapunov", 0),
                "diffusion_exp": metrics.get("diffusion_exponent", 1.0),
                "coherence": metrics.get("coherence_ratio", 1.0),
            })
            print(f"    Step {i}: H={metrics.get('hurst', 0.5):.3f}, "
                  f"λ={metrics.get('lyapunov', 0):.3f}, "
                  f"α={metrics.get('diffusion_exponent', 1.0):.3f}")
    
    # Final metrics
    print("\n[5] Final Analysis Results:")
    final_metrics = scope.get_current_metrics()
    print(f"    Hurst Exponent:     {final_metrics.get('hurst', 0.5):.4f}")
    print(f"    Lyapunov Exponent:  {final_metrics.get('lyapunov', 0):.4f}")
    print(f"    Diffusion Exponent: {final_metrics.get('diffusion_exponent', 1.0):.4f}")
    print(f"    Coherence Ratio:    {final_metrics.get('coherence_ratio', 1.0):.4f}")
    
    # Events summary
    print(f"\n[6] Events Detected: {len(events)}")
    if events:
        for evt in events[:10]:  # Show first 10
            print(f"    Step {evt['step']}: {evt['type']} - {evt['message']}")
        if len(events) > 10:
            print(f"    ... and {len(events) - 10} more events")
    
    # Signal verification
    print("\n[7] Signal Verification:")
    try:
        verification = scope.verify_current_signal()
        print(f"    Signal Class: {verification.signal_class.value}")
        print(f"    Verified: {verification.is_verified}")
        print(f"    Confidence: {verification.confidence:.2f}")
    except Exception as e:
        print(f"    (Verification requires more data: {e})")
    
    # Save analysis results
    results_dir = Path("qrng_results")
    results_dir.mkdir(exist_ok=True)
    results_file = results_dir / f"analysis_{timestamp}.json"
    
    analysis_results = {
        "timestamp": timestamp,
        "samples_processed": len(floats),
        "final_metrics": final_metrics,
        "events_count": len(events),
        "events": events[:50],  # First 50 events
        "metrics_log": metrics_log,
    }
    
    with open(results_file, 'w') as f:
        json.dump(analysis_results, f, indent=2, default=str)
    
    print(f"\n[8] Results saved to: {results_file}")
    print("=" * 60)
    print("Analysis Complete!")
    print("=" * 60)
    
    return analysis_results


if __name__ == "__main__":
    run_full_qrng_analysis(n_samples=2000)
