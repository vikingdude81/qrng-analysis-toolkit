"""
Analyze saved QRNG streams.

Runs trajectory analysis on previously saved QRNG data.
"""
import json
import numpy as np
from datetime import datetime
from pathlib import Path

from helios_anomaly_scope import QRNGStreamScope


def analyze_saved_stream(stream_file: str):
    """Analyze a saved QRNG stream file."""
    print("=" * 60)
    print(f"Analyzing: {stream_file}")
    print("=" * 60)
    
    # Load stream
    with open(stream_file) as f:
        data = json.load(f)
    
    floats = data.get('floats', [])
    print(f"\nLoaded {len(floats)} samples from {data.get('source', 'unknown')}")
    print(f"Timestamp: {data.get('timestamp', 'unknown')}")
    
    # Basic statistics
    arr = np.array(floats)
    print(f"\nBasic Statistics:")
    print(f"  Mean: {arr.mean():.4f} (expected: 0.5)")
    print(f"  Std:  {arr.std():.4f} (expected: ~0.289)")
    print(f"  Min:  {arr.min():.4f}")
    print(f"  Max:  {arr.max():.4f}")
    
    # Initialize trajectory scope (smaller history for faster processing)
    print("\nInitializing QRNGStreamScope...")
    scope = QRNGStreamScope(
        history_len=100,  # Reduced for faster processing
        walk_mode='angle'
    )
    
    # Process samples
    print(f"\nProcessing {len(floats)} samples...")
    events = []
    
    for i, val in enumerate(floats):
        # Use update_from_stream for QRNG values
        result = scope.update_from_stream(val)
        
        # Check for events
        if hasattr(scope, 'events') and scope.events:
            latest_event = scope.events[-1]
            if latest_event.step == i:
                events.append({
                    "step": i,
                    "type": latest_event.event_type,
                    "description": latest_event.description,
                    "confidence": latest_event.confidence,
                })
        
        # Progress every 200 samples
        if i > 0 and i % 200 == 0:
            # Get metrics from latest result
            if isinstance(result, dict):
                print(f"  Step {i}: H={result.get('hurst', 0.5):.3f}, "
                      f"λ={result.get('lyapunov', 0):.3f}, "
                      f"α={result.get('diffusion_exponent', 1.0):.3f}")
    
    # Final results
    print("\n" + "=" * 60)
    print("FINAL ANALYSIS RESULTS")
    print("=" * 60)
    
    # Use last returned metrics
    final_metrics = result if isinstance(result, dict) else {}
    
    print(f"\nTrajectory Metrics:")
    print(f"  Hurst Exponent (H):      {final_metrics.get('hurst', 0.5):.4f}")
    print(f"    → H=0.5: random walk, H>0.5: trending, H<0.5: mean-reverting")
    print(f"  Lyapunov Exponent (λ):   {final_metrics.get('lyapunov', 0):.4f}")
    print(f"    → λ>0: chaotic, λ<0: stable attractor, λ≈0: neutral")
    print(f"  Diffusion Exponent (α):  {final_metrics.get('diffusion_exponent', 1.0):.4f}")
    print(f"    → α=1: normal diffusion, α>1: superdiffusive, α<1: subdiffusive")
    print(f"  Coherence Ratio:         {final_metrics.get('coherence_ratio', 1.0):.4f}")
    
    # Events summary
    print(f"\nEvents Detected: {len(events)}")
    if events:
        event_types = {}
        for evt in events:
            t = evt['type']
            event_types[t] = event_types.get(t, 0) + 1
        print("  Event breakdown:")
        for t, count in sorted(event_types.items(), key=lambda x: -x[1]):
            print(f"    {t}: {count}")
    
    # Interpretation
    print("\n" + "-" * 60)
    print("INTERPRETATION:")
    h = final_metrics.get('hurst', 0.5)
    lam = final_metrics.get('lyapunov', 0)
    alpha = final_metrics.get('diffusion_exponent', 1.0)
    
    if abs(h - 0.5) < 0.05 and abs(alpha - 1.0) < 0.1:
        print("  ✓ Stream appears RANDOM (true QRNG behavior)")
        print("  ✓ No significant patterns or influence detected")
    elif h > 0.6:
        print("  ⚠ Stream shows TRENDING behavior (H > 0.6)")
        print("  → Possible: persistent correlations or influence")
    elif h < 0.4:
        print("  ⚠ Stream shows MEAN-REVERTING behavior (H < 0.4)")
        print("  → Possible: anti-persistent patterns")
    
    if len(events) > 10:
        print(f"  ⚠ Multiple anomaly events detected ({len(events)})")
        print("  → Worth investigating further")
    
    print("-" * 60)
    
    return final_metrics, events


def main():
    """Analyze all saved streams or a specific one."""
    streams_dir = Path("qrng_streams")
    
    if not streams_dir.exists():
        print("No qrng_streams directory found!")
        return
    
    streams = sorted(streams_dir.glob("*.json"))
    
    if not streams:
        print("No stream files found in qrng_streams/")
        return
    
    print(f"Found {len(streams)} stream(s)")
    
    # Analyze the most recent stream
    latest = streams[-1]
    analyze_saved_stream(str(latest))


if __name__ == "__main__":
    main()
