"""
Basic QRNG Analysis Example
===========================

This example demonstrates basic usage of the HELIOS trajectory
analysis system with simulated QRNG data.

It shows:
1. Setting up the QRNGStreamScope
2. Processing values from a stream
3. Detecting anomalies
4. Verifying and classifying signals
"""

import numpy as np
import sys
sys.path.insert(0, '..')

from helios_anomaly_scope import QRNGStreamScope, SignalClass


def run_basic_analysis(num_steps: int = 1000, inject_bias_at: int = None):
    """
    Run a basic trajectory analysis.
    
    Args:
        num_steps: Number of steps to simulate
        inject_bias_at: Step at which to inject bias (or None for pure random)
    """
    print("=" * 60)
    print("HELIOS Basic Analysis Example")
    print("=" * 60)
    
    # Create the scope
    scope = QRNGStreamScope(
        history_len=100,     # Keep 100 steps for analysis
        walk_mode='angle'    # Use angle-based random walk
    )
    
    print(f"\n📊 Configuration:")
    print(f"   Steps: {num_steps}")
    print(f"   History: 100 steps")
    print(f"   Walk mode: angle")
    if inject_bias_at:
        print(f"   ⚠️  Bias injection at step {inject_bias_at}")
    
    print(f"\n🔄 Running analysis...")
    
    anomaly_count = 0
    
    for step in range(num_steps):
        # Generate value (with optional bias)
        if inject_bias_at and step >= inject_bias_at:
            # Inject bias - prefer upper angles
            value = np.random.random() * 0.3 + 0.6
        else:
            value = np.random.random()
        
        # Process the value
        metrics = scope.update_from_stream(value)
        
        # Skip while building history
        if metrics.get('waiting_for_history'):
            continue
        
        # Check for influence
        if metrics.get('influence_detected'):
            anomaly_count += 1
        
        # Print progress every 200 steps
        if step > 0 and step % 200 == 0:
            print(f"   Step {step}: H={metrics.get('hurst', 0):.3f}, "
                  f"λ={metrics.get('lyapunov', 0):.3f}, "
                  f"influence={metrics.get('influence_detected', False)}")
    
    # Final results
    print(f"\n📈 Results:")
    print(f"   Total anomalies detected: {anomaly_count}")
    print(f"   Detection rate: {anomaly_count / num_steps * 100:.1f}%")
    
    # Verify signal
    print(f"\n🔍 Signal Verification:")
    verification = scope.verify_current_signal()
    
    if verification:
        print(f"   Classification: {verification.signal_class.name}")
        print(f"   Is verified: {verification.is_verified}")
        print(f"   Confidence: {verification.confidence:.2f}")
        print(f"   Tests passed: {verification.tests_passed}")
        print(f"   Tests failed: {verification.tests_failed}")
    
    # Get classification summary
    classification = scope.get_signal_classification()
    print(f"\n🎯 Final Classification:")
    print(f"   Type: {classification['signal_type']}")
    print(f"   Is real signal: {classification['is_signal']}")
    print(f"   Summary: {classification['classification_str']}")
    
    return scope


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Basic QRNG Analysis')
    parser.add_argument('--steps', type=int, default=1000, help='Number of steps')
    parser.add_argument('--inject-bias', type=int, default=None,
                        help='Step at which to inject bias')
    
    args = parser.parse_args()
    
    run_basic_analysis(
        num_steps=args.steps,
        inject_bias_at=args.inject_bias
    )
