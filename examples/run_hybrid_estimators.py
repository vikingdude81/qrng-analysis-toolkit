"""
Example: Running Hybrid Entropy Estimators on Helios Data

This script demonstrates how to use the new hybrid entropy estimators
(SWWE and NSAE) for consciousness metrics analysis.
"""

import numpy as np
from src.entropy_estimators import (
    SymbolicWeightedWaveletEntropy,
    NonStationarySymbolicEntropy,
    EntropyEstimatorEnsemble
)
from src.metrics_integration import ConsciousnessMetricsPipeline


def generate_sample_helios_data(size: int = 1000, seed: int = 42):
    """
    Generate sample helios-like data for testing.
    
    Args:
        size: Number of samples
        seed: Random seed for reproducibility
        
    Returns:
        numpy array of sample data
    """
    np.random.seed(seed)
    # Simulate helios data with some structure
    t = np.arange(size)
    base_signal = np.sin(2 * np.pi * 0.1 * t) + np.sin(2 * np.pi * 0.3 * t)
    noise = np.random.randn(size) * 0.5
    helios_data = base_signal + noise
    return helios_data


def main():
    """Main demonstration function."""
    print("="*60)
    print("Helios Hybrid Entropy Estimator Demo")
    print("="*60)
    
    # Generate sample data
    print("\nGenerating sample helios data...")
    helios_data = generate_sample_helios_data(size=500, seed=42)
    print(f"Data shape: {helios_data.shape}")
    print(f"Data range: [{helios_data.min():.4f}, {helios_data.max():.4f}]")
    
    # Method 1: Symbolic-Weighted Wavelet Entropy (SWWE)
    print("\n" + "="*60)
    print("Method 1: Symbolic-Weighted Wavelet Entropy (SWWE)")
    print("="*60)
    
    swwe = SymbolicWeightedWaveletEntropy(
        embedding_dim=3,
        tau=1,
        wavelet_type='db4',
        wavelet_level=2
    )
    entropy_swwe = swwe.fit_transform(helios_data)
    print(f"SWWE Entropy: {entropy_swwe:.4f}")
    
    # Method 2: Non-Stationary Symbolic Entropy (NSAE)
    print("\n" + "="*60)
    print("Method 2: Non-Stationary Symbolic Entropy (NSAE)")
    print("="*60)
    
    nsaes = NonStationarySymbolicEntropy(
        embedding_dim=3,
        tau=1,
        transition_window=10,
        alphabet_size=8
    )
    entropy_nsaes = nsaes.fit_transform(helios_data)
    print(f"NSAE Entropy: {entropy_nsaes:.4f}")
    
    # Method 3: Full Ensemble (all estimators)
    print("\n" + "="*60)
    print("Method 3: Full Ensemble (All Estimators)")
    print("="*60)
    
    ensemble = EntropyEstimatorEnsemble()
    entropies = ensemble.fit_transform(helios_data)
    
    results = {
        'sample_entropy': entropies[0],
        'approximate_entropy': entropies[1],
        'permutation_entropy': entropies[2],
        'swwe_entropy': entropies[3],
        'nsae_entropy': entropies[4],
    }
    
    for name, value in results.items():
        print(f"{name}: {value:.4f}")
    
    # Method 4: Consciousness Metrics Pipeline
    print("\n" + "="*60)
    print("Method 4: Consciousness Metrics Pipeline")
    print("="*60)
    
    pipeline = ConsciousnessMetricsPipeline(use_hybrid=True)
    state = pipeline.classify_state(helios_data)
    
    print(f"State Label: {state.state_label}")
    print(f"Confidence: {state.confidence:.4f}")
    print(f"Sample Entropy: {state.sample_entropy:.4f}")
    print(f"SWWE Entropy: {state.swwe_entropy:.4f}")
    print(f"NSAE Entropy: {state.nsaes_entropy:.4f}")
    
    # Detect state transitions
    print("\n" + "="*60)
    print("State Transition Detection")
    print("="*60)
    
    transitions = pipeline.detect_state_transitions(
        helios_data,
        window_size=50,
        threshold=0.3
    )
    
    if transitions:
        print(f"Detected {len(transitions)} state transitions")
        for ts, label in transitions[:5]:  # Show first 5
            print(f"  Timestamp {ts}: {label}")
    else:
        print("No significant transitions detected")
    
    print("\n" + "="*60)
    print("Demo Complete!")
    print("="*60)


if __name__ == "__main__":
    main()
