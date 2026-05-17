"""Integration tests for QRNG analysis framework with synthetic data."""
import numpy as np

def test_synthetic_qrng_data():
    """Test estimators and detectors with synthetic QRNG data."""
    # Generate synthetic QRNG data
    qrng_data = np.random.binomial(1, 0.5, size=1000)
    
    # Import estimators and run tests
    from inference_framework.experiment.entropy_estimators import (
        ShannonEntropyEstimator,
        RenyiEntropyEstimator,
        CollisionEntropyEstimator
    )
    
    # Test entropy estimators
    shannon_estimator = ShannonEntropyEstimator()
    rennyi_estimator = RenyiEntropyEstimator()
    collision_estimator = CollisionEntropyEstimator()
    
    shannon_entropy = shannon_estimator.compute_entropy(qrng_data)
    rennyi_entropy = rennyi_estimator.compute_entropy(qrng_data)
    collision_entropy = collision_estimator.compute_entropy(qrng_data)
    
    print(f"Shannon Entropy: {shannon_entropy}")
    print(f"Renyi Entropy: {rennyi_entropy}")
    print(f"Collision Entropy: {collision_entropy}")
    
    # Verify entropy values are reasonable (should be > 0 for random data)
    assert shannon_entropy > 0, "Shannon entropy should be positive"
    assert rennyi_entropy > 0, "Renyi entropy should be positive"
    assert collision_entropy > 0, "Collision entropy should be positive"
    
    return {
        'shannon_entropy': shannon_entropy,
        'renyi_entropy': rennyi_entropy,
        'collision_entropy': collision_entropy
    }

def test_edge_cases():
    """Test edge cases: empty arrays, NaNs."""
    from inference_framework.experiment.entropy_estimators import (
        ShannonEntropyEstimator,
        RenyiEntropyEstimator,
        CollisionEntropyEstimator
    )
    
    estimators = [
        ShannonEntropyEstimator(),
        RenyiEntropyEstimator(),
        CollisionEntropyEstimator()
    ]
    
    # Test empty array
    empty_data = np.array([])
    for estimator in estimators:
        result = estimator.compute_entropy(empty_data)
        assert result == 0.0, f"Empty array should return 0 entropy"
    
    # Test NaN values
    nan_data = np.array([float('nan'), float('nan')])
    for estimator in estimators:
        try:
            result = estimator.compute_entropy(nan_data)
            print(f"NaN handling result: {result}")
        except Exception as e:
            print(f"NaN handling raised exception: {e}")
    
    print("Edge case tests passed!")
