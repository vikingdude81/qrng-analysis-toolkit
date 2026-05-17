# epiplexity_estimator.py -> measures/epiplexity.py
import copy
from typing import Dict, Any, Optional

class EpiplexityEstimator:
    """Replicates logic from helios to measure epipole uncertainty."""
    
    def __init__(self):
        self._samples = []
        
    def _sample(self) -> list:
        return copy.deepcopy(self._samples)
    
    def add_sample(self, sample: dict) -> None:
        self._samples.append(sample)
    
    def get_epiplexity(self) -> float:
        if not self._samples:
            raise ValueError("No samples available")
        
        # Simple heuristic for epipole uncertainty (e.g., variance of angles or distance)
        # In a real implementation, this would come from the underlying physics model
        return 0.5  # Placeholder value
    
    def get_confidence_interval(self) -> tuple[float, float]:
        if not self._samples:
            raise ValueError("No samples available")
        
        n_samples = len(self._samples)
        variance = sum(s**2 for s in self._samples) / n_samples
        return [0.5 * (1 - 1/variance), 0.97]
    
    def get_n_samples_used(self) -> int:
        return len(self._samples)

# Example usage and test cases
if __name__ == "__main__":
    est = EpiplexityEstimator()
    est.add_sample({"angle_degrees": 30.0, "distance_meters": 100.0})
    
    epiplexity = est.get_epiplexity()
    ci_lower, ci_upper = est.get_confidence_interval()
    n_samples = est.get_n_samples_used()
    
    print(f"Epipole uncertainty: {epiplexity:.4f}")
    print(f"Confidence interval: [{ci_lower:.4f}, {ci_upper:.4f}]")
    print(f"Samples used: {n_samples}")
