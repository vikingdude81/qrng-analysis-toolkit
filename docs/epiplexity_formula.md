# Epiplexity Formula Documentation

## Overview

The **Epiplexity** metric is a weighted composite score used in Helios pipelines to evaluate model performance during training and inference. It integrates multiple evaluation metrics (accuracy, precision, recall, F1-score) with domain-specific weights to provide a single reliability indicator.

## Formula Definition

### Basic Epiplexity Calculation

```
epiplexity = w_acc * accuracy + w_prec * precision + w_rec * recall + w_f1 * f1_score
```

Where:
- `w_acc`, `w_prec`, `w_rec`, `w_f1` are weights assigned based on domain priorities
- Weights typically sum to 1.0 (or another normalization factor)

### Example Weight Configuration

```python
# Domain-specific priority example
epiplexity = (
    0.4 * accuracy +      # Primary: correct predictions
    0.3 * precision +     # Secondary: avoid false positives
    0.2 * recall +        # Tertiary: avoid false negatives
    0.1 * f1_score        # Bonus: harmonic mean consideration
)
```

## Usage in Helios Pipelines

### 1. Training Phase

Epiplexity is computed during training to:
- Monitor model convergence and stability
- Guide hyperparameter tuning decisions
- Detect performance degradation early

```python
from helios.metrics import calculate_epiplexity

# Compute epiplexity after each epoch
epi = calculate_epiplexity(
    accuracy=0.85,
    precision=0.82,
    recall=0.79,
    f1_score=0.80,
    weights={
        'accuracy': 0.4,
        'precision': 0.3,
        'recall': 0.2,
        'f1_score': 0.1
    }
)
```

### 2. Inference Phase

During inference, epiplexity serves as:
- A reliability indicator for model outputs
- A threshold for confidence-based filtering
- Input to consciousness metrics for awareness monitoring

```python
# Use epiplexity to gate predictions
if epiplexity > THRESHOLD:
    accept_prediction = True
else:
    flag_for_review = True
```

### 3. Anomaly Detection Integration

Epiplexity complements consciousness metrics in anomaly detection:

```python
from helios.metrics import calculate_consciousness

# Combined reliability score
reliability_score = (
    epiplexity * 0.6 +      # Performance component
    calculate_consciousness() * 0.4   # Awareness component
)
```

## Consciousness Metrics Integration

Epiplexity works alongside consciousness metrics to ensure:

1. **Predictive Reliability**: Models maintain high epiplexity scores in critical applications
2. **Decision-Making Robustness**: Low epiplexity triggers fallback mechanisms
3. **Awareness Monitoring**: Consciousness metrics detect when models are overconfident despite low performance

### Example: Decision Gate

```python
def should_trust_model(epiplexity, consciousness):
    """
    Determine if model output should be trusted.
    
    Args:
        epiplexity: Performance-based reliability score (0-1)
        consciousness: Awareness metric from consciousness_metrics.py
    
    Returns:
        True if model is trustworthy, False otherwise
    """
    MIN_EPIPLEXITY = 0.75
    MIN_CONSCIOUSNESS = 0.60
    MAX_OVERCONFIDENCE_THRESHOLD = 0.85
    
    # Check performance threshold
    if epiplexity < MIN_EPIPLEXITY:
        return False, "Low performance reliability"
    
    # Check awareness threshold
    if consciousness < MIN_CONSCIOUSNESS:
        return False, "Insufficient model awareness"
    
    # Check for overconfidence (high confidence but low actual performance)
    if epiplexity > MAX_OVERCONFIDENCE_THRESHOLD and consciousness < 0.5:
        return False, "Overconfident but unaware"
    
    return True, "Model is trustworthy"
```

## Domain-Specific Weight Configurations

### Anomaly Detection

```python
ANOMALY_WEIGHTS = {
    'accuracy': 0.3,
    'precision': 0.4,   # Critical: avoid false positives in anomaly detection
    'recall': 0.2,      # Important: catch anomalies but prioritize precision
    'f1_score': 0.1
}
```

### Classification Tasks

```python
CLASSIFICATION_WEIGHTS = {
    'accuracy': 0.5,
    'precision': 0.2,
    'recall': 0.2,
    'f1_score': 0.1
}
```

### Medical/Diagnostic Applications

```python
MEDICAL_WEIGHTS = {
    'accuracy': 0.3,
    'precision': 0.2,   # Avoid false positives (unnecessary treatment)
    'recall': 0.4,      # Critical: avoid false negatives (missed diagnosis)
    'f1_score': 0.1
}
```

## Implementation Notes

### Numerical Stability

When computing epiplexity:
- Use `np.clip()` to prevent NaN propagation
- Handle edge cases where metrics are undefined
- Normalize weights to sum to 1.0 for fair comparison

```python
import numpy as np

def safe_epiplexity(**metrics):
    """Compute epiplexity with numerical safety."""
    weights = {
        'accuracy': 0.4,
        'precision': 0.3,
        'recall': 0.2,
        'f1_score': 0.1
    }
    
    result = 0.0
    for metric, weight in weights.items():
        value = metrics.get(metric)
        if value is not None:
            # Clip to [0, 1] range
            clipped = np.clip(value, 0.0, 1.0)
            result += clipped * weight
    
    return float(result)
```

### Performance Monitoring

Track epiplexity over time to detect:
- **Drift**: Gradual degradation in performance
- **Instability**: High variance in scores
- **Threshold violations**: Crossing critical boundaries

```python
class EpiplexityMonitor:
    def __init__(self, threshold=0.75):
        self.threshold = threshold
        self.history = []
        
    def record(self, epiplexity):
        self.history.append(epiplexity)
        if epiplexity < self.threshold:
            self.alert()
    
    def alert(self):
        print(f"⚠️  Epiplexity dropped below threshold: {self.history[-1]:.3f}")
```

## Related Files

- `epiplexity_estimator.py` - Core implementation
- `consciousness_metrics.py` - Complementary awareness metrics
- `inference_framework/classifier/` - Integration points
- `measures/` - Additional performance measures

## See Also

- [Consciousness Metrics](../consciousness_metrics.md)
- [Chaos Analysis](../chaos_analysis.py)
- [QRNG Comprehensive Analysis](../qrng_comprehensive_analysis.py)
