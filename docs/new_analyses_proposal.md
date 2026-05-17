# New Analyses Proposal for Helios Trajectory Analysis

## Overview
This document outlines proposed new statistical analyses to enhance the consciousness metrics and chaos analysis capabilities of the Helios framework.

---

## 1. Multiscale Entropy Flow (MSEF)

### Purpose
Analyze how entropy evolves across different temporal scales to detect non-stationary dynamics or chaotic attractors relevant to consciousness.

### Implementation
```python
from typing import List, Tuple
import numpy as np

def multiscale_entropy_flow(
    x: np.ndarray,
    scale_factor: int = 2,
    max_scale: int = 10,
    embedding_dim: int = 3
) -> List[Tuple[int, float]]:
    """
    Calculate entropy across multiple temporal scales.
    
    Args:
        x: Input time series
        scale_factor: Factor by which to downsample at each scale
        max_scale: Maximum number of scales
        embedding_dim: Embedding dimension for permutation entropy
    
    Returns:
        List of (scale, entropy) tuples
    """
    results = []
    current_scale = 1
    
    # Downsample at each scale
    for scale in range(1, max_scale + 1):
        if len(x) < embedding_dim + 2:
            break
        
        # Downsample by taking every nth point
        step = scale_factor ** (scale - 1)
        if step >= len(x):
            break
        
        x_scaled = x[::step]
        entropy, method = permutation_entropy(x_scaled, embedding_dim=embedding_dim)
        results.append((current_scale * scale, entropy))
    
    return results
```

### Consciousness Relevance
- **Scale-dependent dynamics**: Conscious processes may exhibit different entropy patterns at different temporal scales
- **Non-stationarity detection**: Abrupt changes in entropy across scales may indicate phase transitions
- **Chaos attractor identification**: Specific scale patterns may correspond to known chaotic systems

---

## 2. Conditional Entropy Flow (CEF)

### Purpose
Model the conditional distribution of states over time, identifying hidden variables and potential causal pathways in neural activity.

### Implementation
```python
from typing import List, Dict, Tuple
import numpy as np
from scipy.stats import entropy as scipy_entropy

def conditional_entropy_flow(
    x: np.ndarray,
    y: np.ndarray,
    embedding_dim: int = 2,
    time_lags: List[int] = None
) -> Dict[str, float]:
    """
    Calculate conditional entropy to identify causal relationships.
    
    Args:
        x: Input sequence (potential cause)
        y: Output sequence (potential effect)
        embedding_dim: Embedding dimension
        time_lags: List of time lags to consider
    
    Returns:
        Dictionary mapping lag to conditional entropy
    """
    if time_lags is None:
        time_lags = [1, 2, 3, 5, 10]
    
    results = {}
    
    for lag in time_lags:
        if len(x) < embedding_dim + lag or len(y) < embedding_dim + lag:
            continue
        
        # Create joint embedding vectors
        x_embedded = []
        y_embedded = []
        
        for i in range(embedding_dim, min(len(x), len(y)) - lag):
            x_vec = x[i-embedding_dim:i]
            y_vec = y[i:i+embedding_dim]
            x_embedded.append(x_vec)
            y_embedded.append(y_vec)
        
        if not x_embedded:
            continue
        
        # Calculate conditional entropy H(Y|X)
        joint_states, counts = np.unique(
            list(zip(*[np.array(v) for v in zip(x_embedded, y_embedded)])),
            return_counts=True
        )
        
        if len(joint_states) < 2:
            results[str(lag)] = 0.0
            continue
        
        probs = counts / len(counts)
        probs = probs[probs > 0]
        
        # Marginal entropy of Y
        y_counts, _ = np.unique(y_embedded, return_counts=True)
        y_probs = y_counts / len(y_embedded)
        y_probs = y_probs[y_probs > 0]
        h_y = -np.sum(y_probs * np.log2(y_probs))
        
        # Joint entropy H(X,Y)
        joint_probs = probs
        h_xy = -np.sum(joint_probs * np.log2(joint_probs))
        
        # Conditional entropy H(Y|X) = H(X,Y) - H(X)
        x_counts, _ = np.unique(x_embedded, return_counts=True)
        x_probs = x_counts / len(x_embedded)
        x_probs = x_probs[x_probs > 0]
        h_x = -np.sum(x_probs * np.log2(x_probs))
        
        conditional_entropy = h_xy - h_x
        results[str(lag)] = float(conditional_entropy)
    
    return results
```

### Consciousness Relevance
- **Causal pathway identification**: High conditional entropy at specific lags indicates information flow
- **Hidden variable detection**: Low conditional entropy suggests deterministic relationships
- **Temporal hierarchy**: Different lags may correspond to different cognitive processes

---

## 3. Causal Web Complexity (CWC)

### Purpose
Construct a dynamic graph of state transitions based on observed correlations, quantifying emergent complexity that may indicate higher-order consciousness or cognitive integration.

### Implementation
```python
from typing import List, Dict, Tuple, Set
import numpy as np

def causal_web_complexity(
    x: np.ndarray,
    y: np.ndarray,
    embedding_dim: int = 3,
    correlation_threshold: float = 0.5,
    time_lags: List[int] = None
) -> Tuple[np.ndarray, Dict[str, float]]:
    """
    Construct causal web and calculate complexity metrics.
    
    Args:
        x: Input sequence
        y: Output sequence
        embedding_dim: Embedding dimension
        correlation_threshold: Minimum correlation for edge inclusion
        time_lags: Time lags to consider
    
    Returns:
        Tuple of (adjacency matrix, complexity metrics)
    """
    if time_lags is None:
        time_lags = [1, 2, 3, 5]
    
    # Build state space representation
    n_states = min(len(x), len(y))
    
    # Create adjacency matrix
    adj_matrix = np.zeros((embedding_dim + 1, embedding_dim + 1))
    
    # Calculate correlations across time lags
    for lag in time_lags:
        if n_states < embedding_dim + lag:
            continue
        
        x_window = x[lag:]
        y_window = y[:len(x_window)]
        
        # Compute correlation matrix of embeddings
        x_embed = _create_embeddings(x_window, embedding_dim)
        y_embed = _create_embeddings(y_window, embedding_dim)
        
        if len(x_embed) < 2:
            continue
        
        # Correlation between consecutive states
        corr_matrix = np.corrcoef(
            x_embed.reshape(-1, embedding_dim),
            y_embed.reshape(-1, embedding_dim)
        )
        
        # Threshold-based edge creation
        for i in range(embedding_dim + 1):
            for j in range(i + 1, embedding_dim + 1):
                if abs(corr_matrix[i, j]) > correlation_threshold:
                    adj_matrix[i, j] = corr_matrix[i, j]
                    adj_matrix[j, i] = corr_matrix[i, j]
    
    # Calculate complexity metrics
    metrics = _calculate_complexity_metrics(adj_matrix)
    
    return adj_matrix, metrics

def _create_embeddings(x: np.ndarray, dim: int) -> np.ndarray:
    """Create embedding vectors."""
    n_samples = len(x) - dim
    embeddings = []
    for i in range(n_samples):
        embeddings.append(x[i:i+dim])
    return np.array(embeddings)

def _calculate_complexity_metrics(adj_matrix: np.ndarray) -> Dict[str, float]:
    """Calculate complexity metrics from adjacency matrix."""
    n = adj_matrix.shape[0]
    
    # Normalized Laplacian
    degree = np.sum(np.abs(adj_matrix), axis=1)
    degree_norm = np.sqrt(degree)
    laplacian = np.eye(n) - (adj_matrix / (degree_norm.reshape(-1, 1) * degree_norm.reshape(1, -1) + 1e-10))
    
    # Spectral entropy
    eigenvalues = np.linalg.eigvalsh(laplacian)
    eigenvalues = eigenvalues[eigenvalues > 0]
    if len(eigenvalues) > 0:
        spec_entropy = -np.sum((eigenvalues / np.sum(eigenvalues)) * np.log2(eigenvalues + 1e-10))
    else:
        spec_entropy = 0.0
    
    # Assortativity (clustering coefficient)
    clustering = _calculate_clustering(adj_matrix)
    
    return {
        'spectral_entropy': float(spec_entropy),
        'clustering_coefficient': float(clustering),
        'network_density': float(np.sum(adj_matrix) / (n * (n - 1)))
    }
```

### Consciousness Relevance
- **Integration measure**: High spectral entropy indicates distributed information processing
- **Modularity**: Clustering coefficient reflects functional segregation
- **Emergent complexity**: Network density correlates with cognitive integration

---

## Integration Recommendations

1. **Add to `chaos_analysis.py`**: Implement MSEF and CEF as new analysis modules
2. **Add to `consciousness_metrics.py`**: Integrate CWC metrics into consciousness scoring
3. **Create dedicated module**: Consider a new `causal_analysis.py` for all causal inference methods
4. **Update documentation**: Add examples and theoretical background for each method

---

## Testing Strategy

```python
import pytest

def test_multiscale_entropy_flow():
    """Test MSEF with synthetic data."""
    np.random.seed(42)
    x = np.cumsum(np.random.randn(100))  # Random walk
    
    results = multiscale_entropy_flow(x, max_scale=5)
    assert len(results) > 0
    assert all(isinstance(r[1], float) for r in results)

def test_conditional_entropy_flow():
    """Test CEF with synthetic data."""
    np.random.seed(42)
    x = np.cumsum(np.random.randn(50))
    y = np.cumsum(np.random.randn(50) + 0.1 * x[:-1])  # With some dependence
    
    results = conditional_entropy_flow(x, y, time_lags=[1, 2])
    assert all(isinstance(v, float) for v in results.values())

def test_causal_web_complexity():
    """Test CWC with synthetic data."""
    np.random.seed(42)
    x = np.cumsum(np.random.randn(100))
    y = np.cumsum(np.random.randn(100))
    
    adj, metrics = causal_web_complexity(x, y)
    assert adj.shape[0] > 0
    assert 'spectral_entropy' in metrics
```