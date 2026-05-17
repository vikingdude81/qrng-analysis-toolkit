import numpy as np
from typing import Callable, Union, Tuple
import warnings


def kNN_KozachenkoLeonenko(data: np.ndarray, k: int = 2) -> float:
    """
    Kozachenko-Leonenko entropy estimator using k-Nearest Neighbors.
    
    This is a non-parametric entropy estimator that uses the distances
    to k-nearest neighbors in the data space.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    k : int
        Number of nearest neighbors to use (default: 2)
    
    Returns
    -------
    float
        Estimated entropy value in nats
    """
    if len(data) < k + 1:
        raise ValueError(f"Need at least {k + 1} samples for kNN estimator")
    
    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(data, data)
    
    # Get k-nearest neighbor distances (excluding self)
    min_distances = np.sort(dist_matrix[np.arange(len(data)), :, np.newaxis])[:, 1:k+1]
    
    # Kozachenko-Leonenko estimator
    n = len(data)
    d = data.shape[1]  # dimensionality
    
    if d == 0:
        return 0.0
    
    # Volume of k-dimensional unit ball
    from scipy.special import gamma
    c_d = np.pi ** (d / 2) / gamma(d / 2 + 1)
    
    # KL estimator
    kl_entropy = d * np.log(2 ** d) - np.sum(np.log(min_distances))
    kl_entropy -= d * np.log(n) + np.log(k) - np.log(c_d)
    
    return float(kl_entropy)


def biasCorrectedMillerMadow(data: np.ndarray, n: int = None) -> float:
    """
    Bias-corrected Miller-Madow entropy estimator.
    
    This is a histogram-based entropy estimator with bias correction
    for small sample sizes.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array (1D or 2D flattened)
    n : int
        Number of samples (default: len(data))
    
    Returns
    -------
    float
        Bias-corrected entropy estimate in nats
    """
    if n is None:
        n = len(data)
    
    # Flatten data if 2D
    if len(data.shape) > 1:
        data = data.flatten()
    
    # Compute histogram with optimal binning
    hist, _ = np.histogram(data, bins='auto')
    
    # Normalize to get probabilities
    p = hist / n
    
    # Remove zero-probability bins
    p = p[p > 0]
    
    # Shannon entropy
    shannon_entropy = -np.sum(p * np.log2(p))
    
    # Bias correction for small samples (Miller-Madow)
    n_bins = len(p)
    bias_correction = (n_bins - 1) / (2 * n) * np.log(n_bins)
    
    corrected_entropy = shannon_entropy - bias_correction
    
    return float(corrected_entropy)


def renyiAlphaHalfAndTwo(data: np.ndarray) -> Tuple[float, float]:
    """
    Rényi α=0.5 and α=2 entropy estimators via correlation integral.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array of shape (n_samples, n_features)
    
    Returns
    -------
    tuple[float, float]
        (Rényi alpha=0.5 entropy, Rényi alpha=2 entropy) in nats
    """
    if len(data) < 4:
        raise ValueError("Need at least 4 samples for correlation integral estimators")
    
    # Compute pairwise distances
    from scipy.spatial.distance import cdist
    dist_matrix = cdist(data, data)
    
    # Get non-zero distances (excluding diagonal)
    np.fill_diagonal(dist_matrix, np.inf)  # Exclude self-distances
    distances = dist_matrix.flatten()
    distances = distances[distances < np.inf]
    
    if len(distances) == 0:
        return (float('nan'), float('nan'))
    
    # Rényi entropy via correlation integral
    # H_alpha = (1 / (1 - alpha)) * log(sum_{i!=j} d(x_i, x_j)^alpha)
    
    def renyi_from_correlation(distances: np.ndarray, alpha: float) -> float:
        if alpha == 2:
            # For alpha=2, use sum of squared distances
            sum_sq = np.sum(distances ** 2)
            n = len(distances)
            return (1 / (1 - alpha)) * np.log(sum_sq) if sum_sq > 0 else float('nan')
        
        # For other alphas, use correlation integral approximation
        # C(alpha) = sum_{i!=j} d(x_i, x_j)^alpha
        c_alpha = np.sum(distances ** alpha)
        n = len(distances)
        
        if c_alpha <= 0:
            return float('nan')
        
        return (1 / (1 - alpha)) * np.log(c_alpha / n)
    
    alpha_half = renyi_from_correlation(distances, 0.5)
    alpha_two = renyi_from_correlation(distances, 2.0)
    
    return (float(alpha_half), float(alpha_two))


def tsallisQ1_2Entropy(data: np.ndarray) -> float:
    """
    Tsallis q=1.2 entropy estimator.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array (1D or 2D flattened)
    
    Returns
    -------
    float
        Tsallis q=1.2 entropy estimate in nats
    """
    if len(data) < 4:
        raise ValueError("Need at least 4 samples for Tsallis estimator")
    
    # Flatten data if 2D
    if len(data.shape) > 1:
        data = data.flatten()
    
    # Compute histogram with optimal binning
    hist, _ = np.histogram(data, bins='auto')
    
    # Normalize to get probabilities
    p = hist / len(data)
    
    # Remove zero-probability bins
    p = p[p > 0]
    
    if len(p) == 0:
        return float('nan')
    
    q = 1.2
    
    # Tsallis entropy: S_q = (1 - sum(p_i^q)) / (q - 1)
    tsallis_entropy = (1 - np.sum(p ** q)) / (q - 1)
    
    return float(tsallis_entropy)


def gpuHistogramEstimator(data: np.ndarray) -> float:
    """
    GPU-accelerated histogram entropy estimator using CuPy.
    
    Parameters
    ----------
    data : np.ndarray
        Input data array (1D or 2D flattened)
    
    Returns
    -------
    float
        Entropy estimate in nats
    """
    try:
        import cupy as cp
        
        # Check if data is on GPU
        if not hasattr(data, 'device') or data.device.type != 'gpu':
            # Transfer to GPU
            data_gpu = cp.asarray(data)
        else:
            data_gpu = data
        
        # Compute histogram on GPU
        hist_gpu, bin_edges = cp.histogram(data_gpu, bins='auto')
        
        # Normalize
        n_samples = len(data)
        p = hist_gpu / n_samples
        
        # Remove zero-probability bins
        p = p[p > 0]
        
        if len(p) == 0:
            return float('nan')
        
        # Shannon entropy on GPU
        entropy_gpu = -cp.sum(p * cp.log2(p))
        
        return float(entropy_gpu.get())
    except ImportError:
        warnings.warn("CuPy not available, falling back to CPU implementation")
        return biasCorrectedMillerMadow(data)


# Unit-test stubs
def test_kNN_KozachenkoLeonenko():
    """Test Kozachenko-Leonenko entropy estimator."""
    data = np.random.rand(100, 2)
    result = kNN_KozachenkoLeonenko(data)
    assert isinstance(result, float)
    assert not np.isnan(result) and not np.isinf(result)


def test_biasCorrectedMillerMadow():
    """Test bias-corrected Miller-Madow entropy estimator."""
    data = np.random.rand(100)
    result = biasCorrectedMillerMadow(data)
    assert isinstance(result, float)
    assert not np.isnan(result) and not np.isinf(result)


def test_renyiAlphaHalfAndTwo():
    """Test Rényi alpha=0.5 and alpha=2 entropy estimators."""
    data = np.random.rand(100, 2)
    alpha_half, alpha_two = renyiAlphaHalfAndTwo(data)
    assert isinstance(alpha_half, float) and isinstance(alpha_two, float)


def test_tsallisQ1_2Entropy():
    """Test Tsallis q=1.2 entropy estimator."""
    data = np.random.rand(100)
    result = tsallisQ1_2Entropy(data)
    assert isinstance(result, float)


def test_gpuHistogramEstimator():
    """Test GPU histogram entropy estimator."""
    data = np.random.rand(100)
    result = gpuHistogramEstimator(data)
    assert isinstance(result, float) or np.isnan(result)
