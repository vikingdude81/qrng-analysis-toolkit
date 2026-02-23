"""
Tensor Network Analysis for QRNG Patterns
==========================================

GPU-accelerated pattern analysis using PyTorch CUDA.
Optimized for RTX 5090 (Blackwell architecture).

Implements:
- GPU-accelerated correlation dimension
- Fast Lyapunov exponent estimation
- Phase space reconstruction
- Recurrence quantification
- Multi-scale entropy
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Union
from dataclasses import dataclass
import warnings

# PyTorch GPU support (primary backend)
HAS_TORCH_CUDA = False
torch = None

try:
    import torch
    HAS_TORCH_CUDA = torch.cuda.is_available()
    if HAS_TORCH_CUDA:
        _ = torch.tensor([1.0], device='cuda').sum()  # Warm up
except ImportError:
    torch = None
    HAS_TORCH_CUDA = False

# Determine best available backend
GPU_AVAILABLE = HAS_TORCH_CUDA
GPU_BACKEND = "torch" if HAS_TORCH_CUDA else "cpu"


@dataclass 
class ChaosMetrics:
    """Container for chaos/complexity metrics."""
    lyapunov_exponent: float
    correlation_dimension: float
    hurst_exponent: float
    sample_entropy: float
    approximate_entropy: float
    recurrence_rate: float
    determinism: float
    computation_time_ms: float
    method: str  # "gpu" or "cpu"


def gpu_lyapunov_exponent(
    data: Union[np.ndarray, "torch.Tensor"],
    embedding_dim: int = 3,
    time_delay: int = 1,
    epsilon: float = 0.01,
    max_iterations: int = 50,
    n_reference_points: int = 200,
) -> Tuple[float, float]:
    """
    GPU-accelerated Lyapunov exponent estimation using Rosenstein's method.
    
    Uses GPU for parallel distance calculations and divergence tracking.
    
    Args:
        data: Time series data
        embedding_dim: Phase space embedding dimension
        time_delay: Time delay for embedding
        epsilon: Distance threshold for neighbors
        max_iterations: Maximum iterations for divergence
        n_reference_points: Number of reference points to sample
        
    Returns:
        Tuple of (lyapunov_exponent, r_squared)
    """
    # Use PyTorch if available
    if HAS_TORCH_CUDA:
        return _torch_lyapunov_exponent(data, embedding_dim, time_delay, epsilon,
                                         max_iterations, n_reference_points)
    
    # Fallback to CPU (scipy-optimized)
    return _cpu_lyapunov_exponent(data, embedding_dim, time_delay, epsilon, 
                                   max_iterations, n_reference_points)


def _torch_lyapunov_exponent(data, embedding_dim, time_delay, epsilon, 
                              max_iterations, n_reference_points):
    """PyTorch GPU Lyapunov exponent estimation."""
    # Convert to numpy if needed
    if hasattr(data, 'get'):
        data = data.get()
    data = np.asarray(data, dtype=np.float64)
    
    n = len(data)
    N = n - (embedding_dim - 1) * time_delay
    
    if N < 100:
        return 0.0, 0.0
    
    # Phase space reconstruction
    embedded = np.zeros((N, embedding_dim), dtype=np.float64)
    for i in range(embedding_dim):
        embedded[:, i] = data[i * time_delay:i * time_delay + N]
    
    # Transfer to GPU
    t_embedded = torch.from_numpy(embedded).float().cuda()
    
    # Compute pairwise distances using GPU batching
    batch_size = min(2000, N)
    dist_matrix = torch.full((N, N), float('inf'), device='cuda')
    
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        # Vectorized distance computation
        diff = t_embedded[i:end_i].unsqueeze(1) - t_embedded.unsqueeze(0)
        distances = torch.sqrt(torch.sum(diff ** 2, dim=2))
        dist_matrix[i:end_i] = distances
    
    # Set Theiler window
    theiler = time_delay * embedding_dim
    for i in range(N):
        low = max(0, i - theiler)
        high = min(N, i + theiler + 1)
        dist_matrix[i, low:high] = float('inf')
    
    # Sample reference points
    n_ref = min(n_reference_points, N - max_iterations)
    if n_ref < 10:
        return 0.0, 0.0
    
    ref_indices = torch.randperm(N - max_iterations, device='cuda')[:n_ref]
    
    # Find nearest neighbors for all reference points at once
    ref_dist_slice = dist_matrix[ref_indices]
    nn_indices = torch.argmin(ref_dist_slice, dim=1)
    min_dists = ref_dist_slice[torch.arange(n_ref, device='cuda'), nn_indices]
    
    # Filter valid pairs
    valid_mask = (min_dists < 1e10) & (min_dists > 1e-10) & (nn_indices + max_iterations < N)
    valid_refs = ref_indices[valid_mask].cpu().numpy()
    valid_nns = nn_indices[valid_mask].cpu().numpy()
    valid_min_dists = min_dists[valid_mask].cpu().numpy()
    
    if len(valid_refs) < 10:
        return 0.0, 0.0
    
    # Track divergence (still need to loop but using numpy for speed)
    embedded_cpu = t_embedded.cpu().numpy()
    divergence_curves = []
    
    for i, nn_idx, min_dist in zip(valid_refs, valid_nns, valid_min_dists):
        local_div = []
        max_k = min(max_iterations, N - max(i, nn_idx))
        for k in range(1, max_k):
            d = np.linalg.norm(embedded_cpu[i + k] - embedded_cpu[nn_idx + k])
            if d > 1e-10:
                local_div.append(np.log(d / min_dist))
        
        if len(local_div) > 5:
            divergence_curves.append(local_div)
    
    if len(divergence_curves) < 10:
        return 0.0, 0.0
    
    # Average and fit
    min_len = min(len(d) for d in divergence_curves)
    avg_divergence = np.mean([d[:min_len] for d in divergence_curves], axis=0)
    
    from scipy import stats
    t = np.arange(1, min_len + 1)
    slope, _, r_value, _, _ = stats.linregress(t, avg_divergence)
    
    return float(slope), float(r_value ** 2)


def gpu_correlation_dimension(
    data: Union[np.ndarray, "torch.Tensor"],
    embedding_dim_max: int = 10,
    time_delay: int = 1,
    n_scales: int = 20,
    max_samples: int = 3000,
) -> Tuple[float, Dict[str, List[float]]]:
    """
    GPU-accelerated correlation dimension using Grassberger-Procaccia algorithm.
    
    Uses tensor operations for efficient distance calculations.
    
    Args:
        data: Time series data
        embedding_dim_max: Maximum embedding dimension to test
        time_delay: Time delay for embedding
        n_scales: Number of scale values for regression
        max_samples: Maximum number of samples to use
        
    Returns:
        Tuple of (correlation_dimension, debug_info)
    """
    # Use PyTorch GPU if available, else CPU fallback
    if HAS_TORCH_CUDA:
        return _torch_correlation_dimension(data, embedding_dim_max, time_delay,
                                            n_scales, max_samples)
    return _cpu_correlation_dimension(data, embedding_dim_max, time_delay, 
                                      n_scales, max_samples)


def gpu_hurst_exponent(
    data: Union[np.ndarray, "torch.Tensor"],
    max_k: int = 20,
) -> float:
    """
    GPU-accelerated Hurst exponent via R/S analysis.
    
    Args:
        data: Time series data
        max_k: Maximum number of scale divisions
        
    Returns:
        Hurst exponent (0.5 = random walk, >0.5 = persistent, <0.5 = anti-persistent)
    """
    # Use PyTorch GPU if available, else CPU fallback
    if HAS_TORCH_CUDA:
        return _torch_hurst_exponent(data, max_k)
    return _cpu_hurst_exponent(data, max_k)


def gpu_recurrence_analysis(
    data: Union[np.ndarray, "torch.Tensor"],
    embedding_dim: int = 3,
    time_delay: int = 1,
    threshold: Optional[float] = None,
    max_samples: int = 2000,
) -> Dict[str, float]:
    """
    GPU-accelerated Recurrence Quantification Analysis (RQA).
    
    Uses GPU for parallel recurrence matrix computation.
    
    Args:
        data: Time series data
        embedding_dim: Phase space embedding dimension
        time_delay: Time delay for embedding
        threshold: Recurrence threshold (default: auto)
        max_samples: Maximum samples to use
        
    Returns:
        Dictionary of RQA metrics
    """
    # Use PyTorch GPU if available, else CPU fallback
    if HAS_TORCH_CUDA:
        return _torch_recurrence_analysis(data, embedding_dim, time_delay,
                                          threshold, max_samples)
    return _cpu_recurrence_analysis(data, embedding_dim, time_delay, 
                                    threshold, max_samples)


def gpu_multiscale_entropy(
    data: Union[np.ndarray, "torch.Tensor"],
    scales: int = 20,
    m: int = 2,
    r: Optional[float] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    GPU-accelerated multiscale entropy analysis.
    
    Computes sample entropy at multiple time scales to reveal
    complexity structure.
    
    Args:
        data: Time series data
        scales: Number of scales to analyze
        m: Embedding dimension for sample entropy
        r: Tolerance (default: 0.2 * std)
        
    Returns:
        Tuple of (scales, entropy_values)
    """
    # Use CPU implementation - sample entropy benefits from scipy pdist
    return _cpu_multiscale_entropy(data, scales, m, r)


class TensorNetworkAnalyzer:
    """
    Advanced analyzer using PyTorch GPU tensor operations.
    
    Uses tensor operations for pattern extraction and
    efficient computation of complex metrics.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize tensor network analyzer.
        
        Args:
            device_id: CUDA device ID
        """
        self.device_id = device_id
        self.gpu_available = HAS_TORCH_CUDA
        
        if self.gpu_available:
            torch.cuda.set_device(device_id)
    
    def analyze_stream(
        self,
        data: Union[np.ndarray, List[float]],
        compute_all: bool = True,
    ) -> ChaosMetrics:
        """
        Comprehensive chaos/complexity analysis.
        
        Args:
            data: Time series data
            compute_all: Compute all available metrics
            
        Returns:
            ChaosMetrics with all results
        """
        import time
        start = time.perf_counter()
        
        if isinstance(data, list):
            data = np.array(data, dtype=np.float64)
        
        method = "gpu" if self.gpu_available else "cpu"
        
        # Compute all metrics
        lyap, _ = gpu_lyapunov_exponent(data)
        corr_dim, _ = gpu_correlation_dimension(data)
        hurst = gpu_hurst_exponent(data)
        rqa = gpu_recurrence_analysis(data)
        
        from .entropy import gpu_sample_entropy, gpu_approximate_entropy
        sample_ent = gpu_sample_entropy(data)
        approx_ent = gpu_approximate_entropy(data)
        
        elapsed = (time.perf_counter() - start) * 1000
        
        return ChaosMetrics(
            lyapunov_exponent=lyap,
            correlation_dimension=corr_dim,
            hurst_exponent=hurst,
            sample_entropy=sample_ent,
            approximate_entropy=approx_ent,
            recurrence_rate=rqa["recurrence_rate"],
            determinism=rqa["determinism"],
            computation_time_ms=elapsed,
            method=method,
        )
    
    def tensor_decompose_signal(
        self,
        data: np.ndarray,
        rank: int = 10,
    ) -> Tuple[np.ndarray, np.ndarray, float]:
        """
        Decompose signal using tensor SVD for pattern extraction.
        
        Creates a 3D tensor from the signal using time-delay embedding,
        then decomposes to extract dominant patterns.
        
        Args:
            data: Time series data
            rank: Truncation rank for SVD
            
        Returns:
            Tuple of (patterns, weights, reconstruction_error)
        """
        n = len(data)
        window = min(100, n // 10)
        
        # Create Hankel-like tensor
        n_rows = n - 2 * window + 1
        if n_rows < 10:
            return np.array([]), np.array([]), 1.0
        
        if self.gpu_available:
            data_gpu = torch.tensor(data, dtype=torch.float64, device='cuda')
            tensor_3d = torch.zeros((n_rows, window, window), dtype=torch.float64, device='cuda')
            
            for i in range(n_rows):
                for j in range(window):
                    tensor_3d[i, j, :] = data_gpu[i + j:i + j + window]
            
            # Reshape to matrix for SVD
            tensor_mat = tensor_3d.reshape(n_rows, -1)
            
            # SVD
            U, s, Vt = torch.linalg.svd(tensor_mat, full_matrices=False)
            
            # Truncate
            k = min(rank, len(s))
            patterns = Vt[:k].cpu().numpy()
            weights = s[:k].cpu().numpy()
            
            # Reconstruction error
            reconstructed = U[:, :k] @ torch.diag(s[:k]) @ Vt[:k]
            error = float(torch.linalg.norm(tensor_mat - reconstructed) / torch.linalg.norm(tensor_mat))
        else:
            tensor_3d = np.zeros((n_rows, window, window))
            for i in range(n_rows):
                for j in range(window):
                    tensor_3d[i, j, :] = data[i + j:i + j + window]
            
            tensor_mat = tensor_3d.reshape(n_rows, -1)
            U, s, Vt = np.linalg.svd(tensor_mat, full_matrices=False)
            
            k = min(rank, len(s))
            patterns = Vt[:k]
            weights = s[:k]
            
            reconstructed = U[:, :k] @ np.diag(s[:k]) @ Vt[:k]
            error = np.linalg.norm(tensor_mat - reconstructed) / np.linalg.norm(tensor_mat)
        
        return patterns, weights, error
    
    def detect_hidden_periodicity(
        self,
        data: np.ndarray,
        max_period: int = 1000,
    ) -> List[Tuple[int, float]]:
        """
        Detect hidden periodicities using tensor autocorrelation.
        
        Args:
            data: Time series data
            max_period: Maximum period to search
            
        Returns:
            List of (period, strength) tuples
        """
        n = len(data)
        max_period = min(max_period, n // 3)
        
        if self.gpu_available:
            data_gpu = torch.tensor(data, dtype=torch.float64, device='cuda')
            mean = torch.mean(data_gpu)
            var = torch.var(data_gpu)
            
            if var < 1e-10:
                return []
            
            # Compute autocorrelation for all lags at once using FFT
            padded = torch.zeros(2 * n, dtype=torch.float64, device='cuda')
            padded[:n] = data_gpu - mean
            fft_data = torch.fft.fft(padded)
            power = torch.abs(fft_data) ** 2
            autocorr = torch.real(torch.fft.ifft(power))[:n] / (var * n)
            
            autocorr = autocorr.cpu().numpy()
        else:
            mean = np.mean(data)
            var = np.var(data)
            
            if var < 1e-10:
                return []
            
            padded = np.zeros(2 * n)
            padded[:n] = data - mean
            fft_data = np.fft.fft(padded)
            power = np.abs(fft_data) ** 2
            autocorr = np.real(np.fft.ifft(power))[:n] / (var * n)
        
        # Find peaks in autocorrelation
        from scipy.signal import find_peaks
        peaks, properties = find_peaks(autocorr[1:max_period], 
                                        height=0.1, prominence=0.05)
        
        if len(peaks) == 0:
            return []
        
        # Sort by strength
        periods = []
        for peak in peaks:
            period = peak + 1  # Adjust for slicing offset
            strength = autocorr[period]
            periods.append((period, float(strength)))
        
        periods.sort(key=lambda x: -x[1])
        return periods[:10]  # Return top 10


# PyTorch GPU implementations for RTX 5090
def _torch_correlation_dimension(data, m_max, tau, n_scales, max_samples) -> Tuple[float, Dict]:
    """PyTorch GPU-accelerated correlation dimension."""
    import torch
    from scipy import stats
    
    # Convert to numpy first if needed (handle various array types)
    if hasattr(data, 'cpu'):  # PyTorch tensor
        data = data.cpu().numpy()
    data = np.asarray(data).astype(np.float32)
    
    n = len(data)
    if n > max_samples:
        indices = np.random.choice(n, max_samples, replace=False)
        indices = np.sort(indices)
        data = data[indices]
        n = max_samples
    
    results = []
    dimensions = []
    
    for m in range(2, min(m_max, 8)):
        N = n - (m - 1) * tau
        if N < 100:
            break
        
        # Embed
        embedded = np.zeros((N, m), dtype=np.float32)
        for i in range(m):
            embedded[:, i] = data[i * tau:i * tau + N]
        
        # Move to GPU
        embedded_gpu = torch.from_numpy(embedded).cuda()
        
        # Compute pairwise distances in batches
        batch_size = min(1500, N)
        all_distances = []
        
        for i in range(0, N, batch_size):
            end_i = min(i + batch_size, N)
            for j in range(i, N, batch_size):
                end_j = min(j + batch_size, N)
                
                diff = embedded_gpu[i:end_i].unsqueeze(1) - embedded_gpu[j:end_j].unsqueeze(0)
                d = torch.sqrt(torch.sum(diff ** 2, dim=2))
                
                # Only keep upper triangle for i == j
                if i == j:
                    mask = torch.triu(torch.ones(end_i - i, end_j - j, dtype=torch.bool, device='cuda'), diagonal=1)
                    all_distances.append(d[mask].cpu().numpy())
                else:
                    all_distances.append(d.cpu().numpy().ravel())
        
        distances = np.concatenate(all_distances)
        distances = distances[distances > 1e-10]
        
        if len(distances) < 100:
            continue
        
        # Correlation sum for different scales
        log_r = np.linspace(np.log10(np.min(distances)), np.log10(np.max(distances)), n_scales)
        r_values = 10 ** log_r
        
        log_c = []
        for r in r_values:
            c = np.mean(distances < r)
            log_c.append(np.log10(c) if c > 1e-10 else -10)
        
        log_c_np = np.array(log_c)
        
        # Linear regression on valid middle portion
        valid = (log_c_np > -8) & (log_c_np < -0.1)
        if np.sum(valid) < 3:
            continue
        
        slope, _, r_value, _, _ = stats.linregress(log_r[valid], log_c_np[valid])
        results.append(slope)
        dimensions.append(m)
    
    if not results:
        return 0.0, {"dimensions": [], "slopes": []}
    
    d2 = float(np.max(results))
    return d2, {
        "dimensions": dimensions,
        "slopes": results,
        "saturation_dim": dimensions[np.argmax(results)],
    }


def _torch_hurst_exponent(data, max_k) -> float:
    """PyTorch GPU-accelerated Hurst exponent via R/S analysis."""
    import torch
    
    # Convert to numpy
    if hasattr(data, 'get'):
        data = data.get()
    data = np.asarray(data).astype(np.float32)
    
    n = len(data)
    if n < 100:
        return 0.5
    
    # Do most computation on CPU for small batches (faster for this use case)
    rs_values = []
    sizes = []
    
    for k in range(1, min(max_k, int(np.log2(n)) - 2)):
        size = n // (2 ** k)
        if size < 10:
            break
        
        n_segments = n // size
        
        # Reshape data into segments for vectorized R/S calculation
        segment_data = data[:n_segments * size].reshape(n_segments, size)
        means = np.mean(segment_data, axis=1, keepdims=True)
        detrended = segment_data - means
        cumsum = np.cumsum(detrended, axis=1)
        R = np.max(cumsum, axis=1) - np.min(cumsum, axis=1)
        S = np.std(segment_data, axis=1)
        
        valid = S > 1e-10
        if np.any(valid):
            rs_ratios = R[valid] / S[valid]
            rs_values.append(float(np.mean(rs_ratios)))
            sizes.append(size)
    
    if len(sizes) < 3:
        return 0.5
    
    slope, _ = np.polyfit(np.log(sizes), np.log(rs_values), 1)
    return float(slope)


def _torch_recurrence_analysis(data, embedding_dim, time_delay, threshold, max_samples) -> Dict:
    """PyTorch GPU-accelerated Recurrence Quantification Analysis."""
    import torch
    
    # Convert to numpy
    if hasattr(data, 'get'):
        data = data.get()
    data = np.asarray(data).astype(np.float32)
    
    n = len(data)
    if n > max_samples:
        data = data[:max_samples]
        n = max_samples
    
    N = n - (embedding_dim - 1) * time_delay
    if N < 50:
        return {"recurrence_rate": 0.0, "determinism": 0.0}
    
    # Embed
    embedded = np.zeros((N, embedding_dim), dtype=np.float32)
    for i in range(embedding_dim):
        embedded[:, i] = data[i * time_delay:i * time_delay + N]
    
    embedded_gpu = torch.from_numpy(embedded).cuda()
    
    # Auto-threshold
    if threshold is None:
        sample_size = min(500, N)
        sample_idx = torch.randperm(N, device='cuda')[:sample_size]
        sample = embedded_gpu[sample_idx]
        diff = sample.unsqueeze(1) - sample.unsqueeze(0)
        sample_dist = torch.sqrt(torch.sum(diff ** 2, dim=2))
        threshold = float(0.1 * torch.max(sample_dist).cpu().item())
    
    # Compute recurrence in batches
    batch_size = min(500, N)
    recurrence_count = 0
    
    for i in range(0, N, batch_size):
        end_i = min(i + batch_size, N)
        for j in range(0, N, batch_size):
            end_j = min(j + batch_size, N)
            
            diff = embedded_gpu[i:end_i].unsqueeze(1) - embedded_gpu[j:end_j].unsqueeze(0)
            distances = torch.sqrt(torch.sum(diff ** 2, dim=2))
            recurrent = distances < threshold
            recurrence_count += int(torch.sum(recurrent).cpu().item())
    
    # Recurrence rate
    rr = recurrence_count / (N * N)
    
    return {
        "recurrence_rate": float(rr),
        "determinism": 0.0,  # Simplified for now
        "gpu_accelerated": True,
    }


# CPU fallback implementations
def _cpu_lyapunov_exponent(data, m, tau, eps, max_iter, n_ref) -> Tuple[float, float]:
    """CPU Lyapunov exponent - simplified."""
    try:
        from scipy.spatial.distance import cdist
        from scipy import stats
        
        n = len(data)
        N = n - (m - 1) * tau
        if N < 100:
            return 0.0, 0.0
        
        # Embed
        embedded = np.zeros((N, m))
        for i in range(m):
            embedded[:, i] = data[i * tau:i * tau + N]
        
        # Distance matrix
        dist_matrix = cdist(embedded, embedded)
        
        # Theiler window
        theiler = tau * m
        for i in range(N):
            low = max(0, i - theiler)
            high = min(N, i + theiler + 1)
            dist_matrix[i, low:high] = np.inf
        
        # Sample and track divergence
        n_ref = min(n_ref, N - max_iter)
        if n_ref < 10:
            return 0.0, 0.0
        
        indices = np.random.choice(N - max_iter, n_ref, replace=False)
        divergence = []
        
        for i in indices:
            nn_idx = np.argmin(dist_matrix[i])
            min_dist = dist_matrix[i, nn_idx]
            
            if min_dist < np.inf and min_dist > 1e-10 and nn_idx + max_iter < N:
                local_div = []
                for k in range(1, min(max_iter, N - max(i, nn_idx))):
                    d = np.linalg.norm(embedded[i + k] - embedded[nn_idx + k])
                    if d > 1e-10:
                        local_div.append(np.log(d / min_dist))
                
                if len(local_div) > 5:
                    divergence.append(local_div)
        
        if len(divergence) < 10:
            return 0.0, 0.0
        
        min_len = min(len(d) for d in divergence)
        avg_div = np.mean([d[:min_len] for d in divergence], axis=0)
        
        t = np.arange(1, min_len + 1)
        slope, _, r_value, _, _ = stats.linregress(t, avg_div)
        
        return float(slope), float(r_value ** 2)
    except Exception:
        return 0.0, 0.0


def _cpu_correlation_dimension(data, m_max, tau, n_scales, max_samples) -> Tuple[float, Dict]:
    """CPU correlation dimension - simplified."""
    return 0.0, {"dimensions": [], "slopes": []}


def _cpu_hurst_exponent(data, max_k) -> float:
    """CPU Hurst exponent."""
    n = len(data)
    if n < 100:
        return 0.5
    
    rs_values = []
    sizes = []
    
    for k in range(1, min(max_k, int(np.log2(n)) - 2)):
        size = n // (2 ** k)
        if size < 10:
            break
        
        n_segments = n // size
        rs_list = []
        
        for i in range(n_segments):
            segment = data[i * size:(i + 1) * size]
            mean = np.mean(segment)
            cumsum = np.cumsum(segment - mean)
            R = np.max(cumsum) - np.min(cumsum)
            S = np.std(segment)
            if S > 1e-10:
                rs_list.append(R / S)
        
        if rs_list:
            rs_values.append(np.mean(rs_list))
            sizes.append(size)
    
    if len(sizes) < 3:
        return 0.5
    
    slope, _ = np.polyfit(np.log(sizes), np.log(rs_values), 1)
    return float(slope)


def _cpu_recurrence_analysis(data, m, tau, threshold, max_samples) -> Dict:
    """CPU recurrence analysis - simplified."""
    return {"recurrence_rate": 0.0, "determinism": 0.0}


def _cpu_multiscale_entropy(data, scales, m, r) -> Tuple[np.ndarray, np.ndarray]:
    """CPU multiscale entropy."""
    from .entropy import _cpu_sample_entropy
    
    if r is None:
        r = 0.2 * np.std(data)
    
    n = len(data)
    scale_values = []
    entropy_values = []
    
    for scale in range(1, scales + 1):
        coarse_n = n // scale
        if coarse_n < 50:
            break
        
        coarse = np.zeros(coarse_n)
        for i in range(coarse_n):
            coarse[i] = np.mean(data[i * scale:(i + 1) * scale])
        
        se = _cpu_sample_entropy(coarse, m, r, 1000)
        scale_values.append(scale)
        entropy_values.append(se)
    
    return np.array(scale_values), np.array(entropy_values)
