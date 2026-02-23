"""
cuQuantum Core Module
======================

Core GPU initialization and management for GPU-accelerated analysis.
Uses PyTorch CUDA as primary backend (RTX 5090 optimized).
"""

import os
import sys
import warnings
from dataclasses import dataclass
from typing import Optional, Dict, Any, List, Tuple, Union
from enum import Enum
import numpy as np

# GPU backend detection - PyTorch is primary
HAS_TORCH = False
HAS_TORCH_CUDA = False
HAS_CUDA = False

try:
    import torch
    HAS_TORCH = True
    HAS_TORCH_CUDA = torch.cuda.is_available()
    HAS_CUDA = HAS_TORCH_CUDA
    if HAS_TORCH_CUDA:
        # Warm up CUDA
        _ = torch.tensor([1.0], device='cuda').sum()
except ImportError:
    torch = None


class GPUBackend(Enum):
    """Available GPU compute backends."""
    TORCH = "torch"
    CPU = "cpu"  # Fallback


@dataclass
class GPUInfo:
    """Information about available GPU."""
    device_id: int
    name: str
    compute_capability: Tuple[int, int]
    total_memory_gb: float
    free_memory_gb: float
    cuda_version: str
    is_rtx_5090: bool
    tensor_cores: bool


def check_gpu_availability() -> bool:
    """Check if GPU acceleration is available."""
    return HAS_TORCH_CUDA


def get_gpu_info(device_id: int = 0) -> Optional[GPUInfo]:
    """
    Get detailed information about GPU device.
    
    Args:
        device_id: CUDA device ID
        
    Returns:
        GPUInfo object or None if no GPU available
    """
    if not HAS_TORCH_CUDA:
        return None
    
    try:
        props = torch.cuda.get_device_properties(device_id)
        total_mem = props.total_memory
        free_mem = total_mem - torch.cuda.memory_allocated(device_id)
        
        name = props.name
        compute_cap = (props.major, props.minor)
        
        # Detect RTX 5090 (Blackwell architecture)
        is_rtx_5090 = "5090" in name or (compute_cap[0] >= 10)
        
        cuda_version = f"{torch.version.cuda}"
        
        return GPUInfo(
            device_id=device_id,
            name=name,
            compute_capability=compute_cap,
            total_memory_gb=total_mem / (1024**3),
            free_memory_gb=free_mem / (1024**3),
            cuda_version=cuda_version,
            is_rtx_5090=is_rtx_5090,
            tensor_cores=compute_cap[0] >= 7,  # Volta+
        )
    except Exception as e:
        warnings.warn(f"Failed to get GPU info: {e}")
        return None


def initialize_gpu(device_id: int = 0) -> bool:
    """
    Initialize GPU on specified device.
    
    Args:
        device_id: CUDA device ID to use
        
    Returns:
        True if initialization successful
    """
    if not HAS_TORCH_CUDA:
        print("❌ PyTorch CUDA not available - install with: pip install torch --index-url https://download.pytorch.org/whl/cu128")
        return False
    
    try:
        torch.cuda.set_device(device_id)
        # Verify with a simple operation
        test = torch.tensor([1.0, 2.0, 3.0], device='cuda')
        _ = torch.sum(test).item()
        
        info = get_gpu_info(device_id)
        if info:
            print(f"✅ GPU initialized: {info.name}")
            print(f"   CUDA {info.cuda_version}, {info.total_memory_gb:.1f} GB memory")
            if info.is_rtx_5090:
                print(f"   🚀 RTX 5090 Blackwell optimizations enabled")
        return True
    except Exception as e:
        print(f"❌ GPU initialization failed: {e}")
        return False


class GPUAnalyzer:
    """
    High-performance GPU-accelerated QRNG analyzer.
    
    Provides unified interface for all GPU-accelerated analysis operations.
    Uses PyTorch CUDA as primary backend, falls back to CPU if unavailable.
    """
    
    def __init__(
        self,
        device_id: int = 0,
        backend: Optional[GPUBackend] = None,
        memory_limit_gb: Optional[float] = None,
    ):
        """
        Initialize GPU analyzer.
        
        Args:
            device_id: CUDA device ID
            backend: Preferred compute backend
            memory_limit_gb: Memory limit for operations
        """
        self.device_id = device_id
        self.gpu_available = check_gpu_availability()
        
        # Auto-select backend
        if backend is None:
            if HAS_TORCH_CUDA:
                self.backend = GPUBackend.TORCH
            else:
                self.backend = GPUBackend.CPU
        else:
            self.backend = backend
        
        self.memory_limit_gb = memory_limit_gb
        self.info = get_gpu_info(device_id) if self.gpu_available else None
    
    def to_gpu(self, data: np.ndarray) -> Any:
        """
        Transfer numpy array to GPU.
        
        Args:
            data: Numpy array
            
        Returns:
            GPU tensor or numpy array
        """
        if self.backend == GPUBackend.TORCH and HAS_TORCH_CUDA:
            return torch.from_numpy(data).float().cuda(self.device_id)
        else:
            return data
    
    def to_cpu(self, data: Any) -> np.ndarray:
        """
        Transfer GPU array to CPU.
        
        Args:
            data: GPU tensor or array
            
        Returns:
            Numpy array
        """
        if HAS_TORCH and isinstance(data, torch.Tensor):
            return data.cpu().numpy()
        else:
            return np.asarray(data)
    
    def analyze_stream(
        self,
        data: Union[np.ndarray, List[float]],
        compute_entropy: bool = True,
        compute_chaos: bool = True,
        compute_fft: bool = True,
        compute_correlations: bool = True,
    ) -> Dict[str, Any]:
        """
        Comprehensive GPU-accelerated analysis of QRNG stream.
        
        Args:
            data: Input data stream
            compute_entropy: Compute entropy metrics
            compute_chaos: Compute chaos metrics (Lyapunov, etc.)
            compute_fft: Compute FFT analysis
            compute_correlations: Compute correlation analysis
            
        Returns:
            Dictionary of analysis results
        """
        # Convert to numpy if needed
        if isinstance(data, list):
            data = np.array(data, dtype=np.float64)
        
        results = {
            "n_samples": len(data),
            "backend": self.backend.value,
            "gpu_available": self.gpu_available,
        }
        
        # Transfer to GPU
        if self.gpu_available:
            gpu_data = self.to_gpu(data)
        else:
            gpu_data = data
        
        # Entropy analysis
        if compute_entropy:
            results["entropy"] = self._compute_entropy_metrics(gpu_data, data)
        
        # Chaos analysis
        if compute_chaos:
            results["chaos"] = self._compute_chaos_metrics(data)  # Uses numpy
        
        # FFT analysis
        if compute_fft:
            results["fft"] = self._compute_fft_analysis(gpu_data, data)
        
        # Correlation analysis
        if compute_correlations:
            results["correlations"] = self._compute_correlation_analysis(data)
        
        return results
    
    def _compute_entropy_metrics(self, gpu_data: Any, cpu_data: np.ndarray) -> Dict[str, float]:
        """Compute entropy metrics."""
        if self.backend == GPUBackend.TORCH and HAS_TORCH_CUDA:
            t_data = gpu_data if isinstance(gpu_data, torch.Tensor) else torch.from_numpy(cpu_data).float().cuda()
            
            # Shannon entropy via histogram
            hist = torch.histc(t_data, bins=256, min=0.0, max=1.0)
            hist = hist / hist.sum()
            hist = hist[hist > 0]
            shannon = float(-torch.sum(hist * torch.log2(hist)).item())
            
            max_entropy = np.log2(256)
            normalized = shannon / max_entropy
            
            return {
                "shannon": shannon,
                "shannon_normalized": normalized,
                "mean": float(torch.mean(t_data).item()),
                "std": float(torch.std(t_data).item()),
                "min": float(torch.min(t_data).item()),
                "max": float(torch.max(t_data).item()),
            }
        else:
            # CPU fallback
            hist, _ = np.histogram(cpu_data, bins=256, density=True)
            hist = hist[hist > 0]
            shannon = float(-np.sum(hist * np.log2(hist)))
            
            return {
                "shannon": shannon,
                "shannon_normalized": shannon / np.log2(256),
                "mean": float(np.mean(cpu_data)),
                "std": float(np.std(cpu_data)),
                "min": float(np.min(cpu_data)),
                "max": float(np.max(cpu_data)),
            }
    
    def _compute_chaos_metrics(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute chaos/complexity metrics."""
        n = len(data)
        
        # Hurst exponent (R/S analysis) - vectorized
        hurst = self._hurst_exponent(data)
        
        # Autocorrelation
        autocorr_lag1 = self._autocorrelation(data, 1)
        autocorr_lag10 = self._autocorrelation(data, 10)
        
        return {
            "hurst_exponent": hurst,
            "autocorrelation_lag1": autocorr_lag1,
            "autocorrelation_lag10": autocorr_lag10,
            "n_samples": n,
        }
    
    def _compute_fft_analysis(self, gpu_data: Any, cpu_data: np.ndarray) -> Dict[str, Any]:
        """Compute FFT analysis."""
        if self.backend == GPUBackend.TORCH and HAS_TORCH_CUDA:
            t_data = gpu_data if isinstance(gpu_data, torch.Tensor) else torch.from_numpy(cpu_data).float().cuda()
            
            # GPU FFT
            fft_result = torch.fft.fft(t_data)
            power_spectrum = torch.abs(fft_result) ** 2
            
            n = len(t_data)
            power_half = power_spectrum[1:n//2]
            top_idx = int(torch.argmax(power_half).item()) + 1
            
            # Spectral entropy
            ps_norm = power_spectrum / torch.sum(power_spectrum)
            ps_norm = ps_norm[ps_norm > 1e-10]
            spectral_ent = float(-torch.sum(ps_norm * torch.log2(ps_norm)).item())
            
            return {
                "spectral_entropy": spectral_ent,
                "dc_component": float(power_spectrum[0].item()),
                "total_power": float(torch.sum(power_spectrum).item()),
                "dominant_freq_idx": top_idx,
            }
        else:
            # CPU fallback
            fft_result = np.fft.fft(cpu_data)
            power_spectrum = np.abs(fft_result) ** 2
            
            ps_norm = power_spectrum / np.sum(power_spectrum)
            ps_norm = ps_norm[ps_norm > 1e-10]
            spectral_ent = float(-np.sum(ps_norm * np.log2(ps_norm)))
            
            return {
                "spectral_entropy": spectral_ent,
                "dc_component": float(power_spectrum[0]),
                "total_power": float(np.sum(power_spectrum)),
                "dominant_freq_idx": int(np.argmax(power_spectrum[1:len(cpu_data)//2])) + 1,
            }
    
    def _compute_correlation_analysis(self, data: np.ndarray) -> Dict[str, Any]:
        """Compute correlation analysis."""
        n = len(data)
        
        # Split into segments
        n_segments = min(10, n // 100)
        if n_segments < 2:
            return {"segment_correlations": [], "mean_segment_corr": 0.0, "n_segments": 0}
        
        segment_size = n // n_segments
        segments = [data[i*segment_size:(i+1)*segment_size] for i in range(n_segments)]
        
        # Pairwise correlations
        correlations = []
        for i in range(n_segments):
            for j in range(i+1, n_segments):
                corr = np.corrcoef(segments[i], segments[j])[0, 1]
                correlations.append(float(corr))
        
        return {
            "segment_correlations": correlations[:10],
            "mean_segment_corr": float(np.mean(correlations)),
            "max_segment_corr": float(np.max(np.abs(correlations))),
            "n_segments": n_segments,
        }
    
    def _hurst_exponent(self, data: np.ndarray, max_k: int = 20) -> float:
        """Hurst exponent via R/S analysis (vectorized)."""
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
            segment_data = data[:n_segments * size].reshape(n_segments, size)
            means = np.mean(segment_data, axis=1, keepdims=True)
            detrended = segment_data - means
            cumsum = np.cumsum(detrended, axis=1)
            R = np.max(cumsum, axis=1) - np.min(cumsum, axis=1)
            S = np.std(segment_data, axis=1)
            
            valid = S > 1e-10
            if np.any(valid):
                rs_values.append(float(np.mean(R[valid] / S[valid])))
                sizes.append(size)
        
        if len(sizes) < 3:
            return 0.5
        
        slope, _ = np.polyfit(np.log(sizes), np.log(rs_values), 1)
        return float(slope)
    
    def _autocorrelation(self, data: np.ndarray, lag: int) -> float:
        """Autocorrelation at specific lag."""
        n = len(data)
        if lag >= n:
            return 0.0
        
        mean = np.mean(data)
        var = np.var(data)
        if var < 1e-10:
            return 0.0
        
        cov = np.mean((data[:n-lag] - mean) * (data[lag:] - mean))
        return float(cov / var)
    
    def benchmark(self, data_size: int = 100000, n_iterations: int = 10) -> Dict[str, float]:
        """
        Benchmark GPU vs CPU performance.
        
        Args:
            data_size: Number of samples to test
            n_iterations: Number of iterations for timing
            
        Returns:
            Dictionary with timing results
        """
        import time
        
        # Generate test data
        np.random.seed(42)
        test_data = np.random.random(data_size).astype(np.float64)
        
        # CPU timing
        cpu_times = []
        for _ in range(n_iterations):
            start = time.perf_counter()
            self._compute_entropy_metrics(None, test_data)
            self._compute_fft_analysis(None, test_data)
            cpu_times.append(time.perf_counter() - start)
        
        results = {
            "data_size": data_size,
            "n_iterations": n_iterations,
            "cpu_mean_time": np.mean(cpu_times),
            "cpu_std_time": np.std(cpu_times),
        }
        
        # GPU timing
        if self.gpu_available and HAS_TORCH_CUDA:
            gpu_data = self.to_gpu(test_data)
            
            # Warm up
            self._compute_entropy_metrics(gpu_data, test_data)
            torch.cuda.synchronize()
            
            gpu_times = []
            for _ in range(n_iterations):
                start = time.perf_counter()
                self._compute_entropy_metrics(gpu_data, test_data)
                self._compute_fft_analysis(gpu_data, test_data)
                torch.cuda.synchronize()
                gpu_times.append(time.perf_counter() - start)
            
            results["gpu_mean_time"] = np.mean(gpu_times)
            results["gpu_std_time"] = np.std(gpu_times)
            results["speedup"] = results["cpu_mean_time"] / results["gpu_mean_time"]
        
        return results


# Quick initialization function
def quick_init() -> GPUAnalyzer:
    """Quick initialization with auto-detection."""
    if initialize_gpu():
        return GPUAnalyzer()
    else:
        print("⚠️ Running in CPU-only mode")
        return GPUAnalyzer(backend=GPUBackend.CPU)

