"""
GPU Accelerator for QRNG Analysis
===================================

High-performance GPU-accelerated analysis using PyTorch CUDA.
Optimized for RTX 5090 and other NVIDIA GPUs.

Features:
- GPU-accelerated entropy calculation
- Tensor network analysis for pattern detection
- Quantum state simulation for baseline comparison
- Fast Fourier transform on GPU
- Parallel correlation analysis

Requirements:
- NVIDIA GPU with CUDA compute capability 7.0+
- PyTorch with CUDA support

Usage:
    from cuquantum_accelerator import GPUAnalyzer, QuantumStateSimulator
    
    # Accelerate entropy analysis
    analyzer = GPUAnalyzer()
    results = analyzer.analyze_stream(qrng_data)
    
    # Compare against ideal quantum distribution
    simulator = QuantumStateSimulator(n_qubits=16)
    reference = simulator.sample_distribution(n_samples=10000)
"""

from .core import (
    GPUAnalyzer,
    check_gpu_availability,
    get_gpu_info,
    initialize_gpu,
)
from .entropy import (
    GPUEntropyCalculator,
    gpu_shannon_entropy,
    gpu_bientropy,
    gpu_sample_entropy,
)
from .quantum_simulator import (
    QuantumStateSimulator,
    IdealQRNGDistribution,
    NoiseModel,
)
from .tensor_analysis import (
    TensorNetworkAnalyzer,
    gpu_correlation_dimension,
    gpu_lyapunov_exponent,
)
from .benchmarks import (
    QRNGBenchmark,
    compare_to_ideal_quantum,
    run_full_benchmark,
)

__version__ = "1.0.0"
__author__ = "HELIOS Research Team"

__all__ = [
    # Core
    "GPUAnalyzer",
    "check_gpu_availability", 
    "get_gpu_info",
    "initialize_gpu",
    # Entropy
    "GPUEntropyCalculator",
    "gpu_shannon_entropy",
    "gpu_bientropy",
    "gpu_sample_entropy",
    # Quantum Simulation
    "QuantumStateSimulator",
    "IdealQRNGDistribution",
    "NoiseModel",
    # Tensor Analysis
    "TensorNetworkAnalyzer",
    "gpu_correlation_dimension",
    "gpu_lyapunov_exponent",
    # Benchmarks
    "QRNGBenchmark",
    "compare_to_ideal_quantum",
    "run_full_benchmark",
]
