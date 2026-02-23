"""
QRNG Benchmarking and Comparison Tools
========================================

Comprehensive benchmarking suite for comparing physical QRNG sources
against ideal quantum distributions and pseudo-random generators.

Features:
- GPU vs CPU performance benchmarks
- QRNG quality assessment against ideal quantum
- NIST SP 800-22 inspired tests
- Cross-source comparison
- Statistical visualization
"""

import numpy as np
import json
import time
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Union, Any
from dataclasses import dataclass, field, asdict
from datetime import datetime
from collections import defaultdict

# Local imports
try:
    from .core import GPUAnalyzer, check_gpu_availability, get_gpu_info
    from .entropy import GPUEntropyCalculator, EntropyResult
    from .quantum_simulator import (
        QuantumStateSimulator, 
        IdealQRNGDistribution, 
        NoiseModel,
        NoiseParameters,
        SPDCSimulator,
    )
    from .tensor_analysis import TensorNetworkAnalyzer, ChaosMetrics
except ImportError:
    # Running standalone
    pass


@dataclass
class BenchmarkResult:
    """Container for benchmark results."""
    name: str
    n_samples: int
    
    # Timing
    gpu_time_ms: float = 0.0
    cpu_time_ms: float = 0.0
    speedup: float = 1.0
    
    # Memory
    gpu_memory_mb: float = 0.0
    peak_memory_mb: float = 0.0
    
    # Accuracy (vs reference)
    entropy_error: float = 0.0
    mean_error: float = 0.0
    
    # Hardware info
    gpu_name: str = ""
    gpu_compute_capability: str = ""
    
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


@dataclass
class QualityAssessment:
    """QRNG quality assessment results."""
    source_name: str
    n_samples: int
    
    # Basic statistics
    mean: float = 0.0
    std: float = 0.0
    min_val: float = 0.0
    max_val: float = 0.0
    
    # Entropy metrics
    shannon_entropy: float = 0.0
    bientropy: float = 0.0
    sample_entropy: float = 0.0
    permutation_entropy: float = 0.0
    spectral_entropy: float = 0.0
    
    # Chaos metrics
    lyapunov_exponent: float = 0.0
    hurst_exponent: float = 0.0
    correlation_dimension: float = 0.0
    
    # Statistical tests
    ks_test_pvalue: float = 0.0
    chi2_test_pvalue: float = 0.0
    runs_test_pvalue: float = 0.0
    autocorr_test_pvalue: float = 0.0
    
    # Quality scores
    randomness_score: float = 0.0  # 0-100
    quantum_quality_score: float = 0.0  # 0-100
    overall_grade: str = "F"
    
    # Comparison to ideal
    deviation_from_ideal: Dict[str, float] = field(default_factory=dict)
    
    # Metadata
    computation_time_ms: float = 0.0
    method: str = "gpu"
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


class QRNGBenchmark:
    """
    Comprehensive QRNG benchmarking suite.
    
    Tests GPU acceleration performance and QRNG quality.
    """
    
    def __init__(self, device_id: int = 0):
        """
        Initialize benchmark suite.
        
        Args:
            device_id: CUDA device ID
        """
        self.device_id = device_id
        self.gpu_available = check_gpu_availability()
        self.gpu_info = get_gpu_info(device_id) if self.gpu_available else None
        
        # Initialize components
        self.analyzer = GPUAnalyzer(device_id=device_id)
        self.entropy_calc = GPUEntropyCalculator(device_id=device_id)
        self.tensor_analyzer = TensorNetworkAnalyzer(device_id=device_id)
        self.ideal_dist = IdealQRNGDistribution(n_bits=32)
        
        self.results: List[BenchmarkResult] = []
        self.quality_results: List[QualityAssessment] = []
    
    def run_performance_benchmark(
        self,
        data_sizes: Optional[List[int]] = None,
        n_iterations: int = 5,
    ) -> List[BenchmarkResult]:
        """
        Run GPU vs CPU performance benchmark.
        
        Args:
            data_sizes: List of sample sizes to test
            n_iterations: Number of iterations per size
            
        Returns:
            List of BenchmarkResult objects
        """
        if data_sizes is None:
            data_sizes = [1000, 10000, 100000, 500000, 1000000]
        
        results = []
        
        for n in data_sizes:
            print(f"\n📊 Benchmarking with {n:,} samples...")
            
            # Generate test data
            np.random.seed(42)
            test_data = np.random.random(n).astype(np.float64)
            
            # CPU benchmark
            cpu_times = []
            for _ in range(n_iterations):
                start = time.perf_counter()
                _ = self.entropy_calc.compute_all(test_data)
                cpu_times.append((time.perf_counter() - start) * 1000)
            
            cpu_time = np.mean(cpu_times)
            
            # GPU benchmark
            gpu_time = cpu_time  # Default if no GPU
            gpu_memory = 0.0
            
            if self.gpu_available:
                import torch
                
                # Transfer to GPU
                gpu_data = torch.tensor(test_data, device='cuda')
                
                # Warm up
                _ = self.entropy_calc.compute_all(test_data)
                torch.cuda.synchronize()
                
                # Benchmark
                torch.cuda.reset_peak_memory_stats()
                
                gpu_times = []
                for _ in range(n_iterations):
                    start = time.perf_counter()
                    _ = self.entropy_calc.compute_all(test_data)
                    torch.cuda.synchronize()
                    gpu_times.append((time.perf_counter() - start) * 1000)
                
                gpu_time = np.mean(gpu_times)
                gpu_memory = torch.cuda.max_memory_allocated() / (1024 ** 2)
            
            result = BenchmarkResult(
                name=f"entropy_analysis_{n}",
                n_samples=n,
                gpu_time_ms=gpu_time,
                cpu_time_ms=cpu_time,
                speedup=cpu_time / gpu_time if gpu_time > 0 else 1.0,
                gpu_memory_mb=gpu_memory,
                gpu_name=self.gpu_info.name if self.gpu_info else "N/A",
                gpu_compute_capability=(
                    f"{self.gpu_info.compute_capability[0]}.{self.gpu_info.compute_capability[1]}"
                    if self.gpu_info else "N/A"
                ),
            )
            
            results.append(result)
            self.results.append(result)
            
            print(f"   CPU: {cpu_time:.2f} ms")
            print(f"   GPU: {gpu_time:.2f} ms")
            print(f"   Speedup: {result.speedup:.1f}x")
        
        return results
    
    def assess_qrng_quality(
        self,
        data: np.ndarray,
        source_name: str = "unknown",
    ) -> QualityAssessment:
        """
        Comprehensive quality assessment of QRNG data.
        
        Args:
            data: QRNG sample data
            source_name: Name of the QRNG source
            
        Returns:
            QualityAssessment with all metrics
        """
        start = time.perf_counter()
        n = len(data)
        
        # Basic statistics
        mean = float(np.mean(data))
        std = float(np.std(data))
        min_val = float(np.min(data))
        max_val = float(np.max(data))
        
        # Entropy metrics
        entropy_result = self.entropy_calc.compute_all(data)
        
        # Chaos metrics
        chaos_result = self.tensor_analyzer.analyze_stream(data)
        
        # Statistical tests vs ideal uniform
        test_results = self.ideal_dist.compare_to_ideal(data)
        
        # Compute quality scores
        randomness_score = self._compute_randomness_score(
            entropy_result, chaos_result, test_results
        )
        quantum_score = self._compute_quantum_quality_score(
            data, entropy_result, test_results
        )
        grade = self._compute_grade(randomness_score, quantum_score)
        
        # Deviation from ideal
        deviation = {
            "mean": abs(mean - 0.5) / 0.5 * 100,
            "std": abs(std - 1/np.sqrt(12)) / (1/np.sqrt(12)) * 100,
            "shannon": (1 - entropy_result.shannon_normalized) * 100,
        }
        
        elapsed = (time.perf_counter() - start) * 1000
        
        assessment = QualityAssessment(
            source_name=source_name,
            n_samples=n,
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            shannon_entropy=entropy_result.shannon,
            bientropy=entropy_result.bientropy,
            sample_entropy=entropy_result.sample_entropy,
            permutation_entropy=entropy_result.permutation_entropy,
            spectral_entropy=entropy_result.spectral_entropy,
            lyapunov_exponent=chaos_result.lyapunov_exponent,
            hurst_exponent=chaos_result.hurst_exponent,
            correlation_dimension=chaos_result.correlation_dimension,
            ks_test_pvalue=test_results.get("ks", {}).get("p_value", 0),
            chi2_test_pvalue=test_results.get("chi2", {}).get("p_value", 0),
            runs_test_pvalue=test_results.get("runs", {}).get("p_value", 0),
            autocorr_test_pvalue=test_results.get("autocorr", {}).get("p_value", 0),
            randomness_score=randomness_score,
            quantum_quality_score=quantum_score,
            overall_grade=grade,
            deviation_from_ideal=deviation,
            computation_time_ms=elapsed,
            method=entropy_result.method,
        )
        
        self.quality_results.append(assessment)
        return assessment
    
    def _compute_randomness_score(
        self,
        entropy: EntropyResult,
        chaos: ChaosMetrics,
        tests: Dict,
    ) -> float:
        """Compute overall randomness score (0-100)."""
        scores = []
        
        # Entropy contribution (higher = more random)
        scores.append(entropy.shannon_normalized * 100)
        scores.append(entropy.bientropy * 100)
        scores.append(min(100, entropy.spectral_entropy * 100))
        
        # Hurst exponent (should be ~0.5 for random)
        hurst_score = 100 * (1 - abs(chaos.hurst_exponent - 0.5) * 2)
        scores.append(max(0, hurst_score))
        
        # Statistical tests (p-values should be high)
        for test_name in ["ks", "chi2", "runs", "autocorr"]:
            if test_name in tests:
                p_value = tests[test_name].get("p_value", 0)
                # Score based on p-value (higher p = more random)
                scores.append(min(100, p_value * 200))
        
        return float(np.mean(scores))
    
    def _compute_quantum_quality_score(
        self,
        data: np.ndarray,
        entropy: EntropyResult,
        tests: Dict,
    ) -> float:
        """
        Compute quantum quality score (0-100).
        
        Higher score indicates data is closer to ideal quantum source.
        """
        scores = []
        
        # Statistical uniformity
        mean_dev = abs(np.mean(data) - 0.5)
        scores.append(100 * max(0, 1 - mean_dev * 4))
        
        # Variance check (should be 1/12 for uniform)
        var_dev = abs(np.var(data) - 1/12) / (1/12)
        scores.append(100 * max(0, 1 - var_dev))
        
        # Entropy (should be maximal)
        scores.append(entropy.shannon_normalized * 100)
        
        # Independence (low autocorrelation)
        if "autocorr" in tests:
            autocorr = abs(tests["autocorr"].get("lag1_autocorr", 0))
            scores.append(100 * max(0, 1 - autocorr * 10))
        
        # All statistical tests pass
        n_passed = tests.get("summary", {}).get("tests_passed", 0)
        n_tests = tests.get("summary", {}).get("tests_run", 1)
        scores.append(100 * n_passed / n_tests)
        
        return float(np.mean(scores))
    
    def _compute_grade(self, randomness: float, quantum: float) -> str:
        """Compute letter grade from scores."""
        avg = (randomness + quantum) / 2
        
        if avg >= 95:
            return "A+"
        elif avg >= 90:
            return "A"
        elif avg >= 85:
            return "A-"
        elif avg >= 80:
            return "B+"
        elif avg >= 75:
            return "B"
        elif avg >= 70:
            return "B-"
        elif avg >= 65:
            return "C+"
        elif avg >= 60:
            return "C"
        elif avg >= 55:
            return "C-"
        elif avg >= 50:
            return "D"
        else:
            return "F"
    
    def compare_sources(
        self,
        sources: Dict[str, np.ndarray],
    ) -> Dict[str, QualityAssessment]:
        """
        Compare multiple QRNG sources.
        
        Args:
            sources: Dictionary of source_name -> data
            
        Returns:
            Dictionary of assessments
        """
        assessments = {}
        
        for name, data in sources.items():
            print(f"\n🔬 Assessing {name}...")
            assessment = self.assess_qrng_quality(data, name)
            assessments[name] = assessment
            
            print(f"   Randomness Score: {assessment.randomness_score:.1f}/100")
            print(f"   Quantum Quality:  {assessment.quantum_quality_score:.1f}/100")
            print(f"   Grade: {assessment.overall_grade}")
        
        return assessments
    
    def generate_report(
        self,
        output_path: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Generate comprehensive benchmark report.
        
        Args:
            output_path: Path to save JSON report
            
        Returns:
            Report dictionary
        """
        report = {
            "timestamp": datetime.now().isoformat(),
            "gpu_info": asdict(self.gpu_info) if self.gpu_info else None,
            "performance_benchmarks": [asdict(r) for r in self.results],
            "quality_assessments": [asdict(q) for q in self.quality_results],
        }
        
        # Summary statistics
        if self.results:
            speedups = [r.speedup for r in self.results]
            report["summary"] = {
                "mean_speedup": float(np.mean(speedups)),
                "max_speedup": float(np.max(speedups)),
                "total_benchmarks": len(self.results),
            }
        
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, 'w') as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n📝 Report saved to {output_path}")
        
        return report


def compare_to_ideal_quantum(
    data: np.ndarray,
    n_qubits: int = 16,
    noise_model: NoiseModel = NoiseModel.IDEAL,
) -> Dict[str, Any]:
    """
    Compare QRNG data to simulated ideal quantum source.
    
    Args:
        data: QRNG data to compare
        n_qubits: Number of qubits for simulation
        noise_model: Noise model for simulation
        
    Returns:
        Comparison results
    """
    n_samples = len(data)
    
    # Generate ideal quantum reference
    simulator = QuantumStateSimulator(n_qubits=n_qubits)
    ideal_samples = simulator.simulate_and_sample(
        n_samples=n_samples,
        noise_model=noise_model,
    )
    
    # Compare distributions
    from scipy import stats
    
    # KS test between distributions
    ks_stat, ks_pvalue = stats.ks_2samp(data, ideal_samples)
    
    # Compare moments
    moments_comparison = {
        "mean": {
            "data": float(np.mean(data)),
            "ideal": float(np.mean(ideal_samples)),
            "diff": float(abs(np.mean(data) - np.mean(ideal_samples))),
        },
        "std": {
            "data": float(np.std(data)),
            "ideal": float(np.std(ideal_samples)),
            "diff": float(abs(np.std(data) - np.std(ideal_samples))),
        },
        "skewness": {
            "data": float(stats.skew(data)),
            "ideal": float(stats.skew(ideal_samples)),
            "diff": float(abs(stats.skew(data) - stats.skew(ideal_samples))),
        },
        "kurtosis": {
            "data": float(stats.kurtosis(data)),
            "ideal": float(stats.kurtosis(ideal_samples)),
            "diff": float(abs(stats.kurtosis(data) - stats.kurtosis(ideal_samples))),
        },
    }
    
    # Overall similarity score (0-100)
    similarity = 100 * (1 - ks_stat)
    
    return {
        "ks_statistic": float(ks_stat),
        "ks_pvalue": float(ks_pvalue),
        "moments_comparison": moments_comparison,
        "similarity_score": float(similarity),
        "n_samples": n_samples,
        "noise_model": noise_model.value,
    }


def run_full_benchmark(
    qrng_data: Optional[np.ndarray] = None,
    output_dir: Optional[Path] = None,
) -> Dict[str, Any]:
    """
    Run comprehensive benchmark suite.
    
    Args:
        qrng_data: Optional QRNG data to test
        output_dir: Directory to save results
        
    Returns:
        Complete benchmark results
    """
    print("=" * 60)
    print("🚀 cuQuantum QRNG Benchmark Suite")
    print("=" * 60)
    
    benchmark = QRNGBenchmark()
    
    # GPU info
    if benchmark.gpu_info:
        print(f"\n🎮 GPU: {benchmark.gpu_info.name}")
        print(f"   Memory: {benchmark.gpu_info.total_memory_gb:.1f} GB")
        print(f"   Compute: SM {benchmark.gpu_info.compute_capability[0]}.{benchmark.gpu_info.compute_capability[1]}")
        if benchmark.gpu_info.is_rtx_5090:
            print("   🏆 RTX 5090 detected - Maximum performance!")
    else:
        print("\n⚠️ No GPU detected - running CPU benchmarks only")
    
    # Performance benchmarks
    print("\n" + "=" * 60)
    print("📊 Performance Benchmarks")
    print("=" * 60)
    perf_results = benchmark.run_performance_benchmark()
    
    # Quality assessment
    if qrng_data is not None:
        print("\n" + "=" * 60)
        print("🔬 Quality Assessment")
        print("=" * 60)
        quality = benchmark.assess_qrng_quality(qrng_data, "user_data")
        print(f"\n   Randomness Score: {quality.randomness_score:.1f}/100")
        print(f"   Quantum Quality:  {quality.quantum_quality_score:.1f}/100")
        print(f"   Grade: {quality.overall_grade}")
    
    # Compare to ideal quantum
    print("\n" + "=" * 60)
    print("⚛️ Ideal Quantum Comparison")
    print("=" * 60)
    
    test_data = qrng_data if qrng_data is not None else np.random.random(10000)
    comparison = compare_to_ideal_quantum(test_data)
    print(f"   Similarity to ideal: {comparison['similarity_score']:.1f}%")
    print(f"   KS p-value: {comparison['ks_pvalue']:.4f}")
    
    # Generate report
    if output_dir:
        output_dir = Path(output_dir)
        report_path = output_dir / f"benchmark_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        report = benchmark.generate_report(report_path)
    else:
        report = benchmark.generate_report()
    
    print("\n" + "=" * 60)
    print("✅ Benchmark Complete!")
    print("=" * 60)
    
    if perf_results:
        max_speedup = max(r.speedup for r in perf_results)
        print(f"\n   Maximum GPU Speedup: {max_speedup:.1f}x")
    
    return report


# Quick comparison function
def quick_compare(data: np.ndarray) -> None:
    """Quick comparison of data to ideal quantum distribution."""
    print("\n🔍 Quick QRNG Quality Check")
    print("-" * 40)
    
    ideal = IdealQRNGDistribution()
    results = ideal.compare_to_ideal(data)
    
    print(f"Samples: {len(data):,}")
    print(f"\nStatistical Tests:")
    
    for test_name, result in results.items():
        if test_name == "summary":
            continue
        passed = "✅" if result.get("pass", False) else "❌"
        p_value = result.get("p_value", 0)
        print(f"  {test_name:12} p={p_value:.4f} {passed}")
    
    summary = results.get("summary", {})
    pass_rate = summary.get("pass_rate", 0) * 100
    print(f"\nPass Rate: {pass_rate:.0f}%")
    print(f"Overall: {'✅ PASS' if summary.get('overall_pass', False) else '❌ FAIL'}")
