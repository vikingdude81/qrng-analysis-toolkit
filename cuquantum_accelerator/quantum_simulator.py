"""
Quantum State Simulator for QRNG Baseline Comparison
======================================================

Uses PyTorch CUDA for GPU-accelerated quantum state simulation
to compare against physical QRNG sources.

Implements:
- Ideal Hadamard-based QRNG (single qubit measurement)
- Multi-qubit entangled QRNG
- SPDC photon pair simulation
- Noisy channel models (decoherence, detector efficiency)

This provides theoretical baselines for evaluating physical QRNG quality.
"""

import numpy as np
from typing import Optional, List, Dict, Tuple, Union, Callable
from dataclasses import dataclass
from enum import Enum
import warnings

# GPU imports - PyTorch is the primary backend
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

try:
    import cirq
    HAS_CIRQ = True
except ImportError:
    cirq = None
    HAS_CIRQ = False

try:
    import qiskit
    from qiskit.circuit import QuantumCircuit
    from qiskit.circuit.library import QFT
    HAS_QISKIT = True
except ImportError:
    qiskit = None
    QuantumCircuit = None
    HAS_QISKIT = False


class NoiseModel(Enum):
    """Noise models for quantum simulation."""
    IDEAL = "ideal"                    # Perfect quantum operations
    DEPOLARIZING = "depolarizing"      # Random Pauli errors
    AMPLITUDE_DAMPING = "amplitude"    # Energy relaxation (T1)
    PHASE_DAMPING = "phase"            # Dephasing (T2)
    DETECTOR_EFFICIENCY = "detector"   # Imperfect detection
    SPDC_REALISTIC = "spdc"           # Realistic SPDC source model


@dataclass
class NoiseParameters:
    """Parameters for noise models."""
    depolarizing_prob: float = 0.001      # Probability of depolarizing error
    amplitude_damping_prob: float = 0.001  # T1 decay probability
    phase_damping_prob: float = 0.002      # T2 dephasing probability
    detector_efficiency: float = 0.95      # Detection probability
    dark_count_rate: float = 0.0001       # Dark count probability
    coincidence_window_ns: float = 1.0    # For SPDC sources


@dataclass
class QuantumState:
    """Representation of a quantum state."""
    n_qubits: int
    amplitudes: np.ndarray
    probabilities: np.ndarray
    is_pure: bool = True
    entropy: float = 0.0


class QuantumStateSimulator:
    """
    GPU-accelerated quantum state simulator using cuQuantum.
    
    Simulates ideal quantum RNG circuits for baseline comparison.
    Uses tensor network methods for efficient large-scale simulation.
    """
    
    def __init__(
        self,
        n_qubits: int = 16,
        backend: str = "torch",
        device_id: int = 0,
        use_mps: bool = False,
        mps_bond_dim: int = 64,
    ):
        """
        Initialize quantum simulator.
        
        Args:
            n_qubits: Number of qubits
            backend: Compute backend ("torch", "numpy")
            device_id: CUDA device ID
            use_mps: Use Matrix Product State (MPS) for large systems
            mps_bond_dim: Maximum MPS bond dimension
        """
        self.n_qubits = n_qubits
        self.backend = backend
        self.device_id = device_id
        self.use_mps = use_mps
        self.mps_bond_dim = mps_bond_dim
        
        self.has_tensornet = False  # cuQuantum not available with CUDA 13.1
        self.has_cirq = HAS_CIRQ
        self.has_qiskit = HAS_QISKIT
        
        if HAS_TORCH_CUDA:
            torch.cuda.set_device(device_id)
        
        # cuQuantum tensor network configs not available
        self.config = None
    
    def create_hadamard_qrng_circuit(self, n_bits: Optional[int] = None):
        """
        Create a simple Hadamard-based QRNG circuit.
        
        Each qubit is put into superposition with H gate, then measured.
        This is the simplest ideal QRNG.
        
        Args:
            n_bits: Number of random bits (defaults to n_qubits)
            
        Returns:
            Circuit object (Cirq or Qiskit depending on availability)
        """
        n = n_bits or self.n_qubits
        
        if self.has_cirq:
            qubits = cirq.LineQubit.range(n)
            circuit = cirq.Circuit([cirq.H(q) for q in qubits])
            return circuit
        elif self.has_qiskit:
            circuit = QuantumCircuit(n, n)
            for i in range(n):
                circuit.h(i)
            circuit.measure(range(n), range(n))
            return circuit
        else:
            return None
    
    def create_entangled_qrng_circuit(self, n_bits: Optional[int] = None):
        """
        Create an entangled QRNG circuit using GHZ state.
        
        Creates maximally entangled state before measurement.
        Provides correlated randomness across qubits.
        
        Args:
            n_bits: Number of random bits
            
        Returns:
            Circuit object
        """
        n = n_bits or self.n_qubits
        
        if self.has_cirq:
            qubits = cirq.LineQubit.range(n)
            # GHZ state: H on first, then CNOT chain
            ops = [cirq.H(qubits[0])]
            for i in range(n - 1):
                ops.append(cirq.CNOT(qubits[i], qubits[i + 1]))
            return cirq.Circuit(ops)
        elif self.has_qiskit:
            circuit = QuantumCircuit(n, n)
            circuit.h(0)
            for i in range(n - 1):
                circuit.cx(i, i + 1)
            circuit.measure(range(n), range(n))
            return circuit
        else:
            return None
    
    def create_random_circuit(
        self,
        depth: int = 10,
        gate_set: Optional[List[str]] = None,
    ):
        """
        Create a random quantum circuit for complex randomness.
        
        Args:
            depth: Circuit depth
            gate_set: Gates to use (default: ["H", "CNOT", "T", "S"])
            
        Returns:
            Random circuit
        """
        n = self.n_qubits
        
        if self.has_cirq:
            return cirq.testing.random_circuit(
                n, depth, op_density=0.8, random_state=None
            )
        elif self.has_qiskit:
            from qiskit.circuit.random import random_circuit
            return random_circuit(n, depth, max_operands=2, measure=True)
        else:
            return None
    
    def simulate_and_sample(
        self,
        circuit=None,
        n_samples: int = 10000,
        noise_model: NoiseModel = NoiseModel.IDEAL,
        noise_params: Optional[NoiseParameters] = None,
    ) -> np.ndarray:
        """
        Simulate circuit and sample random values.
        
        Uses cuQuantum tensor network for efficient GPU simulation.
        
        Args:
            circuit: Quantum circuit (or creates Hadamard QRNG if None)
            n_samples: Number of random samples to generate
            noise_model: Type of noise to apply
            noise_params: Noise parameters
            
        Returns:
            Array of random float values in [0, 1)
        """
        if circuit is None:
            circuit = self.create_hadamard_qrng_circuit()
        
        if circuit is None:
            # No circuit library available - use pure random simulation
            return self._simulate_ideal_random(n_samples, noise_model, noise_params)
        
        # Use cuQuantum tensor network if available (disabled with CUDA 13.1)
        if self.has_tensornet:
            return self._simulate_with_tensornet(circuit, n_samples, noise_model, noise_params)
        else:
            return self._simulate_with_statevector(circuit, n_samples, noise_model, noise_params)
    
    def _simulate_with_tensornet(
        self,
        circuit,
        n_samples: int,
        noise_model: NoiseModel,
        noise_params: Optional[NoiseParameters],
    ) -> np.ndarray:
        """Simulate using cuQuantum tensor network."""
        try:
            # Create NetworkState from circuit
            state = NetworkState.from_circuit(
                circuit,
                dtype='complex128',
                backend=self.backend,
                config=self.config,
            )
            
            # Sample bitstrings
            samples = state.compute_sampling(n_samples)
            
            # Convert bitstrings to floats
            floats = self._bitstrings_to_floats(samples)
            
            # Apply noise model if not ideal
            if noise_model != NoiseModel.IDEAL and noise_params:
                floats = self._apply_classical_noise(floats, noise_model, noise_params)
            
            state.free()
            return floats
            
        except Exception as e:
            warnings.warn(f"TensorNet simulation failed: {e}, falling back to statevector")
            return self._simulate_with_statevector(circuit, n_samples, noise_model, noise_params)
    
    def _simulate_with_statevector(
        self,
        circuit,
        n_samples: int,
        noise_model: NoiseModel,
        noise_params: Optional[NoiseParameters],
    ) -> np.ndarray:
        """Simulate using direct statevector (for smaller circuits)."""
        n = self.n_qubits
        
        if self.has_cirq:
            # Use Cirq simulator
            simulator = cirq.Simulator()
            result = simulator.simulate(circuit)
            probs = np.abs(result.final_state_vector) ** 2
        elif self.has_qiskit:
            # Use Qiskit statevector
            from qiskit.quantum_info import Statevector
            # Remove measurements for statevector
            circuit_no_measure = circuit.remove_final_measurements(inplace=False)
            sv = Statevector.from_instruction(circuit_no_measure)
            probs = np.abs(sv.data) ** 2
        else:
            # Ideal Hadamard: uniform distribution
            probs = np.ones(2 ** n) / (2 ** n)
        
        # Sample from probability distribution using GPU if available
        if HAS_TORCH_CUDA:
            probs_gpu = torch.tensor(probs, device='cuda')
            indices = torch.multinomial(probs_gpu, n_samples, replacement=True)
            indices = indices.cpu().numpy()
        else:
            indices = np.random.choice(len(probs), size=n_samples, p=probs)
        
        # Convert to floats
        floats = indices.astype(np.float64) / (2 ** n)
        
        # Apply noise
        if noise_model != NoiseModel.IDEAL and noise_params:
            floats = self._apply_classical_noise(floats, noise_model, noise_params)
        
        return floats
    
    def _simulate_ideal_random(
        self,
        n_samples: int,
        noise_model: NoiseModel,
        noise_params: Optional[NoiseParameters],
    ) -> np.ndarray:
        """Generate ideal random values without circuit simulation."""
        n = self.n_qubits
        
        if HAS_TORCH_CUDA:
            # Generate n_qubits random bits per sample
            bits = torch.randint(0, 2, (n_samples, n), dtype=torch.uint8, device='cuda')
            # Convert to integer
            powers = (2 ** torch.arange(n - 1, -1, -1, dtype=torch.int64, device='cuda'))
            integers = torch.sum(bits.to(torch.int64) * powers, dim=1)
            # Normalize to [0, 1)
            floats = (integers.to(torch.float64) / (2 ** n)).cpu().numpy()
        else:
            bits = np.random.randint(0, 2, size=(n_samples, n), dtype=np.uint8)
            powers = 2 ** np.arange(n - 1, -1, -1)
            integers = np.sum(bits * powers, axis=1)
            floats = integers.astype(np.float64) / (2 ** n)
        
        # Apply noise
        if noise_model != NoiseModel.IDEAL and noise_params:
            floats = self._apply_classical_noise(floats, noise_model, noise_params)
        
        return floats
    
    def _bitstrings_to_floats(self, samples) -> np.ndarray:
        """Convert bitstring samples to float values."""
        n = self.n_qubits
        
        if isinstance(samples, dict):
            # Dictionary of bitstring -> count
            floats = []
            for bitstring, count in samples.items():
                if isinstance(bitstring, str):
                    value = int(bitstring, 2)
                else:
                    value = int(bitstring)
                floats.extend([value / (2 ** n)] * count)
            return np.array(floats)
        elif hasattr(samples, '__iter__'):
            # Array of bitstrings
            samples_arr = np.asarray(samples)
            if samples_arr.ndim == 2:
                # Matrix of bits
                powers = 2 ** np.arange(n - 1, -1, -1)
                integers = np.sum(samples_arr * powers, axis=1)
            else:
                integers = samples_arr
            return integers.astype(np.float64) / (2 ** n)
        else:
            return np.array([])
    
    def _apply_classical_noise(
        self,
        floats: np.ndarray,
        noise_model: NoiseModel,
        noise_params: NoiseParameters,
    ) -> np.ndarray:
        """Apply classical noise model to simulate imperfect QRNG."""
        n_samples = len(floats)
        
        if noise_model == NoiseModel.DEPOLARIZING:
            # Random bit flips
            mask = np.random.random(n_samples) < noise_params.depolarizing_prob
            floats[mask] = np.random.random(mask.sum())
            
        elif noise_model == NoiseModel.DETECTOR_EFFICIENCY:
            # Some detections fail, replaced with dark counts
            efficiency = noise_params.detector_efficiency
            dark_rate = noise_params.dark_count_rate
            
            detected = np.random.random(n_samples) < efficiency
            dark_counts = np.random.random(n_samples) < dark_rate
            
            # Failed detections get replaced
            failed = ~detected
            floats[failed] = np.random.random(failed.sum())
            
            # Dark counts add noise
            floats[dark_counts] = (floats[dark_counts] + np.random.random(dark_counts.sum())) % 1.0
            
        elif noise_model == NoiseModel.SPDC_REALISTIC:
            # Realistic SPDC source noise
            # 1. Photon pair timing jitter
            jitter = np.random.normal(0, 0.01, n_samples)
            floats = np.clip(floats + jitter, 0, 1)
            
            # 2. Detector efficiency
            efficiency = noise_params.detector_efficiency
            detected = np.random.random(n_samples) < efficiency
            floats[~detected] = np.random.random((~detected).sum())
            
            # 3. Accidental coincidences
            accidental_rate = noise_params.dark_count_rate
            accidentals = np.random.random(n_samples) < accidental_rate
            floats[accidentals] = np.random.random(accidentals.sum())
        
        return floats
    
    def get_theoretical_distribution(self, circuit=None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get theoretical probability distribution from circuit.
        
        Returns:
            Tuple of (values, probabilities)
        """
        n = self.n_qubits
        
        if circuit is None:
            # Hadamard: uniform distribution
            values = np.arange(2 ** n) / (2 ** n)
            probs = np.ones(2 ** n) / (2 ** n)
            return values, probs
        
        if self.has_cirq:
            simulator = cirq.Simulator()
            result = simulator.simulate(circuit)
            probs = np.abs(result.final_state_vector) ** 2
        elif self.has_qiskit:
            from qiskit.quantum_info import Statevector
            circuit_no_measure = circuit.remove_final_measurements(inplace=False)
            sv = Statevector.from_instruction(circuit_no_measure)
            probs = np.abs(sv.data) ** 2
        else:
            probs = np.ones(2 ** n) / (2 ** n)
        
        values = np.arange(len(probs)) / len(probs)
        return values, probs


class IdealQRNGDistribution:
    """
    Represents ideal QRNG statistical distribution for comparison.
    
    Provides expected values for various statistical tests based on
    perfect quantum randomness.
    """
    
    def __init__(self, n_bits: int = 32):
        """
        Initialize ideal distribution.
        
        Args:
            n_bits: Number of bits per random value
        """
        self.n_bits = n_bits
        self.n_states = 2 ** n_bits
    
    @property
    def expected_mean(self) -> float:
        """Expected mean for uniform [0, 1)."""
        return 0.5
    
    @property
    def expected_variance(self) -> float:
        """Expected variance for uniform [0, 1)."""
        return 1 / 12
    
    @property
    def expected_std(self) -> float:
        """Expected standard deviation."""
        return np.sqrt(self.expected_variance)
    
    @property
    def expected_entropy(self) -> float:
        """Expected Shannon entropy (bits)."""
        return self.n_bits
    
    @property
    def expected_autocorrelation(self) -> float:
        """Expected autocorrelation (should be ~0)."""
        return 0.0
    
    def kolmogorov_smirnov_threshold(self, n_samples: int, alpha: float = 0.05) -> float:
        """
        Get KS test critical value.
        
        Args:
            n_samples: Sample size
            alpha: Significance level
            
        Returns:
            Critical D value for rejecting null hypothesis
        """
        # Approximate critical value
        c_alpha = {0.1: 1.22, 0.05: 1.36, 0.01: 1.63}
        c = c_alpha.get(alpha, 1.36)
        return c / np.sqrt(n_samples)
    
    def chi_squared_threshold(self, n_bins: int, alpha: float = 0.05) -> float:
        """
        Get chi-squared test critical value.
        
        Args:
            n_bins: Number of histogram bins
            alpha: Significance level
            
        Returns:
            Critical chi-squared value
        """
        from scipy import stats
        return stats.chi2.ppf(1 - alpha, n_bins - 1)
    
    def generate_reference_samples(self, n_samples: int) -> np.ndarray:
        """
        Generate reference samples from ideal distribution.
        
        Uses cryptographically secure randomness as proxy for ideal quantum.
        
        Args:
            n_samples: Number of samples
            
        Returns:
            Array of random floats
        """
        import secrets
        
        # Generate cryptographically secure random bytes
        n_bytes = (self.n_bits + 7) // 8
        samples = np.zeros(n_samples, dtype=np.float64)
        
        for i in range(n_samples):
            random_bytes = secrets.token_bytes(n_bytes)
            value = int.from_bytes(random_bytes, 'little') & ((1 << self.n_bits) - 1)
            samples[i] = value / self.n_states
        
        return samples
    
    def compare_to_ideal(
        self,
        samples: np.ndarray,
        tests: Optional[List[str]] = None,
    ) -> Dict[str, Dict[str, float]]:
        """
        Compare samples to ideal distribution.
        
        Args:
            samples: Sample data to test
            tests: Which tests to run (default: all)
            
        Returns:
            Dictionary of test results
        """
        from scipy import stats
        
        if tests is None:
            tests = ["mean", "variance", "ks", "chi2", "runs", "autocorr"]
        
        results = {}
        n = len(samples)
        
        if "mean" in tests:
            sample_mean = np.mean(samples)
            z_score = (sample_mean - self.expected_mean) / (self.expected_std / np.sqrt(n))
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            results["mean"] = {
                "value": sample_mean,
                "expected": self.expected_mean,
                "z_score": z_score,
                "p_value": p_value,
                "pass": p_value > 0.01,
            }
        
        if "variance" in tests:
            sample_var = np.var(samples)
            # Chi-squared test for variance
            chi2_stat = (n - 1) * sample_var / self.expected_variance
            p_value = 2 * min(
                stats.chi2.cdf(chi2_stat, n - 1),
                1 - stats.chi2.cdf(chi2_stat, n - 1)
            )
            results["variance"] = {
                "value": sample_var,
                "expected": self.expected_variance,
                "chi2_stat": chi2_stat,
                "p_value": p_value,
                "pass": p_value > 0.01,
            }
        
        if "ks" in tests:
            # Kolmogorov-Smirnov test against uniform
            d_stat, p_value = stats.kstest(samples, 'uniform')
            results["ks"] = {
                "d_statistic": d_stat,
                "p_value": p_value,
                "threshold": self.kolmogorov_smirnov_threshold(n),
                "pass": p_value > 0.01,
            }
        
        if "chi2" in tests:
            # Chi-squared test with histogram
            n_bins = min(100, int(np.sqrt(n)))
            observed, _ = np.histogram(samples, bins=n_bins, range=(0, 1))
            expected = np.ones(n_bins) * n / n_bins
            chi2_stat, p_value = stats.chisquare(observed, expected)
            results["chi2"] = {
                "chi2_statistic": chi2_stat,
                "p_value": p_value,
                "n_bins": n_bins,
                "pass": p_value > 0.01,
            }
        
        if "runs" in tests:
            # Runs test for randomness
            median = np.median(samples)
            binary = (samples > median).astype(int)
            runs = 1 + np.sum(binary[1:] != binary[:-1])
            n1 = np.sum(binary)
            n0 = n - n1
            expected_runs = (2 * n1 * n0) / n + 1
            var_runs = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n * n * (n - 1))
            z_score = (runs - expected_runs) / np.sqrt(var_runs) if var_runs > 0 else 0
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            results["runs"] = {
                "n_runs": runs,
                "expected": expected_runs,
                "z_score": z_score,
                "p_value": p_value,
                "pass": p_value > 0.01,
            }
        
        if "autocorr" in tests:
            # Autocorrelation test
            mean = np.mean(samples)
            var = np.var(samples)
            if var > 0:
                autocorr = np.mean((samples[:-1] - mean) * (samples[1:] - mean)) / var
            else:
                autocorr = 0
            # Under null, autocorr ~ N(0, 1/n)
            z_score = autocorr * np.sqrt(n)
            p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
            results["autocorr"] = {
                "lag1_autocorr": autocorr,
                "expected": 0.0,
                "z_score": z_score,
                "p_value": p_value,
                "pass": p_value > 0.01,
            }
        
        # Overall pass rate
        n_tests = len(results)
        n_passed = sum(1 for r in results.values() if r.get("pass", False))
        results["summary"] = {
            "tests_run": n_tests,
            "tests_passed": n_passed,
            "pass_rate": n_passed / n_tests if n_tests > 0 else 0,
            "overall_pass": n_passed >= n_tests * 0.9,  # 90% must pass
        }
        
        return results


class SPDCSimulator:
    """
    Specialized simulator for SPDC (Spontaneous Parametric Down-Conversion) QRNG.
    
    Simulates the physics of photon pair generation and detection.
    """
    
    def __init__(
        self,
        n_ring_sections: int = 4,
        coincidence_window_ns: float = 1.0,
        detector_efficiency: float = 0.95,
        dark_count_rate: float = 1e-4,
    ):
        """
        Initialize SPDC simulator.
        
        Args:
            n_ring_sections: Number of detection ring sections
            coincidence_window_ns: Coincidence time window in nanoseconds
            detector_efficiency: Detector quantum efficiency
            dark_count_rate: Dark count probability per detection event
        """
        self.n_ring_sections = n_ring_sections
        self.coincidence_window_ns = coincidence_window_ns
        self.detector_efficiency = detector_efficiency
        self.dark_count_rate = dark_count_rate
    
    def generate_samples(self, n_samples: int) -> np.ndarray:
        """
        Generate random samples simulating SPDC QRNG.
        
        Models:
        - Photon pair emission from nonlinear crystal
        - Detection in opposite ring sections
        - Coincidence timing
        - Bit extraction from which-path information
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            Array of random floats in [0, 1)
        """
        if HAS_TORCH_CUDA:
            return self._generate_gpu(n_samples)
        else:
            return self._generate_cpu(n_samples)
    
    def _generate_gpu(self, n_samples: int) -> np.ndarray:
        """GPU-accelerated SPDC simulation using PyTorch."""
        # Generate photon pair events
        # Each pair goes to opposite ring sections
        n_pairs_needed = int(n_samples / self.detector_efficiency) + 1000
        
        # Random section pairs (opposing)
        sections = torch.randint(0, self.n_ring_sections // 2, (n_pairs_needed,), device='cuda')
        
        # Detection efficiency
        detected_1 = torch.rand(n_pairs_needed, device='cuda') < self.detector_efficiency
        detected_2 = torch.rand(n_pairs_needed, device='cuda') < self.detector_efficiency
        both_detected = detected_1 & detected_2
        
        # Timing (which detected first)
        timing = torch.rand(n_pairs_needed, device='cuda')
        bits = (timing > 0.5).to(torch.uint8)
        
        # Apply coincidence filter
        valid_bits = bits[both_detected]
        
        # Add dark counts
        n_dark = int(len(valid_bits) * self.dark_count_rate)
        if n_dark > 0:
            dark_indices = torch.randperm(len(valid_bits), device='cuda')[:n_dark]
            valid_bits[dark_indices] = torch.randint(0, 2, (n_dark,), dtype=torch.uint8, device='cuda')
        
        # Group bits into floats (use 32 bits per float)
        bits_per_float = 32
        n_floats = len(valid_bits) // bits_per_float
        
        if n_floats < n_samples:
            # Generate more if needed
            return self._generate_gpu(n_samples * 2)[:n_samples]
        
        valid_bits = valid_bits[:n_floats * bits_per_float].reshape(-1, bits_per_float)
        
        # Convert to floats
        powers = (2 ** torch.arange(bits_per_float - 1, -1, -1, dtype=torch.int64, device='cuda'))
        integers = torch.sum(valid_bits.to(torch.int64) * powers, dim=1)
        floats = (integers.to(torch.float64) / (2 ** bits_per_float))
        
        return floats[:n_samples].cpu().numpy()
    
    def _generate_cpu(self, n_samples: int) -> np.ndarray:
        """CPU SPDC simulation."""
        n_pairs_needed = int(n_samples / self.detector_efficiency) + 1000
        
        sections = np.random.randint(0, self.n_ring_sections // 2, size=n_pairs_needed)
        
        detected_1 = np.random.random(n_pairs_needed) < self.detector_efficiency
        detected_2 = np.random.random(n_pairs_needed) < self.detector_efficiency
        both_detected = detected_1 & detected_2
        
        timing = np.random.random(n_pairs_needed)
        bits = (timing > 0.5).astype(np.uint8)
        
        valid_bits = bits[both_detected]
        
        # Dark counts
        n_dark = int(len(valid_bits) * self.dark_count_rate)
        if n_dark > 0:
            dark_indices = np.random.choice(len(valid_bits), n_dark, replace=False)
            valid_bits[dark_indices] = np.random.randint(0, 2, n_dark).astype(np.uint8)
        
        bits_per_float = 32
        n_floats = len(valid_bits) // bits_per_float
        
        if n_floats < n_samples:
            return self._generate_cpu(n_samples * 2)[:n_samples]
        
        valid_bits = valid_bits[:n_floats * bits_per_float].reshape(-1, bits_per_float)
        powers = 2 ** np.arange(bits_per_float - 1, -1, -1)
        integers = np.sum(valid_bits.astype(np.uint64) * powers, axis=1)
        floats = integers / (2 ** bits_per_float)
        
        return floats[:n_samples]
