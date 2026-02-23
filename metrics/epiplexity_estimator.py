"""
Epiplexity Estimator
====================

Implements epiplexity estimation based on the theoretical framework from:
"From Entropy to Epiplexity: Rethinking Information for Computationally Bounded Intelligence"
(Finzi, Qiu, Jiang, Izmailov, Kolter, Wilson - arXiv:2601.03220, 2026)

Key Concepts:
- **Epiplexity (S_T)**: Structural information extractable by a computationally bounded observer
- **Time-Bounded Entropy (H_T)**: Unpredictable/random content for that observer
- **Total Information**: MDL_T(X) = S_T(X) + H_T(X)

For QRNG streams:
- Pure random: High H_T, Low S_T (nothing to learn)
- Consciousness influence: May decrease H_T while increasing S_T (structure emerges!)

Measurement Methods:
1. Loss curve area (simple heuristic)
2. Description length compression ratio
3. Prediction error trajectory
4. Kolmogorov complexity approximation via compression

Integration with HELIOS:
    from epiplexity_estimator import EpiplexityEstimator
    
    estimator = EpiplexityEstimator(window_size=100)
    
    for value in qrng_stream:
        metrics = estimator.update(value)
        if metrics['structural_emergence']:
            print("Structure emerging from randomness!")

Reference:
    arXiv:2601.03220 - Section 3, Definition 8
"""

import numpy as np
import zlib
import lzma
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass, field
from collections import deque
from enum import Enum


class CompressionMethod(Enum):
    """Compression algorithms for Kolmogorov complexity approximation."""
    ZLIB = "zlib"
    LZMA = "lzma"
    DELTA = "delta"  # Custom delta encoding


@dataclass
class EpiplexityMetrics:
    """Results from epiplexity estimation."""
    epiplexity: float           # S_T: Structural information (program size)
    time_bounded_entropy: float  # H_T: Remaining randomness
    total_mdl: float            # S_T + H_T: Total description length
    compression_ratio: float    # How compressible the stream is
    prediction_error: float     # Average prediction error
    structural_emergence: bool  # Flag: is structure emerging?
    emergence_confidence: float # 0-1: confidence in emergence detection
    details: Dict[str, float] = field(default_factory=dict)
    
    def __str__(self) -> str:
        status = "🔮 EMERGENCE" if self.structural_emergence else "📊 RANDOM"
        return (f"{status} S_T={self.epiplexity:.3f}, H_T={self.time_bounded_entropy:.3f}, "
                f"compress={self.compression_ratio:.2%}")


def approximate_kolmogorov_complexity(data: bytes, method: CompressionMethod = CompressionMethod.ZLIB) -> int:
    """
    Approximate Kolmogorov complexity using compression.
    
    K(x) ≈ len(compress(x)) + C
    
    This is an upper bound on true Kolmogorov complexity.
    Random data compresses poorly; structured data compresses well.
    
    Args:
        data: Binary data to measure
        method: Compression algorithm to use
        
    Returns:
        Approximate Kolmogorov complexity in bits
    """
    if len(data) == 0:
        return 0
        
    if method == CompressionMethod.ZLIB:
        compressed = zlib.compress(data, level=9)
    elif method == CompressionMethod.LZMA:
        compressed = lzma.compress(data, preset=9)
    elif method == CompressionMethod.DELTA:
        # Delta encoding: store differences
        arr = np.frombuffer(data, dtype=np.uint8)
        deltas = np.diff(arr.astype(np.int16))
        delta_bytes = deltas.astype(np.int8).tobytes()
        compressed = zlib.compress(delta_bytes, level=9)
    else:
        compressed = zlib.compress(data, level=9)
    
    return len(compressed) * 8  # Convert bytes to bits


def compute_description_length(sequence: np.ndarray, model_complexity: int = 0) -> Tuple[float, float]:
    """
    Compute two-part Minimum Description Length.
    
    MDL = |P| + E[-log P(X)]
    
    Where:
    - |P| is the program/model size (epiplexity)
    - E[-log P(X)] is the entropy given the model
    
    Args:
        sequence: Data sequence
        model_complexity: Known complexity of generating model (if any)
        
    Returns:
        (model_length, data_given_model_length)
    """
    if len(sequence) < 2:
        return 0.0, 0.0
    
    # Convert to bytes for compression
    # Quantize floats to 16-bit for compression
    quantized = (sequence * 65535).astype(np.uint16)
    data_bytes = quantized.tobytes()
    
    # Raw data length
    raw_bits = len(data_bytes) * 8
    
    # Compressed length ≈ K(x)
    compressed_bits = approximate_kolmogorov_complexity(data_bytes)
    
    # The "program" is what's needed beyond raw entropy
    # For random data: compressed_bits ≈ raw_bits (incompressible)
    # For structured data: compressed_bits << raw_bits
    
    # Estimate epiplexity: the learnable/compressible part
    epiplexity = max(0, raw_bits - compressed_bits) / 8  # In bytes for interpretability
    
    # Estimate remaining entropy: what couldn't be compressed
    remaining_entropy = compressed_bits / raw_bits if raw_bits > 0 else 1.0
    
    return epiplexity, remaining_entropy


class OnlinePredictor:
    """
    Simple online predictor for measuring predictability.
    
    Uses exponential moving average - if data becomes predictable,
    prediction error drops, indicating structure is emerging.
    """
    
    def __init__(self, alpha: float = 0.1):
        """
        Args:
            alpha: EMA smoothing factor (higher = more responsive)
        """
        self.alpha = alpha
        self.prediction: Optional[float] = None
        self.error_history: List[float] = []
        self.max_history = 1000
        
    def predict_and_update(self, value: float) -> float:
        """
        Make prediction and update model.
        
        Returns:
            Squared prediction error
        """
        if self.prediction is None:
            self.prediction = value
            error = 0.0
        else:
            error = (value - self.prediction) ** 2
            # Update prediction with EMA
            self.prediction = self.alpha * value + (1 - self.alpha) * self.prediction
            
        self.error_history.append(error)
        if len(self.error_history) > self.max_history:
            self.error_history.pop(0)
            
        return error
    
    def get_error_trend(self, window: int = 50) -> float:
        """
        Get trend in prediction error.
        
        Negative trend = errors decreasing = learning structure
        Positive trend = errors increasing = becoming more random
        
        Returns:
            Slope of error trend (negative = structure emerging)
        """
        if len(self.error_history) < window:
            return 0.0
            
        recent = self.error_history[-window:]
        x = np.arange(len(recent))
        
        try:
            slope, _ = np.polyfit(x, recent, 1)
            return slope
        except:
            return 0.0
    
    def get_mean_error(self, window: int = 50) -> float:
        """Get mean squared error over recent window."""
        if len(self.error_history) < 1:
            return 0.0
        recent = self.error_history[-window:]
        return np.mean(recent)


class LossCurveEstimator:
    """
    Estimate epiplexity from the "loss curve" of an online learner.
    
    From arXiv:2601.03220:
    "A simple heuristic measurement is the area under the loss curve above the final loss"
    
    The area represents how much the observer "learned" - i.e., the structural
    information that was extracted and encoded into the model.
    """
    
    def __init__(self, window_size: int = 200):
        self.window_size = window_size
        self.loss_history: List[float] = []
        
    def update(self, loss: float) -> None:
        """Add new loss value."""
        self.loss_history.append(loss)
        if len(self.loss_history) > self.window_size * 2:
            self.loss_history = self.loss_history[-self.window_size * 2:]
    
    def compute_epiplexity(self) -> Tuple[float, float]:
        """
        Compute epiplexity as area under loss curve above final loss.
        
        Returns:
            (epiplexity_estimate, final_loss)
        """
        if len(self.loss_history) < 10:
            return 0.0, 0.0
            
        losses = np.array(self.loss_history[-self.window_size:])
        final_loss = np.mean(losses[-10:])  # Average of last 10 as "final"
        
        # Area above final loss = learning that occurred
        above_final = np.maximum(0, losses - final_loss)
        # Use trapezoid instead of deprecated trapz (numpy 2.0+)
        try:
            area = np.trapezoid(above_final)
        except AttributeError:
            # Fallback for older numpy versions
            area = np.trapz(above_final)
        
        # Normalize by window size
        epiplexity = area / len(losses)
        
        return epiplexity, final_loss
    
    def get_learning_rate(self, window: int = 50) -> float:
        """
        Measure how fast the observer is learning.
        
        Returns:
            Negative = still learning (loss decreasing)
            Zero = converged
            Positive = losing ground (structure disappearing?)
        """
        if len(self.loss_history) < window:
            return 0.0
            
        recent = self.loss_history[-window:]
        x = np.arange(len(recent))
        
        try:
            slope, _ = np.polyfit(x, recent, 1)
            return slope
        except:
            return 0.0


class EpiplexityEstimator:
    """
    Complete epiplexity estimation system.
    
    Combines multiple methods to estimate:
    - S_T: Epiplexity (structural information for bounded observer)
    - H_T: Time-bounded entropy (irreducible randomness)
    
    For consciousness influence detection:
    - Pure QRNG: High H_T, Low S_T
    - Influenced QRNG: S_T increases as structure emerges
    """
    
    def __init__(self, 
                 window_size: int = 200,
                 compression_method: CompressionMethod = CompressionMethod.ZLIB,
                 emergence_threshold: float = 0.15,
                 learning_rate_threshold: float = -0.001):
        """
        Args:
            window_size: Analysis window
            compression_method: For Kolmogorov complexity approximation
            emergence_threshold: Compression ratio decrease to flag emergence
            learning_rate_threshold: Error decrease rate to flag emergence
        """
        self.window_size = window_size
        self.compression_method = compression_method
        self.emergence_threshold = emergence_threshold
        self.learning_rate_threshold = learning_rate_threshold
        
        # Data buffers
        self.value_buffer: deque = deque(maxlen=window_size * 2)
        self.trajectory_buffer: deque = deque(maxlen=window_size * 2)
        
        # Sub-estimators
        self.predictor = OnlinePredictor(alpha=0.1)
        self.loss_estimator = LossCurveEstimator(window_size)
        
        # History tracking
        self.epiplexity_history: List[float] = []
        self.entropy_history: List[float] = []
        self.compression_history: List[float] = []
        
        # Baseline (computed during warmup)
        self.baseline_compression: Optional[float] = None
        self.baseline_prediction_error: Optional[float] = None
        self.warmup_steps = 100
        self.step_count = 0
        
    def update(self, value: float, 
               trajectory_point: Optional[Tuple[float, float]] = None) -> EpiplexityMetrics:
        """
        Update estimator with new value.
        
        Args:
            value: Scalar value from QRNG or other stream
            trajectory_point: Optional (x, y) for phase space analysis
            
        Returns:
            EpiplexityMetrics with current estimates
        """
        self.step_count += 1
        self.value_buffer.append(value)
        
        if trajectory_point:
            self.trajectory_buffer.append(trajectory_point)
        
        # Update predictor
        pred_error = self.predictor.predict_and_update(value)
        self.loss_estimator.update(pred_error)
        
        # Compute metrics
        metrics = self._compute_metrics()
        
        # Update baseline during warmup
        if self.step_count <= self.warmup_steps:
            self._update_baseline(metrics)
        
        return metrics
    
    def _compute_metrics(self) -> EpiplexityMetrics:
        """Compute all epiplexity metrics."""
        if len(self.value_buffer) < 20:
            return EpiplexityMetrics(
                epiplexity=0.0,
                time_bounded_entropy=1.0,
                total_mdl=1.0,
                compression_ratio=1.0,
                prediction_error=0.0,
                structural_emergence=False,
                emergence_confidence=0.0
            )
        
        # Convert buffer to array
        values = np.array(list(self.value_buffer))
        
        # 1. Compression-based estimation
        epiplexity_compress, entropy_ratio = compute_description_length(values)
        
        # Compression ratio: how much of original data remains after compression
        raw_bits = len(values) * 16  # 16-bit quantization
        quantized = (values * 65535).astype(np.uint16)
        compressed_bits = approximate_kolmogorov_complexity(
            quantized.tobytes(), 
            self.compression_method
        )
        compression_ratio = compressed_bits / raw_bits if raw_bits > 0 else 1.0
        
        # 2. Loss curve based estimation
        loss_epiplexity, final_loss = self.loss_estimator.compute_epiplexity()
        
        # 3. Prediction error analysis
        mean_error = self.predictor.get_mean_error()
        error_trend = self.predictor.get_error_trend()
        learning_rate = self.loss_estimator.get_learning_rate()
        
        # Combine estimates
        # Epiplexity: weight compression and loss-curve methods
        epiplexity = 0.5 * epiplexity_compress + 0.5 * loss_epiplexity
        
        # Time-bounded entropy: what remains unpredictable
        # Higher compression ratio = more entropy (less compressible)
        time_bounded_entropy = compression_ratio
        
        # Total MDL
        total_mdl = epiplexity + time_bounded_entropy * len(values)
        
        # Store history
        self.epiplexity_history.append(epiplexity)
        self.entropy_history.append(time_bounded_entropy)
        self.compression_history.append(compression_ratio)
        
        # Trim history
        max_history = self.window_size * 3
        if len(self.epiplexity_history) > max_history:
            self.epiplexity_history = self.epiplexity_history[-max_history:]
            self.entropy_history = self.entropy_history[-max_history:]
            self.compression_history = self.compression_history[-max_history:]
        
        # Detect structural emergence
        emergence, confidence = self._detect_emergence(
            epiplexity, compression_ratio, error_trend, learning_rate
        )
        
        return EpiplexityMetrics(
            epiplexity=epiplexity,
            time_bounded_entropy=time_bounded_entropy,
            total_mdl=total_mdl,
            compression_ratio=compression_ratio,
            prediction_error=mean_error,
            structural_emergence=emergence,
            emergence_confidence=confidence,
            details={
                'loss_epiplexity': loss_epiplexity,
                'compress_epiplexity': epiplexity_compress,
                'error_trend': error_trend,
                'learning_rate': learning_rate,
                'final_loss': final_loss
            }
        )
    
    def _update_baseline(self, metrics: EpiplexityMetrics) -> None:
        """Update baseline statistics during warmup."""
        if self.step_count == self.warmup_steps:
            # Set baseline from warmup period
            if self.compression_history:
                self.baseline_compression = np.mean(self.compression_history)
            if self.predictor.error_history:
                self.baseline_prediction_error = np.mean(self.predictor.error_history)
    
    def _detect_emergence(self, 
                         epiplexity: float, 
                         compression_ratio: float,
                         error_trend: float,
                         learning_rate: float) -> Tuple[bool, float]:
        """
        Detect if structural information is emerging from randomness.
        
        Emergence indicators:
        1. Compression ratio decreasing (data becoming more compressible)
        2. Prediction error decreasing (data becoming more predictable)
        3. Epiplexity increasing (more structure being extracted)
        
        Returns:
            (is_emerging, confidence)
        """
        if self.step_count < self.warmup_steps + 50:
            return False, 0.0
        
        signals = []
        
        # Signal 1: Compression improvement
        if self.baseline_compression is not None:
            compression_change = compression_ratio - self.baseline_compression
            if compression_change < -self.emergence_threshold:
                signals.append(('compression', min(1.0, abs(compression_change) / 0.3)))
        
        # Signal 2: Prediction error decreasing
        if error_trend < self.learning_rate_threshold:
            signals.append(('prediction', min(1.0, abs(error_trend) / 0.01)))
        
        # Signal 3: Epiplexity increasing
        if len(self.epiplexity_history) > 50:
            early = np.mean(self.epiplexity_history[:25])
            recent = np.mean(self.epiplexity_history[-25:])
            if recent > early * 1.2:  # 20% increase
                signals.append(('epiplexity', min(1.0, (recent - early) / early)))
        
        # Signal 4: Learning still happening (negative learning rate)
        if learning_rate < self.learning_rate_threshold:
            signals.append(('learning', min(1.0, abs(learning_rate) / 0.005)))
        
        # Emergence if multiple signals agree
        is_emerging = len(signals) >= 2
        
        # Confidence based on signal strength and agreement
        if signals:
            confidence = np.mean([s[1] for s in signals]) * (len(signals) / 4)
        else:
            confidence = 0.0
        
        return is_emerging, min(1.0, confidence)
    
    def get_epiplexity_trend(self, window: int = 50) -> float:
        """
        Get trend in epiplexity over time.
        
        Positive trend = structure is accumulating
        Negative trend = structure is dissolving
        """
        if len(self.epiplexity_history) < window:
            return 0.0
        
        recent = self.epiplexity_history[-window:]
        x = np.arange(len(recent))
        
        try:
            slope, _ = np.polyfit(x, recent, 1)
            return slope
        except:
            return 0.0
    
    def get_entropy_trend(self, window: int = 50) -> float:
        """
        Get trend in time-bounded entropy.
        
        Negative trend = randomness decreasing = structure emerging
        Positive trend = randomness increasing
        """
        if len(self.entropy_history) < window:
            return 0.0
        
        recent = self.entropy_history[-window:]
        x = np.arange(len(recent))
        
        try:
            slope, _ = np.polyfit(x, recent, 1)
            return slope
        except:
            return 0.0
    
    def compute_trajectory_epiplexity(self) -> Optional[EpiplexityMetrics]:
        """
        Compute epiplexity specifically for phase space trajectory.
        
        Returns None if no trajectory data available.
        """
        if len(self.trajectory_buffer) < 20:
            return None
        
        # Convert trajectory to flattened array
        points = np.array(list(self.trajectory_buffer))
        trajectory_flat = points.flatten()
        
        # Compute description length
        epiplexity, entropy_ratio = compute_description_length(trajectory_flat)
        
        # Compression
        quantized = (trajectory_flat * 65535).astype(np.uint16)
        raw_bits = len(quantized) * 16
        compressed_bits = approximate_kolmogorov_complexity(
            quantized.tobytes(), self.compression_method
        )
        compression_ratio = compressed_bits / raw_bits if raw_bits > 0 else 1.0
        
        return EpiplexityMetrics(
            epiplexity=epiplexity,
            time_bounded_entropy=compression_ratio,
            total_mdl=epiplexity + compression_ratio * len(trajectory_flat),
            compression_ratio=compression_ratio,
            prediction_error=0.0,
            structural_emergence=compression_ratio < 0.7,  # Simple threshold
            emergence_confidence=max(0, 1.0 - compression_ratio),
            details={'trajectory_points': len(self.trajectory_buffer)}
        )


# Convenience functions

def estimate_stream_epiplexity(values: List[float], 
                               window_size: int = 100) -> EpiplexityMetrics:
    """
    Estimate epiplexity for a complete stream of values.
    
    Args:
        values: List of scalar values
        window_size: Analysis window
        
    Returns:
        Final EpiplexityMetrics
    """
    estimator = EpiplexityEstimator(window_size=window_size)
    
    metrics = None
    for v in values:
        metrics = estimator.update(v)
    
    return metrics if metrics else EpiplexityMetrics(
        epiplexity=0.0,
        time_bounded_entropy=1.0,
        total_mdl=1.0,
        compression_ratio=1.0,
        prediction_error=0.0,
        structural_emergence=False,
        emergence_confidence=0.0
    )


def compare_epiplexity(stream_a: List[float], 
                       stream_b: List[float]) -> Dict[str, float]:
    """
    Compare epiplexity between two streams.
    
    Useful for A/B testing QRNG vs PRNG, or influenced vs baseline.
    
    Returns:
        Dict with comparison metrics
    """
    metrics_a = estimate_stream_epiplexity(stream_a)
    metrics_b = estimate_stream_epiplexity(stream_b)
    
    return {
        'epiplexity_diff': metrics_b.epiplexity - metrics_a.epiplexity,
        'entropy_diff': metrics_b.time_bounded_entropy - metrics_a.time_bounded_entropy,
        'compression_diff': metrics_b.compression_ratio - metrics_a.compression_ratio,
        'a_epiplexity': metrics_a.epiplexity,
        'b_epiplexity': metrics_b.epiplexity,
        'a_entropy': metrics_a.time_bounded_entropy,
        'b_entropy': metrics_b.time_bounded_entropy,
        'a_emergence': metrics_a.structural_emergence,
        'b_emergence': metrics_b.structural_emergence
    }
