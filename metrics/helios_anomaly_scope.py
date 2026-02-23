"""
HELIOS Anomaly Scope
====================

Attaches to the HELIOS Processor to visualize the 'Trajectory' of its thought process.
Detects if the CORGI Junction is behaving randomly (Exploration) or if it has locked 
onto a pattern (Emergent Behavior/Influence).

This module implements the core detection algorithms:
- Phase Space Reconstruction (Takens' Embedding)
- Mean Squared Displacement (MSD) tracking
- Coherence/Entropy monitoring
- Real-time influence detection

Integration with LOCAL_Ai HELIOS:
    from helios_cuda.helios_model import HeliosProcessor
    from helios_anomaly_scope import HeliosAnomalyScope
    
    model = HeliosProcessor(dim=64)
    scope = HeliosAnomalyScope(history_len=50)
    
    for step in range(200):
        readout, indices = model(input_signal)
        active_batch = model.ring_state[0, indices[0]]
        scope.update(active_batch)
        
        if scope.detect_influence():
            print("EMERGENCE DETECTED")

Theory Reference:
    - QFC (Quantum-interacting Fundamental Consciousness) framework
    - Consciousness influence detection via trajectory analysis
    - Signal vs. Noise discrimination in QRNG streams
    - Epiplexity framework (arXiv:2601.03220) for structural information detection
"""

import numpy as np
import torch
from typing import Optional, Tuple, List, Dict
from dataclasses import dataclass, field
from collections import deque
from enum import Enum

# Import epiplexity estimator for structural information detection
try:
    from epiplexity_estimator import EpiplexityEstimator, EpiplexityMetrics
    EPIPLEXITY_AVAILABLE = True
except ImportError:
    EPIPLEXITY_AVAILABLE = False
    EpiplexityEstimator = None
    EpiplexityMetrics = None

# Import validation utilities
try:
    from validation import (
        validate_positive_int, validate_positive_float,
        validate_choice, validate_tensor, ValidationError
    )
except ImportError:
    # Fallback if validation module not available
    ValidationError = ValueError
    def validate_positive_int(v, n, **kw): return int(v)
    def validate_positive_float(v, n, **kw): return float(v)
    def validate_choice(v, n, c): return v
    def validate_tensor(v, n, **kw): return v


class SignalClass(Enum):
    """Classification of detected signal types."""
    NOISE = "noise"                    # Random fluctuation, no real signal
    DRIFT = "drift"                    # Sustained directional movement
    ATTRACTOR = "attractor"            # Locked into stable orbit/pattern
    PERIODIC = "periodic"              # Oscillating/cyclical behavior
    CHAOTIC = "chaotic"                # Sensitive dependence, emerging structure
    ANOMALOUS = "anomalous"            # Unusual but unclassified
    INFLUENCE = "influence"            # Strong multi-metric coherent signal
    EMERGENCE = "emergence"            # Structural information emerging (epiplexity increasing)


@dataclass
class SignalVerification:
    """Results of signal verification tests."""
    is_verified: bool
    confidence: float  # 0-1, how confident we are this is a real signal
    signal_class: SignalClass
    persistence_score: float  # How long the signal persists
    multi_metric_agreement: float  # How many metrics agree
    statistical_significance: float  # p-value or similar
    tests_passed: List[str]  # Which verification tests passed
    tests_failed: List[str]  # Which tests failed
    details: Dict[str, float] = field(default_factory=dict)
    
    def __str__(self) -> str:
        status = "✓ VERIFIED" if self.is_verified else "✗ UNVERIFIED"
        return (f"{status} [{self.signal_class.value.upper()}] "
                f"confidence={self.confidence:.2f}, "
                f"persistence={self.persistence_score:.2f}, "
                f"agreement={self.multi_metric_agreement:.2f}")


@dataclass
class AnomalyEvent:
    """Records a detected anomaly/emergence event."""
    step: int
    event_type: str  # 'emergence', 'attractor_lock', 'drift', 'coherence_spike', 'ballistic_motion'
    confidence: float
    metrics: Dict[str, float] = field(default_factory=dict)
    description: str = ""
    verification: Optional[SignalVerification] = None  # Added verification result


def compute_hurst_exponent(series: np.ndarray, max_lag: int = 20) -> float:
    """
    Compute Hurst Exponent using R/S analysis.
    
    H ≈ 0.5: Random walk (no memory)
    H > 0.5: Trending/persistent (positive autocorrelation)
    H < 0.5: Mean-reverting (negative autocorrelation)
    
    Args:
        series: Time series data
        max_lag: Maximum lag for analysis
        
    Returns:
        Hurst exponent estimate
    """
    if len(series) < max_lag * 2:
        return 0.5  # Not enough data
        
    lags = range(2, max_lag)
    rs_values = []
    
    for lag in lags:
        # Divide series into chunks of size 'lag'
        chunks = len(series) // lag
        if chunks < 2:
            continue
            
        rs_chunk = []
        for i in range(chunks):
            chunk = series[i * lag:(i + 1) * lag]
            mean = np.mean(chunk)
            cumdev = np.cumsum(chunk - mean)
            R = np.max(cumdev) - np.min(cumdev)
            S = np.std(chunk)
            if S > 0:
                rs_chunk.append(R / S)
                
        if rs_chunk:
            rs_values.append((lag, np.mean(rs_chunk)))
    
    if len(rs_values) < 3:
        return 0.5
        
    # Linear fit in log-log space: log(R/S) = H * log(n)
    log_lags = np.log([x[0] for x in rs_values])
    log_rs = np.log([x[1] for x in rs_values])
    
    try:
        H, _ = np.polyfit(log_lags, log_rs, 1)
        return np.clip(H, 0, 1)
    except:
        return 0.5


def compute_msd_from_trajectory(x: List[float], y: List[float], max_lag: int = 50) -> Tuple[np.ndarray, np.ndarray, float]:
    """
    Compute proper Mean Squared Displacement.
    
    MSD(τ) = ⟨|r(t+τ) - r(t)|²⟩ averaged over all t
    
    For diffusive motion: MSD ∝ τ (slope = 1 in log-log)
    For ballistic motion: MSD ∝ τ² (slope = 2 in log-log)
    
    Args:
        x, y: Trajectory coordinates
        max_lag: Maximum time lag to compute
        
    Returns:
        (lags, msd_values, diffusion_exponent)
    """
    n = len(x)
    if n < 10:
        return np.array([1]), np.array([0]), 1.0
        
    x = np.array(x)
    y = np.array(y)
    
    lags = np.arange(1, min(max_lag, n // 2))
    msd_values = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        # Compute squared displacements for this lag
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        squared_displacements = dx**2 + dy**2
        msd_values[i] = np.mean(squared_displacements)
    
    # Compute diffusion exponent (slope in log-log space)
    valid = msd_values > 0
    if np.sum(valid) < 3:
        return lags, msd_values, 1.0
        
    log_lags = np.log(lags[valid])
    log_msd = np.log(msd_values[valid])
    
    try:
        alpha, _ = np.polyfit(log_lags, log_msd, 1)
        # alpha ≈ 1 for diffusive, alpha ≈ 2 for ballistic
        return lags, msd_values, alpha
    except:
        return lags, msd_values, 1.0


def compute_lyapunov_exponent(x: List[float], y: List[float], 
                               min_neighbors: int = 5,
                               evolution_steps: int = 10) -> float:
    """
    Estimate the largest Lyapunov exponent from a 2D trajectory.
    
    The Lyapunov exponent measures the rate of separation of infinitesimally
    close trajectories - a key signature of chaos vs. order.
    
    λ > 0: Chaotic (nearby trajectories diverge exponentially - faster than diffusive)
    λ ≈ 0: Stable/neutral (normal random walk divergence)
    λ < 0: Convergent (trajectories collapse to attractor)
    
    For consciousness influence detection:
    - Random QRNG should show λ ≈ 0 (diffusive spread)
    - Influenced trajectory may show λ > 0 (sensitive dependence) or λ < 0 (attractor)
    
    Uses the Rosenstein algorithm with normalization for diffusive baseline:
    1. For each point, find nearest neighbor in phase space
    2. Track how the distance evolves over time
    3. Compare to expected √t diffusive growth
    4. λ = deviation from diffusive expectation
    
    Args:
        x, y: Trajectory coordinates
        min_neighbors: Minimum temporal separation for "neighbor" (avoid autocorrelation)
        evolution_steps: How many steps to track divergence
        
    Returns:
        Estimated largest Lyapunov exponent (normalized to 0 for random walk)
    """
    n = len(x)
    if n < 50:  # Need sufficient data for reliable estimate
        return 0.0
        
    x = np.array(x)
    y = np.array(y)
    
    # Estimate step size from trajectory
    dx = np.diff(x)
    dy = np.diff(y)
    step_sizes = np.sqrt(dx**2 + dy**2)
    mean_step = np.mean(step_sizes) if len(step_sizes) > 0 else 0.1
    
    # Stack into 2D points
    trajectory = np.column_stack([x, y])

    divergence_rates = []

    # Optimized neighbor search using vectorized operations
    # Sample points for efficiency (every 3rd point)
    sample_indices = np.arange(0, n - evolution_steps - min_neighbors, 3)

    if len(sample_indices) == 0:
        return 0.0

    for i in sample_indices:
        # Vectorized distance computation to all valid neighbors
        valid_mask = np.abs(np.arange(n - evolution_steps) - i) >= min_neighbors
        valid_indices = np.where(valid_mask)[0]

        if len(valid_indices) == 0:
            continue

        # Compute distances to all valid points at once
        diffs = trajectory[valid_indices] - trajectory[i]
        distances = np.sqrt(np.sum(diffs**2, axis=1))

        # Filter by distance constraints
        dist_mask = (distances > mean_step * 0.5) & (distances < mean_step * 20)
        valid_distances = distances[dist_mask]
        valid_neighbor_indices = valid_indices[dist_mask]

        if len(valid_distances) == 0:
            continue

        # Get nearest neighbor
        min_idx = np.argmin(valid_distances)
        best_dist = valid_distances[min_idx]
        best_j = valid_neighbor_indices[min_idx]

        # Track how distance evolves (vectorized)
        initial_dist = best_dist
        k_range = np.arange(1, evolution_steps + 1)
        valid_k = (i + k_range < n) & (best_j + k_range < n)
        k_valid = k_range[valid_k]

        if len(k_valid) < 5:
            continue

        # Vectorized distance evolution computation
        evolved_diffs = trajectory[i + k_valid] - trajectory[best_j + k_valid]
        evolved_distances = np.sqrt(np.sum(evolved_diffs**2, axis=1))

        # Compute log ratios (vectorized)
        valid_evolved = (evolved_distances > 1e-10) & (initial_dist > 1e-10)
        if np.sum(valid_evolved) < 3:
            continue

        log_ratios = np.log(evolved_distances[valid_evolved] / initial_dist)
        times = k_valid[valid_evolved]

        # Fit: log(d/d₀) = λ * t
        try:
            lambda_est, _ = np.polyfit(times, log_ratios, 1)
            divergence_rates.append(lambda_est)
        except Exception:
            pass
    
    if len(divergence_rates) < 5:
        return 0.0
    
    # The raw Lyapunov will be positive for random walk due to diffusive spread
    # Normalize by subtracting expected diffusive rate: λ_diff ≈ 0.5/t_scale for √t growth
    raw_lyapunov = float(np.mean(divergence_rates))
    
    # Expected diffusive rate: for √t growth, log(d) ~ 0.5*log(t)
    # At our evolution_steps scale, this is approximately 0.5 / evolution_steps
    # But more accurately, we compare to theoretical random walk
    expected_diffusive = 0.5 / evolution_steps * 2  # Empirical calibration
    
    # Normalized Lyapunov: 0 for random walk, positive for chaos, negative for attractor
    normalized_lyapunov = raw_lyapunov - expected_diffusive
    
    return normalized_lyapunov


# =============================================================================
# SIGNAL VERIFICATION AND CLASSIFICATION
# =============================================================================

def compute_runs_test(series: np.ndarray) -> Tuple[float, bool]:
    """
    Wald-Wolfowitz runs test for randomness.
    
    Counts the number of "runs" (consecutive sequences above/below median).
    Too few runs = trending, too many = oscillating.
    
    Returns:
        (z_score, is_random) where is_random is True if |z| < 1.96 (p > 0.05)
    """
    if len(series) < 20:
        return 0.0, True
        
    median = np.median(series)
    binary = (series > median).astype(int)
    
    # Count runs
    runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i-1]:
            runs += 1
    
    # Expected runs for random sequence
    n1 = np.sum(binary)
    n0 = len(binary) - n1
    
    if n0 == 0 or n1 == 0:
        return 0.0, True
        
    expected_runs = (2 * n0 * n1) / (n0 + n1) + 1
    std_runs = np.sqrt((2 * n0 * n1 * (2 * n0 * n1 - n0 - n1)) / 
                       ((n0 + n1)**2 * (n0 + n1 - 1)))
    
    if std_runs < 1e-10:
        return 0.0, True
        
    z_score = (runs - expected_runs) / std_runs
    is_random = abs(z_score) < 1.96  # 95% confidence
    
    return z_score, is_random


def compute_autocorrelation(series: np.ndarray, max_lag: int = 20) -> np.ndarray:
    """
    Compute autocorrelation coefficients up to max_lag.
    
    For random series, autocorrelation should be near 0 for all lags > 0.
    Significant positive autocorrelation = trending/persistent
    Significant negative autocorrelation = oscillating/mean-reverting
    """
    if len(series) < max_lag + 10:
        return np.zeros(max_lag)
        
    series = series - np.mean(series)
    n = len(series)
    
    autocorr = np.zeros(max_lag)
    var = np.var(series)
    
    if var < 1e-10:
        return autocorr
        
    for lag in range(1, max_lag + 1):
        if lag >= n:
            break
        autocorr[lag-1] = np.mean(series[:-lag] * series[lag:]) / var
        
    return autocorr


def compute_spectral_entropy(series: np.ndarray) -> float:
    """
    Compute spectral entropy of the signal.
    
    High entropy = noise-like (random)
    Low entropy = structured (periodic or patterned)
    
    Returns:
        Normalized spectral entropy (0-1), where 1 = maximum randomness
    """
    if len(series) < 16:
        return 1.0
        
    # FFT
    fft = np.fft.rfft(series - np.mean(series))
    power = np.abs(fft)**2
    
    # Normalize to probability distribution
    power = power / (np.sum(power) + 1e-10)
    
    # Shannon entropy
    power = power[power > 1e-10]  # Remove zeros
    entropy = -np.sum(power * np.log2(power))
    
    # Normalize by maximum entropy (uniform distribution)
    max_entropy = np.log2(len(power))
    if max_entropy < 1e-10:
        return 1.0
        
    return entropy / max_entropy


def detect_periodicity(series: np.ndarray) -> Tuple[bool, float, int]:
    """
    Detect if series has significant periodic component.
    
    Returns:
        (is_periodic, period_strength, dominant_period)
    """
    if len(series) < 32:
        return False, 0.0, 0
        
    # FFT
    fft = np.fft.rfft(series - np.mean(series))
    power = np.abs(fft)**2
    
    # Find dominant frequency (excluding DC)
    if len(power) < 3:
        return False, 0.0, 0
        
    dominant_idx = np.argmax(power[1:]) + 1
    dominant_power = power[dominant_idx]
    total_power = np.sum(power[1:])
    
    if total_power < 1e-10:
        return False, 0.0, 0
        
    # Strength is ratio of dominant to total
    strength = dominant_power / total_power
    
    # Period in samples
    period = len(series) // dominant_idx if dominant_idx > 0 else 0
    
    # Significant if > 30% of power in one frequency
    is_periodic = strength > 0.3 and period > 2
    
    return is_periodic, strength, period


def verify_signal(trajectory_x: List[float], trajectory_y: List[float],
                  hurst_log: List[float], lyapunov_log: List[float],
                  diffusion_log: List[float], coherence_log: List[float],
                  baseline_hurst: float = 0.5,
                  window_size: int = 50) -> SignalVerification:
    """
    Comprehensive signal verification using multiple statistical tests.
    
    This function determines:
    1. Is the detected signal real or just noise?
    2. What type of signal is it?
    3. How confident are we?
    
    Args:
        trajectory_x, trajectory_y: Phase space trajectory
        hurst_log: History of Hurst exponent values
        lyapunov_log: History of Lyapunov exponent values
        diffusion_log: History of diffusion exponent (α) values
        coherence_log: History of coherence values
        baseline_hurst: Expected Hurst for random (typically 0.5)
        window_size: Size of window to analyze
        
    Returns:
        SignalVerification with classification and confidence
    """
    tests_passed = []
    tests_failed = []
    details = {}
    
    # Get recent data
    n = min(window_size, len(trajectory_x))
    if n < 20:
        return SignalVerification(
            is_verified=False, confidence=0.0, signal_class=SignalClass.NOISE,
            persistence_score=0.0, multi_metric_agreement=0.0,
            statistical_significance=1.0,
            tests_passed=[], tests_failed=["insufficient_data"],
            details={"n_samples": n}
        )
    
    x = np.array(trajectory_x[-n:])
    y = np.array(trajectory_y[-n:])
    dx = np.diff(x)
    dy = np.diff(y)
    
    # =========================================================================
    # TEST 1: Runs Test on displacements
    # =========================================================================
    z_runs_x, is_random_x = compute_runs_test(dx)
    z_runs_y, is_random_y = compute_runs_test(dy)
    
    details['runs_z_x'] = z_runs_x
    details['runs_z_y'] = z_runs_y
    
    if not is_random_x or not is_random_y:
        tests_passed.append("runs_test_nonrandom")
    else:
        tests_failed.append("runs_test_random")
    
    # =========================================================================
    # TEST 2: Autocorrelation analysis
    # =========================================================================
    autocorr = compute_autocorrelation(dx, max_lag=min(20, n//3))
    significant_lags = np.sum(np.abs(autocorr) > 2/np.sqrt(n))  # 95% threshold
    details['significant_autocorr_lags'] = int(significant_lags)
    details['max_autocorr'] = float(np.max(np.abs(autocorr))) if len(autocorr) > 0 else 0.0
    
    if significant_lags > 2:
        tests_passed.append("autocorrelation_significant")
    else:
        tests_failed.append("autocorrelation_insignificant")
    
    # =========================================================================
    # TEST 3: Spectral Entropy
    # =========================================================================
    spectral_ent = compute_spectral_entropy(dx)
    details['spectral_entropy'] = spectral_ent
    
    if spectral_ent < 0.8:  # Less than 80% of max = structured
        tests_passed.append("low_spectral_entropy")
    else:
        tests_failed.append("high_spectral_entropy")
    
    # =========================================================================
    # TEST 4: Periodicity Detection
    # =========================================================================
    is_periodic, period_strength, period = detect_periodicity(dx)
    details['is_periodic'] = is_periodic
    details['period_strength'] = period_strength
    details['dominant_period'] = period
    
    if is_periodic:
        tests_passed.append("periodicity_detected")
    
    # =========================================================================
    # TEST 5: Multi-metric Agreement
    # =========================================================================
    agreement_score = 0.0
    n_metrics = 0
    
    # Hurst: deviation from 0.5
    if len(hurst_log) >= 10:
        recent_hurst = np.mean(hurst_log[-10:])
        hurst_deviation = abs(recent_hurst - baseline_hurst)
        if hurst_deviation > 0.1:
            agreement_score += 1
            details['hurst_signal'] = True
        else:
            details['hurst_signal'] = False
        n_metrics += 1
        details['recent_hurst'] = recent_hurst
    
    # Lyapunov: deviation from 0
    if len(lyapunov_log) >= 10:
        recent_lyap = np.mean(lyapunov_log[-10:])
        if abs(recent_lyap) > 0.1:
            agreement_score += 1
            details['lyapunov_signal'] = True
        else:
            details['lyapunov_signal'] = False
        n_metrics += 1
        details['recent_lyapunov'] = recent_lyap
    
    # Diffusion: deviation from 1
    if len(diffusion_log) >= 10:
        recent_alpha = np.mean(diffusion_log[-10:])
        if abs(recent_alpha - 1.0) > 0.2:
            agreement_score += 1
            details['diffusion_signal'] = True
        else:
            details['diffusion_signal'] = False
        n_metrics += 1
        details['recent_alpha'] = recent_alpha
    
    # Coherence: significantly above baseline
    if len(coherence_log) >= 20:
        recent_coh = np.mean(coherence_log[-10:])
        baseline_coh = np.mean(coherence_log[:10])
        if recent_coh > baseline_coh * 1.5:
            agreement_score += 1
            details['coherence_signal'] = True
        else:
            details['coherence_signal'] = False
        n_metrics += 1
    
    multi_metric = agreement_score / max(n_metrics, 1)
    details['multi_metric_agreement'] = multi_metric
    
    if multi_metric >= 0.5:
        tests_passed.append("multi_metric_agreement")
    else:
        tests_failed.append("multi_metric_disagreement")
    
    # =========================================================================
    # TEST 6: Persistence Check
    # =========================================================================
    # Check if signal persists across multiple windows
    persistence = 0.0
    if len(hurst_log) >= 30:
        # Check consistency: std of Hurst should be low for persistent signal
        hurst_std = np.std(hurst_log[-30:])
        hurst_mean_dev = abs(np.mean(hurst_log[-30:]) - 0.5)
        if hurst_std < 0.1 and hurst_mean_dev > 0.05:
            persistence = 1.0 - hurst_std / 0.2
            tests_passed.append("signal_persistent")
        else:
            tests_failed.append("signal_transient")
    
    details['persistence_score'] = persistence
    
    # =========================================================================
    # CLASSIFICATION
    # =========================================================================
    signal_class = SignalClass.NOISE
    
    recent_hurst = details.get('recent_hurst', 0.5)
    recent_lyap = details.get('recent_lyapunov', 0.0)
    recent_alpha = details.get('recent_alpha', 1.0)
    
    # Classification logic based on metric signatures
    if len(tests_passed) < 2:
        signal_class = SignalClass.NOISE
    elif is_periodic:
        signal_class = SignalClass.PERIODIC
    elif recent_lyap < -0.1:
        signal_class = SignalClass.ATTRACTOR
    elif recent_lyap > 0.1:
        signal_class = SignalClass.CHAOTIC
    elif recent_alpha > 1.5:
        signal_class = SignalClass.DRIFT
    elif recent_hurst > 0.7 and multi_metric >= 0.75:
        signal_class = SignalClass.INFLUENCE
    elif recent_hurst > 0.6 or recent_hurst < 0.4:
        signal_class = SignalClass.ANOMALOUS
    else:
        signal_class = SignalClass.NOISE
    
    # =========================================================================
    # CONFIDENCE CALCULATION
    # =========================================================================
    base_confidence = len(tests_passed) / max(len(tests_passed) + len(tests_failed), 1)
    
    # Weight by multi-metric agreement
    confidence = base_confidence * 0.5 + multi_metric * 0.3 + persistence * 0.2
    
    # Boost if multiple strong indicators
    if signal_class in [SignalClass.INFLUENCE, SignalClass.ATTRACTOR]:
        confidence = min(1.0, confidence * 1.2)
    
    # Is it verified?
    is_verified = (
        confidence > 0.5 and 
        signal_class != SignalClass.NOISE and
        len(tests_passed) >= 2
    )
    
    # Statistical significance (simplified - based on runs test)
    stat_sig = 1.0 - min(1.0, (abs(z_runs_x) + abs(z_runs_y)) / 4)
    
    return SignalVerification(
        is_verified=is_verified,
        confidence=confidence,
        signal_class=signal_class,
        persistence_score=persistence,
        multi_metric_agreement=multi_metric,
        statistical_significance=stat_sig,
        tests_passed=tests_passed,
        tests_failed=tests_failed,
        details=details
    )


def classify_event(event: AnomalyEvent, verification: SignalVerification) -> str:
    """
    Generate a human-readable classification string for an event.
    
    Returns formatted string with classification and key evidence.
    """
    if not verification.is_verified:
        return f"[UNVERIFIED {event.event_type}] - likely noise fluctuation"
    
    class_str = verification.signal_class.value.upper()
    conf_str = f"{verification.confidence*100:.0f}%"
    
    # Key evidence
    evidence = []
    if verification.details.get('hurst_signal'):
        evidence.append(f"H={verification.details.get('recent_hurst', 0):.2f}")
    if verification.details.get('lyapunov_signal'):
        evidence.append(f"λ={verification.details.get('recent_lyapunov', 0):.3f}")
    if verification.details.get('diffusion_signal'):
        evidence.append(f"α={verification.details.get('recent_alpha', 1):.2f}")
    if verification.details.get('is_periodic'):
        evidence.append(f"period={verification.details.get('dominant_period', 0)}")
    
    evidence_str = ", ".join(evidence) if evidence else "weak evidence"
    
    return f"[{class_str}] {conf_str} confidence ({evidence_str})"


class HeliosAnomalyScope:
    """
    Attaches to the Helios Processor to visualize the 'Trajectory' of its thought process.
    Detects if the CORGI Junction is behaving randomly (Exploration) or 
    if it has locked onto a pattern (Emergent Behavior/Influence).
    
    The scope tracks the "Center of Mass" of the active batch to see where the 
    processor is focusing in the high-dimensional vector space.
    
    Detection Methods:
    1. Phase Space Trajectory - tracks movement in reduced 2D space
    2. MSD (Mean Squared Displacement) - measures diffusion vs. ballistic motion
    3. Coherence Metric - measures velocity variance (low = attractor lock)
    4. Hurst Exponent - measures memory/persistence in trajectory
    5. Diffusion Exponent - distinguishes random walk (α=1) from ballistic (α=2)
    6. Lyapunov Exponent - measures trajectory divergence (chaos vs. order)
    """
    
    def __init__(self,
                 history_len: int = 100,
                 influence_threshold: float = None,  # Now auto-calculated
                 msd_window: int = 20,
                 projection_method: str = 'mean_split'):
        """
        Args:
            history_len: Number of steps to keep in trajectory history
            influence_threshold: Volatility threshold (auto-calculated if None)
            msd_window: Window size for MSD trend analysis
            projection_method: How to reduce high-dim state to 2D
                - 'mean_split': Mean of first half vs second half of dimensions
                - 'pca': Principal Component Analysis (requires more compute)
                - 'random_projection': Random orthogonal projection
        """
        # Input validation
        self.history_len = validate_positive_int(history_len, 'history_len')
        self.msd_window = validate_positive_int(msd_window, 'msd_window')
        self.projection_method = validate_choice(
            projection_method, 'projection_method',
            ['mean_split', 'pca', 'random_projection']
        )

        if influence_threshold is not None:
            influence_threshold = validate_positive_float(
                influence_threshold, 'influence_threshold'
            )
        
        # Trajectory tracking (Phase Space)
        self.trajectory_x: List[float] = []
        self.trajectory_y: List[float] = []
        
        # Metrics history
        self.entropy_log: List[float] = []  # Velocity/movement history
        self.msd_log: List[float] = []      # Proper MSD values at different lags
        self.coherence_log: List[float] = [] # How aligned are movements
        self.hurst_log: List[float] = []     # Hurst exponent over time
        self.diffusion_exponent_log: List[float] = []  # α from MSD ~ t^α
        self.lyapunov_log: List[float] = []  # Lyapunov exponent over time
        
        # Event detection
        self.events: List[AnomalyEvent] = []
        self.step_count = 0
        
        # For random projection method
        self._projection_matrix: Optional[np.ndarray] = None
        
        # Baseline statistics (computed during "burn-in" period)
        self.baseline_velocity_std: Optional[float] = None
        self.baseline_msd_slope: Optional[float] = None
        self.baseline_hurst: Optional[float] = None
        self.burn_in_steps = 50
        
        # Warmup period - suppress detections during pipeline initialization
        # This prevents false positives from detector "waking up" artifacts
        # Set to 60 to account for: 50 (trajectory requirement) + 5 (averaging) + buffer
        self.warmup_steps = 60
        
        # Influence threshold (auto-calculated from baseline if not provided)
        self._user_threshold = influence_threshold
        self.influence_threshold = influence_threshold if influence_threshold else 0.01  # temporary
        
        # Epiplexity estimation (arXiv:2601.03220)
        # Tracks structural information emergence from random stream
        self.epiplexity_estimator: Optional['EpiplexityEstimator'] = None
        self.epiplexity_log: List[float] = []
        self.time_bounded_entropy_log: List[float] = []
        if EPIPLEXITY_AVAILABLE:
            self.epiplexity_estimator = EpiplexityEstimator(
                window_size=history_len,
                emergence_threshold=0.15,
                learning_rate_threshold=-0.001
            )

    @property
    def step(self) -> int:
        """Current step count (alias for step_count for API compatibility)."""
        return self.step_count
        
    def _project_to_2d(self, state_vector: np.ndarray) -> Tuple[float, float]:
        """
        Project high-dimensional state to 2D for phase space visualization.
        
        Args:
            state_vector: Flattened state [Dim] or [N, Dim]
            
        Returns:
            (x, y) coordinates in phase space
        """
        if state_vector.ndim > 1:
            state_vector = state_vector.flatten()
            
        dim = len(state_vector)
        
        if self.projection_method == 'mean_split':
            # Simple but effective: mean of each half
            x = np.mean(state_vector[:dim//2])
            y = np.mean(state_vector[dim//2:])
            
        elif self.projection_method == 'random_projection':
            # Random orthogonal projection (consistent across calls)
            if self._projection_matrix is None or self._projection_matrix.shape[0] != dim:
                # Create random orthogonal projection matrix
                random_matrix = np.random.randn(dim, 2)
                q, _ = np.linalg.qr(random_matrix)
                self._projection_matrix = q
            
            projected = state_vector @ self._projection_matrix
            x, y = projected[0], projected[1]
            
        elif self.projection_method == 'pca':
            # PCA requires history - fall back to mean_split if not enough data
            if len(self.trajectory_x) < 10:
                x = np.mean(state_vector[:dim//2])
                y = np.mean(state_vector[dim//2:])
            else:
                # Use accumulated states for PCA
                # This is expensive - consider caching
                x = np.mean(state_vector[:dim//2])
                y = np.mean(state_vector[dim//2:])
        else:
            raise ValueError(f"Unknown projection method: {self.projection_method}")
            
        return float(x), float(y)
    
    def update(self, active_batch: torch.Tensor) -> Dict[str, float]:
        """
        Update the scope with new state from HELIOS processor.

        Args:
            active_batch: [Batch, Logic_Size, Dim] or [Logic_Size, Dim]
                         The ions selected by CORGI junction

        Returns:
            Dict with current metrics: {x, y, velocity, msd, coherence, influence_detected}
        """
        # Input validation
        active_batch = validate_tensor(active_batch, 'active_batch', min_dims=2, max_dims=3)

        self.step_count += 1

        # Convert to numpy
        if isinstance(active_batch, torch.Tensor):
            state = active_batch.detach().cpu().numpy()
        else:
            state = np.array(active_batch)
            
        # Compute center of mass (mean across batch and logic dimensions)
        if state.ndim == 3:
            state = state.mean(axis=(0, 1))  # [Dim]
        elif state.ndim == 2:
            state = state.mean(axis=0)  # [Dim]
            
        # Project to 2D phase space
        x, y = self._project_to_2d(state)
        
        # Store trajectory
        self.trajectory_x.append(x)
        self.trajectory_y.append(y)
        
        # Trim to history length
        if len(self.trajectory_x) > self.history_len:
            self.trajectory_x.pop(0)
            self.trajectory_y.pop(0)
            
        # Calculate metrics
        metrics = self._compute_metrics()
        
        # Check for anomalies
        if self.step_count > self.burn_in_steps:
            self._check_for_anomalies(metrics)
            
        return metrics
    
    def _compute_metrics(self) -> Dict[str, float]:
        """Compute all tracking metrics from current trajectory."""
        metrics = {
            'x': self.trajectory_x[-1],
            'y': self.trajectory_y[-1],
            'step': self.step_count,
            'influence_detected': False
        }
        
        if len(self.trajectory_x) < 2:
            metrics.update({'velocity': 0, 'msd': 0, 'coherence': 0, 
                           'hurst': 0.5, 'diffusion_exponent': 1.0, 'lyapunov': 0.0})
            return metrics
            
        # Velocity (instantaneous movement)
        dx = self.trajectory_x[-1] - self.trajectory_x[-2]
        dy = self.trajectory_y[-1] - self.trajectory_y[-2]
        velocity = np.sqrt(dx**2 + dy**2)
        
        # Store directional components for Hurst analysis
        # (velocity magnitude may be constant in angle mode, but direction varies)
        self.entropy_log.append(dx)  # Track x-displacement, not velocity magnitude
        if len(self.entropy_log) > self.history_len:
            self.entropy_log.pop(0)
            
        metrics['velocity'] = velocity
        metrics['dx'] = dx
        metrics['dy'] = dy
        
        # Proper MSD calculation (distance from starting point of window)
        # This measures how far we've traveled, not just current position
        if len(self.trajectory_x) >= 10:
            _, msd_values, alpha = compute_msd_from_trajectory(
                self.trajectory_x, self.trajectory_y, 
                max_lag=min(50, len(self.trajectory_x) // 2)
            )
            # Use MSD at lag=10 as the representative value
            msd = msd_values[min(9, len(msd_values)-1)] if len(msd_values) > 0 else 0
            self.diffusion_exponent_log.append(alpha)
            if len(self.diffusion_exponent_log) > self.history_len:
                self.diffusion_exponent_log.pop(0)
            metrics['diffusion_exponent'] = alpha
        else:
            msd = self.trajectory_x[-1]**2 + self.trajectory_y[-1]**2
            metrics['diffusion_exponent'] = 1.0
            
        self.msd_log.append(msd)
        if len(self.msd_log) > self.history_len:
            self.msd_log.pop(0)
            
        metrics['msd'] = msd
        
        # Hurst Exponent (memory/persistence detection)
        # Calculate on directional displacements (dx), not velocity magnitude
        if len(self.entropy_log) >= 30:
            hurst = compute_hurst_exponent(np.array(self.entropy_log), max_lag=20)
            self.hurst_log.append(hurst)
            if len(self.hurst_log) > self.history_len:
                self.hurst_log.pop(0)
            metrics['hurst'] = hurst
        else:
            metrics['hurst'] = 0.5
        
        # Lyapunov Exponent (chaos/order detection)
        # Measures rate of trajectory divergence - key indicator from QFC theory
        # λ > 0: Chaotic (sensitive to initial conditions)
        # λ ≈ 0: Stable/neutral (random walk)
        # λ < 0: Convergent (attractor lock)
        if len(self.trajectory_x) >= 50:
            lyapunov = compute_lyapunov_exponent(
                self.trajectory_x, self.trajectory_y,
                min_neighbors=5, evolution_steps=10
            )
            self.lyapunov_log.append(lyapunov)
            if len(self.lyapunov_log) > self.history_len:
                self.lyapunov_log.pop(0)
            metrics['lyapunov'] = lyapunov
        else:
            metrics['lyapunov'] = 0.0
        
        # Coherence (inverse of displacement variance - high = stable orbit)
        # Uses directional displacements which vary even in angle mode
        if len(self.entropy_log) >= 5:
            recent_dx_std = np.std(self.entropy_log[-20:]) if len(self.entropy_log) >= 20 else np.std(self.entropy_log)
            coherence = 1.0 / (recent_dx_std + 1e-8)
            self.coherence_log.append(coherence)
            if len(self.coherence_log) > self.history_len:
                self.coherence_log.pop(0)
            metrics['coherence'] = coherence
        else:
            metrics['coherence'] = 0
            
        # Compute baseline during burn-in
        if self.step_count == self.burn_in_steps:
            self.baseline_velocity_std = np.std(self.entropy_log)
            self.baseline_hurst = np.mean(self.hurst_log) if self.hurst_log else 0.5
            
            if len(self.msd_log) > 10:
                # Fit linear slope to log-log MSD (should be ~1 for random walk)
                log_time = np.log(np.arange(1, len(self.msd_log) + 1))
                log_msd = np.log(np.array(self.msd_log) + 1e-8)
                self.baseline_msd_slope = np.polyfit(log_time, log_msd, 1)[0]
            
            # Auto-calculate threshold: 3σ below baseline std
            if self._user_threshold is None and self.baseline_velocity_std:
                self.influence_threshold = self.baseline_velocity_std * 0.3
                
        # Influence detection (multiple criteria)
        if self.baseline_velocity_std is not None:
            recent_std = np.std(self.entropy_log[-20:]) if len(self.entropy_log) >= 20 else np.std(self.entropy_log)
            
            # Check multiple conditions:
            # 1. Low velocity variance (attractor lock)
            low_variance = recent_std < self.influence_threshold
            
            # 2. High Hurst exponent (trending behavior)
            high_hurst = metrics.get('hurst', 0.5) > 0.65
            
            # 3. High diffusion exponent (ballistic motion)
            ballistic = metrics.get('diffusion_exponent', 1.0) > 1.5
            
            metrics['influence_detected'] = low_variance or high_hurst or ballistic
            metrics['influence_reason'] = []
            if low_variance:
                metrics['influence_reason'].append('low_variance')
            if high_hurst:
                metrics['influence_reason'].append('high_hurst')
            if ballistic:
                metrics['influence_reason'].append('ballistic')
        
        # Epiplexity estimation (arXiv:2601.03220)
        # Tracks structural information emerging from random stream
        # S_T = epiplexity (structural info), H_T = time-bounded entropy (randomness)
        if self.epiplexity_estimator is not None:
            # Feed the displacement magnitude to epiplexity estimator
            value = velocity if velocity > 0 else 0.0
            trajectory_point = (metrics.get('x', 0), metrics.get('y', 0))
            epi_metrics = self.epiplexity_estimator.update(value, trajectory_point)
            
            metrics['epiplexity'] = epi_metrics.epiplexity
            metrics['time_bounded_entropy'] = epi_metrics.time_bounded_entropy
            metrics['compression_ratio'] = epi_metrics.compression_ratio
            metrics['structural_emergence'] = epi_metrics.structural_emergence
            metrics['emergence_confidence'] = epi_metrics.emergence_confidence
            
            self.epiplexity_log.append(epi_metrics.epiplexity)
            self.time_bounded_entropy_log.append(epi_metrics.time_bounded_entropy)
            if len(self.epiplexity_log) > self.history_len:
                self.epiplexity_log.pop(0)
                self.time_bounded_entropy_log.pop(0)
            
            # Add structural emergence to influence detection
            if epi_metrics.structural_emergence:
                metrics['influence_detected'] = True
                metrics['influence_reason'].append('structural_emergence')
                
        return metrics
    
    def _check_for_anomalies(self, metrics: Dict[str, float]):
        """Check current state for anomaly events."""
        
        # Attractor Lock Detection
        # If velocity variance drops significantly below baseline
        # Skip during warmup period to avoid pipeline boundary artifacts
        if self.step_count > self.warmup_steps and self.baseline_velocity_std is not None and len(self.entropy_log) >= 20:
            recent_std = np.std(self.entropy_log[-20:])
            if recent_std < self.baseline_velocity_std * 0.3:
                event = AnomalyEvent(
                    step=self.step_count,
                    event_type='attractor_lock',
                    confidence=1 - (recent_std / self.baseline_velocity_std),
                    metrics={'velocity_std': recent_std, 'baseline': self.baseline_velocity_std},
                    description=f"Trajectory locked into attractor (velocity std {recent_std:.4f} << baseline {self.baseline_velocity_std:.4f})"
                )
                self.events.append(event)
        
        # Ballistic Motion Detection (MSD growing faster than linear)
        # Skip during warmup period to avoid pipeline boundary artifacts
        if self.step_count > self.warmup_steps and len(self.diffusion_exponent_log) >= 5:
            recent_alpha = np.mean(self.diffusion_exponent_log[-5:])
            if recent_alpha > 1.5:  # Significantly above diffusive (α=1)
                event = AnomalyEvent(
                    step=self.step_count,
                    event_type='ballistic_motion',
                    confidence=min(1.0, (recent_alpha - 1) / 1.0),
                    metrics={'diffusion_exponent': recent_alpha},
                    description=f"Ballistic motion detected (α={recent_alpha:.2f} > 1.5) - directed drift"
                )
                self.events.append(event)
        
        # Hurst Exponent Anomaly (trending behavior)
        # Skip during warmup period to avoid pipeline boundary artifacts
        if self.step_count > self.warmup_steps and len(self.hurst_log) >= 5:
            recent_hurst = np.mean(self.hurst_log[-5:])
            if recent_hurst > 0.65:  # Significantly above random (H=0.5)
                event = AnomalyEvent(
                    step=self.step_count,
                    event_type='trending_behavior',
                    confidence=min(1.0, (recent_hurst - 0.5) / 0.3),
                    metrics={'hurst': recent_hurst},
                    description=f"Trending behavior detected (H={recent_hurst:.3f} > 0.65) - memory in sequence"
                )
                self.events.append(event)
                
        # Coherence Spike Detection
        # Skip during warmup period to avoid pipeline boundary artifacts
        if self.step_count > self.warmup_steps and len(self.coherence_log) >= 10:
            recent_coherence = np.mean(self.coherence_log[-5:])
            baseline_coherence = np.mean(self.coherence_log[:10]) if len(self.coherence_log) > 10 else recent_coherence
            if recent_coherence > baseline_coherence * 3:
                event = AnomalyEvent(
                    step=self.step_count,
                    event_type='coherence_spike',
                    confidence=min(1.0, recent_coherence / (baseline_coherence * 5)),
                    metrics={'coherence': recent_coherence},
                    description=f"Coherence spike detected - system synchronizing"
                )
                self.events.append(event)
                
        # Drift Detection (center of mass moving away from origin)
        # Skip during warmup period to avoid pipeline boundary artifacts
        if self.step_count > self.warmup_steps and len(self.msd_log) >= 30:
            msd_trend = np.mean(self.msd_log[-10:]) - np.mean(self.msd_log[-30:-20])
            if msd_trend > np.std(self.msd_log) * 2:
                event = AnomalyEvent(
                    step=self.step_count,
                    event_type='drift',
                    confidence=min(1.0, msd_trend / (np.std(self.msd_log) * 3)),
                    metrics={'msd_trend': msd_trend},
                    description=f"Directional drift detected - external field influence possible"
                )
                self.events.append(event)
        
        # Lyapunov Exponent Anomaly Detection
        # λ > 0.1: Chaotic behavior (trajectories diverging)
        # λ < -0.1: Attractor lock (trajectories converging)
        # Skip during warmup period to avoid pipeline boundary artifacts
        if self.step_count > self.warmup_steps and len(self.lyapunov_log) >= 5:
            recent_lyapunov = np.mean(self.lyapunov_log[-5:])
            
            if recent_lyapunov > 0.1:  # Positive = chaotic/sensitive
                event = AnomalyEvent(
                    step=self.step_count,
                    event_type='chaotic_sensitivity',
                    confidence=min(1.0, recent_lyapunov / 0.3),
                    metrics={'lyapunov': recent_lyapunov},
                    description=f"Chaotic sensitivity detected (λ={recent_lyapunov:.3f} > 0.1) - emerging structure"
                )
                self.events.append(event)
                
            elif recent_lyapunov < -0.1:  # Negative = convergent
                event = AnomalyEvent(
                    step=self.step_count,
                    event_type='convergent_attractor',
                    confidence=min(1.0, abs(recent_lyapunov) / 0.3),
                    metrics={'lyapunov': recent_lyapunov},
                    description=f"Convergent attractor detected (λ={recent_lyapunov:.3f} < -0.1) - locked orbit"
                )
                self.events.append(event)
        
        # Structural Emergence Detection (arXiv:2601.03220)
        # High epiplexity + decreasing entropy = structure emerging from noise
        # This is the theoretical signature of consciousness influence
        if self.step_count > self.warmup_steps and self.epiplexity_estimator is not None:
            if len(self.epiplexity_log) >= 10:
                recent_epiplexity = np.mean(self.epiplexity_log[-5:])
                baseline_epiplexity = np.mean(self.epiplexity_log[:10]) if len(self.epiplexity_log) > 10 else recent_epiplexity
                recent_entropy = np.mean(self.time_bounded_entropy_log[-5:])
                baseline_entropy = np.mean(self.time_bounded_entropy_log[:10]) if len(self.time_bounded_entropy_log) > 10 else recent_entropy
                
                # Structural emergence: epiplexity rising while entropy dropping
                epiplexity_rising = recent_epiplexity > baseline_epiplexity * 1.3
                entropy_dropping = recent_entropy < baseline_entropy * 0.7
                
                if epiplexity_rising and entropy_dropping:
                    # Avoid division by zero
                    if baseline_epiplexity > 0 and baseline_entropy > 0:
                        confidence = min(1.0, (recent_epiplexity / baseline_epiplexity - 1) + (1 - recent_entropy / baseline_entropy))
                    else:
                        confidence = 0.5
                    event = AnomalyEvent(
                        step=self.step_count,
                        event_type='structural_emergence',
                        confidence=confidence,
                        metrics={
                            'epiplexity': recent_epiplexity,
                            'baseline_epiplexity': baseline_epiplexity,
                            'time_bounded_entropy': recent_entropy,
                            'baseline_entropy': baseline_entropy
                        },
                        description=f"Structural emergence detected (S_T={recent_epiplexity:.3f}↑, H_T={recent_entropy:.3f}↓) - arXiv:2601.03220 signature"
                    )
                    self.events.append(event)
    
    def detect_influence(self, threshold: Optional[float] = None) -> bool:
        """
        Returns True if the system has collapsed into an ordered state.
        
        This is the primary detection method for external consciousness influence.
        Low velocity variance = Stable Orbit (Attractor Detected)
        
        Args:
            threshold: Override default influence_threshold
            
        Returns:
            True if influence/emergence detected
        """
        if len(self.entropy_log) < 10:
            return False
            
        threshold = threshold or self.influence_threshold
        
        # Low velocity variance = Stable Orbit (Attractor Detected)
        recent_volatility = np.std(self.entropy_log[-20:]) if len(self.entropy_log) >= 20 else np.std(self.entropy_log)
        return recent_volatility < threshold
    
    def get_trajectory(self) -> Tuple[List[float], List[float]]:
        """Get current trajectory for visualization."""
        return self.trajectory_x.copy(), self.trajectory_y.copy()
    
    def get_events(self, last_n: Optional[int] = None) -> List[AnomalyEvent]:
        """Get detected anomaly events."""
        if last_n is None:
            return self.events.copy()
        return self.events[-last_n:]
    
    def verify_current_signal(self, window_size: int = 50) -> SignalVerification:
        """
        Verify if the current trajectory shows a real signal.
        
        This runs comprehensive statistical tests to determine:
        1. Is the detected pattern real or just noise?
        2. What type of signal is it?
        3. How confident should we be?
        
        Args:
            window_size: Number of recent steps to analyze
            
        Returns:
            SignalVerification object with classification and confidence
        """
        return verify_signal(
            trajectory_x=self.trajectory_x,
            trajectory_y=self.trajectory_y,
            hurst_log=self.hurst_log,
            lyapunov_log=self.lyapunov_log,
            diffusion_log=self.diffusion_exponent_log,
            coherence_log=self.coherence_log,
            baseline_hurst=self.baseline_hurst or 0.5,
            window_size=window_size
        )
    
    def verify_event(self, event: AnomalyEvent, window_size: int = 50) -> AnomalyEvent:
        """
        Verify a specific detected event and attach verification results.
        
        Args:
            event: The event to verify
            window_size: Analysis window size
            
        Returns:
            The same event with verification attached
        """
        verification = self.verify_current_signal(window_size)
        event.verification = verification
        return event
    
    def get_verified_events(self, min_confidence: float = 0.5) -> List[AnomalyEvent]:
        """
        Get only events that pass verification with minimum confidence.
        
        Args:
            min_confidence: Minimum confidence threshold (0-1)
            
        Returns:
            List of verified events
        """
        verified = []
        for event in self.events:
            if event.verification is None:
                # Run verification if not already done
                self.verify_event(event)
            if event.verification and event.verification.is_verified:
                if event.verification.confidence >= min_confidence:
                    verified.append(event)
        return verified
    
    def get_signal_classification(self) -> Dict:
        """
        Get comprehensive signal classification for current state.
        
        Returns dict with:
        - verification: SignalVerification object
        - classification_str: Human-readable classification
        - is_signal: Boolean if real signal detected
        - signal_type: SignalClass enum value
        - evidence: Dict of supporting evidence
        """
        verification = self.verify_current_signal()
        
        return {
            'verification': verification,
            'classification_str': str(verification),
            'is_signal': verification.is_verified,
            'signal_type': verification.signal_class.value,
            'confidence': verification.confidence,
            'evidence': {
                'tests_passed': verification.tests_passed,
                'tests_failed': verification.tests_failed,
                'multi_metric_agreement': verification.multi_metric_agreement,
                'persistence': verification.persistence_score,
                'details': verification.details
            }
        }
    
    def get_summary(self) -> Dict:
        """Get summary statistics for the current session."""
        return {
            'total_steps': self.step_count,
            'events_detected': len(self.events),
            'event_types': {et: sum(1 for e in self.events if e.event_type == et) 
                          for et in ['emergence', 'attractor_lock', 'drift', 'coherence_spike', 
                                    'ballistic_motion', 'trending_behavior', 
                                    'chaotic_sensitivity', 'convergent_attractor']},
            'current_state': {
                'x': self.trajectory_x[-1] if self.trajectory_x else 0,
                'y': self.trajectory_y[-1] if self.trajectory_y else 0,
                'velocity': self.entropy_log[-1] if self.entropy_log else 0,
                'msd': self.msd_log[-1] if self.msd_log else 0,
                'hurst': self.hurst_log[-1] if self.hurst_log else 0.5,
                'diffusion_exponent': self.diffusion_exponent_log[-1] if self.diffusion_exponent_log else 1.0,
                'lyapunov': self.lyapunov_log[-1] if self.lyapunov_log else 0.0,
            },
            'baseline': {
                'velocity_std': self.baseline_velocity_std,
                'msd_slope': self.baseline_msd_slope,
                'hurst': self.baseline_hurst,
            },
            'interpretation': self._get_interpretation()
        }
    
    def _get_interpretation(self) -> str:
        """Get human-readable interpretation of current state."""
        if not self.diffusion_exponent_log:
            return "Collecting baseline data..."
            
        alpha = np.mean(self.diffusion_exponent_log[-10:]) if len(self.diffusion_exponent_log) >= 10 else 1.0
        hurst = np.mean(self.hurst_log[-10:]) if len(self.hurst_log) >= 10 else 0.5
        lyapunov = np.mean(self.lyapunov_log[-10:]) if len(self.lyapunov_log) >= 10 else 0.0
        
        if alpha < 0.8:
            motion = "SUB-DIFFUSIVE (constrained/confined motion)"
        elif alpha < 1.2:
            motion = "DIFFUSIVE (normal random walk)"
        elif alpha < 1.8:
            motion = "SUPER-DIFFUSIVE (directed drift)"
        else:
            motion = "BALLISTIC (strong directed motion)"
            
        if hurst < 0.4:
            memory = "ANTI-PERSISTENT (mean-reverting)"
        elif hurst < 0.6:
            memory = "NO MEMORY (truly random)"
        elif hurst < 0.75:
            memory = "PERSISTENT (trending)"
        else:
            memory = "STRONGLY PERSISTENT (strong memory/influence)"
        
        if lyapunov > 0.1:
            stability = "CHAOTIC (sensitive dependence)"
        elif lyapunov < -0.1:
            stability = "CONVERGENT (attractor lock)"
        else:
            stability = "STABLE (neutral)"
            
        return f"{motion} | {memory} | {stability}"
    
    def reset(self):
        """Reset the scope for a new session."""
        self.trajectory_x.clear()
        self.trajectory_y.clear()
        self.entropy_log.clear()
        self.msd_log.clear()
        self.coherence_log.clear()
        self.hurst_log.clear()
        self.diffusion_exponent_log.clear()
        self.lyapunov_log.clear()
        self.events.clear()
        self.step_count = 0
        self.baseline_velocity_std = None
        self.baseline_msd_slope = None
        self.baseline_hurst = None


class QRNGStreamScope(HeliosAnomalyScope):
    """
    Specialized scope for analyzing raw QRNG streams.
    
    Uses Time-Delay Embedding (Takens' Theorem) to unfold 
    a 1D number stream into phase space.
    
    The key insight from QFC (Quantum-interacting Fundamental Consciousness):
    - Pure QRNG should produce diffusive random walk (R ~ sqrt(t))
    - Consciousness influence creates ballistic motion (R ~ t)
    - Or collapses trajectory into structured attractors
    """
    
    def __init__(self,
                 embedding_dim: int = 2,
                 embedding_delay: int = 1,
                 walk_mode: str = 'angle',  # 'angle', 'xy_independent', or 'takens'
                 **kwargs):
        """
        Args:
            embedding_dim: Dimension of phase space reconstruction (2 or 3)
            embedding_delay: Time delay for embedding (tau)
            walk_mode: How to construct the 2D random walk:
                - 'angle': Use QRNG value as angle (0-2π), fixed step size
                - 'xy_independent': Use pairs of values for x,y (requires 2x samples)
                - 'takens': Pure time-delay embedding (may show diagonal bias)
            **kwargs: Passed to HeliosAnomalyScope
        """
        super().__init__(**kwargs)

        # Input validation
        self.embedding_dim = validate_positive_int(embedding_dim, 'embedding_dim')
        if self.embedding_dim < 2 or self.embedding_dim > 3:
            raise ValidationError("embedding_dim must be 2 or 3")

        self.embedding_delay = validate_positive_int(embedding_delay, 'embedding_delay')
        self.walk_mode = validate_choice(
            walk_mode, 'walk_mode',
            ['angle', 'xy_independent', 'takens']
        )

        self.raw_stream: List[float] = []
        
        # Accumulator for random walk position
        self._walk_position = np.zeros(2)
        self._step_size = 0.1  # Scale factor for random walk steps
        self._pending_value: Optional[float] = None  # For xy_independent mode
        
    def update_from_stream(self, value: float) -> Dict[str, float]:
        """
        Update from a single QRNG value.
        
        Args:
            value: Single random number from QRNG (should be in [0, 1))
            
        Returns:
            Metrics dict
        """
        self.raw_stream.append(value)
        self.step_count += 1
        
        # Handle different walk modes
        if self.walk_mode == 'angle':
            # Convert value to angle: full [0,1) → [0, 2π)
            angle = value * 2 * np.pi
            dx = np.cos(angle) * self._step_size
            dy = np.sin(angle) * self._step_size
            
        elif self.walk_mode == 'xy_independent':
            # Use pairs of values for independent x and y
            if self._pending_value is None:
                self._pending_value = value
                return {'x': 0, 'y': 0, 'step': self.step_count, 'waiting_for_history': True}
            else:
                dx = (self._pending_value - 0.5) * self._step_size * 2
                dy = (value - 0.5) * self._step_size * 2
                self._pending_value = None
                
        else:  # 'takens' mode - original behavior
            # Need enough history for embedding
            min_history = self.embedding_delay * (self.embedding_dim - 1) + 1
            if len(self.raw_stream) < min_history:
                return {'x': 0, 'y': 0, 'step': self.step_count, 'waiting_for_history': True}
                
            # Create embedded state vector using time-delay embedding
            embedded = []
            for d in range(self.embedding_dim):
                idx = -(1 + d * self.embedding_delay)
                embedded.append(self.raw_stream[idx])
            
            dx = (embedded[0] - 0.5) * self._step_size
            dy = (embedded[1] - 0.5) * self._step_size
        
        # Update walk position (accumulate movement)
        self._walk_position[0] += dx
        self._walk_position[1] += dy
        
        x, y = self._walk_position[0], self._walk_position[1]
        
        # Store trajectory
        self.trajectory_x.append(float(x))
        self.trajectory_y.append(float(y))
        
        # Trim to history length
        if len(self.trajectory_x) > self.history_len:
            self.trajectory_x.pop(0)
            self.trajectory_y.pop(0)
            
        # Calculate metrics using parent's method
        metrics = self._compute_metrics()
        
        # Check for anomalies after burn-in
        if self.step_count > self.burn_in_steps:
            self._check_for_anomalies(metrics)
            
        return metrics
    
    def update_batch(self, values: List[float]) -> List[Dict[str, float]]:
        """Update from a batch of QRNG values."""
        return [self.update_from_stream(v) for v in values]


# Convenience function for quick testing
def create_scope_for_helios(helios_processor, **kwargs) -> HeliosAnomalyScope:
    """
    Create an appropriately configured scope for a HELIOS processor.
    
    Args:
        helios_processor: HeliosProcessor instance from LOCAL_Ai
        **kwargs: Override default scope parameters
        
    Returns:
        Configured HeliosAnomalyScope
    """
    # Infer good defaults from processor config
    dim = getattr(helios_processor, 'dim', 256)
    
    defaults = {
        'history_len': 100,
        'influence_threshold': 0.01,
        'msd_window': 20,
        'projection_method': 'mean_split' if dim > 64 else 'random_projection'
    }
    defaults.update(kwargs)
    
    return HeliosAnomalyScope(**defaults)


if __name__ == "__main__":
    # Quick test with synthetic data
    print("Testing HeliosAnomalyScope with synthetic data...")
    
    scope = HeliosAnomalyScope(history_len=50)
    
    # Simulate HELIOS-like state evolution
    # Phase 1: Random exploration (steps 0-100)
    # Phase 2: Attractor lock (steps 100-200)
    
    for step in range(200):
        if step < 100:
            # Random walk
            state = torch.randn(1, 16, 64)
        else:
            # Structured orbit (attractor)
            t = step * 0.1
            base = torch.zeros(1, 16, 64)
            base[:, :, :32] = np.sin(t) * 0.5
            base[:, :, 32:] = np.cos(t) * 0.5
            state = base + torch.randn_like(base) * 0.1
            
        metrics = scope.update(state)
        
        if step % 50 == 0:
            print(f"Step {step}: x={metrics['x']:.3f}, y={metrics['y']:.3f}, "
                  f"velocity={metrics['velocity']:.4f}, influence={metrics['influence_detected']}")
            
    print(f"\nSummary: {scope.get_summary()}")
    print(f"Events: {[(e.step, e.event_type, e.confidence) for e in scope.get_events()]}")
