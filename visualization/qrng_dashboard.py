#!/usr/bin/env python3
"""
QRNG Comprehensive Dashboard
============================

A unified dashboard integrating all analysis capabilities:

1. Longitudinal Tracking - Compare metrics across sessions
2. Real-time Anomaly Alerts - Sliding window detection  
3. NIST SP 800-22 Test Suite - Full randomness validation
4. Correlation Dimension Tracking - Chaos analysis
5. Consciousness Metrics - H_mode, PR, coherence, criticality
6. Multi-source Comparison - QRNG vs CSPRNG vs PRNG

Usage:
    python qrng_dashboard.py                    # Analyze latest stream
    python qrng_dashboard.py --all              # Analyze all streams (longitudinal)
    python qrng_dashboard.py --live             # Real-time monitoring mode
    python qrng_dashboard.py --compare          # Multi-source comparison
"""

import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from dataclasses import dataclass, field, asdict
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing
import argparse
import secrets
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# CPU count for parallel processing
N_WORKERS = min(multiprocessing.cpu_count(), 64)

# Import project modules
from helios_anomaly_scope import (
    compute_hurst_exponent,
    compute_lyapunov_exponent,
    compute_msd_from_trajectory,
    compute_runs_test,
    compute_spectral_entropy,
    compute_autocorrelation,
    detect_periodicity
)

# NIST tests
try:
    from tests.nist_tests import NIST800_22, NISTTestResult
    NIST_AVAILABLE = True
except ImportError:
    NIST_AVAILABLE = False
    print("⚠ NIST tests not available")

# Chaos detector
try:
    from chaos_detector import ChaosDetector
    CHAOS_AVAILABLE = True
    _chaos_detector = ChaosDetector()
except ImportError:
    CHAOS_AVAILABLE = False
    _chaos_detector = None

# Consciousness metrics
try:
    from consciousness_metrics import ConsciousnessMetrics
    CONSCIOUSNESS_AVAILABLE = True
except ImportError:
    CONSCIOUSNESS_AVAILABLE = False


@dataclass
class SessionMetrics:
    """Metrics for a single QRNG session."""
    timestamp: str
    n_samples: int
    source: str
    
    # Basic statistics
    mean: float
    std: float
    min_val: float
    max_val: float
    
    # Bit-level
    bit_bias: float
    worst_bit_bias: float
    min_entropy_ratio: float
    
    # Hurst analysis (centered)
    hurst: float
    hurst_ci_low: float = 0.0
    hurst_ci_high: float = 0.0
    
    # Other metrics
    spectral_entropy: float = 0.0
    runs_z: float = 0.0
    runs_random: bool = True
    lyapunov: float = 0.0
    msd_alpha: float = 0.0
    
    # Chaos metrics
    correlation_dimension: float = 0.0
    n_phase_transitions: int = 0
    
    # Consciousness metrics
    h_mode: float = 0.0
    participation_ratio: float = 0.0
    phase_coherence: float = 0.0
    criticality_index: float = 0.0
    consciousness_state: str = "UNKNOWN"
    
    # NIST results
    nist_tests_passed: int = 0
    nist_tests_total: int = 0
    nist_details: Dict = field(default_factory=dict)
    
    # Anomaly detection
    anomalies_detected: List[str] = field(default_factory=list)
    change_points: List[int] = field(default_factory=list)


@dataclass 
class AnomalyAlert:
    """Real-time anomaly alert."""
    timestamp: str
    alert_type: str
    severity: str  # LOW, MEDIUM, HIGH, CRITICAL
    metric: str
    value: float
    threshold: float
    message: str


class QRNGDashboard:
    """Comprehensive QRNG analysis dashboard."""
    
    def __init__(self, streams_dir: str = "qrng_streams", 
                 results_dir: str = "qrng_dashboard_results"):
        self.streams_dir = Path(streams_dir)
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(exist_ok=True)
        
        self.sessions: List[SessionMetrics] = []
        self.alerts: List[AnomalyAlert] = []
        
        # Anomaly thresholds
        self.thresholds = {
            'bit_bias': 0.01,           # Max deviation from 0.5
            'min_entropy_ratio': 0.90,  # Min acceptable ratio
            'hurst_deviation': 0.15,    # Max deviation from expected
            'spectral_entropy': 0.85,   # Min acceptable
            'runs_z': 2.576,            # 99% confidence
            'autocorr_lag1': 0.1,       # Max lag-1 autocorrelation
        }
    
    def load_stream(self, filepath: Path) -> Tuple[np.ndarray, dict]:
        """Load QRNG stream file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        
        if 'raw_integers' in data:
            raw = np.array(data['raw_integers'], dtype=np.uint32)
            values = raw.astype(np.float64) / (2**32)
            data['_format'] = 'uint32'
        elif 'floats' in data:
            values = np.array(data['floats'])
            data['_format'] = 'float64'
        else:
            raise ValueError("Unknown data format")
        
        return values, data
    
    def compute_bit_stats(self, values: np.ndarray) -> Dict:
        """Compute bit-level statistics."""
        uint_vals = (values * (2**32)).astype(np.uint64)
        n_samples = len(uint_vals)
        
        total_ones = 0
        total_bits = 0
        bit_biases = []
        
        for bit_pos in range(32):
            bit_vals = (uint_vals >> bit_pos) & 1
            p1 = np.mean(bit_vals)
            bit_biases.append(p1)
            total_ones += np.sum(bit_vals)
            total_bits += n_samples
        
        overall_p1 = total_ones / total_bits
        worst_bias = max(abs(p - 0.5) for p in bit_biases)
        
        return {
            'overall_p1': overall_p1,
            'bias_from_half': abs(overall_p1 - 0.5),
            'worst_bit_bias': worst_bias,
        }
    
    def compute_min_entropy(self, values: np.ndarray, n_bins: int = 256) -> float:
        """Compute min-entropy ratio."""
        hist, _ = np.histogram(values, bins=n_bins, range=(0, 1))
        probs = hist / np.sum(hist)
        p_max = np.max(probs)
        
        if p_max > 0:
            h_min = -np.log2(p_max)
            h_max = np.log2(n_bins)
            return h_min / h_max
        return 0.0
    
    def detect_change_points(self, values: np.ndarray, window: int = 100) -> List[int]:
        """Simple change-point detection."""
        if len(values) < window * 2:
            return []
        
        change_points = []
        threshold = 3.0
        
        for i in range(window, len(values) - window):
            left = values[i-window:i]
            right = values[i:i+window]
            
            mean_diff = abs(np.mean(left) - np.mean(right))
            pooled_std = np.sqrt((np.var(left) + np.var(right)) / 2)
            
            if pooled_std > 0:
                z = mean_diff / (pooled_std / np.sqrt(window))
                if z > threshold:
                    change_points.append(i)
        
        # Cluster nearby points
        if len(change_points) > 1:
            clustered = [change_points[0]]
            for cp in change_points[1:]:
                if cp - clustered[-1] > window:
                    clustered.append(cp)
            return clustered
        
        return change_points
    
    def check_anomalies(self, metrics: SessionMetrics) -> List[AnomalyAlert]:
        """Check for anomalies and generate alerts."""
        alerts = []
        ts = datetime.now().isoformat()
        
        # Bit bias check
        if metrics.bit_bias > self.thresholds['bit_bias']:
            alerts.append(AnomalyAlert(
                timestamp=ts,
                alert_type="BIT_BIAS",
                severity="MEDIUM" if metrics.bit_bias < 0.02 else "HIGH",
                metric="bit_bias",
                value=metrics.bit_bias,
                threshold=self.thresholds['bit_bias'],
                message=f"Bit bias {metrics.bit_bias:.4f} exceeds threshold"
            ))
        
        # Min entropy check
        if metrics.min_entropy_ratio < self.thresholds['min_entropy_ratio']:
            alerts.append(AnomalyAlert(
                timestamp=ts,
                alert_type="LOW_ENTROPY",
                severity="HIGH",
                metric="min_entropy_ratio",
                value=metrics.min_entropy_ratio,
                threshold=self.thresholds['min_entropy_ratio'],
                message=f"Min entropy ratio {metrics.min_entropy_ratio:.4f} below threshold"
            ))
        
        # Runs test
        if abs(metrics.runs_z) > self.thresholds['runs_z']:
            alerts.append(AnomalyAlert(
                timestamp=ts,
                alert_type="RUNS_ANOMALY",
                severity="HIGH",
                metric="runs_z",
                value=metrics.runs_z,
                threshold=self.thresholds['runs_z'],
                message=f"Runs test z={metrics.runs_z:.2f} indicates non-random structure"
            ))
        
        # Spectral entropy
        if metrics.spectral_entropy < self.thresholds['spectral_entropy']:
            alerts.append(AnomalyAlert(
                timestamp=ts,
                alert_type="SPECTRAL_ANOMALY",
                severity="MEDIUM",
                metric="spectral_entropy",
                value=metrics.spectral_entropy,
                threshold=self.thresholds['spectral_entropy'],
                message=f"Spectral entropy {metrics.spectral_entropy:.4f} indicates structure"
            ))
        
        # NIST failures
        if metrics.nist_tests_total > 0:
            fail_rate = 1 - (metrics.nist_tests_passed / metrics.nist_tests_total)
            if fail_rate > 0.1:  # More than 10% failures
                alerts.append(AnomalyAlert(
                    timestamp=ts,
                    alert_type="NIST_FAILURES",
                    severity="HIGH" if fail_rate > 0.2 else "MEDIUM",
                    metric="nist_fail_rate",
                    value=fail_rate,
                    threshold=0.1,
                    message=f"NIST tests: {metrics.nist_tests_passed}/{metrics.nist_tests_total} passed"
                ))
        
        return alerts
    
    def analyze_stream(self, filepath: Path, run_nist: bool = True,
                       run_chaos: bool = True, run_consciousness: bool = True) -> SessionMetrics:
        """Full analysis of a single QRNG stream."""
        values, metadata = self.load_stream(filepath)
        n = len(values)
        
        # Extract timestamp from filename
        ts = filepath.stem.replace("qrng_stream_", "")
        
        # Basic stats
        mean = float(np.mean(values))
        std = float(np.std(values))
        min_val = float(np.min(values))
        max_val = float(np.max(values))
        
        # Bit-level
        bit_stats = self.compute_bit_stats(values)
        min_entropy_ratio = self.compute_min_entropy(values)
        
        # Center for Hurst
        centered = values - 0.5
        hurst = compute_hurst_exponent(centered)
        
        # Other metrics
        spectral_entropy = compute_spectral_entropy(centered)
        runs_z, runs_random = compute_runs_test(values)
        
        # Build trajectory for MSD/Lyapunov
        angles = values * 2 * np.pi
        dx = np.cos(angles) * 0.1
        dy = np.sin(angles) * 0.1
        x = np.cumsum(dx).tolist()
        y = np.cumsum(dy).tolist()
        
        _, _, msd_alpha = compute_msd_from_trajectory(x, y)
        lyapunov = compute_lyapunov_exponent(x, y)
        
        # Change points
        change_points = self.detect_change_points(values)
        
        # Initialize metrics
        metrics = SessionMetrics(
            timestamp=ts,
            n_samples=n,
            source=metadata.get('source', 'unknown'),
            mean=mean,
            std=std,
            min_val=min_val,
            max_val=max_val,
            bit_bias=bit_stats['bias_from_half'],
            worst_bit_bias=bit_stats['worst_bit_bias'],
            min_entropy_ratio=min_entropy_ratio,
            hurst=hurst,
            spectral_entropy=spectral_entropy,
            runs_z=runs_z,
            runs_random=runs_random,
            lyapunov=lyapunov,
            msd_alpha=msd_alpha,
            change_points=change_points,
        )
        
        # Chaos metrics
        if run_chaos and CHAOS_AVAILABLE and _chaos_detector is not None:
            try:
                metrics.correlation_dimension = _chaos_detector.compute_correlation_dimension(centered)
                transitions = _chaos_detector.detect_phase_transition(centered)
                metrics.n_phase_transitions = len(transitions) if transitions else 0
            except Exception as e:
                pass
        
        # Consciousness metrics
        if run_consciousness and CONSCIOUSNESS_AVAILABLE:
            try:
                cm = ConsciousnessMetrics()
                logits_history = [np.array([v, 1-v]) for v in values[:200]]
                result = cm.compute(logits_history)
                metrics.h_mode = result.get('h_mode', 0.0)
                metrics.participation_ratio = result.get('pr', 0.0)
                metrics.phase_coherence = result.get('r', 0.0)
                metrics.criticality_index = result.get('kappa', 0.0)
                metrics.consciousness_state = result.get('state', 'UNKNOWN')
            except Exception as e:
                pass
        
        # NIST tests
        if run_nist and NIST_AVAILABLE and n >= 1000:
            try:
                bits = [1 if v > 0.5 else 0 for v in values]
                nist = NIST800_22(bits)
                results = nist.run_all_tests()
                
                passed = sum(1 for r in results if r.passed)
                metrics.nist_tests_passed = passed
                metrics.nist_tests_total = len(results)
                metrics.nist_details = {
                    r.test_name: {'p_value': r.p_value, 'passed': r.passed}
                    for r in results
                }
            except Exception as e:
                pass
        
        # Check for anomalies
        alerts = self.check_anomalies(metrics)
        metrics.anomalies_detected = [a.alert_type for a in alerts]
        self.alerts.extend(alerts)
        
        return metrics
    
    def analyze_all_streams(self) -> List[SessionMetrics]:
        """Analyze all QRNG streams for longitudinal tracking."""
        streams = sorted(self.streams_dir.glob("qrng_stream_*.json"))
        
        if not streams:
            print("No QRNG stream files found")
            return []
        
        print(f"Found {len(streams)} QRNG streams")
        
        for i, stream in enumerate(streams):
            print(f"  [{i+1}/{len(streams)}] Analyzing {stream.name}...")
            metrics = self.analyze_stream(stream)
            self.sessions.append(metrics)
        
        return self.sessions
    
    def generate_csprng_control(self, n_samples: int) -> np.ndarray:
        """Generate CSPRNG control data."""
        raw = np.array([
            int.from_bytes(secrets.token_bytes(4), 'little')
            for _ in range(n_samples)
        ], dtype=np.uint32)
        return raw.astype(np.float64) / (2**32)
    
    def generate_prng_control(self, n_samples: int, seed: int = 42) -> np.ndarray:
        """Generate PRNG (Mersenne Twister) control data."""
        rng = np.random.Generator(np.random.MT19937(seed))
        return rng.random(n_samples)
    
    def compare_sources(self, n_samples: int = 1000) -> Dict:
        """Compare QRNG with CSPRNG and PRNG."""
        print("📊 MULTI-SOURCE COMPARISON")
        print("-" * 60)
        
        # Load latest QRNG
        streams = sorted(self.streams_dir.glob("qrng_stream_*.json"))
        if not streams:
            print("No QRNG stream found")
            return {}
        
        qrng_values, _ = self.load_stream(streams[-1])
        qrng_values = qrng_values[:n_samples]
        
        # Generate controls
        print("Generating CSPRNG control (os.urandom)...")
        csprng_values = self.generate_csprng_control(n_samples)
        
        print("Generating PRNG control (Mersenne Twister)...")
        prng_values = self.generate_prng_control(n_samples)
        
        # Analyze all three
        sources = {
            'QRNG': qrng_values,
            'CSPRNG': csprng_values,
            'PRNG': prng_values
        }
        
        results = {}
        
        for name, values in sources.items():
            centered = values - 0.5
            
            bit_stats = self.compute_bit_stats(values)
            runs_z, runs_random = compute_runs_test(values)
            
            results[name] = {
                'mean': float(np.mean(values)),
                'std': float(np.std(values)),
                'bit_bias': bit_stats['bias_from_half'],
                'hurst': compute_hurst_exponent(centered),
                'spectral_entropy': compute_spectral_entropy(centered),
                'runs_z': runs_z,
                'runs_random': runs_random,
                'min_entropy_ratio': self.compute_min_entropy(values),
            }
            
            if NIST_AVAILABLE:
                bits = [1 if v > 0.5 else 0 for v in values]
                nist = NIST800_22(bits)
                nist_results = nist.run_all_tests()
                passed = sum(1 for r in nist_results if r.passed)
                results[name]['nist_passed'] = passed
                results[name]['nist_total'] = len(nist_results)
        
        # Print comparison table
        print("\n" + "=" * 70)
        print(f"{'Metric':<25} {'QRNG':>12} {'CSPRNG':>12} {'PRNG':>12}")
        print("=" * 70)
        
        metrics_to_show = [
            ('Mean', 'mean', '.6f'),
            ('Std Dev', 'std', '.6f'),
            ('Bit Bias', 'bit_bias', '.6f'),
            ('Hurst (centered)', 'hurst', '.4f'),
            ('Spectral Entropy', 'spectral_entropy', '.4f'),
            ('Runs Z-score', 'runs_z', '.3f'),
            ('Min Entropy Ratio', 'min_entropy_ratio', '.4f'),
        ]
        
        for label, key, fmt in metrics_to_show:
            vals = [results[s][key] for s in ['QRNG', 'CSPRNG', 'PRNG']]
            print(f"{label:<25} {vals[0]:>12{fmt}} {vals[1]:>12{fmt}} {vals[2]:>12{fmt}}")
        
        if NIST_AVAILABLE:
            print("-" * 70)
            for s in ['QRNG', 'CSPRNG', 'PRNG']:
                p, t = results[s]['nist_passed'], results[s]['nist_total']
                print(f"NIST Tests ({s}): {p}/{t} passed ({100*p/t:.1f}%)")
        
        print("=" * 70)
        
        return results
    
    def plot_longitudinal(self, output_file: Optional[str] = None):
        """Plot longitudinal trends across sessions."""
        if not self.sessions:
            print("No sessions to plot")
            return
        
        plt.style.use('dark_background')
        fig, axes = plt.subplots(3, 2, figsize=(14, 12), dpi=150)
        
        timestamps = [s.timestamp for s in self.sessions]
        x = range(len(timestamps))
        
        # Hurst over time
        ax = axes[0, 0]
        hursts = [s.hurst for s in self.sessions]
        ax.plot(x, hursts, 'co-', markersize=6)
        ax.axhline(0.5, color='red', linestyle='--', alpha=0.7, label='Random (0.5)')
        ax.fill_between(x, 0.45, 0.55, alpha=0.2, color='gray')
        ax.set_ylabel('Hurst Exponent')
        ax.set_title('Hurst Exponent Over Sessions')
        ax.legend()
        
        # Bit bias
        ax = axes[0, 1]
        biases = [s.bit_bias for s in self.sessions]
        ax.bar(x, biases, color='cyan', alpha=0.7)
        ax.axhline(self.thresholds['bit_bias'], color='red', linestyle='--', 
                   label=f'Threshold ({self.thresholds["bit_bias"]})')
        ax.set_ylabel('Bit Bias (|p-0.5|)')
        ax.set_title('Bit Bias Over Sessions')
        ax.legend()
        
        # Spectral entropy
        ax = axes[1, 0]
        entropies = [s.spectral_entropy for s in self.sessions]
        ax.plot(x, entropies, 'go-', markersize=6)
        ax.axhline(1.0, color='lime', linestyle=':', alpha=0.5, label='White noise')
        ax.axhline(self.thresholds['spectral_entropy'], color='red', linestyle='--',
                   label=f'Threshold ({self.thresholds["spectral_entropy"]})')
        ax.set_ylabel('Spectral Entropy')
        ax.set_title('Spectral Entropy Over Sessions')
        ax.legend()
        
        # Runs Z-score
        ax = axes[1, 1]
        runs = [s.runs_z for s in self.sessions]
        colors = ['green' if s.runs_random else 'red' for s in self.sessions]
        ax.bar(x, runs, color=colors, alpha=0.7)
        ax.axhline(1.96, color='orange', linestyle='--', alpha=0.7)
        ax.axhline(-1.96, color='orange', linestyle='--', alpha=0.7)
        ax.set_ylabel('Runs Z-score')
        ax.set_title('Runs Test Over Sessions')
        
        # NIST pass rate
        ax = axes[2, 0]
        if self.sessions[0].nist_tests_total > 0:
            pass_rates = [s.nist_tests_passed / max(s.nist_tests_total, 1) 
                         for s in self.sessions]
            ax.bar(x, pass_rates, color='cyan', alpha=0.7)
            ax.axhline(0.9, color='red', linestyle='--', label='90% threshold')
            ax.set_ylabel('NIST Pass Rate')
            ax.set_title('NIST Test Pass Rate')
            ax.set_ylim(0, 1.1)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'NIST tests not run', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
        
        # Anomaly count
        ax = axes[2, 1]
        anomaly_counts = [len(s.anomalies_detected) for s in self.sessions]
        colors = ['green' if c == 0 else 'orange' if c < 3 else 'red' 
                  for c in anomaly_counts]
        ax.bar(x, anomaly_counts, color=colors, alpha=0.7)
        ax.set_ylabel('Anomaly Count')
        ax.set_title('Anomalies Detected Per Session')
        ax.set_xlabel('Session')
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, facecolor='black')
            print(f"📈 Saved: {output_file}")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = self.results_dir / f"longitudinal_{ts}.png"
            plt.savefig(out, facecolor='black')
            print(f"📈 Saved: {out}")
        
        plt.close()
    
    def plot_comparison(self, comparison_results: Dict, output_file: Optional[str] = None):
        """Plot multi-source comparison."""
        plt.style.use('dark_background')
        fig, axes = plt.subplots(2, 2, figsize=(12, 10), dpi=150)
        
        sources = ['QRNG', 'CSPRNG', 'PRNG']
        colors = ['cyan', 'lime', 'orange']
        
        # Bar chart: Key metrics
        ax = axes[0, 0]
        metrics = ['hurst', 'spectral_entropy', 'min_entropy_ratio']
        labels = ['Hurst', 'Spectral Ent.', 'Min Ent. Ratio']
        x = np.arange(len(metrics))
        width = 0.25
        
        for i, (src, color) in enumerate(zip(sources, colors)):
            vals = [comparison_results[src][m] for m in metrics]
            ax.bar(x + i*width, vals, width, label=src, color=color, alpha=0.7)
        
        ax.set_xticks(x + width)
        ax.set_xticklabels(labels)
        ax.set_ylabel('Value')
        ax.set_title('Key Metrics Comparison')
        ax.legend()
        
        # Bit bias comparison
        ax = axes[0, 1]
        biases = [comparison_results[s]['bit_bias'] for s in sources]
        ax.bar(sources, biases, color=colors, alpha=0.7)
        ax.axhline(0.01, color='red', linestyle='--', label='Threshold')
        ax.set_ylabel('Bit Bias')
        ax.set_title('Bit Bias Comparison')
        ax.legend()
        
        # Runs Z-score
        ax = axes[1, 0]
        runs = [comparison_results[s]['runs_z'] for s in sources]
        bar_colors = ['green' if abs(r) < 1.96 else 'red' for r in runs]
        ax.bar(sources, runs, color=bar_colors, alpha=0.7)
        ax.axhline(1.96, color='orange', linestyle='--', alpha=0.7)
        ax.axhline(-1.96, color='orange', linestyle='--', alpha=0.7)
        ax.set_ylabel('Runs Z-score')
        ax.set_title('Runs Test Comparison')
        
        # NIST pass rate
        ax = axes[1, 1]
        if 'nist_passed' in comparison_results['QRNG']:
            pass_rates = [comparison_results[s]['nist_passed'] / 
                         comparison_results[s]['nist_total'] for s in sources]
            ax.bar(sources, pass_rates, color=colors, alpha=0.7)
            ax.axhline(0.9, color='red', linestyle='--', label='90% threshold')
            ax.set_ylabel('Pass Rate')
            ax.set_title('NIST Test Pass Rate')
            ax.set_ylim(0, 1.1)
            ax.legend()
        else:
            ax.text(0.5, 0.5, 'NIST tests not available', ha='center', va='center',
                   transform=ax.transAxes, fontsize=14)
        
        plt.tight_layout()
        
        if output_file:
            plt.savefig(output_file, facecolor='black')
            print(f"📈 Saved: {output_file}")
        else:
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            out = self.results_dir / f"comparison_{ts}.png"
            plt.savefig(out, facecolor='black')
            print(f"📈 Saved: {out}")
        
        plt.close()
    
    def print_alerts(self):
        """Print all anomaly alerts."""
        if not self.alerts:
            print("\n✅ No anomalies detected")
            return
        
        print(f"\n⚠️ ANOMALY ALERTS ({len(self.alerts)} total)")
        print("-" * 60)
        
        for alert in self.alerts:
            icon = {'LOW': '🟡', 'MEDIUM': '🟠', 'HIGH': '🔴', 'CRITICAL': '⛔'}
            print(f"{icon.get(alert.severity, '❓')} [{alert.severity}] {alert.alert_type}")
            print(f"   {alert.message}")
            print(f"   Value: {alert.value:.4f}, Threshold: {alert.threshold:.4f}")
            print()
    
    def save_results(self):
        """Save all results to JSON."""
        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Convert sessions to dicts
        sessions_data = []
        for s in self.sessions:
            d = asdict(s)
            # Handle non-serializable types
            d['change_points'] = list(d['change_points'])
            d['anomalies_detected'] = list(d['anomalies_detected'])
            sessions_data.append(d)
        
        alerts_data = [asdict(a) for a in self.alerts]
        
        results = {
            'timestamp': ts,
            'n_sessions': len(self.sessions),
            'n_alerts': len(self.alerts),
            'sessions': sessions_data,
            'alerts': alerts_data,
        }
        
        out_file = self.results_dir / f"dashboard_results_{ts}.json"
        with open(out_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        print(f"📊 Results saved to: {out_file}")
        return out_file
    
    def print_summary(self):
        """Print dashboard summary."""
        if not self.sessions:
            return
        
        print("\n" + "=" * 70)
        print("DASHBOARD SUMMARY")
        print("=" * 70)
        
        latest = self.sessions[-1]
        
        print(f"""
Sessions analyzed: {len(self.sessions)}
Latest session: {latest.timestamp}
  Samples: {latest.n_samples}
  Source: {latest.source}

Key Metrics (latest):
  Hurst (centered):    {latest.hurst:.4f}
  Bit Bias:            {latest.bit_bias:.6f}
  Min Entropy Ratio:   {latest.min_entropy_ratio:.4f}
  Spectral Entropy:    {latest.spectral_entropy:.4f}
  Runs Z-score:        {latest.runs_z:+.4f} {'✓' if latest.runs_random else '⚠'}
  NIST:                {latest.nist_tests_passed}/{latest.nist_tests_total} passed

Chaos Metrics:
  Correlation Dim:     {latest.correlation_dimension:.4f}
  Phase Transitions:   {latest.n_phase_transitions}
  Lyapunov:            {latest.lyapunov:.4f}
  MSD Alpha:           {latest.msd_alpha:.4f}

Consciousness Metrics:
  H_mode:              {latest.h_mode:.4f}
  Participation Ratio: {latest.participation_ratio:.4f}
  Phase Coherence:     {latest.phase_coherence:.4f}
  Criticality Index:   {latest.criticality_index:.4f}
  State:               {latest.consciousness_state}

Anomalies: {len(latest.anomalies_detected)} detected
Change Points: {len(latest.change_points)} detected
        """)


def main():
    parser = argparse.ArgumentParser(description="QRNG Comprehensive Dashboard")
    parser.add_argument('--all', action='store_true', help='Analyze all streams (longitudinal)')
    parser.add_argument('--compare', action='store_true', help='Multi-source comparison')
    parser.add_argument('--live', action='store_true', help='Real-time monitoring mode')
    parser.add_argument('--no-nist', action='store_true', help='Skip NIST tests')
    parser.add_argument('--no-chaos', action='store_true', help='Skip chaos metrics')
    parser.add_argument('--no-consciousness', action='store_true', help='Skip consciousness metrics')
    
    args = parser.parse_args()
    
    dashboard = QRNGDashboard()
    
    print("=" * 70)
    print("QRNG COMPREHENSIVE DASHBOARD")
    print("=" * 70)
    print(f"Using {N_WORKERS} CPU cores")
    print(f"NIST available: {NIST_AVAILABLE}")
    print(f"Chaos available: {CHAOS_AVAILABLE}")
    print(f"Consciousness available: {CONSCIOUSNESS_AVAILABLE}")
    print()
    
    if args.compare:
        # Multi-source comparison
        comparison = dashboard.compare_sources()
        if comparison:
            dashboard.plot_comparison(comparison)
    
    elif args.all:
        # Longitudinal analysis
        dashboard.analyze_all_streams()
        dashboard.plot_longitudinal()
        dashboard.print_alerts()
        dashboard.print_summary()
        dashboard.save_results()
    
    else:
        # Single latest stream
        streams = sorted(Path("qrng_streams").glob("qrng_stream_*.json"))
        if streams:
            print(f"Analyzing: {streams[-1].name}")
            metrics = dashboard.analyze_stream(
                streams[-1],
                run_nist=not args.no_nist,
                run_chaos=not args.no_chaos,
                run_consciousness=not args.no_consciousness
            )
            dashboard.sessions.append(metrics)
            dashboard.print_summary()
            dashboard.print_alerts()
            dashboard.save_results()
        else:
            print("No QRNG streams found in qrng_streams/")


if __name__ == "__main__":
    main()
