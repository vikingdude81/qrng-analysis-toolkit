"""
QRNG Daily Collector and Analysis Suite
========================================

Automated daily collection and longitudinal analysis of QRNG streams
from the Outshift API for consciousness influence research.

Features:
- Daily QRNG stream collection with metadata
- Statistical significance testing (bootstrap, permutation)
- Longitudinal session comparison
- Anomaly detection across sessions
- Automated report generation

Usage:
    # Collect daily stream
    python qrng_daily_collector.py collect

    # Analyze all collected streams
    python qrng_daily_collector.py analyze

    # Compare specific sessions
    python qrng_daily_collector.py compare 20260114 20260102

    # Generate longitudinal report
    python qrng_daily_collector.py report
"""

import json
import os
import argparse
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, asdict, field
import numpy as np
from scipy import stats
from collections import defaultdict

# Try rich for better terminal output
try:
    from rich.console import Console
    from rich.table import Table
    from rich.panel import Panel
    from rich.progress import Progress, SpinnerColumn, TextColumn
    from rich import print as rprint
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False
    rprint = print

# Import our modules
from qrng_outshift_client import OutshiftQRNGClient, get_qrng_client
from helios_anomaly_scope import (
    compute_hurst_exponent,
    compute_lyapunov_exponent,
    compute_msd_from_trajectory,
    compute_runs_test,
    compute_autocorrelation,
    compute_spectral_entropy,
    detect_periodicity
)
from chaos_detector import ChaosDetector
from consciousness_metrics import ConsciousnessMetrics


@dataclass
class SignificanceResult:
    """Result of a statistical significance test."""
    observed_value: float
    null_mean: float
    null_std: float
    p_value: float
    z_score: float
    is_significant: bool  # p < 0.05
    is_highly_significant: bool  # p < 0.01
    test_name: str
    n_permutations: int = 1000

    def __str__(self) -> str:
        sig = "***" if self.is_highly_significant else ("**" if self.is_significant else "")
        return (f"{self.test_name}: {self.observed_value:.4f} "
                f"(null: {self.null_mean:.4f}±{self.null_std:.4f}, "
                f"p={self.p_value:.4f}{sig}, z={self.z_score:.2f})")


@dataclass
class SessionMetrics:
    """Complete metrics for a QRNG session."""
    session_id: str
    timestamp: str
    source: str
    sample_count: int

    # Core trajectory metrics
    hurst_exponent: float
    lyapunov_exponent: float
    diffusion_alpha: float
    spectral_entropy: float

    # Statistical tests
    runs_test_z: float
    runs_test_random: bool
    autocorr_lag1: float
    autocorr_lag5: float

    # Chaos metrics
    correlation_dimension: float
    criticality_index: float
    phase_transitions: int

    # Consciousness metrics
    consciousness_functional: float
    consciousness_state: str
    mode_entropy: float
    phase_coherence: float

    # Distribution metrics
    mean: float
    std: float
    skewness: float
    kurtosis: float

    # Significance tests (added after bootstrap)
    hurst_significance: Optional[SignificanceResult] = None
    lyapunov_significance: Optional[SignificanceResult] = None

    # Anomaly flags
    anomalies: List[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary, handling nested dataclasses and numpy types."""
        def convert_numpy(obj):
            """Recursively convert numpy types to Python native types."""
            if isinstance(obj, dict):
                return {k: convert_numpy(v) for k, v in obj.items()}
            elif isinstance(obj, list):
                return [convert_numpy(v) for v in obj]
            elif isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, (np.integer, np.int64, np.int32)):
                return int(obj)
            elif isinstance(obj, (np.floating, np.float64, np.float32)):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj
        
        d = asdict(self)
        d = convert_numpy(d)
        if self.hurst_significance:
            d['hurst_significance'] = convert_numpy(asdict(self.hurst_significance))
        if self.lyapunov_significance:
            d['lyapunov_significance'] = convert_numpy(asdict(self.lyapunov_significance))
        return d


class StatisticalSignificanceTester:
    """
    Bootstrap and permutation tests for QRNG metric significance.

    Tests whether observed metrics are significantly different from
    what would be expected from a purely random process.
    """

    def __init__(self, n_permutations: int = 1000, seed: Optional[int] = None):
        self.n_permutations = n_permutations
        self.rng = np.random.RandomState(seed)

    def bootstrap_hurst_test(self, series: np.ndarray) -> SignificanceResult:
        """
        Test if Hurst exponent is significantly different from 0.5 (pure random).

        Uses bootstrap resampling of a shuffled series to establish null distribution.
        """
        observed_h = compute_hurst_exponent(series)

        # Generate null distribution by shuffling (destroys temporal structure)
        null_distribution = []
        for _ in range(self.n_permutations):
            shuffled = self.rng.permutation(series)
            null_distribution.append(compute_hurst_exponent(shuffled))

        null_distribution = np.array(null_distribution)
        null_mean = np.mean(null_distribution)
        null_std = np.std(null_distribution)

        # Two-tailed p-value
        if null_std > 0:
            z_score = (observed_h - null_mean) / null_std
            # Count how many null values are more extreme
            p_value = np.mean(np.abs(null_distribution - null_mean) >= np.abs(observed_h - null_mean))
        else:
            z_score = 0.0
            p_value = 1.0

        return SignificanceResult(
            observed_value=observed_h,
            null_mean=null_mean,
            null_std=null_std,
            p_value=p_value,
            z_score=z_score,
            is_significant=p_value < 0.05,
            is_highly_significant=p_value < 0.01,
            test_name="Hurst Bootstrap",
            n_permutations=self.n_permutations
        )

    def bootstrap_lyapunov_test(self, x: List[float], y: List[float]) -> SignificanceResult:
        """
        Test if Lyapunov exponent is significantly different from 0 (stable).
        """
        observed_l = compute_lyapunov_exponent(x, y)

        # Generate null distribution by shuffling trajectory points
        null_distribution = []
        for _ in range(self.n_permutations):
            # Shuffle both coordinates together to preserve (x,y) pairing
            indices = self.rng.permutation(len(x))
            shuffled_x = [x[i] for i in indices]
            shuffled_y = [y[i] for i in indices]
            null_distribution.append(compute_lyapunov_exponent(shuffled_x, shuffled_y))

        null_distribution = np.array(null_distribution)
        null_mean = np.mean(null_distribution)
        null_std = np.std(null_distribution)

        if null_std > 0:
            z_score = (observed_l - null_mean) / null_std
            p_value = np.mean(np.abs(null_distribution - null_mean) >= np.abs(observed_l - null_mean))
        else:
            z_score = 0.0
            p_value = 1.0

        return SignificanceResult(
            observed_value=observed_l,
            null_mean=null_mean,
            null_std=null_std,
            p_value=p_value,
            z_score=z_score,
            is_significant=p_value < 0.05,
            is_highly_significant=p_value < 0.01,
            test_name="Lyapunov Bootstrap",
            n_permutations=self.n_permutations
        )

    def permutation_correlation_test(self, series: np.ndarray, lag: int = 1) -> SignificanceResult:
        """
        Test if autocorrelation at given lag is significant.
        """
        if len(series) < lag + 10:
            return SignificanceResult(
                observed_value=0.0, null_mean=0.0, null_std=0.0,
                p_value=1.0, z_score=0.0, is_significant=False,
                is_highly_significant=False, test_name=f"Autocorr(lag={lag})"
            )

        observed_corr = np.corrcoef(series[:-lag], series[lag:])[0, 1]

        null_distribution = []
        for _ in range(self.n_permutations):
            shuffled = self.rng.permutation(series)
            null_distribution.append(np.corrcoef(shuffled[:-lag], shuffled[lag:])[0, 1])

        null_distribution = np.array(null_distribution)
        null_mean = np.mean(null_distribution)
        null_std = np.std(null_distribution)

        if null_std > 0:
            z_score = (observed_corr - null_mean) / null_std
            p_value = np.mean(np.abs(null_distribution) >= np.abs(observed_corr))
        else:
            z_score = 0.0
            p_value = 1.0

        return SignificanceResult(
            observed_value=observed_corr,
            null_mean=null_mean,
            null_std=null_std,
            p_value=p_value,
            z_score=z_score,
            is_significant=p_value < 0.05,
            is_highly_significant=p_value < 0.01,
            test_name=f"Autocorr(lag={lag}) Permutation",
            n_permutations=self.n_permutations
        )


class QRNGSessionAnalyzer:
    """
    Analyze a single QRNG session with full metrics suite.
    """

    def __init__(self, significance_tester: Optional[StatisticalSignificanceTester] = None):
        self.sig_tester = significance_tester or StatisticalSignificanceTester()
        self.chaos_detector = ChaosDetector()
        self.consciousness = ConsciousnessMetrics()

    def analyze_stream(self,
                       floats: List[float],
                       session_id: str,
                       timestamp: str,
                       source: str = "unknown",
                       run_significance_tests: bool = True) -> SessionMetrics:
        """
        Perform complete analysis on a QRNG stream.
        """
        arr = np.array(floats)
        n = len(arr)

        # Build trajectory using angle-based random walk
        x, y = [0.0], [0.0]
        for val in floats[:-1]:  # Leave one for delay
            angle = val * 2 * np.pi
            step = 0.1
            x.append(x[-1] + step * np.cos(angle))
            y.append(y[-1] + step * np.sin(angle))

        # Core trajectory metrics
        hurst = compute_hurst_exponent(arr)
        lyapunov = compute_lyapunov_exponent(x, y)

        # MSD analysis - returns (lags, msd_values, diffusion_exponent)
        msd_result = compute_msd_from_trajectory(x, y)
        if isinstance(msd_result, tuple) and len(msd_result) >= 3:
            diffusion_alpha = msd_result[2]  # Third element is diffusion_exponent
        else:
            diffusion_alpha = 1.0

        # Statistical tests - returns (z_score, is_random)
        runs_result = compute_runs_test(arr)
        if isinstance(runs_result, tuple) and len(runs_result) >= 2:
            runs_z = float(runs_result[0])
            runs_random = bool(runs_result[1])  # Convert numpy.bool_ to Python bool
        else:
            runs_z = 0.0
            runs_random = True

        autocorr = compute_autocorrelation(arr, max_lag=10)
        autocorr_lag1 = autocorr[1] if len(autocorr) > 1 else 0.0
        autocorr_lag5 = autocorr[5] if len(autocorr) > 5 else 0.0

        spectral_ent = compute_spectral_entropy(arr)

        # Chaos metrics - use individual methods
        chaos = {
            'correlation_dimension': self.chaos_detector.compute_correlation_dimension(arr),
            'criticality_index': self.chaos_detector.compute_criticality_index(arr),
            'phase_transitions': len(self.chaos_detector.detect_phase_transition(arr).get('transitions', []))
        }

        # Consciousness metrics
        # Convert QRNG values to simulated logit distributions for consciousness analysis
        # Chunk into windows and treat each as a "logit distribution"
        window_size = 32
        logits_history = []
        for i in range(0, n - window_size, window_size // 2):
            window = arr[i:i + window_size]
            # Convert to pseudo-logits (just the values normalized)
            logits_history.append(window)

        if len(logits_history) >= 2:
            cons_result = self.consciousness.compute(logits_history)
        else:
            cons_result = {
                'consciousness': 0.0,
                'state': 'unknown',
                'h_mode': 0.0,
                'r': 0.0
            }

        # Distribution metrics
        mean_val = float(np.mean(arr))
        std_val = float(np.std(arr))
        skew = float(stats.skew(arr))
        kurt = float(stats.kurtosis(arr))

        # Build anomalies list
        anomalies = []
        if hurst > 0.6:
            anomalies.append(f"Hurst={hurst:.3f} suggests persistent correlations")
        elif hurst < 0.4:
            anomalies.append(f"Hurst={hurst:.3f} suggests anti-persistent behavior")

        if abs(lyapunov) > 0.1:
            if lyapunov > 0:
                anomalies.append(f"Lyapunov={lyapunov:.3f} indicates chaotic dynamics")
            else:
                anomalies.append(f"Lyapunov={lyapunov:.3f} indicates attractor convergence")

        if diffusion_alpha < 0.5:
            anomalies.append(f"Alpha={diffusion_alpha:.2f} indicates confined/sub-diffusive motion")
        elif diffusion_alpha > 1.5:
            anomalies.append(f"Alpha={diffusion_alpha:.2f} indicates super-diffusive/ballistic motion")

        if not runs_random:
            anomalies.append(f"Runs test failed (z={runs_z:.2f})")

        if abs(autocorr_lag1) > 0.1:
            anomalies.append(f"Significant lag-1 autocorrelation: {autocorr_lag1:.3f}")

        # Create metrics object
        metrics = SessionMetrics(
            session_id=session_id,
            timestamp=timestamp,
            source=source,
            sample_count=n,
            hurst_exponent=hurst,
            lyapunov_exponent=lyapunov,
            diffusion_alpha=diffusion_alpha,
            spectral_entropy=spectral_ent,
            runs_test_z=runs_z,
            runs_test_random=runs_random,
            autocorr_lag1=autocorr_lag1,
            autocorr_lag5=autocorr_lag5,
            correlation_dimension=chaos.get('correlation_dimension', 0.0),
            criticality_index=chaos.get('criticality_index', 0.0),
            phase_transitions=chaos.get('phase_transitions', 0),
            consciousness_functional=cons_result.get('consciousness', 0.0),
            consciousness_state=cons_result.get('state', 'unknown'),
            mode_entropy=cons_result.get('mode_entropy', 0.0),
            phase_coherence=cons_result.get('phase_coherence', 0.0),
            mean=mean_val,
            std=std_val,
            skewness=skew,
            kurtosis=kurt,
            anomalies=anomalies
        )

        # Run significance tests if requested
        if run_significance_tests:
            metrics.hurst_significance = self.sig_tester.bootstrap_hurst_test(arr)
            metrics.lyapunov_significance = self.sig_tester.bootstrap_lyapunov_test(x, y)

            # Update anomalies based on significance
            if metrics.hurst_significance.is_significant:
                anomalies.append(
                    f"Hurst significantly different from random (p={metrics.hurst_significance.p_value:.4f})"
                )
            if metrics.lyapunov_significance.is_significant:
                anomalies.append(
                    f"Lyapunov significantly different from null (p={metrics.lyapunov_significance.p_value:.4f})"
                )

        return metrics


class QRNGDailyCollector:
    """
    Collect and manage daily QRNG streams from Outshift API.
    """

    def __init__(self,
                 output_dir: str = "qrng_streams",
                 analysis_dir: str = "qrng_analysis"):
        self.output_dir = Path(output_dir)
        self.analysis_dir = Path(analysis_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.analysis_dir.mkdir(parents=True, exist_ok=True)

        self.analyzer = QRNGSessionAnalyzer()

    def collect_daily_stream(self,
                            count: int = 1000,
                            bits_per_number: int = 32) -> Tuple[str, Dict]:
        """
        Collect a daily QRNG stream from Outshift API.

        Returns:
            Tuple of (session_id, stream_data)
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session_id = f"qrng_stream_{timestamp}"

        rprint(f"[bold blue]Collecting QRNG stream from Outshift API...[/bold blue]" if RICH_AVAILABLE
               else "Collecting QRNG stream from Outshift API...")

        client = get_qrng_client()

        # Get raw integers and normalized floats
        result = client.generate(
            bits_per_block=bits_per_number,
            number_of_blocks=count,
            encoding='raw',
            format='decimal'
        )

        raw_integers = [int(block['decimal']) for block in result['random_numbers']]
        max_value = 2 ** bits_per_number
        floats = [n / max_value for n in raw_integers]

        stream_data = {
            'timestamp': timestamp,
            'source': 'outshift_qrng_api',
            'bits_per_block': bits_per_number,
            'count': count,
            'raw_integers': raw_integers,
            'floats': floats,
            'encoding': result.get('encoding', 'raw')
        }

        # Save raw stream
        stream_path = self.output_dir / f"{session_id}.json"
        with open(stream_path, 'w') as f:
            json.dump(stream_data, f, indent=2)

        rprint(f"[green]✓ Saved stream to {stream_path}[/green]" if RICH_AVAILABLE
               else f"Saved stream to {stream_path}")

        return session_id, stream_data

    def analyze_stream(self,
                      session_id: str,
                      run_significance: bool = True) -> SessionMetrics:
        """
        Analyze a collected stream.
        """
        # Load stream
        stream_path = self.output_dir / f"{session_id}.json"
        with open(stream_path) as f:
            data = json.load(f)

        # Analyze
        metrics = self.analyzer.analyze_stream(
            floats=data['floats'],
            session_id=session_id,
            timestamp=data['timestamp'],
            source=data['source'],
            run_significance_tests=run_significance
        )

        # Save analysis
        analysis_path = self.analysis_dir / f"analysis_{session_id}.json"
        with open(analysis_path, 'w') as f:
            json.dump(metrics.to_dict(), f, indent=2)

        return metrics

    def collect_and_analyze(self,
                           count: int = 1000,
                           run_significance: bool = True) -> SessionMetrics:
        """
        Collect a new stream and immediately analyze it.
        """
        session_id, data = self.collect_daily_stream(count=count)

        rprint("[bold blue]Analyzing stream...[/bold blue]" if RICH_AVAILABLE
               else "Analyzing stream...")

        metrics = self.analyze_stream(session_id, run_significance=run_significance)

        # Print summary
        self._print_metrics_summary(metrics)

        return metrics

    def _print_metrics_summary(self, metrics: SessionMetrics):
        """Print a summary of session metrics."""
        if RICH_AVAILABLE:
            console = Console()

            table = Table(title=f"QRNG Session Analysis: {metrics.session_id}")
            table.add_column("Metric", style="cyan")
            table.add_column("Value", style="magenta")
            table.add_column("Interpretation", style="green")

            # Hurst
            h_interp = "random walk" if 0.45 < metrics.hurst_exponent < 0.55 else \
                       "persistent" if metrics.hurst_exponent > 0.55 else "anti-persistent"
            h_sig = ""
            if metrics.hurst_significance:
                h_sig = f" (p={metrics.hurst_significance.p_value:.4f})"
            table.add_row("Hurst Exponent", f"{metrics.hurst_exponent:.4f}{h_sig}", h_interp)

            # Lyapunov
            l_interp = "stable" if abs(metrics.lyapunov_exponent) < 0.05 else \
                       "chaotic" if metrics.lyapunov_exponent > 0 else "convergent"
            l_sig = ""
            if metrics.lyapunov_significance:
                l_sig = f" (p={metrics.lyapunov_significance.p_value:.4f})"
            table.add_row("Lyapunov Exponent", f"{metrics.lyapunov_exponent:.4f}{l_sig}", l_interp)

            # Diffusion
            d_interp = "diffusive" if 0.8 < metrics.diffusion_alpha < 1.2 else \
                       "sub-diffusive" if metrics.diffusion_alpha < 0.8 else "super-diffusive"
            table.add_row("Diffusion α", f"{metrics.diffusion_alpha:.4f}", d_interp)

            # Spectral entropy
            table.add_row("Spectral Entropy", f"{metrics.spectral_entropy:.4f}",
                         "high randomness" if metrics.spectral_entropy > 0.8 else "structured")

            # Consciousness
            table.add_row("Consciousness", f"{metrics.consciousness_functional:.4f}",
                         metrics.consciousness_state)

            # Runs test
            table.add_row("Runs Test",
                         "PASS" if metrics.runs_test_random else f"FAIL (z={metrics.runs_test_z:.2f})",
                         "random" if metrics.runs_test_random else "non-random")

            console.print(table)

            # Anomalies
            if metrics.anomalies:
                console.print(Panel(
                    "\n".join([f"• {a}" for a in metrics.anomalies]),
                    title="[bold red]Anomalies Detected[/bold red]",
                    border_style="red"
                ))
        else:
            print(f"\n=== Session Analysis: {metrics.session_id} ===")
            print(f"  Hurst: {metrics.hurst_exponent:.4f}")
            print(f"  Lyapunov: {metrics.lyapunov_exponent:.4f}")
            print(f"  Diffusion α: {metrics.diffusion_alpha:.4f}")
            print(f"  Spectral Entropy: {metrics.spectral_entropy:.4f}")
            print(f"  Consciousness: {metrics.consciousness_functional:.4f} ({metrics.consciousness_state})")
            print(f"  Runs Test: {'PASS' if metrics.runs_test_random else 'FAIL'}")
            if metrics.anomalies:
                print("  Anomalies:")
                for a in metrics.anomalies:
                    print(f"    • {a}")

    def list_sessions(self) -> List[str]:
        """List all collected session IDs."""
        sessions = []
        for f in self.output_dir.glob("qrng_stream_*.json"):
            sessions.append(f.stem)
        return sorted(sessions)

    def load_session_metrics(self, session_id: str) -> Optional[SessionMetrics]:
        """Load previously computed metrics for a session."""
        analysis_path = self.analysis_dir / f"analysis_{session_id}.json"
        if not analysis_path.exists():
            return None

        with open(analysis_path) as f:
            data = json.load(f)

        # Reconstruct SignificanceResult if present
        hurst_sig = None
        if data.get('hurst_significance'):
            hurst_sig = SignificanceResult(**data['hurst_significance'])

        lyap_sig = None
        if data.get('lyapunov_significance'):
            lyap_sig = SignificanceResult(**data['lyapunov_significance'])

        return SessionMetrics(
            session_id=data['session_id'],
            timestamp=data['timestamp'],
            source=data['source'],
            sample_count=data['sample_count'],
            hurst_exponent=data['hurst_exponent'],
            lyapunov_exponent=data['lyapunov_exponent'],
            diffusion_alpha=data['diffusion_alpha'],
            spectral_entropy=data['spectral_entropy'],
            runs_test_z=data['runs_test_z'],
            runs_test_random=data['runs_test_random'],
            autocorr_lag1=data['autocorr_lag1'],
            autocorr_lag5=data['autocorr_lag5'],
            correlation_dimension=data['correlation_dimension'],
            criticality_index=data['criticality_index'],
            phase_transitions=data['phase_transitions'],
            consciousness_functional=data['consciousness_functional'],
            consciousness_state=data['consciousness_state'],
            mode_entropy=data['mode_entropy'],
            phase_coherence=data['phase_coherence'],
            mean=data['mean'],
            std=data['std'],
            skewness=data['skewness'],
            kurtosis=data['kurtosis'],
            hurst_significance=hurst_sig,
            lyapunov_significance=lyap_sig,
            anomalies=data.get('anomalies', [])
        )


class LongitudinalAnalyzer:
    """
    Analyze trends across multiple QRNG sessions.
    """

    def __init__(self, collector: QRNGDailyCollector):
        self.collector = collector

    def load_all_sessions(self) -> List[SessionMetrics]:
        """Load metrics from all analyzed sessions."""
        sessions = []
        for sid in self.collector.list_sessions():
            metrics = self.collector.load_session_metrics(sid)
            if metrics is None:
                # Analyze if not already done
                metrics = self.collector.analyze_stream(sid, run_significance=False)
            sessions.append(metrics)
        return sorted(sessions, key=lambda m: m.timestamp)

    def compute_trends(self, sessions: List[SessionMetrics]) -> Dict[str, Any]:
        """
        Compute trends across sessions.
        """
        if len(sessions) < 2:
            return {'error': 'Need at least 2 sessions for trend analysis'}

        # Extract time series of metrics
        hursts = [s.hurst_exponent for s in sessions]
        lyapunovs = [s.lyapunov_exponent for s in sessions]
        alphas = [s.diffusion_alpha for s in sessions]
        entropies = [s.spectral_entropy for s in sessions]
        consciousness = [s.consciousness_functional for s in sessions]

        # Compute statistics
        def trend_stats(values: List[float], name: str) -> dict:
            arr = np.array(values)
            n = len(arr)

            # Linear trend
            slope, intercept, r_value, p_value, std_err = stats.linregress(range(n), arr)

            return {
                'metric': name,
                'mean': float(np.mean(arr)),
                'std': float(np.std(arr)),
                'min': float(np.min(arr)),
                'max': float(np.max(arr)),
                'trend_slope': float(slope),
                'trend_r_squared': float(r_value ** 2),
                'trend_p_value': float(p_value),
                'trend_significant': p_value < 0.05
            }

        return {
            'n_sessions': len(sessions),
            'date_range': {
                'start': sessions[0].timestamp,
                'end': sessions[-1].timestamp
            },
            'metrics': {
                'hurst': trend_stats(hursts, 'Hurst Exponent'),
                'lyapunov': trend_stats(lyapunovs, 'Lyapunov Exponent'),
                'diffusion_alpha': trend_stats(alphas, 'Diffusion Alpha'),
                'spectral_entropy': trend_stats(entropies, 'Spectral Entropy'),
                'consciousness': trend_stats(consciousness, 'Consciousness Functional')
            },
            'anomaly_summary': self._summarize_anomalies(sessions)
        }

    def _summarize_anomalies(self, sessions: List[SessionMetrics]) -> dict:
        """Summarize anomalies across all sessions."""
        all_anomalies = []
        sessions_with_anomalies = 0

        for s in sessions:
            if s.anomalies:
                sessions_with_anomalies += 1
                all_anomalies.extend(s.anomalies)

        # Count anomaly types
        anomaly_counts = defaultdict(int)
        for a in all_anomalies:
            # Extract anomaly type (first word pattern)
            if 'Hurst' in a:
                anomaly_counts['Hurst anomaly'] += 1
            elif 'Lyapunov' in a:
                anomaly_counts['Lyapunov anomaly'] += 1
            elif 'Alpha' in a:
                anomaly_counts['Diffusion anomaly'] += 1
            elif 'Runs' in a:
                anomaly_counts['Runs test failure'] += 1
            elif 'autocorrelation' in a:
                anomaly_counts['Autocorrelation'] += 1
            else:
                anomaly_counts['Other'] += 1

        return {
            'total_anomalies': len(all_anomalies),
            'sessions_with_anomalies': sessions_with_anomalies,
            'anomaly_rate': sessions_with_anomalies / len(sessions) if sessions else 0,
            'anomaly_types': dict(anomaly_counts)
        }

    def compare_sessions(self, session_id_1: str, session_id_2: str) -> dict:
        """
        Compare two specific sessions.
        """
        m1 = self.collector.load_session_metrics(session_id_1)
        m2 = self.collector.load_session_metrics(session_id_2)

        if m1 is None:
            m1 = self.collector.analyze_stream(session_id_1)
        if m2 is None:
            m2 = self.collector.analyze_stream(session_id_2)

        # Compute differences
        def diff(a: float, b: float) -> dict:
            return {
                'value_1': a,
                'value_2': b,
                'difference': b - a,
                'percent_change': 100 * (b - a) / a if a != 0 else 0
            }

        return {
            'session_1': session_id_1,
            'session_2': session_id_2,
            'timestamp_1': m1.timestamp,
            'timestamp_2': m2.timestamp,
            'comparisons': {
                'hurst': diff(m1.hurst_exponent, m2.hurst_exponent),
                'lyapunov': diff(m1.lyapunov_exponent, m2.lyapunov_exponent),
                'diffusion_alpha': diff(m1.diffusion_alpha, m2.diffusion_alpha),
                'spectral_entropy': diff(m1.spectral_entropy, m2.spectral_entropy),
                'consciousness': diff(m1.consciousness_functional, m2.consciousness_functional)
            },
            'anomalies_1': m1.anomalies,
            'anomalies_2': m2.anomalies
        }

    def generate_report(self, output_path: Optional[str] = None) -> str:
        """
        Generate a comprehensive longitudinal report.
        """
        sessions = self.load_all_sessions()

        if not sessions:
            return "No sessions found for analysis."

        trends = self.compute_trends(sessions)

        # Build report
        lines = []
        lines.append("=" * 70)
        lines.append("HELIOS QRNG LONGITUDINAL ANALYSIS REPORT")
        lines.append("=" * 70)
        lines.append(f"\nGenerated: {datetime.now().isoformat()}")
        lines.append(f"Sessions Analyzed: {trends['n_sessions']}")
        lines.append(f"Date Range: {trends['date_range']['start']} to {trends['date_range']['end']}")

        lines.append("\n" + "-" * 70)
        lines.append("METRIC TRENDS")
        lines.append("-" * 70)

        for metric_name, metric_data in trends['metrics'].items():
            lines.append(f"\n{metric_data['metric']}:")
            lines.append(f"  Mean: {metric_data['mean']:.4f} ± {metric_data['std']:.4f}")
            lines.append(f"  Range: [{metric_data['min']:.4f}, {metric_data['max']:.4f}]")
            lines.append(f"  Trend: slope={metric_data['trend_slope']:.6f}, R²={metric_data['trend_r_squared']:.4f}")
            if metric_data['trend_significant']:
                lines.append(f"  ⚠ SIGNIFICANT TREND (p={metric_data['trend_p_value']:.4f})")

        lines.append("\n" + "-" * 70)
        lines.append("ANOMALY SUMMARY")
        lines.append("-" * 70)

        anomaly_data = trends['anomaly_summary']
        lines.append(f"\nTotal Anomalies: {anomaly_data['total_anomalies']}")
        lines.append(f"Sessions with Anomalies: {anomaly_data['sessions_with_anomalies']} ({anomaly_data['anomaly_rate']*100:.1f}%)")

        if anomaly_data['anomaly_types']:
            lines.append("\nAnomaly Types:")
            for atype, count in sorted(anomaly_data['anomaly_types'].items(), key=lambda x: -x[1]):
                lines.append(f"  {atype}: {count}")

        lines.append("\n" + "-" * 70)
        lines.append("SESSION DETAILS")
        lines.append("-" * 70)

        for s in sessions:
            sig_marker = ""
            if s.hurst_significance and s.hurst_significance.is_significant:
                sig_marker = " *"
            lines.append(f"\n{s.session_id} ({s.timestamp}):")
            lines.append(f"  H={s.hurst_exponent:.3f}{sig_marker}, λ={s.lyapunov_exponent:.4f}, α={s.diffusion_alpha:.3f}")
            lines.append(f"  S_ent={s.spectral_entropy:.3f}, C={s.consciousness_functional:.4f} ({s.consciousness_state})")
            if s.anomalies:
                lines.append(f"  Anomalies: {len(s.anomalies)}")

        lines.append("\n" + "=" * 70)
        lines.append("END OF REPORT")
        lines.append("=" * 70)

        report = "\n".join(lines)

        # Save if path provided
        if output_path:
            with open(output_path, 'w') as f:
                f.write(report)

        return report


def main():
    parser = argparse.ArgumentParser(
        description='QRNG Daily Collector and Analysis Suite',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Commands:
  collect   - Collect a new QRNG stream from Outshift API
  analyze   - Analyze all collected streams
  compare   - Compare two specific sessions
  report    - Generate longitudinal analysis report
  list      - List all collected sessions

Examples:
  python qrng_daily_collector.py collect
  python qrng_daily_collector.py collect --count 2000
  python qrng_daily_collector.py analyze
  python qrng_daily_collector.py compare qrng_stream_20260114_190247 qrng_stream_20260102_071149
  python qrng_daily_collector.py report --output report.txt
  python qrng_daily_collector.py list
        """
    )

    parser.add_argument('command', choices=['collect', 'analyze', 'compare', 'report', 'list'],
                       help='Command to execute')
    parser.add_argument('--count', type=int, default=1000,
                       help='Number of QRNG samples to collect (default: 1000)')
    parser.add_argument('--output', '-o', type=str, default=None,
                       help='Output file path for reports')
    parser.add_argument('--no-significance', action='store_true',
                       help='Skip significance testing (faster)')
    parser.add_argument('sessions', nargs='*',
                       help='Session IDs for compare command')

    args = parser.parse_args()

    collector = QRNGDailyCollector()
    longitudinal = LongitudinalAnalyzer(collector)

    if args.command == 'collect':
        try:
            metrics = collector.collect_and_analyze(
                count=args.count,
                run_significance=not args.no_significance
            )
            rprint(f"\n[bold green]✓ Collection complete: {metrics.session_id}[/bold green]"
                   if RICH_AVAILABLE else f"\nCollection complete: {metrics.session_id}")
        except Exception as e:
            rprint(f"[bold red]Error: {e}[/bold red]" if RICH_AVAILABLE else f"Error: {e}")
            import traceback
            traceback.print_exc()

    elif args.command == 'analyze':
        sessions = collector.list_sessions()
        rprint(f"[bold blue]Analyzing {len(sessions)} sessions...[/bold blue]"
               if RICH_AVAILABLE else f"Analyzing {len(sessions)} sessions...")

        for sid in sessions:
            rprint(f"  Processing {sid}..." if RICH_AVAILABLE else f"  Processing {sid}...")
            collector.analyze_stream(sid, run_significance=not args.no_significance)

        rprint(f"[bold green]✓ Analysis complete[/bold green]" if RICH_AVAILABLE else "Analysis complete")

    elif args.command == 'compare':
        if len(args.sessions) != 2:
            rprint("[bold red]Error: compare requires exactly 2 session IDs[/bold red]"
                   if RICH_AVAILABLE else "Error: compare requires exactly 2 session IDs")
            return

        comparison = longitudinal.compare_sessions(args.sessions[0], args.sessions[1])

        rprint(f"\n[bold]Comparison: {comparison['session_1']} vs {comparison['session_2']}[/bold]"
               if RICH_AVAILABLE else f"\nComparison: {comparison['session_1']} vs {comparison['session_2']}")

        for metric, data in comparison['comparisons'].items():
            change = data['percent_change']
            arrow = "↑" if change > 0 else "↓" if change < 0 else "→"
            print(f"  {metric}: {data['value_1']:.4f} → {data['value_2']:.4f} ({arrow} {abs(change):.1f}%)")

    elif args.command == 'report':
        output_path = args.output or f"qrng_report_{datetime.now().strftime('%Y%m%d')}.txt"
        report = longitudinal.generate_report(output_path)
        print(report)
        rprint(f"\n[bold green]✓ Report saved to {output_path}[/bold green]"
               if RICH_AVAILABLE else f"\nReport saved to {output_path}")

    elif args.command == 'list':
        sessions = collector.list_sessions()
        if not sessions:
            rprint("[yellow]No sessions found[/yellow]" if RICH_AVAILABLE else "No sessions found")
        else:
            rprint(f"[bold]Found {len(sessions)} sessions:[/bold]" if RICH_AVAILABLE
                   else f"Found {len(sessions)} sessions:")
            for sid in sessions:
                print(f"  {sid}")


if __name__ == "__main__":
    main()
