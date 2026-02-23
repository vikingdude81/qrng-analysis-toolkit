"""
HELIOS QRNG Trajectory Analysis
================================

Combines SPDC-based quantum random number generation with
the Helios Anomaly Scope for consciousness influence detection.

This integrates:
1. SPDC QRNG (arXiv:2410.00440) - True quantum randomness source
2. Phase Space Reconstruction - Takens' embedding of QRNG stream
3. Trajectory Analysis - MSD, coherence, attractor detection
4. Real-time Visualization - Terminal display and plots

Theory:
- Pure QRNG produces diffusive random walk in phase space
- Consciousness influence could create:
  * Ballistic motion (directed drift)
  * Attractor collapse (orbit stabilization)
  * Coherence spikes (synchronized behavior)
  
Usage:
    python run_qrng_analysis.py --steps 5000 --output-dir qrng_results
"""

import os
import json
import argparse
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import asdict

from qrng_spdc_source import SPDCQuantumSource, SPDCSourceConfig
from helios_anomaly_scope import QRNGStreamScope, HeliosAnomalyScope
from logger_config import setup_analysis_logger, get_logger
from file_utils import atomic_write_json

# Optional: Outshift QRNG client for real quantum data
try:
    from qrng_outshift_client import get_qrng_client, OutshiftQRNGClient
    OUTSHIFT_AVAILABLE = True
except ImportError:
    OUTSHIFT_AVAILABLE = False


class QRNGTerminalVisualizer:
    """ASCII visualization for QRNG trajectory analysis."""
    
    def __init__(self, width: int = 60, height: int = 20):
        self.width = width
        self.height = height
        self.x_range = (-0.5, 0.5)
        self.y_range = (-0.5, 0.5)
        
    def _scale_point(self, x: float, y: float) -> tuple:
        """Scale point to terminal coordinates."""
        x = max(self.x_range[0], min(self.x_range[1], x))
        y = max(self.y_range[0], min(self.y_range[1], y))
        
        col = int((x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * (self.width - 1))
        row = int((self.y_range[1] - y) / (self.y_range[1] - self.y_range[0]) * (self.height - 1))
        
        return row, col
    
    def render(self, x_points: List[float], y_points: List[float], 
               metrics: Dict, qrng_stats: Dict) -> str:
        """Render trajectory with QRNG statistics."""
        # Create grid
        grid = [[' ' for _ in range(self.width)] for _ in range(self.height)]
        
        # Draw axes
        center_row = self.height // 2
        center_col = self.width // 2
        for c in range(self.width):
            grid[center_row][c] = '─'
        for r in range(self.height):
            grid[r][center_col] = '│'
        grid[center_row][center_col] = '┼'
        
        # Auto-scale
        if x_points and y_points:
            x_max = max(abs(min(x_points)), abs(max(x_points)), 0.3)
            y_max = max(abs(min(y_points)), abs(max(y_points)), 0.3)
            scale = max(x_max, y_max) * 1.2
            self.x_range = (-scale, scale)
            self.y_range = (-scale, scale)
        
        # Draw trajectory
        trail_chars = ['░', '▒', '▓', '█']
        n_points = len(x_points)
        for i, (x, y) in enumerate(zip(x_points, y_points)):
            row, col = self._scale_point(x, y)
            if 0 <= row < self.height and 0 <= col < self.width:
                char_idx = min(3, int((i / max(n_points, 1)) * 4))
                if grid[row][col] in ' ─│┼':
                    grid[row][col] = trail_chars[char_idx]
        
        # Mark current position
        if x_points and y_points:
            row, col = self._scale_point(x_points[-1], y_points[-1])
            if 0 <= row < self.height and 0 <= col < self.width:
                grid[row][col] = '●'
        
        # Build output
        lines = []
        lines.append("┌" + "─" * self.width + "┐")
        for row in grid:
            lines.append("│" + "".join(row) + "│")
        lines.append("└" + "─" * self.width + "┘")
        
        # Metrics
        influence = metrics.get('influence_detected', False)
        velocity = metrics.get('velocity', 0)
        msd = metrics.get('msd', 0)
        step = metrics.get('step', 0)
        hurst = metrics.get('hurst', 0.5)
        alpha = metrics.get('diffusion_exponent', 1.0)
        lyapunov = metrics.get('lyapunov', 0.0)
        
        influence_str = "\033[92m● EMERGENCE DETECTED\033[0m" if influence else "○ Normal (Random Walk)"
        
        # Interpret metrics
        if alpha < 0.8:
            motion_type = "Sub-diffusive"
        elif alpha < 1.2:
            motion_type = "Diffusive"
        elif alpha < 1.8:
            motion_type = "Super-diffusive"
        else:
            motion_type = "Ballistic"
        
        # Interpret Lyapunov
        if lyapunov > 0.1:
            lyap_type = "chaotic"
        elif lyapunov < -0.1:
            lyap_type = "convergent"
        else:
            lyap_type = "stable"
            
        lines.append(f"  Step: {step:6d}  │  {influence_str}")
        lines.append(f"  Phase Space: ({metrics.get('x', 0):+.4f}, {metrics.get('y', 0):+.4f})")
        lines.append(f"  Velocity: {velocity:.6f}  │  MSD: {msd:.6f}")
        lines.append(f"  Hurst: {hurst:.3f} (0.5=random)  │  α: {alpha:.2f} ({motion_type})")
        lines.append(f"  Lyapunov: {lyapunov:+.4f} ({lyap_type})  │  0=stable, +chaotic, -attractor")
        
        # QRNG stats
        lines.append("")
        lines.append("  ⚛ SPDC QRNG Statistics:")
        lines.append(f"    Bits generated: {qrng_stats.get('total_bits', 0):,}")
        lines.append(f"    Bit rate: {qrng_stats.get('bit_rate_bps', 0)/1e6:.2f} Mbps")
        lines.append(f"    Coincidences: {qrng_stats.get('coincidences', 0):,}")
        
        return "\n".join(lines)
    
    def render_progress_bar(self, current: int, total: int, width: int = 40) -> str:
        """Render progress bar."""
        pct = current / total
        filled = int(width * pct)
        bar = "█" * filled + "░" * (width - filled)
        return f"  Progress: [{bar}] {current}/{total} ({pct*100:.1f}%)"


class QRNGPlotGenerator:
    """Generate analysis plots for QRNG trajectory data."""
    
    def __init__(self, output_dir: str = "qrng_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def generate_all_plots(self, run_id: str, data: Dict) -> List[str]:
        """Generate all analysis plots."""
        plt.style.use('dark_background')
        
        plots = []
        plots.append(self._plot_trajectory(run_id, data))
        plots.append(self._plot_metrics(run_id, data))
        plots.append(self._plot_bit_distribution(run_id, data))
        plots.append(self._plot_phase_analysis(run_id, data))
        
        return plots
    
    def _plot_trajectory(self, run_id: str, data: Dict) -> str:
        """Plot phase space trajectory with embedding."""
        filepath = self.output_dir / f"run_{run_id}_trajectory.png"
        
        fig, axes = plt.subplots(1, 2, figsize=(14, 7))
        
        # Left: 2D trajectory
        ax1 = axes[0]
        x = data['trajectory']['x']
        y = data['trajectory']['y']
        n = len(x)
        
        colors = plt.cm.plasma(np.linspace(0, 1, n))
        for i in range(n - 1):
            ax1.plot([x[i], x[i+1]], [y[i], y[i+1]], color=colors[i], linewidth=0.5, alpha=0.7)
        
        ax1.scatter([x[0]], [y[0]], color='green', s=200, marker='o', label='Start', zorder=5)
        ax1.scatter([x[-1]], [y[-1]], color='red', s=200, marker='*', label='End', zorder=5)
        
        ax1.set_xlabel('X (Embedding Dim 1)')
        ax1.set_ylabel('Y (Embedding Dim 2)')
        ax1.set_title('Phase Space Trajectory (Takens Embedding)')
        ax1.legend()
        ax1.set_aspect('equal')
        ax1.grid(True, alpha=0.3)
        
        # Right: Raw QRNG stream
        ax2 = axes[1]
        raw_values = data['raw_stream'][:500] if len(data['raw_stream']) > 500 else data['raw_stream']
        ax2.plot(raw_values, linewidth=0.5, color='cyan', alpha=0.7)
        ax2.axhline(y=0.5, color='yellow', linestyle='--', alpha=0.5, label='Expected mean')
        ax2.set_xlabel('Sample')
        ax2.set_ylabel('QRNG Value')
        ax2.set_title('Raw QRNG Stream (First 500 samples)')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.suptitle(f'SPDC QRNG Trajectory Analysis - {run_id}', fontsize=14)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        
        return str(filepath)
    
    def _plot_metrics(self, run_id: str, data: Dict) -> str:
        """Plot all metrics over time."""
        filepath = self.output_dir / f"run_{run_id}_metrics.png"
        
        fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)
        
        metrics = data['metrics']
        n = len(metrics['velocity'])
        steps = range(n)
        
        # Velocity
        axes[0].plot(steps, metrics['velocity'], color='cyan', linewidth=0.8)
        axes[0].fill_between(steps, 0, metrics['velocity'], alpha=0.3, color='cyan')
        axes[0].set_ylabel('Velocity')
        axes[0].set_title('Phase Space Velocity (Movement Speed)')
        axes[0].grid(True, alpha=0.3)
        
        # MSD
        axes[1].plot(steps, metrics['msd'], color='yellow', linewidth=0.8)
        # Theoretical random walk: MSD ~ t
        theoretical_msd = np.array(metrics['msd'][:10]).mean() * np.arange(1, n+1) / 10
        axes[1].plot(steps, theoretical_msd, '--', color='white', alpha=0.5, label='Diffusive (R²~t)')
        axes[1].set_ylabel('MSD')
        axes[1].set_title('Mean Squared Displacement')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)
        
        # Coherence
        axes[2].plot(steps, metrics['coherence'], color='magenta', linewidth=0.8)
        axes[2].fill_between(steps, 0, metrics['coherence'], alpha=0.3, color='magenta')
        axes[2].set_ylabel('Coherence')
        axes[2].set_title('Movement Coherence (High = Stable Orbit)')
        axes[2].grid(True, alpha=0.3)
        
        # Influence detection
        influence = [1 if x else 0 for x in metrics['influence_detected']]
        axes[3].fill_between(steps, 0, influence, alpha=0.7, color='green', step='mid')
        axes[3].set_ylabel('Detection')
        axes[3].set_xlabel('Step')
        axes[3].set_title('Emergence/Influence Detection')
        axes[3].set_ylim(-0.1, 1.1)
        axes[3].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        
        return str(filepath)
    
    def _plot_bit_distribution(self, run_id: str, data: Dict) -> str:
        """Plot bit distribution and randomness quality metrics."""
        filepath = self.output_dir / f"run_{run_id}_randomness.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        raw_values = np.array(data['raw_stream'])
        
        # Histogram
        ax1 = axes[0, 0]
        ax1.hist(raw_values, bins=50, color='cyan', alpha=0.7, edgecolor='white')
        ax1.axvline(x=0.5, color='yellow', linestyle='--', linewidth=2, label='Expected mean')
        ax1.set_xlabel('Value')
        ax1.set_ylabel('Count')
        ax1.set_title('QRNG Value Distribution')
        ax1.legend()
        
        # Autocorrelation
        ax2 = axes[0, 1]
        n_lags = min(50, len(raw_values) // 4)
        autocorr = [np.corrcoef(raw_values[:-lag], raw_values[lag:])[0,1] 
                   if lag > 0 else 1.0 for lag in range(n_lags)]
        ax2.bar(range(n_lags), autocorr, color='magenta', alpha=0.7)
        ax2.axhline(y=0, color='white', linewidth=0.5)
        ax2.axhline(y=2/np.sqrt(len(raw_values)), color='yellow', linestyle='--', alpha=0.5)
        ax2.axhline(y=-2/np.sqrt(len(raw_values)), color='yellow', linestyle='--', alpha=0.5)
        ax2.set_xlabel('Lag')
        ax2.set_ylabel('Autocorrelation')
        ax2.set_title('Autocorrelation (Should be ~0 for random)')
        
        # Successive differences
        ax3 = axes[1, 0]
        diffs = np.diff(raw_values)
        ax3.hist(diffs, bins=50, color='green', alpha=0.7, edgecolor='white')
        ax3.set_xlabel('Δ Value')
        ax3.set_ylabel('Count')
        ax3.set_title('Successive Differences')
        
        # 2D scatter (consecutive pairs)
        ax4 = axes[1, 1]
        ax4.scatter(raw_values[:-1], raw_values[1:], alpha=0.3, s=1, color='yellow')
        ax4.set_xlabel('Value(n)')
        ax4.set_ylabel('Value(n+1)')
        ax4.set_title('Successive Pairs (Should fill uniformly)')
        ax4.set_aspect('equal')
        
        plt.suptitle(f'QRNG Quality Analysis - {run_id}', fontsize=14)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        
        return str(filepath)
    
    def _plot_phase_analysis(self, run_id: str, data: Dict) -> str:
        """Detailed phase space analysis."""
        filepath = self.output_dir / f"run_{run_id}_phase_analysis.png"
        
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        x = np.array(data['trajectory']['x'])
        y = np.array(data['trajectory']['y'])
        
        # Position density
        ax1 = axes[0, 0]
        h = ax1.hist2d(x, y, bins=30, cmap='hot')
        plt.colorbar(h[3], ax=ax1, label='Density')
        ax1.set_xlabel('X')
        ax1.set_ylabel('Y')
        ax1.set_title('Position Density Heatmap')
        
        # Radial distance over time
        ax2 = axes[0, 1]
        r = np.sqrt(x**2 + y**2)
        ax2.plot(r, color='cyan', linewidth=0.5)
        ax2.axhline(y=np.mean(r), color='yellow', linestyle='--', label=f'Mean: {np.mean(r):.4f}')
        ax2.set_xlabel('Step')
        ax2.set_ylabel('Distance from Origin')
        ax2.set_title('Radial Distance Evolution')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Angular position
        ax3 = axes[1, 0]
        theta = np.arctan2(y, x)
        ax3.scatter(range(len(theta)), theta, alpha=0.5, s=1, color='magenta')
        ax3.set_xlabel('Step')
        ax3.set_ylabel('Angle (radians)')
        ax3.set_title('Angular Position Evolution')
        ax3.grid(True, alpha=0.3)
        
        # Angular velocity
        ax4 = axes[1, 1]
        dtheta = np.diff(theta)
        # Handle wraparound
        dtheta = np.where(dtheta > np.pi, dtheta - 2*np.pi, dtheta)
        dtheta = np.where(dtheta < -np.pi, dtheta + 2*np.pi, dtheta)
        ax4.hist(dtheta, bins=50, color='green', alpha=0.7)
        ax4.set_xlabel('Angular Velocity')
        ax4.set_ylabel('Count')
        ax4.set_title('Angular Velocity Distribution')
        
        plt.suptitle(f'Phase Space Analysis - {run_id}', fontsize=14)
        plt.tight_layout()
        plt.savefig(filepath, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        
        return str(filepath)


def run_qrng_analysis(steps: int = 1000,
                      output_dir: str = "qrng_results",
                      visualize: bool = True,
                      embedding_delay: int = 1,
                      ring_sections: int = 4,
                      pump_power_mw: float = 17.0,
                      walk_mode: str = 'angle',
                      use_outshift: bool = False,
                      outshift_stream_file: Optional[str] = None) -> Dict:
    """
    Run QRNG-based trajectory analysis.

    Args:
        steps: Number of QRNG samples to analyze
        output_dir: Directory for output files
        visualize: Show terminal visualization
        embedding_delay: Delay for Takens embedding (tau)
        ring_sections: SPDC ring sections (4, 8, 16...)
        pump_power_mw: Pump laser power
        walk_mode: Random walk construction mode ('angle', 'xy_independent', 'takens')
        use_outshift: Use real Outshift QRNG API instead of simulation
        outshift_stream_file: Path to pre-downloaded Outshift stream JSON file

    Returns:
        Dict with all collected data
    """
    # Handle Outshift stream file if provided
    outshift_floats = None
    if outshift_stream_file:
        print(f"\n  Loading Outshift stream from: {outshift_stream_file}")
        with open(outshift_stream_file) as f:
            stream_data = json.load(f)
        outshift_floats = stream_data.get('floats', [])
        steps = min(steps, len(outshift_floats))
        print(f"  Loaded {len(outshift_floats)} values, using {steps}")
    elif use_outshift:
        if not OUTSHIFT_AVAILABLE:
            raise RuntimeError("Outshift client not available. Install with: pip install requests")
        print("\n  Fetching from Outshift QRNG API...")
        client = get_qrng_client()
        result = client.generate(bits_per_block=32, number_of_blocks=steps)
        outshift_floats = [int(b['decimal']) / (2**32) for b in result['random_numbers']]
        print(f"  Received {len(outshift_floats)} quantum random values")

    # Initialize QRNG source
    qrng = SPDCQuantumSource(
        ring_sections=ring_sections,
        pump_power_mw=pump_power_mw,
        use_extraction=True
    )
    qrng_source_name = "outshift_qrng" if (use_outshift or outshift_stream_file) else "spdc_simulation"
    
    scope = QRNGStreamScope(
        embedding_dim=2,
        embedding_delay=embedding_delay,
        history_len=min(100, steps // 2),
        walk_mode=walk_mode
    )
    
    viz = QRNGTerminalVisualizer(width=60, height=20)
    plotter = QRNGPlotGenerator(output_dir)
    
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Data collection
    raw_stream: List[float] = []
    all_metrics = {
        'velocity': [],
        'msd': [],
        'coherence': [],
        'hurst': [],
        'diffusion_exponent': [],
        'lyapunov': [],
        'influence_detected': []
    }
    
    print("\n" + "=" * 70)
    print("  HELIOS QRNG TRAJECTORY ANALYSIS")
    print("  Based on SPDC Source (arXiv:2410.00440)")
    print("=" * 70)
    print(f"  Run ID: {run_id}")
    print(f"  Steps: {steps}")
    print(f"  Ring Sections: {ring_sections}")
    print(f"  Pump Power: {pump_power_mw} mW")
    print("=" * 70 + "\n")
    
    # Run analysis
    for step in range(steps):
        # Get quantum random value
        if outshift_floats is not None:
            value = outshift_floats[step]
        else:
            value = qrng.get_random()
        raw_stream.append(value)
        
        # Update scope (Takens embedding + trajectory analysis)
        metrics = scope.update_from_stream(value)
        
        # Skip if waiting for embedding history
        if metrics.get('waiting_for_history'):
            continue
            
        # Collect metrics
        all_metrics['velocity'].append(metrics.get('velocity', 0))
        all_metrics['msd'].append(metrics.get('msd', 0))
        all_metrics['coherence'].append(metrics.get('coherence', 0))
        all_metrics['hurst'].append(metrics.get('hurst', 0.5))
        all_metrics['diffusion_exponent'].append(metrics.get('diffusion_exponent', 1.0))
        all_metrics['lyapunov'].append(metrics.get('lyapunov', 0.0))
        all_metrics['influence_detected'].append(metrics.get('influence_detected', False))
        
        # Visualization
        if visualize and (step % max(1, steps // 50) == 0 or step == steps - 1):
            print("\033[2J\033[H", end="")
            
            x_traj, y_traj = scope.get_trajectory()
            qrng_stats = qrng.get_statistics()
            
            print(viz.render(x_traj, y_traj, metrics, qrng_stats))
            print(viz.render_progress_bar(step + 1, steps))
    
    # Collect final data
    x_traj, y_traj = scope.get_trajectory()
    summary = scope.get_summary()
    qrng_stats = qrng.get_statistics()
    
    events_list = []
    for event in scope.get_events():
        events_list.append({
            'step': event.step,
            'event_type': event.event_type,
            'confidence': float(event.confidence),
            'description': event.description
        })
    
    # Full data package
    data = {
        'run_id': run_id,
        'timestamp': datetime.now().isoformat(),
        'config': {
            'steps': steps,
            'embedding_delay': embedding_delay,
            'ring_sections': ring_sections,
            'pump_power_mw': pump_power_mw
        },
        'trajectory': {
            'x': x_traj,
            'y': y_traj
        },
        'raw_stream': raw_stream,
        'metrics': all_metrics,
        'events': events_list,
        'summary': summary,
        'qrng_stats': {
            'total_bits': qrng_stats['total_bits'],
            'bit_rate_bps': qrng_stats['bit_rate_bps'],
            'coincidences': qrng_stats['coincidences'],
            'coincidence_ratio': qrng_stats['coincidence_ratio']
        }
    }
    
    # Export
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    print("\n" + "=" * 70)
    print("  EXPORTING DATA")
    print("=" * 70)
    
    # JSON export
    json_path = output_path / f"run_{run_id}.json"
    
    def convert(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        elif isinstance(obj, (np.int32, np.int64)):
            return int(obj)
        elif isinstance(obj, dict):
            return {k: convert(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert(v) for v in obj]
        return obj
    
    try:
        atomic_write_json(convert(data), json_path)
        print(f"  ✅ JSON: {json_path}")
    except Exception as e:
        print(f"  ❌ JSON write failed: {e}")
        logger = get_logger()
        logger.error(f"Failed to write JSON: {e}")
    
    # Excel export
    try:
        import pandas as pd
        excel_path = output_path / f"run_{run_id}.xlsx"
        
        with pd.ExcelWriter(excel_path, engine='openpyxl') as writer:
            # Raw stream
            pd.DataFrame({'value': raw_stream}).to_excel(writer, sheet_name='Raw_Stream', index=False)
            
            # Trajectory
            pd.DataFrame({
                'step': range(len(x_traj)),
                'x': x_traj,
                'y': y_traj
            }).to_excel(writer, sheet_name='Trajectory', index=False)
            
            # Metrics
            min_len = min(len(v) for v in all_metrics.values())
            pd.DataFrame({
                'step': range(min_len),
                **{k: v[:min_len] for k, v in all_metrics.items()}
            }).to_excel(writer, sheet_name='Metrics', index=False)
            
            # Events
            if events_list:
                pd.DataFrame(events_list).to_excel(writer, sheet_name='Events', index=False)
        
        print(f"  ✅ Excel: {excel_path}")
    except ImportError:
        print("  ⚠️  Excel export skipped (pandas not installed)")
    
    # Generate plots
    print("\n  Generating plots...")
    plot_paths = plotter.generate_all_plots(run_id, data)
    for path in plot_paths:
        print(f"  ✅ Plot: {path}")
    
    # Final summary
    print("\n" + "=" * 70)
    print("  ANALYSIS COMPLETE")
    print("=" * 70)
    print(f"  Total Steps: {steps}")
    print(f"  Events Detected: {summary.get('events_detected', 0)}")
    print(f"  QRNG Bits Generated: {qrng_stats['total_bits']:,}")
    print(f"  QRNG Bit Rate: {qrng_stats['bit_rate_bps']/1e6:.2f} Mbps")
    
    # Randomness quality
    raw_arr = np.array(raw_stream)
    print(f"\n  Randomness Quality:")
    print(f"    Mean: {np.mean(raw_arr):.4f} (expected: 0.5)")
    print(f"    Std: {np.std(raw_arr):.4f} (expected: {1/np.sqrt(12):.4f})")
    
    # Trajectory dynamics
    print(f"\n  Trajectory Dynamics:")
    if all_metrics['hurst']:
        avg_hurst = np.mean(all_metrics['hurst'][-50:])
        print(f"    Hurst Exponent: {avg_hurst:.3f} ", end="")
        if avg_hurst < 0.45:
            print("(anti-persistent)")
        elif avg_hurst < 0.55:
            print("(random walk ✓)")
        else:
            print("\033[93m(persistent/trending)\033[0m")
            
    if all_metrics['diffusion_exponent']:
        avg_alpha = np.mean(all_metrics['diffusion_exponent'][-50:])
        print(f"    Diffusion Exponent (α): {avg_alpha:.2f} ", end="")
        if avg_alpha < 0.8:
            print("(sub-diffusive/confined)")
        elif avg_alpha < 1.2:
            print("(diffusive ✓)")
        elif avg_alpha < 1.5:
            print("\033[93m(super-diffusive)\033[0m")
        else:
            print("\033[91m(ballistic - strong drift!)\033[0m")
    
    if all_metrics['lyapunov']:
        avg_lyap = np.mean(all_metrics['lyapunov'][-50:])
        print(f"    Lyapunov Exponent (λ): {avg_lyap:+.4f} ", end="")
        if avg_lyap > 0.1:
            print("\033[91m(chaotic - sensitive dependence!)\033[0m")
        elif avg_lyap < -0.1:
            print("\033[93m(convergent - attractor lock)\033[0m")
        else:
            print("(stable ✓)")
    
    # Interpretation
    interpretation = summary.get('interpretation', '')
    if interpretation:
        print(f"\n  Interpretation: {interpretation}")
    
    # =========================================================================
    # SIGNAL VERIFICATION
    # =========================================================================
    print(f"\n  Signal Verification:")
    signal_class = scope.get_signal_classification()
    verification = signal_class['verification']
    
    if verification.is_verified:
        print(f"    \033[92m✓ SIGNAL VERIFIED\033[0m")
        print(f"    Classification: \033[93m{verification.signal_class.value.upper()}\033[0m")
        print(f"    Confidence: {verification.confidence*100:.1f}%")
        print(f"    Persistence: {verification.persistence_score*100:.1f}%")
        print(f"    Multi-metric Agreement: {verification.multi_metric_agreement*100:.1f}%")
        
        if verification.tests_passed:
            print(f"    Tests Passed: {', '.join(verification.tests_passed)}")
        
        # Show key evidence
        details = verification.details
        evidence_parts = []
        if details.get('recent_hurst'):
            evidence_parts.append(f"H={details['recent_hurst']:.3f}")
        if details.get('recent_lyapunov') is not None:
            evidence_parts.append(f"λ={details['recent_lyapunov']:.4f}")
        if details.get('recent_alpha'):
            evidence_parts.append(f"α={details['recent_alpha']:.2f}")
        if details.get('spectral_entropy'):
            evidence_parts.append(f"S_ent={details['spectral_entropy']:.3f}")
        if evidence_parts:
            print(f"    Evidence: {', '.join(evidence_parts)}")
    else:
        print(f"    ✗ No verified signal")
        print(f"    Classification: {verification.signal_class.value}")
        if verification.tests_failed:
            print(f"    Failed: {', '.join(verification.tests_failed[:3])}")
    
    # Check for influence
    n_influence = sum(all_metrics['influence_detected'])
    pct_influence = 100 * n_influence / len(all_metrics['influence_detected']) if all_metrics['influence_detected'] else 0
    print(f"\n  Influence Detection:")
    print(f"    Steps with influence: {n_influence} ({pct_influence:.1f}%)")
    
    if verification.is_verified and verification.signal_class.value in ['influence', 'attractor', 'chaotic']:
        print(f"    \033[92m⚠ SIGNIFICANT VERIFIED SIGNAL\033[0m")
    elif pct_influence > 10:
        print(f"    \033[93m⚠ PATTERN DETECTED (unverified)\033[0m")
    else:
        print(f"    ○ Random behavior (as expected)")
    
    print("=" * 70 + "\n")
    
    return data


def main():
    parser = argparse.ArgumentParser(
        description='HELIOS QRNG Trajectory Analysis',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python run_qrng_analysis.py --steps 5000
  python run_qrng_analysis.py --steps 10000 --ring-sections 8 --pump-power 25
  python run_qrng_analysis.py --steps 20000 --no-viz --output-dir my_results
  python run_qrng_analysis.py --steps 5000 --walk-mode angle
        """
    )
    
    parser.add_argument('--steps', type=int, default=2000,
                       help='Number of QRNG samples (default: 2000)')
    parser.add_argument('--output-dir', type=str, default='qrng_results',
                       help='Output directory (default: qrng_results)')
    parser.add_argument('--embedding-delay', type=int, default=1,
                       help='Takens embedding delay (default: 1)')
    parser.add_argument('--ring-sections', type=int, default=4,
                       help='SPDC ring sections (default: 4)')
    parser.add_argument('--pump-power', type=float, default=17.0,
                       help='Pump power in mW (default: 17.0)')
    parser.add_argument('--walk-mode', type=str, default='angle',
                       choices=['angle', 'xy_independent', 'takens'],
                       help='Random walk mode: angle (best), xy_independent, takens (default: angle)')
    parser.add_argument('--no-viz', action='store_true',
                       help='Disable terminal visualization')
    parser.add_argument('--use-outshift', action='store_true',
                       help='Use real Outshift QRNG API (requires API key)')
    parser.add_argument('--outshift-stream', type=str, default=None,
                       help='Path to pre-downloaded Outshift stream JSON file')

    args = parser.parse_args()

    run_qrng_analysis(
        steps=args.steps,
        output_dir=args.output_dir,
        visualize=not args.no_viz,
        embedding_delay=args.embedding_delay,
        ring_sections=args.ring_sections,
        pump_power_mw=args.pump_power,
        walk_mode=args.walk_mode,
        use_outshift=args.use_outshift,
        outshift_stream_file=args.outshift_stream
    )


if __name__ == "__main__":
    main()
