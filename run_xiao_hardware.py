"""
XIAO Hardware QRNG — Helios Trajectory Analysis
=================================================
Runs the full Helios pipeline (phase space reconstruction, Takens embedding,
Hurst/Lyapunov/MSD metrics, influence detection) using the Seeed XIAO SAMD21
hardware QRNG on COM4 as the entropy source.

This is a drop-in replacement for run_qrng_analysis.py that uses real
hardware instead of the SPDC simulation.

Usage:
    python run_xiao_hardware.py --steps 5000
    python run_xiao_hardware.py --steps 10000 --no-viz --output-dir hw_results
    python run_xiao_hardware.py --port COM4 --steps 2000 --walk-mode angle
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

# Ensure the toolkit root and collectors/ are on sys.path regardless of cwd
_HERE = Path(__file__).resolve().parent
for _p in [str(_HERE), str(_HERE / "collectors"), str(_HERE / "metrics")]:
    if _p not in sys.path:
        sys.path.insert(0, _p)

import numpy as np
import matplotlib
matplotlib.use("Agg")          # headless - saves PNGs without needing a display
import matplotlib.pyplot as plt

from collectors.xiao_serial_source import XIAOQuantumSource

try:
    from helios_anomaly_scope import QRNGStreamScope
    HELIOS_AVAILABLE = True
except ImportError as _e:
    HELIOS_AVAILABLE = False
    print(f"[WARN] helios_anomaly_scope not importable ({_e}); running in entropy-only mode.")


# ------------------------------------------------------------------ #
# Visualizer & plotter (inlined — no dependency on run_qrng_analysis) #
# ------------------------------------------------------------------ #

class QRNGTerminalVisualizer:
    def __init__(self, width: int = 60, height: int = 20):
        self.width = width
        self.height = height
        self.x_range = (-0.5, 0.5)
        self.y_range = (-0.5, 0.5)

    def _scale_point(self, x, y):
        x = max(self.x_range[0], min(self.x_range[1], x))
        y = max(self.y_range[0], min(self.y_range[1], y))
        col = int((x - self.x_range[0]) / (self.x_range[1] - self.x_range[0]) * (self.width - 1))
        row = int((self.y_range[1] - y) / (self.y_range[1] - self.y_range[0]) * (self.height - 1))
        return row, col

    def render(self, x_points, y_points, metrics, hw_stats):
        grid = [[' '] * self.width for _ in range(self.height)]
        cr, cc = self.height // 2, self.width // 2
        for c in range(self.width): grid[cr][c] = '-'
        for r in range(self.height): grid[r][cc] = '|'
        grid[cr][cc] = '+'
        if x_points and y_points:
            xm = max(abs(min(x_points)), abs(max(x_points)), 0.3)
            ym = max(abs(min(y_points)), abs(max(y_points)), 0.3)
            s = max(xm, ym) * 1.2
            self.x_range = (-s, s); self.y_range = (-s, s)
        trail = ['.', '+', '#', '@']
        n = len(x_points)
        for i, (x, y) in enumerate(zip(x_points, y_points)):
            r, c = self._scale_point(x, y)
            if 0 <= r < self.height and 0 <= c < self.width:
                if grid[r][c] in ' -|+':
                    grid[r][c] = trail[min(3, int(i / max(n, 1) * 4))]
        if x_points:
            r, c = self._scale_point(x_points[-1], y_points[-1])
            if 0 <= r < self.height and 0 <= c < self.width:
                grid[r][c] = 'O'
        lines = ['+' + '-' * self.width + '+']
        for row in grid: lines.append('|' + ''.join(row) + '|')
        lines.append('+' + '-' * self.width + '+')
        inf = metrics.get('influence_detected', False)
        lines.append(f"  Step: {metrics.get('step',0):6d}  | {'*** EMERGENCE ***' if inf else 'Normal (Random Walk)'}")
        lines.append(f"  Hurst: {metrics.get('hurst',0.5):.3f}  | MSD: {metrics.get('msd',0):.6f}  | Lyapunov: {metrics.get('lyapunov',0):+.4f}")
        lines.append(f"  Bytes rx: {hw_stats.get('total_bits',0)//8:,}  | Port: {hw_stats.get('port','?')}")
        return '\n'.join(lines)

    def render_progress_bar(self, current, total, width=40):
        pct = current / total
        filled = int(width * pct)
        bar = '#' * filled + '.' * (width - filled)
        return f"  [{bar}] {current}/{total} ({pct*100:.1f}%)"


class QRNGPlotGenerator:
    def __init__(self, output_dir: str = "xiao_hw_results"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_all_plots(self, run_id: str, data: dict) -> List[str]:
        plt.style.use('dark_background')
        paths = []
        paths.append(self._plot_trajectory(run_id, data))
        paths.append(self._plot_randomness(run_id, data))
        return paths

    def _plot_trajectory(self, run_id, data):
        fp = self.output_dir / f"xiao_{run_id}_trajectory.png"
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        x, y = data['trajectory']['x'], data['trajectory']['y']
        n = len(x)
        colors = plt.cm.plasma(np.linspace(0, 1, max(n-1, 1)))
        for i in range(n - 1):
            axes[0].plot([x[i], x[i+1]], [y[i], y[i+1]], color=colors[i], lw=0.5, alpha=0.7)
        if x:
            axes[0].scatter([x[0]], [y[0]], color='green', s=100, zorder=5, label='Start')
            axes[0].scatter([x[-1]], [y[-1]], color='red', s=100, marker='*', zorder=5, label='End')
        axes[0].set_title('Phase Space Trajectory (XIAO Hardware)')
        axes[0].set_xlabel('X'); axes[0].set_ylabel('Y')
        axes[0].legend(); axes[0].grid(True, alpha=0.3)
        raw = data['raw_stream'][:500]
        axes[1].plot(raw, lw=0.5, color='cyan', alpha=0.7)
        axes[1].axhline(0.5, color='yellow', ls='--', alpha=0.5)
        axes[1].set_title('Raw QRNG Stream (first 500)'); axes[1].set_xlabel('Sample')
        axes[1].grid(True, alpha=0.3)
        plt.suptitle(f'XIAO QRNG Hardware — {run_id}', fontsize=13)
        plt.tight_layout()
        plt.savefig(fp, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        return str(fp)

    def _plot_randomness(self, run_id, data):
        fp = self.output_dir / f"xiao_{run_id}_randomness.png"
        fig, axes = plt.subplots(2, 2, figsize=(12, 9))
        arr = np.array(data['raw_stream'])
        axes[0,0].hist(arr, bins=50, color='cyan', alpha=0.7)
        axes[0,0].axvline(0.5, color='yellow', ls='--'); axes[0,0].set_title('Value Distribution')
        n_lag = min(50, len(arr)//4)
        ac = [np.corrcoef(arr[:-l], arr[l:])[0,1] if l else 1.0 for l in range(n_lag)]
        axes[0,1].bar(range(n_lag), ac, color='magenta', alpha=0.7); axes[0,1].set_title('Autocorrelation')
        axes[1,0].hist(np.diff(arr), bins=50, color='green', alpha=0.7); axes[1,0].set_title('Successive Differences')
        axes[1,1].scatter(arr[:-1], arr[1:], alpha=0.3, s=1, color='yellow')
        axes[1,1].set_title('Consecutive Pairs'); axes[1,1].set_aspect('equal')
        plt.suptitle(f'XIAO QRNG Randomness Quality — {run_id}')
        plt.tight_layout()
        plt.savefig(fp, dpi=150, bbox_inches='tight', facecolor='black')
        plt.close()
        return str(fp)

try:
    from file_utils import atomic_write_json
except ImportError:
    def atomic_write_json(data, path):
        Path(path).write_text(json.dumps(data, indent=2))


# ------------------------------------------------------------------ #
# Stub stats for display (replaces qrng.get_statistics())            #
# ------------------------------------------------------------------ #

def _make_hw_stats(source: XIAOQuantumSource) -> Dict:
    return {
        "total_bits": source.total_bytes_received * 8,
        "bit_rate_bps": 115200,          # hardware baud rate
        "coincidences": source.total_bytes_received,
        "coincidence_ratio": 1.0,
        "source": source.SOURCE_NAME,
        "port": source.port,
    }


# ------------------------------------------------------------------ #
# Entropy-only fallback (no Helios deps required)                    #
# ------------------------------------------------------------------ #

def run_entropy_only(source: XIAOQuantumSource, steps: int, output_dir: str) -> Dict:
    """Collect raw values + compute basic stats without Helios."""
    import math
    from collections import Counter

    print(f"\nCollecting {steps} quantum floats from {source.port}...")
    raw: List[float] = []
    for i in range(steps):
        raw.append(source.get_random())
        if (i + 1) % max(1, steps // 20) == 0:
            print(f"  {i+1}/{steps}", end="\r")

    arr = np.array(raw)
    # Convert to bytes for Shannon entropy
    byte_vals = (arr * 255).astype(np.uint8)
    counts = Counter(byte_vals.tolist())
    n = len(byte_vals)
    ent = -sum((c / n) * math.log2(c / n) for c in counts.values() if c > 0)

    print(f"\n  Shannon entropy : {ent:.4f} bits/byte")
    print(f"  Mean            : {arr.mean():.4f}  (expected 0.5000)")
    print(f"  Std             : {arr.std():.4f}  (expected {1/np.sqrt(12):.4f})")

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    result = {
        "run_id": run_id,
        "source": "xiao_serial",
        "port": source.port,
        "steps": steps,
        "entropy_bits_per_byte": ent,
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "raw_stream": raw,
    }
    atomic_write_json(result, out / f"xiao_entropy_{run_id}.json")
    print(f"\n  Saved -> {out / f'xiao_entropy_{run_id}.json'}")
    return result


# ------------------------------------------------------------------ #
# Full Helios run                                                     #
# ------------------------------------------------------------------ #

def run_xiao_analysis(
    port: str = "COM4",
    steps: int = 2000,
    output_dir: str = "xiao_hw_results",
    visualize: bool = True,
    embedding_delay: int = 1,
    walk_mode: str = "angle",
) -> Dict:

    print("\n" + "=" * 70)
    print("  XIAO HARDWARE QRNG - HELIOS TRAJECTORY ANALYSIS")
    print("=" * 70)
    print(f"  Source : Seeed XIAO SAMD21 on {port} @ 115200 baud")
    print(f"  Steps  : {steps}")
    print(f"  Walk   : {walk_mode}")
    print("=" * 70)

    print(f"\nConnecting to XIAO QRNG on {port}...")
    with XIAOQuantumSource(port=port) as source:
        print(f"  Connected. Buffer ready ({source.buffer_size} bytes).\n")

        if not HELIOS_AVAILABLE:
            return run_entropy_only(source, steps, output_dir)

        scope = QRNGStreamScope(
            embedding_dim=2,
            embedding_delay=embedding_delay,
            history_len=min(100, steps // 2),
            walk_mode=walk_mode,
        )

        viz = QRNGTerminalVisualizer(width=60, height=20)
        plotter = QRNGPlotGenerator(output_dir)
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        raw_stream: List[float] = []
        all_metrics: Dict[str, List] = {
            "velocity": [], "msd": [], "coherence": [],
            "hurst": [], "diffusion_exponent": [], "lyapunov": [],
            "influence_detected": [],
        }

        for step in range(steps):
            value = source.get_random()
            raw_stream.append(value)
            metrics = scope.update_from_stream(value)

            if metrics.get("waiting_for_history"):
                continue

            for key in all_metrics:
                all_metrics[key].append(metrics.get(key, 0 if key != "influence_detected" else False))

            if step % max(1, steps // 50) == 0 or step == steps - 1:
                h = metrics.get("hurst", 0.5)
                ly = metrics.get("lyapunov", 0.0)
                inf = "*** EMERGENCE ***" if metrics.get("influence_detected") else "normal"
                pct = (step + 1) / steps * 100
                print(f"  [{pct:5.1f}%] step {step+1:6d}/{steps}  "
                      f"Hurst={h:.3f}  Lyapunov={ly:+.4f}  {inf}  "
                      f"buf={source.buffer_size}B", flush=True)

        # ---- final summary ----
        x_traj, y_traj = scope.get_trajectory()
        summary = scope.get_summary()
        hw_stats = _make_hw_stats(source)
        signal_class = scope.get_signal_classification()
        verification = signal_class["verification"]

        events_list = [
            {
                "step": e.step,
                "event_type": e.event_type,
                "confidence": float(e.confidence),
                "description": e.description,
            }
            for e in scope.get_events()
        ]

        data = {
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "source": "xiao_serial",
            "port": port,
            "config": {"steps": steps, "embedding_delay": embedding_delay, "walk_mode": walk_mode},
            "trajectory": {"x": x_traj, "y": y_traj},
            "raw_stream": raw_stream,
            "metrics": all_metrics,
            "events": events_list,
            "summary": summary,
            "hardware_stats": hw_stats,
        }

        # ---- export ----
        out = Path(output_dir)
        out.mkdir(parents=True, exist_ok=True)

        def _convert(obj):
            if isinstance(obj, np.ndarray): return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)): return float(obj)
            if isinstance(obj, (np.int32, np.int64)): return int(obj)
            if isinstance(obj, (np.bool_, bool)): return bool(obj)
            if isinstance(obj, dict): return {k: _convert(v) for k, v in obj.items()}
            if isinstance(obj, list): return [_convert(v) for v in obj]
            return obj

        json_path = out / f"xiao_run_{run_id}.json"
        atomic_write_json(_convert(data), json_path)
        print(f"\n  JSON  -> {json_path}")

        print("  Generating plots...")
        for p in plotter.generate_all_plots(run_id, data):
            print(f"  Plot  -> {p}")

        # ---- print summary identical to run_qrng_analysis.py ----
        print("\n" + "=" * 70)
        print("  ANALYSIS COMPLETE  (hardware source: XIAO SAMD21)")
        print("=" * 70)
        raw_arr = np.array(raw_stream)
        print(f"  Mean: {raw_arr.mean():.4f}  Std: {raw_arr.std():.4f}")

        if all_metrics["hurst"]:
            h = np.mean(all_metrics["hurst"][-50:])
            label = "random walk" if 0.45 < h < 0.55 else ("persistent" if h > 0.55 else "anti-persistent")
            print(f"  Hurst: {h:.3f} ({label})")

        if all_metrics["diffusion_exponent"]:
            a = np.mean(all_metrics["diffusion_exponent"][-50:])
            label = "diffusive" if 0.8 < a < 1.2 else ("sub-diffusive" if a < 0.8 else "super-diffusive")
            print(f"  Diffusion alpha: {a:.2f} ({label})")

        if verification.is_verified:
            print(f"\n  SIGNAL VERIFIED — {verification.signal_class.value.upper()} "
                  f"({verification.confidence*100:.1f}% confidence)")
        else:
            print(f"\n  No verified signal — {verification.signal_class.value}")

        print("=" * 70 + "\n")
        return data


def main():
    p = argparse.ArgumentParser(description="XIAO Hardware QRNG — Helios analysis")
    p.add_argument("--port", default="COM4", help="Serial port (default: COM4)")
    p.add_argument("--steps", type=int, default=2000)
    p.add_argument("--output-dir", default="xiao_hw_results")
    p.add_argument("--embedding-delay", type=int, default=1)
    p.add_argument("--walk-mode", choices=["angle", "xy_independent", "takens"], default="angle")
    p.add_argument("--no-viz", action="store_true")
    args = p.parse_args()

    run_xiao_analysis(
        port=args.port,
        steps=args.steps,
        output_dir=args.output_dir,
        visualize=not args.no_viz,
        embedding_delay=args.embedding_delay,
        walk_mode=args.walk_mode,
    )


if __name__ == "__main__":
    main()
