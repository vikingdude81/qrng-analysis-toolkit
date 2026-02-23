"""
Phase Space Visualizer
======================

Real-time visualization of trajectory in phase space.
Shows the walker's path and key metrics.

Usage:
    python phase_space_visualizer.py --simulate
    python phase_space_visualizer.py --input stream.csv
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from collections import deque
import argparse
from typing import Optional, Callable, Generator
import time


class PhaseSpaceVisualizer:
    """
    Real-time phase space trajectory visualizer.
    
    Shows:
    - Left: 2D phase space trajectory
    - Right: Metrics over time (MSD, Hurst, Lyapunov)
    """
    
    def __init__(self, history_len: int = 200, update_interval: int = 50):
        self.history_len = history_len
        self.update_interval = update_interval
        
        # Data storage
        self.x_history = deque(maxlen=history_len)
        self.y_history = deque(maxlen=history_len)
        self.msd_history = deque(maxlen=history_len)
        self.hurst_history = deque(maxlen=history_len)
        self.lyapunov_history = deque(maxlen=history_len)
        self.influence_flags = deque(maxlen=history_len)
        
        # Setup figure
        self.fig, (self.ax_phase, self.ax_metrics) = plt.subplots(1, 2, figsize=(14, 6))
        self.fig.suptitle('HELIOS Trajectory Analysis', fontsize=14)
        
        # Phase space plot
        self.ax_phase.set_xlabel('X (Phase 1)')
        self.ax_phase.set_ylabel('Y (Phase 2)')
        self.ax_phase.set_title('Phase Space Trajectory')
        self.ax_phase.set_aspect('equal')
        self.ax_phase.grid(True, alpha=0.3)
        
        # Initialize plot elements
        self.trajectory_line, = self.ax_phase.plot([], [], 'b-', alpha=0.5, linewidth=0.5)
        self.current_point, = self.ax_phase.plot([], [], 'ro', markersize=8)
        self.influence_scatter = self.ax_phase.scatter([], [], c='red', s=20, alpha=0.5)
        
        # Metrics plot
        self.ax_metrics.set_xlabel('Step')
        self.ax_metrics.set_ylabel('Metric Value')
        self.ax_metrics.set_title('Real-time Metrics')
        self.ax_metrics.grid(True, alpha=0.3)
        
        self.msd_line, = self.ax_metrics.plot([], [], 'g-', label='MSD', linewidth=1)
        self.hurst_line, = self.ax_metrics.plot([], [], 'b-', label='Hurst', linewidth=1)
        self.lyapunov_line, = self.ax_metrics.plot([], [], 'r-', label='Lyapunov', linewidth=1)
        self.ax_metrics.axhline(y=0.5, color='blue', linestyle='--', alpha=0.5, label='H=0.5 (random)')
        self.ax_metrics.axhline(y=0, color='red', linestyle='--', alpha=0.5, label='λ=0 (stable)')
        self.ax_metrics.legend(loc='upper left', fontsize=8)
        
        # Status text
        self.status_text = self.ax_phase.text(0.02, 0.98, '', transform=self.ax_phase.transAxes,
                                               verticalalignment='top', fontsize=10,
                                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        
        plt.tight_layout()
        
    def update_data(self, x: float, y: float, msd: float = 0, 
                    hurst: float = 0.5, lyapunov: float = 0,
                    influence: bool = False):
        """Add new data point."""
        self.x_history.append(x)
        self.y_history.append(y)
        self.msd_history.append(msd)
        self.hurst_history.append(hurst)
        self.lyapunov_history.append(lyapunov)
        self.influence_flags.append(influence)
        
    def update_plot(self, frame=None):
        """Update the visualization."""
        if len(self.x_history) < 2:
            return self.trajectory_line, self.current_point
        
        x = list(self.x_history)
        y = list(self.y_history)
        
        # Update trajectory
        self.trajectory_line.set_data(x, y)
        self.current_point.set_data([x[-1]], [y[-1]])
        
        # Update phase space limits
        margin = 0.1
        x_range = max(x) - min(x) + 0.1
        y_range = max(y) - min(y) + 0.1
        max_range = max(x_range, y_range)
        center_x = (max(x) + min(x)) / 2
        center_y = (max(y) + min(y)) / 2
        self.ax_phase.set_xlim(center_x - max_range/2 - margin, center_x + max_range/2 + margin)
        self.ax_phase.set_ylim(center_y - max_range/2 - margin, center_y + max_range/2 + margin)
        
        # Highlight influence points
        influence_x = [x[i] for i in range(len(x)) if self.influence_flags[i]]
        influence_y = [y[i] for i in range(len(y)) if self.influence_flags[i]]
        self.influence_scatter.set_offsets(np.c_[influence_x, influence_y] if influence_x else np.empty((0, 2)))
        
        # Update metrics
        steps = list(range(len(self.msd_history)))
        self.msd_line.set_data(steps, list(self.msd_history))
        self.hurst_line.set_data(steps, list(self.hurst_history))
        self.lyapunov_line.set_data(steps, list(self.lyapunov_history))
        
        if steps:
            self.ax_metrics.set_xlim(0, max(steps) + 1)
            all_vals = list(self.msd_history) + list(self.hurst_history) + list(self.lyapunov_history)
            if all_vals:
                self.ax_metrics.set_ylim(min(all_vals) - 0.1, max(all_vals) + 0.1)
        
        # Update status
        if self.hurst_history:
            status = f"H: {self.hurst_history[-1]:.3f}  λ: {self.lyapunov_history[-1]:.3f}"
            if self.influence_flags[-1]:
                status = "⚠️ INFLUENCE\n" + status
            self.status_text.set_text(status)
        
        return self.trajectory_line, self.current_point, self.msd_line, self.hurst_line, self.lyapunov_line
    
    def run_animation(self, data_generator: Generator, interval: int = 50):
        """
        Run the animation with a data generator.
        
        Args:
            data_generator: Yields (x, y, msd, hurst, lyapunov, influence) tuples
            interval: Update interval in ms
        """
        def animate(frame):
            try:
                data = next(data_generator)
                self.update_data(*data)
                return self.update_plot()
            except StopIteration:
                return self.trajectory_line, self.current_point
        
        self.anim = FuncAnimation(self.fig, animate, interval=interval, blit=False)
        plt.show()


def simulate_random_walk(steps: int = 500, influence_start: Optional[int] = None,
                         step_size: float = 0.1) -> Generator:
    """
    Generate simulated random walk data.
    
    Args:
        steps: Number of steps
        influence_start: Step at which to introduce influence (drift)
        step_size: Size of each step
    """
    from helios_anomaly_scope import HeliosAnomalyScope, QRNGStreamScope
    
    scope = QRNGStreamScope(history_len=100, walk_mode='angle')
    
    x, y = 0.0, 0.0
    
    for step in range(steps):
        # Generate random value
        if influence_start and step >= influence_start:
            # Add bias toward one direction
            value = np.random.random() * 0.3 + 0.6  # Bias toward upper angles
        else:
            value = np.random.random()
        
        # Update scope
        metrics = scope.update_from_stream(value)
        
        if metrics.get('waiting_for_history'):
            continue
        
        yield (
            metrics.get('x', 0),
            metrics.get('y', 0),
            metrics.get('msd', 0),
            metrics.get('hurst', 0.5),
            metrics.get('lyapunov', 0),
            metrics.get('influence_detected', False)
        )


def run_phase_space_animation(simulate: bool = True, influence_start: Optional[int] = None,
                               steps: int = 500):
    """
    Run the phase space animation.
    
    Args:
        simulate: If True, use simulated data
        influence_start: Step to introduce artificial influence
        steps: Number of steps
    """
    viz = PhaseSpaceVisualizer(history_len=200)
    
    if simulate:
        gen = simulate_random_walk(steps=steps, influence_start=influence_start)
        viz.run_animation(gen, interval=50)
    else:
        print("Live data mode not yet implemented")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Phase Space Visualizer')
    parser.add_argument('--simulate', action='store_true', help='Use simulated data')
    parser.add_argument('--influence-start', type=int, default=None,
                        help='Step at which to introduce artificial influence')
    parser.add_argument('--steps', type=int, default=500, help='Number of steps')
    
    args = parser.parse_args()
    
    run_phase_space_animation(
        simulate=args.simulate or True,  # Default to simulate
        influence_start=args.influence_start,
        steps=args.steps
    )
