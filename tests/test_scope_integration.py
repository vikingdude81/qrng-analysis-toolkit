"""
Integration tests for QRNGStreamScope.

Tests the full pipeline:
- Stream input processing
- Metric calculation
- Event generation
- Signal verification
"""

import pytest
import numpy as np
import sys
from helios_anomaly_scope import (
    QRNGStreamScope,
    AnomalyEvent,
    SignalClass,
    SignalVerification,
)


class TestScopeBasics:
    """Basic scope functionality tests."""
    
    def test_scope_initialization(self):
        """Scope should initialize with default parameters."""
        scope = QRNGStreamScope()
        assert scope.step == 0
        assert len(scope.events) == 0
        assert scope.trajectory_x == []
        assert scope.trajectory_y == []
    
    def test_scope_accepts_stream_values(self, scope):
        """Scope should accept stream values without error."""
        for _ in range(100):
            result = scope.update_from_stream(np.random.random())
            assert isinstance(result, dict)
    
    def test_scope_tracks_step_count(self, scope):
        """Scope should correctly count steps."""
        for i in range(50):
            scope.update_from_stream(0.5)
        assert scope.step == 50
    
    def test_scope_builds_trajectory(self, scope):
        """Scope should build 2D trajectory from 1D stream."""
        for _ in range(100):
            scope.update_from_stream(np.random.random())
        
        # Should have trajectory points
        assert len(scope.trajectory_x) > 0
        assert len(scope.trajectory_y) > 0
        assert len(scope.trajectory_x) == len(scope.trajectory_y)


class TestScopeMetrics:
    """Test metric calculation in scope."""
    
    def test_metrics_returned(self, scope):
        """Update should return metrics dict."""
        # Need enough steps for metrics
        for _ in range(100):
            result = scope.update_from_stream(np.random.random())
        
        # Should have various metrics
        assert 'step' in result
        assert isinstance(result.get('hurst', 0), (int, float))
        assert isinstance(result.get('lyapunov', 0), (int, float))
    
    def test_hurst_in_valid_range(self, scope):
        """Hurst exponent should be in [0, 1]."""
        for _ in range(200):
            result = scope.update_from_stream(np.random.random())
        
        hurst = result.get('hurst', 0.5)
        assert 0 <= hurst <= 1, f"Hurst={hurst} out of range"
    
    def test_diffusion_exponent_reasonable(self, scope):
        """Diffusion exponent should be reasonable."""
        for _ in range(200):
            result = scope.update_from_stream(np.random.random())
        
        alpha = result.get('diffusion_exponent', 1.0)
        # For random walk, should be near 1
        assert -1 <= alpha <= 5, f"Alpha={alpha} unreasonable"


class TestScopeEvents:
    """Test event generation."""
    
    def test_events_are_anomaly_event_type(self, scope):
        """All events should be AnomalyEvent instances."""
        # Force some events by injecting bias
        for _ in range(80):
            scope.update_from_stream(np.random.random())
        for _ in range(100):
            scope.update_from_stream(0.9)  # Strong bias
        
        for event in scope.events:
            assert isinstance(event, AnomalyEvent)
    
    def test_event_has_required_fields(self, scope):
        """Events should have step, event_type, confidence."""
        for _ in range(80):
            scope.update_from_stream(np.random.random())
        for _ in range(100):
            scope.update_from_stream(0.9)
        
        if scope.events:
            event = scope.events[0]
            assert hasattr(event, 'step')
            assert hasattr(event, 'event_type')
            assert hasattr(event, 'confidence')
            assert isinstance(event.step, int)
            assert isinstance(event.event_type, str)
    
    def test_no_events_in_warmup(self, scope):
        """No events should occur in warmup period (step <= 60)."""
        for _ in range(200):
            scope.update_from_stream(np.random.random())
        
        warmup_events = [e for e in scope.events if e.step <= 60]
        assert len(warmup_events) == 0, f"Found {len(warmup_events)} events in warmup"


class TestScopeWalkModes:
    """Test different random walk embedding modes."""
    
    def test_angle_walk_mode(self):
        """Angle walk mode should work."""
        scope = QRNGStreamScope(walk_mode='angle')
        for _ in range(100):
            scope.update_from_stream(np.random.random())
        assert len(scope.trajectory_x) > 0
    
    def test_xy_independent_walk_mode(self):
        """XY independent mode should work."""
        scope = QRNGStreamScope(walk_mode='xy_independent')
        for _ in range(100):
            scope.update_from_stream(np.random.random())
        assert len(scope.trajectory_x) > 0
    
    def test_takens_walk_mode(self):
        """Takens embedding mode should work."""
        scope = QRNGStreamScope(walk_mode='takens')
        for _ in range(100):
            scope.update_from_stream(np.random.random())
        assert len(scope.trajectory_x) > 0


class TestScopeReset:
    """Test scope reset functionality."""
    
    def test_reset_clears_state(self, scope):
        """Reset should clear all state."""
        for _ in range(100):
            scope.update_from_stream(np.random.random())
        
        scope.reset()
        
        assert scope.step == 0
        assert len(scope.events) == 0
        assert len(scope.trajectory_x) == 0


class TestScopeEdgeCases:
    """Test edge cases."""
    
    def test_value_at_boundary_zero(self, scope):
        """Value of exactly 0 should work."""
        scope.update_from_stream(0.0)
        assert scope.step == 1
    
    def test_value_at_boundary_one(self, scope):
        """Value of exactly 1 should work (clipped to < 1)."""
        scope.update_from_stream(1.0)
        assert scope.step == 1
    
    def test_negative_value_handled(self, scope):
        """Negative values should be handled (clipped to 0)."""
        scope.update_from_stream(-0.5)
        assert scope.step == 1
    
    def test_value_above_one_handled(self, scope):
        """Values > 1 should be handled (clipped)."""
        scope.update_from_stream(1.5)
        assert scope.step == 1
    
    def test_many_updates(self, scope):
        """Should handle many updates without issue."""
        for _ in range(5000):
            scope.update_from_stream(np.random.random())
        
        assert scope.step == 5000
        assert len(scope.trajectory_x) <= 5000  # May have windowing
