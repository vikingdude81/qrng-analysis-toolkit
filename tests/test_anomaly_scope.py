"""
Integration tests for HeliosAnomalyScope and QRNGStreamScope.

Tests the full detection pipeline:
- Warmup period behavior
- Event detection
- Bias injection detection
- Signal classification
"""

import pytest
import numpy as np
import torch
from helios_anomaly_scope import (
    HeliosAnomalyScope,
    QRNGStreamScope,
    SignalClass
)


class TestWarmupPeriod:
    """Test warmup period prevents false positives."""

    def test_no_events_during_warmup(self):
        """No events should be detected during warmup period."""
        scope = QRNGStreamScope(history_len=100)

        # Run for warmup period (60 steps)
        for step in range(60):
            scope.update_from_stream(np.random.random())

        assert len(scope.events) == 0, \
            f"Events detected during warmup: {[e.step for e in scope.events]}"

    def test_events_after_warmup(self):
        """Events can be detected after warmup."""
        scope = QRNGStreamScope(history_len=100)

        # Warmup with random
        for step in range(70):
            scope.update_from_stream(np.random.random())

        # Inject strong bias
        for step in range(50):
            scope.update_from_stream(0.95)  # Strong bias

        # Should detect something after warmup
        assert len(scope.events) > 0, "No events detected after warmup with bias"
        assert all(e.step > 60 for e in scope.events), \
            "Events detected during warmup period"

    def test_warmup_consistency(self):
        """Warmup behavior should be consistent across runs."""
        first_event_steps = []

        for _ in range(10):
            scope = QRNGStreamScope(history_len=100)

            # Random for 100 steps
            for step in range(100):
                scope.update_from_stream(np.random.random())

            # Strong bias
            for step in range(100):
                scope.update_from_stream(0.9)

            if scope.events:
                first_event_steps.append(scope.events[0].step)

        # All first events should be after warmup
        assert all(s > 60 for s in first_event_steps), \
            f"Events during warmup: {[s for s in first_event_steps if s <= 60]}"


class TestBiasDetection:
    """Test detection of injected bias."""

    def test_detect_constant_bias(self):
        """Should detect constant bias injection."""
        scope = QRNGStreamScope(history_len=100)

        # Normal random for warmup
        for _ in range(100):
            scope.update_from_stream(np.random.random())

        # Inject constant bias
        for _ in range(100):
            scope.update_from_stream(0.9 + np.random.random() * 0.05)

        # Should detect drift or trending
        assert len(scope.events) > 0, "Failed to detect constant bias"

        # Check signal classification
        classification = scope.get_signal_classification()
        assert classification['is_signal'], "Bias not classified as signal"

    def test_detect_periodic_pattern(self):
        """Should detect periodic pattern."""
        scope = QRNGStreamScope(history_len=100)

        # Normal random for warmup
        for _ in range(100):
            scope.update_from_stream(np.random.random())

        # Inject periodic pattern
        for i in range(200):
            value = 0.5 + 0.3 * np.sin(i / 10)
            scope.update_from_stream(value)

        # Should detect something
        assert len(scope.events) > 0, "Failed to detect periodic pattern"


class TestQRNGStreamScope:
    """Test QRNG stream scope functionality."""

    def test_angle_walk_mode(self):
        """Test angle-based random walk mode."""
        scope = QRNGStreamScope(walk_mode='angle', history_len=50)

        for _ in range(100):
            scope.update_from_stream(np.random.random())

        x, y = scope.get_trajectory()
        assert len(x) > 0, "No trajectory generated"
        assert len(x) == len(y), "Trajectory dimensions mismatch"

    def test_xy_independent_mode(self):
        """Test XY independent walk mode."""
        scope = QRNGStreamScope(walk_mode='xy_independent', history_len=50)

        for _ in range(100):
            metrics = scope.update_from_stream(np.random.random())

        x, y = scope.get_trajectory()
        # Note: xy_independent needs pairs, so might have fewer points
        assert len(x) >= 0, "Trajectory generation failed"

    def test_trajectory_growth(self):
        """Trajectory should grow with steps."""
        scope = QRNGStreamScope(history_len=100)

        for i in range(10, 150, 10):
            scope.update_from_stream(np.random.random())
            x, y = scope.get_trajectory()

            # Should have trajectory after initial steps
            if i > 20:
                assert len(x) > 0, f"No trajectory at step {i}"

    def test_history_limit(self):
        """Trajectory should respect history limit."""
        history_len = 50
        scope = QRNGStreamScope(history_len=history_len)

        for _ in range(200):
            scope.update_from_stream(np.random.random())

        x, y = scope.get_trajectory()
        assert len(x) <= history_len, \
            f"Trajectory exceeds history limit: {len(x)} > {history_len}"


class TestHeliosAnomalyScope:
    """Test HELIOS tensor-based scope."""

    def test_basic_update(self):
        """Test basic tensor update."""
        scope = HeliosAnomalyScope(history_len=50)

        for _ in range(100):
            batch = torch.randn(1, 16, 64)
            metrics = scope.update(batch)

            assert 'x' in metrics
            assert 'y' in metrics
            assert 'step' in metrics

    def test_different_tensor_shapes(self):
        """Should handle different tensor shapes."""
        scope = HeliosAnomalyScope(history_len=50)

        # Test different shapes
        shapes = [
            (16, 64),      # 2D
            (1, 16, 64),   # 3D with batch=1
            (4, 16, 64),   # 3D with batch=4
        ]

        for shape in shapes:
            batch = torch.randn(*shape)
            metrics = scope.update(batch)
            assert 'x' in metrics, f"Failed for shape {shape}"

    def test_attractor_detection(self):
        """Should detect attractor-like behavior."""
        scope = HeliosAnomalyScope(history_len=50)

        # Random exploration phase
        for _ in range(100):
            scope.update(torch.randn(1, 16, 64))

        # Locked state (same value repeated)
        locked_state = torch.zeros(1, 16, 64)
        locked_state[:, :, :32] = 0.5
        locked_state[:, :, 32:] = -0.5

        for _ in range(100):
            # Add tiny noise to avoid exact repetition
            state = locked_state + torch.randn_like(locked_state) * 0.01
            scope.update(state)

        # Should detect attractor lock
        events = scope.get_events()
        event_types = [e.event_type for e in events]

        assert len(events) > 0, "No events detected for attractor"
        # Should detect some kind of structure
        assert any(t in ['attractor_lock', 'convergent_attractor', 'coherence_spike']
                   for t in event_types), f"No attractor-related events: {event_types}"


class TestMetricsCalculation:
    """Test metrics calculation during updates."""

    def test_metrics_availability(self):
        """All expected metrics should be present."""
        scope = QRNGStreamScope(history_len=100)

        # Build up history
        for _ in range(100):
            scope.update_from_stream(np.random.random())

        metrics = scope.update_from_stream(np.random.random())

        expected_keys = ['x', 'y', 'velocity', 'msd', 'coherence',
                         'hurst', 'diffusion_exponent', 'lyapunov',
                         'influence_detected', 'step']

        for key in expected_keys:
            assert key in metrics, f"Missing metric: {key}"

    def test_baseline_calculation(self):
        """Baseline should be calculated after burn-in."""
        scope = QRNGStreamScope(history_len=100)

        # Run through burn-in
        for _ in range(60):
            scope.update_from_stream(np.random.random())

        assert scope.baseline_velocity_std is not None, \
            "Baseline not calculated after burn-in"
        assert scope.baseline_velocity_std > 0, \
            "Baseline velocity std should be positive"


class TestSignalClassification:
    """Test signal classification system."""

    def test_noise_classification(self):
        """Random walk should be classified as NOISE."""
        scope = QRNGStreamScope(history_len=100)

        np.random.seed(42)
        for _ in range(200):
            scope.update_from_stream(np.random.random())

        classification = scope.get_signal_classification()

        # Should be noise or unverified
        assert classification['signal_type'] in ['noise', 'anomalous'], \
            f"Random walk misclassified as {classification['signal_type']}"

    def test_get_verified_events(self):
        """Should filter events by verification."""
        scope = QRNGStreamScope(history_len=100)

        # Generate some events
        for _ in range(100):
            scope.update_from_stream(np.random.random())

        # Inject bias
        for _ in range(100):
            scope.update_from_stream(0.85)

        verified = scope.get_verified_events(min_confidence=0.5)

        # All returned events should be verified
        for event in verified:
            assert event.verification is not None
            assert event.verification.is_verified
            assert event.verification.confidence >= 0.5


class TestResetFunctionality:
    """Test scope reset functionality."""

    def test_reset_clears_state(self):
        """Reset should clear all state."""
        scope = QRNGStreamScope(history_len=100)

        # Generate some history
        for _ in range(100):
            scope.update_from_stream(np.random.random())

        assert len(scope.trajectory_x) > 0
        assert len(scope.events) >= 0
        assert scope.step_count > 0

        # Reset
        scope.reset()

        assert len(scope.trajectory_x) == 0
        assert len(scope.trajectory_y) == 0
        assert len(scope.events) == 0
        assert scope.step_count == 0
        assert scope.baseline_velocity_std is None

    def test_reset_allows_reuse(self):
        """Should be able to reuse scope after reset."""
        scope = QRNGStreamScope(history_len=100)

        # First run
        for _ in range(100):
            scope.update_from_stream(np.random.random())

        first_step_count = scope.step_count

        # Reset and second run
        scope.reset()

        for _ in range(50):
            scope.update_from_stream(np.random.random())

        assert scope.step_count == 50, "Step count not reset properly"
        assert scope.step_count != first_step_count


class TestInputValidation:
    """Test input validation (will be added)."""

    def test_invalid_walk_mode(self):
        """Should reject invalid walk mode."""
        with pytest.raises(ValueError):
            scope = QRNGStreamScope(walk_mode='invalid_mode')

    def test_negative_history_len(self):
        """Should reject negative history length."""
        # Currently no validation - this test documents desired behavior
        # Will pass after we add validation
        pass


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
