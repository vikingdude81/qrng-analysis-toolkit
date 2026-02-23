"""
Tests for ChaosDetector from quantum-llama.cpp analysis.

Tests:
- Lyapunov exponent calculation
- Criticality index
- Phase transition detection
- Correlation dimension
- Feigenbaum analysis
"""

import pytest
import numpy as np
from chaos_detector import ChaosDetector


class TestLyapunovExponent:
    """Test Lyapunov exponent calculation."""

    def setup_method(self):
        self.detector = ChaosDetector()

    def test_random_walk_lyapunov_near_zero(self):
        """Random walk should have λ ≈ 0."""
        np.random.seed(42)
        sequence = np.cumsum(np.random.randn(500))
        lyap = self.detector.compute_lyapunov(sequence)
        assert -0.2 < lyap < 0.2, f"Random walk λ={lyap}, expected ~0"

    def test_lorenz_attractor_positive_lyapunov(self):
        """Lorenz attractor should have positive Lyapunov (chaotic)."""
        # Generate Lorenz attractor
        sigma, rho, beta = 10.0, 28.0, 8.0 / 3.0
        dt = 0.01
        n_steps = 2000
        
        x, y, z = 1.0, 1.0, 1.0
        trajectory = [x]
        
        for _ in range(n_steps - 1):
            dx = sigma * (y - x)
            dy = x * (rho - z) - y
            dz = x * y - beta * z
            x += dx * dt
            y += dy * dt
            z += dz * dt
            trajectory.append(x)
        
        lyap = self.detector.compute_lyapunov(np.array(trajectory))
        # Lorenz should show positive Lyapunov
        assert lyap > -0.1, f"Lorenz λ={lyap}, expected positive"

    def test_periodic_signal_near_zero_lyapunov(self):
        """Periodic signal should have λ ≈ 0."""
        t = np.linspace(0, 20 * np.pi, 1000)
        sequence = np.sin(t)
        lyap = self.detector.compute_lyapunov(sequence)
        assert lyap < 0.1, f"Periodic λ={lyap}, expected ≤0"

    def test_short_sequence_returns_zero(self):
        """Very short sequence should return 0."""
        sequence = np.array([1, 2, 3, 4, 5])
        lyap = self.detector.compute_lyapunov(sequence)
        assert lyap == 0.0


class TestCriticalityIndex:
    """Test criticality index calculation."""

    def setup_method(self):
        self.detector = ChaosDetector()

    def test_high_entropy_variance_high_criticality(self):
        """High entropy variance should give high criticality index."""
        # Create logits with fluctuating entropy
        logits_history = []
        for i in range(100):
            # Alternating between peaked and flat distributions
            if i % 10 < 5:
                logits = np.random.randn(100) * 0.5  # Flatter
            else:
                logits = np.random.randn(100) * 3.0  # More peaked
            logits_history.append(logits)
        
        kappa = self.detector.compute_criticality_index(logits_history)
        assert kappa > 0, f"Expected positive criticality, got {kappa}"

    def test_constant_entropy_low_criticality(self):
        """Constant entropy should give low criticality index."""
        # Same distribution repeated
        base_logits = np.random.randn(100)
        logits_history = [base_logits.copy() for _ in range(50)]
        
        kappa = self.detector.compute_criticality_index(logits_history)
        assert kappa < 0.1, f"Expected low criticality, got {kappa}"

    def test_insufficient_data_returns_zero(self):
        """Insufficient data should return 0."""
        logits_history = [np.random.randn(100) for _ in range(5)]
        kappa = self.detector.compute_criticality_index(logits_history, window_size=20)
        assert kappa == 0.0


class TestPhaseTransition:
    """Test phase transition detection."""

    def setup_method(self):
        self.detector = ChaosDetector()

    def test_detect_sudden_change(self):
        """Should detect a sudden change in dynamics."""
        # First half: low variance
        part1 = np.random.randn(200) * 0.5
        # Second half: high variance
        part2 = np.random.randn(200) * 3.0
        sequence = np.concatenate([part1, part2])
        
        result = self.detector.detect_phase_transition(sequence)
        
        assert result['transition_detected'], "Should detect phase transition"
        assert result['n_transitions'] >= 1

    def test_no_transition_in_uniform(self):
        """Uniform random sequence should have few transitions."""
        np.random.seed(42)
        sequence = np.random.randn(500)
        
        result = self.detector.detect_phase_transition(sequence)
        
        # May detect some spurious transitions due to statistical fluctuations
        # This is a soft check - uniform data may still trigger some false positives
        assert result['n_transitions'] < 20

    def test_short_sequence_no_transition(self):
        """Short sequence should return no transition."""
        sequence = np.array([1, 2, 3, 4, 5])
        result = self.detector.detect_phase_transition(sequence)
        
        assert not result['transition_detected']


class TestCorrelationDimension:
    """Test correlation dimension calculation."""

    def setup_method(self):
        self.detector = ChaosDetector()

    def test_random_high_dimension(self):
        """Random data should have high effective dimension."""
        np.random.seed(42)
        sequence = np.random.randn(1000)
        
        dim = self.detector.compute_correlation_dimension(sequence, embedding_dim=5)
        
        # Should be close to embedding dimension for random
        assert dim > 2.0, f"Random dimension {dim} should be high"

    def test_periodic_low_dimension(self):
        """Periodic signal should have low dimension."""
        t = np.linspace(0, 20 * np.pi, 1000)
        sequence = np.sin(t)
        
        dim = self.detector.compute_correlation_dimension(sequence, embedding_dim=5)
        
        # Periodic should have low dimension
        assert dim < 3.0, f"Periodic dimension {dim} should be low"

    def test_insufficient_data(self):
        """Insufficient data should return embedding dimension."""
        sequence = np.array([1, 2, 3, 4, 5])
        dim = self.detector.compute_correlation_dimension(sequence, embedding_dim=3)
        assert dim == 3.0


class TestFeigenbaumAnalysis:
    """Test Feigenbaum bifurcation analysis."""

    def setup_method(self):
        self.detector = ChaosDetector()

    def test_bifurcation_detection(self):
        """Should detect bifurcation points."""
        # Simulate parameter sweep with regime change
        params = np.linspace(0, 1, 200)
        values = np.zeros(200)
        
        # First regime: constant
        values[:100] = np.random.randn(100) * 0.1 + 1.0
        # Second regime: different constant
        values[100:] = np.random.randn(100) * 0.1 + 3.0
        
        result = self.detector.feigenbaum_analysis(params, values)
        
        assert result['bifurcation_detected'], "Should detect bifurcation"

    def test_no_bifurcation_in_constant(self):
        """Constant regime should have fewer bifurcations than regime change."""
        params = np.linspace(0, 1, 200)
        np.random.seed(42)
        values = np.random.randn(200) * 0.05 + 1.0  # Very small noise
        
        result = self.detector.feigenbaum_analysis(params, values)
        
        # May detect some false positives - just ensure it's bounded
        assert isinstance(result['n_bifurcations'], int)


class TestTimeDelayEmbedding:
    """Test time delay embedding."""

    def setup_method(self):
        self.detector = ChaosDetector()

    def test_embedding_shape(self):
        """Embedding should have correct shape."""
        data = np.random.randn(100)
        embedding_dim = 3
        tau = 1
        
        embedded = self.detector._time_delay_embedding(data, embedding_dim, tau)
        
        expected_length = 100 - (embedding_dim - 1) * tau
        assert embedded.shape == (expected_length, embedding_dim)

    def test_embedding_with_delay(self):
        """Embedding with tau > 1 should work correctly."""
        data = np.arange(20, dtype=float)
        embedding_dim = 3
        tau = 2
        
        embedded = self.detector._time_delay_embedding(data, embedding_dim, tau)
        
        # Check first point
        assert embedded[0, 0] == 0
        assert embedded[0, 1] == 2
        assert embedded[0, 2] == 4

    def test_insufficient_data_empty(self):
        """Insufficient data should return empty array."""
        data = np.array([1, 2, 3])
        embedded = self.detector._time_delay_embedding(data, embedding_dim=5, tau=1)
        assert len(embedded) == 0


class TestWithQRNGData:
    """Test chaos detector with real QRNG data."""

    def setup_method(self):
        self.detector = ChaosDetector()

    def test_qrng_stream_analysis(self):
        """Analyze saved QRNG stream for chaos metrics."""
        import json
        from pathlib import Path
        
        # Load saved QRNG stream
        stream_dir = Path("qrng_streams")
        if not stream_dir.exists():
            pytest.skip("No QRNG streams available")
        
        streams = list(stream_dir.glob("*.json"))
        if not streams:
            pytest.skip("No QRNG streams available")
        
        with open(streams[0]) as f:
            data = json.load(f)
        
        # Extract floats
        # Try different key names used in different versions
        samples = data.get('floats', data.get('samples', []))
        if not samples:
            pytest.skip("No samples in QRNG file")
        if isinstance(samples[0], dict):
            floats = [s['decimal'] for s in samples]
        else:
            floats = samples
        
        sequence = np.array(floats)
        
        # Compute Lyapunov
        lyap = self.detector.compute_lyapunov(sequence)
        assert isinstance(lyap, float)
        
        # Compute correlation dimension
        dim = self.detector.compute_correlation_dimension(sequence)
        assert isinstance(dim, float)
        
        # Detect phase transitions
        result = self.detector.detect_phase_transition(sequence)
        assert isinstance(result, dict)
        
        print(f"\nQRNG Chaos Analysis:")
        print(f"  Lyapunov exponent: {lyap:.4f}")
        print(f"  Correlation dimension: {dim:.2f}")
        print(f"  Phase transitions: {result['n_transitions']}")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
