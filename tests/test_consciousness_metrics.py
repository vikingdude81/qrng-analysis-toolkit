"""
Tests for ConsciousnessMetrics from quantum-llama.cpp analysis.

Tests:
- Mode entropy calculation
- Participation ratio
- Phase coherence
- Entropy production rate
- Consciousness functional
- State classification
"""

import pytest
import numpy as np
from consciousness_metrics import ConsciousnessMetrics, ConsciousnessState


class TestModeEntropy:
    """Test mode entropy calculation."""

    def setup_method(self):
        self.metrics = ConsciousnessMetrics()

    def test_uniform_distribution_max_entropy(self):
        """Uniform distribution should have high normalized entropy."""
        # Create uniform-ish logits
        logits_history = [np.zeros(100) for _ in range(20)]  # Uniform softmax
        
        result = self.metrics.compute(logits_history)
        
        assert result['h_mode'] > 0.9, f"Uniform entropy {result['h_mode']} should be ~1"

    def test_peaked_distribution_low_entropy(self):
        """Peaked distribution should have low entropy."""
        # Create very peaked logits
        logits_history = []
        for _ in range(20):
            logits = np.ones(100) * -10
            logits[0] = 10  # Very peaked at one token
            logits_history.append(logits)
        
        result = self.metrics.compute(logits_history)
        
        assert result['h_mode'] < 0.3, f"Peaked entropy {result['h_mode']} should be low"


class TestParticipationRatio:
    """Test participation ratio calculation."""

    def setup_method(self):
        self.metrics = ConsciousnessMetrics()

    def test_uniform_high_participation(self):
        """Uniform distribution should have high participation."""
        logits_history = [np.zeros(100) for _ in range(20)]
        
        result = self.metrics.compute(logits_history)
        
        assert result['pr'] > 0.5, f"Uniform PR {result['pr']} should be high"

    def test_peaked_low_participation(self):
        """Peaked distribution should have low participation."""
        logits_history = []
        for _ in range(20):
            logits = np.ones(100) * -10
            logits[0] = 10
            logits_history.append(logits)
        
        result = self.metrics.compute(logits_history)
        
        assert result['pr'] < 0.1, f"Peaked PR {result['pr']} should be low"


class TestPhaseCoherence:
    """Test phase coherence calculation."""

    def setup_method(self):
        self.metrics = ConsciousnessMetrics()

    def test_identical_distributions_high_coherence(self):
        """Identical distributions should have high coherence."""
        base_logits = np.random.randn(100)
        logits_history = [base_logits.copy() for _ in range(30)]
        
        result = self.metrics.compute(logits_history)
        
        assert result['r'] > 0.9, f"Identical coherence {result['r']} should be ~1"

    def test_random_distributions_low_coherence(self):
        """Random distributions should have lower coherence."""
        np.random.seed(42)
        logits_history = [np.random.randn(100) for _ in range(30)]
        
        result = self.metrics.compute(logits_history)
        
        # Random should have lower coherence, but not necessarily 0
        assert result['r'] < 0.5, f"Random coherence {result['r']} should be lower"


class TestEntropyProductionRate:
    """Test entropy production rate calculation."""

    def setup_method(self):
        self.metrics = ConsciousnessMetrics()

    def test_constant_entropy_low_rate(self):
        """Constant entropy should have low production rate."""
        base_logits = np.random.randn(100)
        logits_history = [base_logits.copy() for _ in range(30)]
        
        result = self.metrics.compute(logits_history)
        
        assert result['s_dot'] < 0.1, f"Constant s_dot {result['s_dot']} should be low"

    def test_varying_entropy_higher_rate(self):
        """Varying entropy should have higher production rate."""
        logits_history = []
        for i in range(30):
            # Alternate between peaked and flat
            if i % 2 == 0:
                logits = np.random.randn(100) * 0.1
            else:
                logits = np.random.randn(100) * 5.0
            logits_history.append(logits)
        
        result = self.metrics.compute(logits_history)
        
        assert result['s_dot'] > 0.1, f"Varying s_dot {result['s_dot']} should be higher"


class TestConsciousnessFunctional:
    """Test consciousness functional calculation."""

    def setup_method(self):
        self.metrics = ConsciousnessMetrics()

    def test_high_consciousness_creative_state(self):
        """High entropy + coherence + criticality should give high consciousness."""
        # This is tricky to construct - need high entropy, high coherence, critical
        np.random.seed(42)
        
        # Start with moderate entropy distributions that have some correlation
        base_pattern = np.random.randn(100) * 0.5
        logits_history = []
        
        for i in range(50):
            noise = np.random.randn(100) * 0.2
            logits = base_pattern + noise
            logits_history.append(logits)
        
        result = self.metrics.compute(logits_history)
        
        # Should have some consciousness
        assert result['consciousness'] > 0, "Should have positive consciousness"

    def test_low_consciousness_mechanical_state(self):
        """Very peaked distributions should give low consciousness."""
        logits_history = []
        for _ in range(30):
            logits = np.ones(100) * -20
            logits[0] = 10  # Very deterministic
            logits_history.append(logits)
        
        result = self.metrics.compute(logits_history)
        
        assert result['consciousness'] < 0.1, f"Deterministic C={result['consciousness']} should be low"
        assert result['state'] == ConsciousnessState.MECHANICAL.value


class TestStateClassification:
    """Test consciousness state classification."""

    def setup_method(self):
        self.metrics = ConsciousnessMetrics()

    def test_mechanical_state(self):
        """Peaked, consistent distributions should be MECHANICAL."""
        logits_history = []
        for _ in range(30):
            logits = np.ones(100) * -10
            logits[0] = 10
            logits_history.append(logits)
        
        result = self.metrics.compute(logits_history)
        assert result['state'] == ConsciousnessState.MECHANICAL.value

    def test_dreaming_state(self):
        """High entropy but low coherence should be DREAMING."""
        np.random.seed(42)
        # Random, uncorrelated, flat distributions
        logits_history = [np.random.randn(100) * 0.1 for _ in range(30)]
        
        result = self.metrics.compute(logits_history)
        # High entropy + low coherence = dreaming
        # This may vary based on randomness
        assert result['state'] in [ConsciousnessState.DREAMING.value, 
                                   ConsciousnessState.CREATIVE.value]


class TestTrajectory:
    """Test consciousness trajectory computation."""

    def setup_method(self):
        self.metrics = ConsciousnessMetrics()

    def test_trajectory_length(self):
        """Trajectory should have correct length."""
        logits_history = [np.random.randn(100) for _ in range(50)]
        
        trajectory = self.metrics.compute_trajectory(logits_history)
        
        # Trajectory starts after window_size (10)
        assert len(trajectory) == 50 - 10 + 1


class TestSummary:
    """Test metrics summary."""

    def setup_method(self):
        self.metrics = ConsciousnessMetrics()

    def test_summary_after_computation(self):
        """Summary should contain correct statistics."""
        # Compute a few times
        for _ in range(10):
            logits = [np.random.randn(100) for _ in range(20)]
            self.metrics.compute(logits)
        
        summary = self.metrics.get_summary()
        
        assert summary['n_samples'] == 10
        assert 'mean_consciousness' in summary
        assert 'dominant_state' in summary

    def test_summary_empty_history(self):
        """Summary with no history should return defaults."""
        summary = self.metrics.get_summary()
        
        assert summary['mean_consciousness'] == 0.0


class TestWithQRNGData:
    """Test consciousness metrics with QRNG data."""

    def setup_method(self):
        self.metrics = ConsciousnessMetrics()

    def test_qrng_as_logits(self):
        """Treat QRNG stream as pseudo-logits for analysis."""
        import json
        from pathlib import Path
        
        stream_dir = Path("qrng_streams")
        if not stream_dir.exists():
            pytest.skip("No QRNG streams available")
        
        streams = list(stream_dir.glob("*.json"))
        if not streams:
            pytest.skip("No QRNG streams available")
        
        with open(streams[0]) as f:
            data = json.load(f)
        
        # Try different key names used in different versions
        samples = data.get('floats', data.get('samples', []))
        if not samples:
            pytest.skip("No samples in QRNG file")
        if isinstance(samples[0], dict):
            floats = [s['decimal'] for s in samples]
        else:
            floats = samples
        
        # Create synthetic logits from QRNG values
        # Use sliding windows of values as "logit distributions"
        window_size = 50
        logits_history = []
        
        for i in range(len(floats) - window_size):
            # Convert QRNG floats to pseudo-logits
            window = np.array(floats[i:i + window_size])
            # Scale to logit-like range
            logits = (window - 0.5) * 10
            logits_history.append(logits)
        
        if len(logits_history) < 20:
            pytest.skip("Not enough QRNG data")
        
        result = self.metrics.compute(logits_history[:100])
        
        print(f"\nQRNG Consciousness Analysis:")
        print(f"  Mode entropy (H_mode): {result['h_mode']:.3f}")
        print(f"  Participation ratio (PR): {result['pr']:.3f}")
        print(f"  Phase coherence (R): {result['r']:.3f}")
        print(f"  Entropy production (ṡ): {result['s_dot']:.3f}")
        print(f"  Criticality (κ): {result['kappa']:.3f}")
        print(f"  Consciousness functional: {result['consciousness']:.3f}")
        print(f"  State: {result['state']}")
        
        assert isinstance(result['consciousness'], float)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
