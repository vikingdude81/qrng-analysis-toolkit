"""
Tests for ConsciousnessMetrics extractor.
"""

import pytest
from typing import Any, Dict

from inference_framework.consciousness_metrics_extractor import ConsciousnessMetrics


class TestConsciousnessMetrics:
    """Test cases for ConsciousnessMetrics class."""

    def test_init_default(self):
        """Test initialization with default values."""
        m = ConsciousnessMetrics()
        assert m._source_file == "consciousness_metrics.py"
        assert isinstance(m._metrics_dict, dict)

    def test_set_source_file(self):
        """Test setting source file path."""
        m = ConsciousnessMetrics()
        m.set_source_file("/path/to/consciousness_metrics.py")
        assert m._source_file == "/path/to/consciousness_metrics.py"

    def test_add_metric(self):
        """Test adding a metric."""
        m = ConsciousnessMetrics()
        m.add_metric("test_metric", 42.0)
        assert "test_metric" in m._metrics_dict
        assert m._metrics_dict["test_metric"] == 42.0

    def test_clear_metrics(self):
        """Test clearing all metrics."""
        m = ConsciousnessMetrics()
        m.add_metric("metric1", 1.0)
        m.add_metric("metric2", 2.0)
        m.clear_metrics()
        assert len(m._metrics_dict) == 0

    def test_get_integrated_information_missing(self):
        """Test accessing missing integrated_information raises ValueError."""
        m = ConsciousnessMetrics()
        with pytest.raises(ValueError, match="integrated_information"):
            _ = m.integrated_information

    def test_get_epiplexity_score_missing(self):
        """Test accessing missing epiplexity_score raises ValueError."""
        m = ConsciousnessMetrics()
        with pytest.raises(ValueError, match="epiplexity_score"):
            _ = m.epiplexity_score

    def test_get_other_metrics_missing(self):
        """Test accessing missing other_metrics raises ValueError."""
        m = ConsciousnessMetrics()
        with pytest.raises(ValueError, match="other_metrics"):
            _ = m.other_metrics

    def test_get_all_metrics_missing(self):
        """Test accessing missing all_metrics raises ValueError."""
        m = ConsciousnessMetrics()
        with pytest.raises(ValueError, match="all_metrics"):
            _ = m.get_all_metrics()

    def test_repr(self):
        """Test string representation."""
        m = ConsciousnessMetrics()
        repr_str = repr(m)
        assert "ConsciousnessMetrics" in repr_str
        assert "consciousness_metrics.py" in repr_str
