from typing import Any, Dict, List, Optional, Union
import inspect
import copy

# =============================================================================
# Top-level metrics extracted from consciousness_metrics.py (e.g., integrated_information, epiplexity_score)
# =============================================================================

class ConsciousnessMetrics:
    """
    A class to extract top-level metrics from the `consciousness_metrics` module.
    
    This class provides a standardized interface for accessing key performance indicators
    derived from the `consciousness_metrics.py` source code, including:
        - integrated_information (a measure of information flow)
        - epiplexity_score (a metric representing system complexity or entanglement)
        - other potentially relevant metrics if available in the module
    
    Attributes:
        _metrics_dict: A dictionary mapping specific metric names to their values.
        _source_file: The file path where these metrics were extracted from.
        
    Examples:
        >>> m = ConsciousnessMetrics()
        >>> print(m.get_integrated_information())  # Returns the value of integrated_information
        >>> print(m.get_epiplexity_score())         # Returns the value of epiplexity_score
    """

    def __init__(self):
        self._metrics_dict: Dict[str, Any] = {}
        self._source_file: str = "consciousness_metrics.py"

    @property
    def integrated_information(self) -> float:
        """
        Returns the value of `integrated_information` from the source code.
        
        Raises:
            ValueError: If no value is available for this metric in the source file.
        """
        if "integrated_information" not in self._metrics_dict:
            raise ValueError(
                f"No value found for 'integrated_information' in {self._source_file}. "
                "Please check the source code or add a comment defining it."
            )
        
        return self._metrics_dict["integrated_information"]

    @property
    def epiplexity_score(self) -> float:
        """
        Returns the value of `epiplexity_score` from the source code.
        
        Raises:
            ValueError: If no value is available for this metric in the source file.
        """
        if "epiplexity_score" not in self._metrics_dict:
            raise ValueError(
                f"No value found for 'epiplexity_score' in {self._source_file}. "
                "Please check the source code or add a comment defining it."
            )
        
        return self._metrics_dict["epiplexity_score"]

    @property
    def other_metrics(self) -> Dict[str, Any]:
        """
        Returns all available metrics from the source code.
        
        Raises:
            ValueError: If no value is available for any metric in the source file.
        """
        if "other_metrics" not in self._metrics_dict:
            raise ValueError(
                f"No value found for 'other_metrics' in {self._source_file}. "
                "Please check the source code or add a comment defining it."
            )
        
        return self._metrics_dict["other_metrics"]

    def get_all_metrics(self) -> Dict[str, Any]:
        """
        Returns all available metrics from the source code as a dictionary.
        
        Raises:
            ValueError: If no value is available for any metric in the source file.
        """
        if "all_metrics" not in self._metrics_dict:
            raise ValueError(
                f"No value found for 'all_metrics' in {self._source_file}. "
                "Please check the source code or add a comment defining it."
            )
        
        return self._metrics_dict["all_metrics"]

    def set_source_file(self, file_path: str) -> None:
        """
        Sets the file path where these metrics were extracted from.
        
        Args:
            file_path (str): The path to the source file containing the metrics.
        """
        self._source_file = file_path

    def add_metric(self, name: str, value: Any) -> None:
        """
        Adds a new metric to the metrics dictionary.
        
        Args:
            name (str): The name of the metric.
            value (Any): The value of the metric.
        """
        self._metrics_dict[name] = value

    def clear_metrics(self) -> None:
        """
        Clears all metrics from the metrics dictionary.
        """
        self._metrics_dict.clear()

    def __repr__(self) -> str:
        """
        Returns a string representation of the ConsciousnessMetrics object.
        """
        return f"ConsciousnessMetrics(source_file={self._source_file}, metrics={list(self._metrics_dict.keys())})"
