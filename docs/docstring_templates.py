"""Documentation templates for public classes and functions in `qrng_comprehensive_analysis.py` and `epiplexity_estimator.py`.

All docstrings follow NumPy-style format (Triple quotes, parameters, return value, examples).
"""

# Example function template
def example_function(
    param1: str,
    param2: int = 0,
    param3: float = 1.0
) -> bool:
    """Example function description.

    Parameters
    ----------
    param1 : str
        Description of param1.
    param2 : int, optional
        Description of param2 (default: 0).
    param3 : float, optional
        Description of param3 (default: 1.0).

    Returns
    -------
    bool
        True if condition met, False otherwise.

    Examples
    --------
    >>> example_function("test")
    True
    """
    return True

# Example class template
class ExampleClass:
    """Example class description."""

    def __init__(self, value: int):
        """Initialize the class with a value."""
        self.value = value

    def get_value(self) -> int:
        """Return the stored value."""
        return self.value
