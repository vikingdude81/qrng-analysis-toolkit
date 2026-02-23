"""
Input validation utilities for HELIOS trajectory analysis.

Provides validators for public API parameters to ensure
robust error handling and clear error messages.
"""

from typing import Any, Union
import numpy as np
import torch


class ValidationError(ValueError):
    """Raised when input validation fails."""
    pass


def validate_positive_int(value: Any, name: str, allow_zero: bool = False) -> int:
    """
    Validate positive integer parameter.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        allow_zero: Whether to allow zero

    Returns:
        Validated integer

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, np.integer)):
        raise ValidationError(f"{name} must be an integer, got {type(value).__name__}")

    value = int(value)

    if allow_zero:
        if value < 0:
            raise ValidationError(f"{name} must be non-negative, got {value}")
    else:
        if value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")

    return value


def validate_positive_float(value: Any, name: str, allow_zero: bool = False) -> float:
    """
    Validate positive float parameter.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        allow_zero: Whether to allow zero

    Returns:
        Validated float

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float, np.number)):
        raise ValidationError(f"{name} must be a number, got {type(value).__name__}")

    value = float(value)

    if not np.isfinite(value):
        raise ValidationError(f"{name} must be finite, got {value}")

    if allow_zero:
        if value < 0:
            raise ValidationError(f"{name} must be non-negative, got {value}")
    else:
        if value <= 0:
            raise ValidationError(f"{name} must be positive, got {value}")

    return value


def validate_probability(value: Any, name: str) -> float:
    """
    Validate probability (value in [0, 1]).

    Args:
        value: Value to validate
        name: Parameter name for error messages

    Returns:
        Validated probability

    Raises:
        ValidationError: If validation fails
    """
    value = validate_positive_float(value, name, allow_zero=True)

    if value > 1.0:
        raise ValidationError(f"{name} must be in [0, 1], got {value}")

    return value


def validate_tensor(value: Any, name: str, min_dims: int = 1, max_dims: int = None,
                   allow_empty: bool = False) -> torch.Tensor:
    """
    Validate PyTorch tensor.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_dims: Minimum number of dimensions
        max_dims: Maximum number of dimensions (None for no limit)
        allow_empty: Whether to allow empty tensors

    Returns:
        Validated tensor

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, torch.Tensor):
        raise ValidationError(f"{name} must be a torch.Tensor, got {type(value).__name__}")

    if not allow_empty and value.numel() == 0:
        raise ValidationError(f"{name} cannot be empty")

    ndim = value.ndim
    if ndim < min_dims:
        raise ValidationError(f"{name} must have at least {min_dims} dimensions, got {ndim}")

    if max_dims is not None and ndim > max_dims:
        raise ValidationError(f"{name} must have at most {max_dims} dimensions, got {ndim}")

    return value


def validate_array(value: Any, name: str, min_length: int = 0,
                   dtype: type = None, allow_empty: bool = False) -> np.ndarray:
    """
    Validate numpy array.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_length: Minimum array length
        dtype: Required dtype (None for no restriction)
        allow_empty: Whether to allow empty arrays

    Returns:
        Validated array

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (np.ndarray, list)):
        raise ValidationError(f"{name} must be array-like, got {type(value).__name__}")

    arr = np.asarray(value)

    if not allow_empty and arr.size == 0:
        raise ValidationError(f"{name} cannot be empty")

    if len(arr) < min_length:
        raise ValidationError(f"{name} must have at least {min_length} elements, got {len(arr)}")

    if dtype is not None and arr.dtype != dtype:
        raise ValidationError(f"{name} must have dtype {dtype}, got {arr.dtype}")

    return arr


def validate_choice(value: Any, name: str, choices: list) -> Any:
    """
    Validate value is in allowed choices.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        choices: List of allowed values

    Returns:
        Validated value

    Raises:
        ValidationError: If validation fails
    """
    if value not in choices:
        raise ValidationError(
            f"{name} must be one of {choices}, got {value!r}"
        )

    return value


def validate_range(value: Union[int, float], name: str,
                   min_value: Union[int, float] = None,
                   max_value: Union[int, float] = None) -> Union[int, float]:
    """
    Validate numeric value is within range.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_value: Minimum allowed value (inclusive)
        max_value: Maximum allowed value (inclusive)

    Returns:
        Validated value

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (int, float, np.number)):
        raise ValidationError(f"{name} must be numeric, got {type(value).__name__}")

    if min_value is not None and value < min_value:
        raise ValidationError(f"{name} must be >= {min_value}, got {value}")

    if max_value is not None and value > max_value:
        raise ValidationError(f"{name} must be <= {max_value}, got {value}")

    return value


def validate_list_of_floats(value: Any, name: str, min_length: int = 0) -> list:
    """
    Validate list of floats.

    Args:
        value: Value to validate
        name: Parameter name for error messages
        min_length: Minimum list length

    Returns:
        Validated list

    Raises:
        ValidationError: If validation fails
    """
    if not isinstance(value, (list, tuple)):
        raise ValidationError(f"{name} must be a list, got {type(value).__name__}")

    if len(value) < min_length:
        raise ValidationError(f"{name} must have at least {min_length} elements, got {len(value)}")

    try:
        return [float(x) for x in value]
    except (ValueError, TypeError) as e:
        raise ValidationError(f"{name} must contain numeric values: {e}")
