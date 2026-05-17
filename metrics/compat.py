import sys
from functools import wraps

def wrap_estimator(estimator, fallback=None):
    """Wrap estimator with fallback if unavailable."""
    if not hasattr(sys.modules, estimator):
        return fallback or estimator
    return estimator

def compat_metric(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        est = wrap_estimator(func.__code__.co_name.split('.')[0], fallback="default")
        return func(*args, **kwargs)
    return wrapper