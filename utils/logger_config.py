"""
Logging configuration for HELIOS Trajectory Analysis.

Provides centralized logging setup with configurable levels,
formats, and output handlers.
"""

import logging
import logging.handlers
import sys
from pathlib import Path
from typing import Optional


# Default log format
DEFAULT_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
DETAILED_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s'


def setup_logger(
    name: str = 'helios',
    level: int = logging.INFO,
    log_file: Optional[str] = None,
    console: bool = True,
    detailed: bool = False
) -> logging.Logger:
    """
    Setup logger with console and optional file output.

    Args:
        name: Logger name
        level: Logging level (logging.DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Path to log file (None for no file logging)
        console: Whether to log to console
        detailed: Whether to use detailed log format

    Returns:
        Configured logger instance

    Example:
        >>> logger = setup_logger('helios', level=logging.DEBUG, log_file='helios.log')
        >>> logger.info("Analysis started")
        >>> logger.warning("Unusual pattern detected")
    """
    logger = logging.getLogger(name)
    logger.setLevel(level)

    # Remove existing handlers to avoid duplicates
    logger.handlers.clear()

    # Choose format
    fmt = DETAILED_FORMAT if detailed else DEFAULT_FORMAT
    formatter = logging.Formatter(fmt)

    # Console handler
    if console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    # File handler with rotation
    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)

        # Rotating file handler (10MB max, keep 5 backups)
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10 MB
            backupCount=5
        )
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    # Prevent propagation to root logger
    logger.propagate = False

    return logger


def get_logger(name: str = 'helios') -> logging.Logger:
    """
    Get existing logger or create new one with default settings.

    Args:
        name: Logger name

    Returns:
        Logger instance
    """
    logger = logging.getLogger(name)

    # If logger has no handlers, set up with defaults
    if not logger.handlers:
        return setup_logger(name)

    return logger


# Pre-configured loggers for different components
def setup_qrng_logger(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger for QRNG source."""
    return setup_logger('helios.qrng', level=level, log_file=log_file)


def setup_scope_logger(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger for anomaly scope."""
    return setup_logger('helios.scope', level=level, log_file=log_file)


def setup_analysis_logger(level: int = logging.INFO, log_file: Optional[str] = None) -> logging.Logger:
    """Setup logger for analysis runners."""
    return setup_logger('helios.analysis', level=level, log_file=log_file)


# Context manager for temporary log level changes
class LogLevel:
    """
    Context manager to temporarily change log level.

    Example:
        >>> logger = get_logger()
        >>> logger.info("Normal logging")
        >>> with LogLevel(logger, logging.DEBUG):
        ...     logger.debug("This will be logged")
        >>> logger.debug("This won't be logged")
    """

    def __init__(self, logger: logging.Logger, level: int):
        self.logger = logger
        self.new_level = level
        self.old_level = None

    def __enter__(self):
        self.old_level = self.logger.level
        self.logger.setLevel(self.new_level)
        return self.logger

    def __exit__(self, *args):
        self.logger.setLevel(self.old_level)


# Utility functions
def log_function_call(logger: logging.Logger):
    """
    Decorator to log function calls.

    Example:
        >>> logger = get_logger()
        >>> @log_function_call(logger)
        ... def analyze_data(x):
        ...     return x * 2
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            logger.debug(f"Calling {func.__name__} with args={args}, kwargs={kwargs}")
            try:
                result = func(*args, **kwargs)
                logger.debug(f"{func.__name__} returned {result}")
                return result
            except Exception as e:
                logger.error(f"{func.__name__} raised {type(e).__name__}: {e}")
                raise
        return wrapper
    return decorator


def setup_all_loggers(level: int = logging.INFO, log_dir: str = 'logs') -> dict:
    """
    Setup all component loggers at once.

    Args:
        level: Logging level for all loggers
        log_dir: Directory for log files

    Returns:
        Dictionary of logger instances
    """
    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    loggers = {
        'main': setup_logger('helios', level=level,
                             log_file=str(log_path / 'helios.log')),
        'qrng': setup_qrng_logger(level=level,
                                   log_file=str(log_path / 'qrng.log')),
        'scope': setup_scope_logger(level=level,
                                     log_file=str(log_path / 'scope.log')),
        'analysis': setup_analysis_logger(level=level,
                                          log_file=str(log_path / 'analysis.log')),
    }

    return loggers


if __name__ == "__main__":
    # Demo/test
    print("Setting up logging demo...")

    logger = setup_logger('helios.demo', level=logging.DEBUG,
                          log_file='logs/demo.log', detailed=True)

    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    logger.critical("This is a critical message")

    print("\nLog levels demo:")
    with LogLevel(logger, logging.ERROR):
        logger.info("This won't show (level temporarily ERROR)")
        logger.error("This will show")

    logger.info("Back to normal INFO level")

    print("\nDone! Check logs/demo.log for file output")
