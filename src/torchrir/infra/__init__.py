"""Infrastructure utilities (logging configuration and helpers).

Example:
    >>> from torchrir import LoggingConfig, get_logger, setup_logging
    >>> setup_logging(LoggingConfig(level="INFO"))
    >>> logger = get_logger("examples")
    >>> logger.info("running torchrir example")
"""

from .logging import LoggingConfig, get_logger, setup_logging

__all__ = [
    "LoggingConfig",
    "get_logger",
    "setup_logging",
]
