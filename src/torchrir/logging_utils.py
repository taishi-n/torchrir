from __future__ import annotations

"""Logging helpers for torchrir."""

from dataclasses import dataclass, replace
import logging
from typing import Optional


@dataclass(frozen=True)
class LoggingConfig:
    """Configuration for torchrir logging.

    Example:
        >>> config = LoggingConfig(level="INFO")
        >>> logger = setup_logging(config)
    """

    level: str | int = "INFO"
    format: str = "%(levelname)s:%(name)s:%(message)s"
    datefmt: Optional[str] = None
    propagate: bool = False

    def resolve_level(self) -> int:
        """Resolve level to a logging integer constant."""
        if isinstance(self.level, int):
            return self.level
        if not isinstance(self.level, str):
            raise TypeError("level must be str or int")
        name = self.level.upper()
        if name not in logging._nameToLevel:
            raise ValueError(f"unknown log level: {self.level}")
        return logging._nameToLevel[name]

    def replace(self, **kwargs) -> "LoggingConfig":
        """Return a new config with updated fields."""
        return replace(self, **kwargs)


def setup_logging(config: LoggingConfig, *, name: str = "torchrir") -> logging.Logger:
    """Configure and return the base torchrir logger.

    Example:
        >>> logger = setup_logging(LoggingConfig(level="DEBUG"))
        >>> logger.info("ready")
    """
    logger = logging.getLogger(name)
    level = config.resolve_level()
    logger.setLevel(level)
    logger.propagate = config.propagate
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setLevel(level)
        handler.setFormatter(logging.Formatter(config.format, datefmt=config.datefmt))
        logger.addHandler(handler)
    return logger


def get_logger(name: Optional[str] = None) -> logging.Logger:
    """Return a torchrir logger, namespaced under the torchrir root.

    Example:
        >>> logger = get_logger("examples.static")
    """
    if not name:
        return logging.getLogger("torchrir")
    if name.startswith("torchrir"):
        return logging.getLogger(name)
    return logging.getLogger(f"torchrir.{name}")
