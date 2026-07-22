"""Logging helpers for torchrir."""

from __future__ import annotations

from dataclasses import dataclass, replace
import logging
import threading
from typing import Any


_setup_lock = threading.RLock()


@dataclass(frozen=True, slots=True, kw_only=True)
class LoggingConfig:
    """Configuration for torchrir logging.

    Examples:
        ```python
        config = LoggingConfig(level="INFO")
        logger = setup_logging(config)
        ```
    """

    level: str | int = "INFO"
    format: str = "%(levelname)s:%(name)s:%(message)s"
    datefmt: str | None = None

    def __post_init__(self) -> None:
        if isinstance(self.level, bool) or not isinstance(self.level, (str, int)):
            raise TypeError("level must be str or int")
        if not isinstance(self.format, str):
            raise TypeError("format must be a string")
        if not self.format:
            raise ValueError("format must be a non-empty string")
        if self.datefmt is not None and not isinstance(self.datefmt, str):
            raise TypeError("datefmt must be a string or None")
        self.resolve_level()
        try:
            logging.Formatter(self.format, datefmt=self.datefmt)
        except ValueError as exc:
            raise ValueError("format must be a valid logging format") from exc

    def resolve_level(self) -> int:
        """Resolve level to a logging integer constant."""
        if isinstance(self.level, int):
            return self.level
        name = self.level.strip().upper()
        levels = logging.getLevelNamesMapping()
        if name not in levels:
            raise ValueError(f"unknown log level: {self.level}")
        return levels[name]

    def replace(self, **kwargs: Any) -> LoggingConfig:
        """Return a new config with updated fields."""
        return replace(self, **kwargs)


def setup_logging(config: LoggingConfig) -> logging.Logger:
    """Configure and return the root ``torchrir`` logger.

    Examples:
        ```python
        logger = setup_logging(LoggingConfig(level="DEBUG"))
        logger.info("ready")
        ```
    """
    if not isinstance(config, LoggingConfig):
        raise TypeError("config must be a LoggingConfig")
    level = config.resolve_level()
    formatter = logging.Formatter(config.format, datefmt=config.datefmt)
    logger = logging.getLogger("torchrir")
    with _setup_lock:
        managed_handlers = [
            candidate
            for candidate in logger.handlers
            if getattr(candidate, "_torchrir_managed", False)
        ]
        if managed_handlers:
            handler = managed_handlers[0]
            for duplicate in managed_handlers[1:]:
                logger.removeHandler(duplicate)
                duplicate.close()
        else:
            handler = logging.StreamHandler()
            setattr(handler, "_torchrir_managed", True)
            logger.addHandler(handler)
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.setLevel(level)
        # The managed handler is the only sink for the torchrir namespace.
        # Keeping propagation disabled prevents duplicate records when an
        # application has also configured Python's process-wide root logger.
        logger.propagate = False
    return logger


def get_logger(name: str | None = None) -> logging.Logger:
    """Return a torchrir logger, namespaced under the torchrir root.

    Examples:
        ```python
        logger = get_logger("examples.static")
        ```
    """
    if name is None:
        return logging.getLogger("torchrir")
    if not isinstance(name, str):
        raise TypeError("name must be a string or None")
    if not name:
        raise ValueError("name must be a non-empty string or None")
    if name == "torchrir" or name.startswith("torchrir."):
        return logging.getLogger(name)
    return logging.getLogger(f"torchrir.{name}")


def _validate_info_logger(logger: object | None) -> None:
    """Validate the optional logger protocol before any caller side effects."""

    if logger is not None and not callable(getattr(logger, "info", None)):
        raise TypeError("logger must provide a callable info method or be None")


__all__ = ["LoggingConfig", "get_logger", "setup_logging"]
