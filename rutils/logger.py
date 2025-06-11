"""Logging utilities for Franka automation."""

import logging


def setup_logging(level: str = "INFO") -> logging.Logger:
    """Configure application logging."""
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%H:%M:%S"
    )
    return logging.getLogger(__name__)