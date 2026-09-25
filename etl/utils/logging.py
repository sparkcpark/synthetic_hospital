"""Structured logging for ETL pipeline. (spec §9: etl/utils/logging.py)"""

import logging
import sys

_configured = False


def get_logger(name: str) -> logging.Logger:
    """Return a logger with structured formatting."""
    global _configured
    if not _configured:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(
            logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%dT%H:%M:%S",
            )
        )
        root = logging.getLogger("etl")
        root.addHandler(handler)
        root.setLevel(logging.INFO)
        _configured = True
    return logging.getLogger(name)
