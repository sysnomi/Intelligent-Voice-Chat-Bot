"""
utils/logging.py — Structured JSON logger using structlog.
All modules should use: from app.utils.logging import get_logger
"""
import logging
import sys
import structlog
from app.config import settings


def setup_logging() -> None:
    """Configure structlog with JSON rendering for production."""
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.ConsoleRenderer() if settings.log_level == "DEBUG"
            else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__):
    """Return a bound structlog logger."""
    return structlog.get_logger(name)
