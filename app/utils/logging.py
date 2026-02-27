"""
structlog configuration.
Call configure_logging() once at application startup.
Every logger returned by get_logger() will emit JSON-structured logs
with timestamps, log level, and any bound context (e.g. trace_id).
"""

import logging
import sys

import structlog


def configure_logging() -> None:
    """Configure structlog for JSON output. Call once at startup."""
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.dev.set_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(logging.DEBUG),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(file=sys.stdout),
        cache_logger_on_first_use=True,
    )


def get_logger(name: str = __name__) -> structlog.BoundLogger:
    """Return a structlog bound logger. Bind trace_id via structlog.contextvars."""
    return structlog.get_logger(name)
