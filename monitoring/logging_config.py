"""
Structlog configuration — call configure_logging() once at application startup.

Sets up a single JSON-per-line output chain used by both:
  - native structlog loggers (structlog.get_logger())
  - stdlib bridge loggers (logging.getLogger())

Static fields (service, environment) are bound once here via bind_contextvars.
Dynamic fields (request_id) are bound per-request by RequestIdMiddleware and
merge into every log line automatically via the merge_contextvars processor.
"""
from __future__ import annotations
import logging
import sys

import structlog
from config.settings import settings

_SHARED_PROCESSORS = [
    structlog.contextvars.merge_contextvars,
    structlog.stdlib.add_log_level,
    structlog.stdlib.add_logger_name,
    structlog.processors.TimeStamper(fmt="iso", utc=True),
    structlog.processors.StackInfoRenderer(),
    structlog.processors.format_exc_info,
    structlog.processors.UnicodeDecoder(),
]


def configure_logging() -> None:
    structlog.configure(
        processors=[
            *_SHARED_PROCESSORS,
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        foreign_pre_chain=_SHARED_PROCESSORS,
        processors=[
            structlog.stdlib.ProcessorFormatter.remove_processors_meta,
            structlog.processors.JSONRenderer(),
        ],
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root = logging.getLogger()
    root.handlers.clear()
    root.addHandler(handler)
    root.setLevel(getattr(logging, settings.log_level.upper(), logging.INFO))

    structlog.contextvars.bind_contextvars(
        service="ragcore",
        environment=settings.environment,
    )
