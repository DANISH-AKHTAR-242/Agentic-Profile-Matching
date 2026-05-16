"""Structured logging helpers."""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from typing import Any


class JsonFormatter(logging.Formatter):
    """Format log records as JSON for operational observability."""

    def format(self, record: logging.LogRecord) -> str:
        payload: dict[str, Any] = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        for attr in ("event", "node", "query", "candidate_id", "details"):
            if hasattr(record, attr):
                payload[attr] = getattr(record, attr)
        return json.dumps(payload, ensure_ascii=True)


def get_logger(name: str, level: str = "INFO") -> logging.Logger:
    """Create or fetch a configured logger."""
    logger = logging.getLogger(name)
    logger.setLevel(level.upper())
    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(JsonFormatter())
        logger.addHandler(handler)
        logger.propagate = False
    return logger

