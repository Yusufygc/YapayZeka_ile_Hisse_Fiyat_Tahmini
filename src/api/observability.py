# -*- coding: utf-8 -*-
"""Structured logging helpers for AI_Core serving."""

from __future__ import annotations

import json
import logging
import os
from datetime import datetime
from logging.handlers import RotatingFileHandler
from typing import Any


def get_logger(project_root: str) -> logging.Logger:
    logger = logging.getLogger("ai_core")
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    log_dir = os.path.join(project_root, "logs")
    os.makedirs(log_dir, exist_ok=True)
    handler = RotatingFileHandler(
        os.path.join(log_dir, "ai_core.log"),
        maxBytes=2_000_000,
        backupCount=3,
        encoding="utf-8",
    )
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.propagate = False
    return logger


def log_event(project_root: str, event: str, **fields: Any) -> None:
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,
        **fields,
    }
    try:
        get_logger(project_root).info(json.dumps(payload, ensure_ascii=False, sort_keys=True))
    except Exception:
        # Logging must never break analysis serving.
        return
