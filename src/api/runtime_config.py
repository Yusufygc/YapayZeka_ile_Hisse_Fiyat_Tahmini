# -*- coding: utf-8 -*-
"""Runtime configuration helpers for the FastAPI serving layer."""

from __future__ import annotations

import os
from dataclasses import dataclass


LOCAL_ORIGIN_REGEX = r"^http://(localhost|127\.0\.0\.1)(:\d+)?$"


@dataclass(frozen=True)
class CorsSettings:
    allow_origins: list[str]
    allow_origin_regex: str | None
    mode: str


def get_cors_settings() -> CorsSettings:
    raw = os.getenv("AI_CORE_CORS_ORIGINS", "")
    extra = [item.strip() for item in raw.split(",") if item.strip() and item.strip() != "*"]
    origins = ["http://localhost", "http://127.0.0.1", *extra]
    mode = "local-only" if not extra else "local-plus-env"
    return CorsSettings(
        allow_origins=origins,
        allow_origin_regex=LOCAL_ORIGIN_REGEX,
        mode=mode,
    )
