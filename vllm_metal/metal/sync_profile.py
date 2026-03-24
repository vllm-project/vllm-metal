# SPDX-License-Identifier: Apache-2.0
"""Lightweight sync profiling for raw Metal paged-attention calls."""

from __future__ import annotations

import atexit
import logging
import os
import threading
from collections import defaultdict

logger = logging.getLogger(__name__)

_ENABLED = os.getenv("VLLM_METAL_PROFILE_SYNC") == "1"
_LOCK = threading.Lock()
_STATS: dict[str, list[float]] = defaultdict(lambda: [0.0, 0.0])
_REGISTERED = False


def enabled() -> bool:
    return _ENABLED


def record(name: str, elapsed_s: float) -> None:
    if not _ENABLED:
        return
    with _LOCK:
        stat = _STATS[name]
        stat[0] += 1.0
        stat[1] += elapsed_s


def _emit_summary() -> None:
    if not _STATS:
        return
    logger.info("Metal sync profile summary:")
    for name in sorted(_STATS):
        count, total_s = _STATS[name]
        avg_ms = (total_s / count) * 1000.0 if count else 0.0
        total_ms = total_s * 1000.0
        logger.info(
            "  %s calls=%d total_ms=%.3f avg_ms=%.3f",
            name,
            int(count),
            total_ms,
            avg_ms,
        )


def ensure_registered() -> None:
    global _REGISTERED
    if not _ENABLED or _REGISTERED:
        return
    atexit.register(_emit_summary)
    _REGISTERED = True
