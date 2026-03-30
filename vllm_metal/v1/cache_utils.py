# SPDX-License-Identifier: Apache-2.0
"""Utilities for KV cache introspection."""

from __future__ import annotations

from typing import Any

from mlx_lm.models.cache import KVCache, RotatingKVCache, make_prompt_cache
from vllm.logger import init_logger

logger = init_logger(__name__)


def supports_batched_decode(model: Any, *, is_vlm: bool = False) -> bool:
    """Check whether *model*'s prompt cache is compatible with ``BatchKVCache``.

    Hybrid models (e.g. Qwen3.5 with mixed attention + linear/SSM layers)
    produce caches containing ``ArraysCache`` entries whose attention code
    uses ``cache.offset`` as a Python ``int`` for mask slicing.
    ``BatchKVCache.offset`` returns an ``mx.array``, which causes::

        ValueError: Slice indices must be integers or None.

    Returns ``True`` when every cache layer is a ``KVCache`` or
    ``RotatingKVCache`` (safe to batch), ``False`` otherwise.

    NOTE: This is an interim workaround for the mlx-native (non-paged) path.
    The proper fix is per-layer attention dispatching (#201) combined with a
    paged linear attention kernel (roadmap #148).
    """
    cache_model = (
        model.language_model
        if is_vlm and hasattr(model, "language_model")
        else model
    )
    try:
        cache = make_prompt_cache(cache_model)
        return all(isinstance(c, (KVCache, RotatingKVCache)) for c in cache)
    except Exception:
        # If we can't determine, default to sequential (safe).
        return False
