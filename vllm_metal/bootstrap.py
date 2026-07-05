# SPDX-License-Identifier: Apache-2.0
"""Deterministic platform bootstrap for the vLLM Metal plugin.

Problem
-------
vLLM resolves and *caches* ``vllm.platforms.current_platform`` lazily on first
access. Importing ``vllm`` (directly or transitively) can trigger that first
access *before* the out-of-tree ``metal`` platform plugin has been loaded, in
which case vLLM caches the built-in ``CpuPlatform`` and never re-resolves. The
result is silent: ``LLM(...)`` runs on vLLM's generic CPU torch backend instead
of the Metal / MLX runtime, with no error.

The ``vllm serve`` CLI avoids this because it calls ``load_general_plugins()``
early. Code that uses the ``LLM`` Python API directly has no such guarantee.

Fix
---
Call :func:`bootstrap_metal_platform` *before* constructing ``LLM`` (and before
any code path that reads ``current_platform``). It loads vLLM's general plugins
and forces the platform to re-resolve, so the Metal plugin wins on Apple
Silicon.

Usage::

    from vllm_metal.bootstrap import bootstrap_metal_platform
    bootstrap_metal_platform()
    from vllm import LLM
    llm = LLM(model="Qwen/Qwen3-0.6B")
"""

from __future__ import annotations

import os

from vllm.logger import init_logger

logger = init_logger(__name__)


def bootstrap_metal_platform(*, require_metal: bool = False) -> str:
    """Force vLLM to resolve the Metal platform plugin.

    Loads general plugins and clears any prematurely-cached platform so the
    out-of-tree Metal plugin is (re-)resolved. Also defaults
    ``VLLM_ENABLE_V1_MULTIPROCESSING=0`` so the engine runs in-process (the
    common single-node Apple-Silicon case), which keeps the plugin resolution
    and worker in the same interpreter.

    Args:
        require_metal: When True, raise ``RuntimeError`` if the resolved
            platform is not the Metal platform (e.g. non-Apple-Silicon host).

    Returns:
        The resolved platform class name.
    """
    os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

    from vllm.plugins import load_general_plugins

    load_general_plugins()

    import vllm.platforms as _platforms

    # Clear a prematurely-cached platform so resolution reruns after plugins
    # have been registered.
    _platforms._current_platform = None  # type: ignore[attr-defined]
    resolved = type(_platforms.current_platform).__name__

    if "Metal" in resolved:
        logger.info("bootstrap_metal_platform: Metal platform active (%s)", resolved)
    else:
        msg = (
            f"bootstrap_metal_platform: resolved {resolved}, not the Metal platform. "
            "On Apple Silicon ensure vllm-metal is installed; on other hosts "
            "this is expected."
        )
        if require_metal:
            raise RuntimeError(msg)
        logger.warning(msg)

    return resolved
