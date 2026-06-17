# SPDX-License-Identifier: Apache-2.0
"""Shared lifecycle base for the paged attention runtimes.

The three concrete runtimes (MHA, MLA, hybrid) differ only in which caches they
allocate and how they wrap layers; their initialise-guard, warm-up, and
block-count plumbing is identical.  That shared plumbing lives here so there is
one copy instead of three.
"""

from __future__ import annotations

from typing import Any

from vllm_metal.metal import warm_up_kernels


class PagedAttentionRuntimeBase:
    """Common lifecycle for paged attention runtimes.

    Subclasses allocate their primary paged cache onto ``self._cache`` in
    ``initialize()``.  The primary cache must expose ``num_blocks``.  Secondary
    caches (e.g. the hybrid GDN state cache) remain the subclass's concern.
    """

    # Primary paged cache; ``None`` until ``initialize()`` runs.  Subclasses set
    # it to their concrete cache type in ``__init__`` / ``initialize``.  The
    # class-level default ensures the guard below raises a clear RuntimeError
    # (not AttributeError) even for instances built via ``__new__``.
    _cache: Any = None

    def _require_initialized(self, caller: str) -> Any:
        if self._cache is None:
            raise RuntimeError(f"{caller}() called before initialize()")
        return self._cache

    def warm_up(self) -> None:
        self._require_initialized("warm_up")
        warm_up_kernels()

    def num_blocks(self) -> int:
        return self._require_initialized("num_blocks").num_blocks
