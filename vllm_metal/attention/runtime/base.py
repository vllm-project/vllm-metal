# SPDX-License-Identifier: Apache-2.0
"""Shared lifecycle base for the paged attention runtimes.

The three concrete runtimes (MHA, MLA, hybrid) differ only in which caches they
allocate and how they wrap layers; their initialise-guard, warm-up, and
block-count plumbing is identical.  That shared plumbing lives here so there is
one copy instead of three.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import mlx.core as mx

from vllm_metal.metal import warm_up_kernels

if TYPE_CHECKING:
    from vllm_metal.attention.context import PagedAttentionContext


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

    def needs_step_context(self) -> bool:
        """Return whether this runtime attaches request-ordered step metadata."""
        return False

    def populate_step_context(
        self, *, req_ids: list[str], ctx: PagedAttentionContext
    ) -> None:
        """Attach runtime-specific metadata to one forward-pass context."""
        del req_ids, ctx

    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        """Append runtime-owned side-effect arrays that must be eval'd."""
        del outputs

    def release_requests(self, req_ids: set[str]) -> None:
        """Release runtime-owned state for requests whose state is invalid."""
        del req_ids

    def materialize_pending_state(self) -> None:
        """Detach deferred runtime state from the lazy graph."""
