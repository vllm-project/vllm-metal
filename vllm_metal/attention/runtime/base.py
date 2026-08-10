# SPDX-License-Identifier: Apache-2.0
"""Shared lifecycle base for the paged attention runtimes.

The three concrete runtimes (MHA, MLA, hybrid) differ only in which caches they
allocate and how they wrap layers; their initialise-guard, warm-up, and
block-count plumbing is identical.  That shared plumbing lives here so there is
one copy instead of three.
"""

from __future__ import annotations

from collections.abc import Sequence
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

    def kv_scheduler_group_indices(self) -> tuple[int, ...]:
        """Return scheduler KV groups consumed by this runtime."""
        self._require_initialized("kv_scheduler_group_indices")
        return (0,)

    def kv_group_block_sizes(self) -> tuple[int, ...]:
        """Return scheduler page sizes owned by this runtime."""
        return (self._require_initialized("kv_group_block_sizes").block_size,)

    # Scheduler groups holding per-request state (mamba) blocks — empty for
    # runtimes whose layers are all block-table addressed.  The hybrid runtime
    # overwrites this when it adopts mamba cache groups so the runner forwards
    # those groups' block ids alongside the KV block tables.
    _state_group_indices: tuple[int, ...] = ()

    def state_scheduler_group_indices(self) -> tuple[int, ...]:
        """Return scheduler groups holding per-request state (mamba) blocks."""
        return self._state_group_indices

    def copy_blocks(self, block_copies: Sequence[tuple[int, int]]) -> None:
        """Apply scheduler copy-on-write operations to the primary cache."""
        self._require_initialized("copy_blocks").copy_blocks(block_copies)

    def needs_step_context(self) -> bool:
        """Return whether this runtime attaches request-ordered step metadata."""
        return False

    def populate_step_context(
        self,
        *,
        req_ids: list[str],
        ctx: PagedAttentionContext,
        state_block_ids: list[list[list[int]]] | None = None,
        step_positions: list[tuple[int, int]] | None = None,
    ) -> None:
        """Attach runtime-specific metadata to one forward-pass context."""
        del req_ids, ctx, state_block_ids, step_positions

    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        """Append runtime-owned side-effect arrays that must be eval'd."""
        del outputs

    def release_requests(self, req_ids: set[str]) -> None:
        """Release runtime-owned state for requests whose state is invalid."""
        del req_ids

    def materialize_pending_state(self) -> None:
        """Detach deferred runtime state from the lazy graph."""


class StateHybridRuntimeBase(PagedAttentionRuntimeBase):
    """Shared lifecycle for hybrid runtimes that own per-request state layers.

    Both hybrid families — GDN linear attention (Qwen3.5/3.6) and ShortConv
    (LFM2/LFM2.5) — pair a paged SDPA KV cache with a per-request state cache
    driven by one of the state managers.  Everything about that pairing is
    identical between them: which scheduler groups they adopt, how the state
    cache and manager are built for ``none`` vs ``align`` mode, and how the
    per-step hooks delegate to the manager.  Only the state cache's own type,
    the layer classification, and layer wrapping differ, so subclasses
    override ``_new_state_cache`` and implement ``patch_model``.
    """

    #: Scheduler-side state caching strategies this runtime implements.
    SUPPORTED_MAMBA_CACHE_MODES: tuple[str, ...] = ("none",)

    def _init_state_hybrid(
        self, *, block_size: int, max_num_seqs: int, mamba_cache_mode: str
    ) -> None:
        """Validate the caching mode and set up shared state-hybrid fields."""
        if mamba_cache_mode not in self.SUPPORTED_MAMBA_CACHE_MODES:
            supported = " and ".join(
                repr(mode) for mode in self.SUPPORTED_MAMBA_CACHE_MODES
            )
            raise NotImplementedError(
                f"{type(self).__name__} does not support "
                f"mamba_cache_mode={mamba_cache_mode!r} (only {supported})"
            )
        self._mamba_cache_mode = mamba_cache_mode
        self._block_size = block_size
        self._max_num_seqs = max_num_seqs
        self._scheduler_group_indices: tuple[int, ...] = (0,)
        self._group_block_sizes: tuple[int, ...] = (block_size,)
        self._state_cache: Any = None
        self._state_manager: Any = None

    def _new_state_cache(self, *, max_seqs: int, initial_seqs: int) -> Any:
        """Build this runtime's per-request state cache."""
        raise NotImplementedError

    def _init_sdpa_kv_cache(self, num_blocks: int) -> None:
        """Allocate the paged KV cache backing this runtime's SDPA layers."""
        from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache

        self._cache = MetalPagedKVCache(
            num_layers=len(self._sdpa_indices),
            num_kv_heads=self._num_kv_heads,
            head_dim=self._head_dim,
            num_blocks=num_blocks,
            block_size=self._block_size,
            dtype=self._dtype,
            turboquant=self._turboquant,
            k_quant=self._k_quant,
            v_quant=self._v_quant,
        )

    def _initialize_state_lifecycle(self, num_blocks: int) -> None:
        """Build the state cache and the manager its caching mode requires.

        Align-mode slabs are addressed directly by scheduler block id; any of
        the pool's blocks can become a state block (the block pool is fungible
        across cache groups), so the id space is ``[0, num_blocks)``.  The
        paged plan charges every block for its state bytes (admission worst
        case), but the pool materializes lazily by high-water block id — vLLM's
        BlockPool hands out low ids first, so resident state memory tracks the
        live + cached set instead of wiring the whole worst case up front.
        Start empty so the scheduler's shared tensor layout is adopted before
        any physical pool exists.  None mode keeps one slab per resident
        request and grows on demand.
        """
        from vllm_metal.attention.state import (
            AlignGDNStateManager,
            HybridGDNStateManager,
        )

        align = self._mamba_cache_mode == "align"
        self._state_cache = self._new_state_cache(
            max_seqs=num_blocks if align else self._max_num_seqs,
            initial_seqs=0,
        )
        self._state_manager = (
            AlignGDNStateManager(self._state_cache, self._block_size)
            if align
            else HybridGDNStateManager(self._state_cache)
        )

    @property
    def state_cache(self) -> Any:
        if self._state_cache is None:
            raise RuntimeError("state_cache accessed before initialize()")
        return self._state_cache

    @property
    def state_manager(self) -> Any:
        if self._state_manager is None:
            raise RuntimeError("state_manager accessed before initialize()")
        return self._state_manager

    def adopt_scheduler_group(
        self,
        group_index: int,
        block_size: int,
        *,
        state_group_indices: tuple[int, ...] = (),
        layer_group_ordinals: list[int] | None = None,
        layer_pool_ordinals: list[int] | None = None,
    ) -> None:
        """Select the vLLM scheduler groups backing this runtime.

        ``group_index`` is the group owning SDPA KV blocks (kernel block
        tables); ``state_group_indices`` are the mamba cache groups whose
        block ids key state slabs under align mode;
        ``layer_group_ordinals[cache_idx]`` records which of those groups each
        state layer belongs to (the engine stripes same-spec layers across
        several groups) and ``layer_pool_ordinals[cache_idx]`` which physical
        pool it shares (one pool per within-group position, following
        ``kv_cache_tensors.shared_by``).  None mode keeps a private
        per-request pool and receives none of them.
        """
        self._require_initialized("adopt_scheduler_group")
        if block_size != self._block_size:
            raise NotImplementedError(
                f"{type(self).__name__} requires the SDPA scheduler group "
                f"block size to stay {self._block_size}, got {block_size}"
            )
        self._scheduler_group_indices = (group_index,)
        self._group_block_sizes = (block_size,)
        self._state_group_indices = tuple(state_group_indices)
        if layer_group_ordinals is not None:
            pool_ordinals = (
                layer_pool_ordinals
                if layer_pool_ordinals is not None
                else list(range(len(layer_group_ordinals)))
            )
            self.state_cache.set_layer_layout(layer_group_ordinals, pool_ordinals)

    def kv_scheduler_group_indices(self) -> tuple[int, ...]:
        """Return scheduler KV groups consumed by SDPA layers."""
        self._require_initialized("kv_scheduler_group_indices")
        return self._scheduler_group_indices

    def kv_group_block_sizes(self) -> tuple[int, ...]:
        """Return SDPA scheduler group page sizes."""
        self._require_initialized("kv_group_block_sizes")
        return self._group_block_sizes

    def needs_step_context(self) -> bool:
        return True

    def copy_blocks(self, block_copies: Sequence[tuple[int, int]]) -> None:
        """Apply scheduler CoW copies to SDPA KV and align-mode state."""
        self._require_initialized("copy_blocks").copy_blocks(block_copies)
        if self._mamba_cache_mode == "align":
            self.state_cache.copy_blocks(block_copies)

    def populate_step_context(
        self,
        *,
        req_ids: list[str],
        ctx: PagedAttentionContext,
        state_block_ids: list[list[list[int]]] | None = None,
        step_positions: list[tuple[int, int]] | None = None,
    ) -> None:
        self.state_manager.populate_step_context(
            req_ids=req_ids,
            ctx=ctx,
            state_block_ids=state_block_ids,
            step_positions=step_positions,
        )

    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        self.state_manager.extend_forward_eval_outputs(outputs)

    def release_requests(self, req_ids: set[str]) -> None:
        self.state_manager.release_requests(req_ids)

    def materialize_pending_state(self) -> None:
        self.state_manager.materialize_pending_state()
