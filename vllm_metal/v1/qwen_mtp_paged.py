# SPDX-License-Identifier: Apache-2.0
"""Scheduler-owned cache transaction for native Qwen MTP on Metal.

The target hybrid runtime owns three pieces of speculative state:

* the ordinary target SDPA/GDN state;
* a dedicated, scheduler-addressed MTP-head KV cache;
* one target pre-norm hidden-state checkpoint per completed target cache block.

The MTP group is an EAGLE-style cache group and therefore recomputes a full
hash unit on a reusable prefix hit. The boundary checkpoint shares the target
block lifecycle and validates that warm-prefix drafting never combines restored
target state with a fresh or incomplete MTP cache.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.kv_cache_interface import HiddenStateCacheSpec
from vllm.v1.kv_cache_spec_registry import KVCacheSpecRegistry

from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.context import (
    PagedAttentionContext,
    clear_context,
    prepare_unified,
)
from vllm_metal.attention.impls.sdpa_wrapper import patch_sdpa_attention


@dataclass(frozen=True, kw_only=True)
class QwenMTPAttentionSpec(HiddenStateCacheSpec):
    """A distinct full-attention cache group for the native Qwen MTP head.

    The cache-only marker makes vLLM pull this layer out before hybrid
    grouping, so a single MTP layer does not collapse every target/GDN group to
    size one. ``num_kv_heads`` is reported as combined K+V head slots; the
    inherited latent-page byte formula therefore equals the physical dense
    K+V page owned by :class:`MetalPagedKVCache`.
    """


# Initialize the built-in registry first, then register this cache-only
# subtype as its own uniform base. Cache policy inserts it first in the spec
# map, so vLLM's early uniformity probe cannot absorb it into target attention.
KVCacheSpecRegistry._ensure_registered()
KVCacheSpecRegistry.register(
    QwenMTPAttentionSpec,
    FullAttentionManager,
    uniform_type_base_spec=QwenMTPAttentionSpec,
)


class QwenMTPBoundaryHiddenCache:
    """One pre-final-norm hidden checkpoint per completed target block.

    Prefix-cache adoption is block-aligned. EAGLE drops one hash unit and
    recomputes it, so only the hidden state at the end of the retained target
    block is durable state. Keeping one vector per block avoids multiplying the
    memory cost by ``block_size`` while retaining an exact, fail-closed lineage
    check.
    """

    def __init__(
        self,
        *,
        num_blocks: int,
        block_size: int,
        hidden_size: int,
        dtype: mx.Dtype,
    ) -> None:
        if num_blocks <= 0 or block_size <= 0 or hidden_size <= 0:
            raise ValueError("Qwen MTP boundary cache dimensions must be positive")
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.cache = mx.zeros((num_blocks, hidden_size), dtype=dtype)
        self.valid = mx.zeros((num_blocks,), dtype=mx.uint8)
        mx.eval(self.cache, self.valid)

    @property
    def bytes_per_block(self) -> int:
        return self.hidden_size * self.dtype.size + mx.uint8.size

    def store(self, slot_mapping: Sequence[int], hidden_states: mx.array) -> None:
        hidden_states = hidden_states.reshape(-1, self.hidden_size)
        if len(slot_mapping) != hidden_states.shape[0]:
            raise RuntimeError(
                "Qwen MTP boundary-hidden slot/row mismatch: "
                f"{len(slot_mapping)} != {hidden_states.shape[0]}"
            )
        if not slot_mapping:
            return
        if any(slot < 0 for slot in slot_mapping):
            raise RuntimeError("Qwen MTP boundary cache received a negative slot")
        max_slot = self.num_blocks * self.block_size
        if any(slot >= max_slot for slot in slot_mapping):
            raise RuntimeError("Qwen MTP boundary cache slot is out of range")

        # A write to offset zero begins a new physical block lifetime. Clear a
        # previous occupant's boundary marker before the block can be cached.
        clear_blocks = sorted(
            {
                slot // self.block_size
                for slot in slot_mapping
                if slot % self.block_size == 0
            }
        )
        if clear_blocks:
            clear_ids = mx.array(clear_blocks, dtype=mx.int32)
            self.valid[clear_ids] = mx.zeros((len(clear_blocks),), dtype=mx.uint8)

        boundary_rows = [
            row
            for row, slot in enumerate(slot_mapping)
            if slot % self.block_size == self.block_size - 1
        ]
        if not boundary_rows:
            return
        block_ids = mx.array(
            [slot_mapping[row] // self.block_size for row in boundary_rows],
            dtype=mx.int32,
        )
        row_ids = mx.array(boundary_rows, dtype=mx.int32)
        self.cache[block_ids] = hidden_states[row_ids].astype(self.dtype)
        self.valid[block_ids] = mx.ones((len(boundary_rows),), dtype=mx.uint8)

    def read(self, block_ids: Sequence[int], token_position: int) -> mx.array:
        if token_position < 0:
            raise RuntimeError("Qwen MTP boundary token position must be non-negative")
        if token_position % self.block_size != self.block_size - 1:
            raise RuntimeError(
                "Qwen MTP warm-prefix boundary is not target-block aligned"
            )
        block_index = token_position // self.block_size
        if block_index >= len(block_ids):
            raise RuntimeError(
                "Qwen MTP target prefix is missing the boundary-hidden block"
            )
        block_id = int(block_ids[block_index])
        if block_id < 0 or block_id >= self.num_blocks:
            raise RuntimeError("Qwen MTP boundary block id is out of range")
        valid = self.valid[block_id]
        mx.eval(valid)
        if int(valid.item()) != 1:
            raise RuntimeError(
                "Qwen MTP target prefix has no valid boundary-hidden checkpoint"
            )
        value = self.cache[block_id : block_id + 1]
        mx.eval(value)
        return value

    def copy_blocks(self, block_copies: Sequence[tuple[int, int]]) -> None:
        if not block_copies:
            return
        src_ids, dst_ids = zip(*block_copies, strict=True)
        if any(
            block_id < 0 or block_id >= self.num_blocks
            for block_id in (*src_ids, *dst_ids)
        ):
            raise RuntimeError(
                "Qwen MTP boundary cache copy contains an out-of-range block id"
            )
        src = mx.array(src_ids, dtype=mx.int32)
        dst = mx.array(dst_ids, dtype=mx.int32)
        self.cache[dst] = self.cache[src]
        self.valid[dst] = self.valid[src]


class QwenMTPPagedState:
    """Dedicated paged MTP KV plus target-boundary hidden-state shadow."""

    def __init__(
        self,
        *,
        model: Any,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        hidden_size: int,
        dtype: mx.Dtype,
    ) -> None:
        if num_layers <= 0:
            raise ValueError("native Qwen MTP requires at least one MTP layer")
        mtp_module = getattr(model, "mtp", None)
        if mtp_module is None:
            raise ValueError("native Qwen MTP model does not expose model.mtp")
        self.model = model
        self.mtp_module = mtp_module
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.hidden_size = hidden_size
        self.dtype = dtype
        self.mtp_cache: MetalPagedKVCache | None = None
        self.boundary_cache: QwenMTPBoundaryHiddenCache | None = None
        self._target_group_index: int | None = None
        self._mtp_group_index: int | None = None
        self._target_block_size: int | None = None
        self._mtp_block_size: int | None = None

    def initialize(self, *, num_blocks: int, block_size: int) -> None:
        self.mtp_cache = MetalPagedKVCache(
            num_layers=self.num_layers,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_dim,
            num_blocks=num_blocks,
            block_size=block_size,
            dtype=self.dtype,
        )
        self.boundary_cache = QwenMTPBoundaryHiddenCache(
            num_blocks=num_blocks,
            block_size=block_size,
            hidden_size=self.hidden_size,
            dtype=self.dtype,
        )
        patched = patch_sdpa_attention(
            self.mtp_module,
            self.mtp_cache,
            block_size,
        )
        if patched != self.num_layers:
            raise RuntimeError(
                "Qwen MTP paged runtime patched "
                f"{patched} attention layers, expected {self.num_layers}"
            )

    def configure_groups(
        self,
        *,
        target_group_index: int,
        mtp_group_index: int,
        target_block_size: int,
        mtp_block_size: int,
    ) -> None:
        if target_group_index == mtp_group_index:
            raise RuntimeError("Qwen MTP KV must use a distinct scheduler group")
        if target_block_size != mtp_block_size:
            raise NotImplementedError(
                "Qwen MTP currently requires target and MTP groups to use the "
                "same block size"
            )
        cache = self._require_mtp_cache()
        boundary = self._require_boundary_cache()
        if (
            cache.block_size != mtp_block_size
            or boundary.block_size != target_block_size
        ):
            raise RuntimeError("Qwen MTP scheduler block size changed after allocation")
        self._target_group_index = target_group_index
        self._mtp_group_index = mtp_group_index
        self._target_block_size = target_block_size
        self._mtp_block_size = mtp_block_size

    @property
    def ready(self) -> bool:
        return (
            self.mtp_cache is not None
            and self.boundary_cache is not None
            and self._target_group_index is not None
            and self._mtp_group_index is not None
        )

    @property
    def scheduler_group_indices(self) -> tuple[int, int]:
        self._require_ready()
        assert self._target_group_index is not None
        assert self._mtp_group_index is not None
        return self._target_group_index, self._mtp_group_index

    @property
    def group_block_sizes(self) -> tuple[int, int]:
        self._require_ready()
        assert self._target_block_size is not None
        assert self._mtp_block_size is not None
        return self._target_block_size, self._mtp_block_size

    @property
    def mtp_group_ordinal(self) -> int:
        # ``HybridPagedAttentionRuntime`` always publishes target first.
        self._require_ready()
        return 1

    def store_target_hidden(
        self,
        ctx: PagedAttentionContext,
        hidden_states: mx.array,
    ) -> None:
        self._require_ready()
        if ctx.kv_groups is None or not ctx.kv_groups:
            slot_mapping = ctx.slot_mapping
        else:
            # Runtime-local group zero is the target SDPA group.
            slot_mapping = ctx.kv_groups[0].slot_mapping
        self._require_boundary_cache().store(slot_mapping, hidden_states)

    def read_boundary_hidden(
        self,
        block_ids_by_group: Sequence[Sequence[int]],
        token_position: int,
    ) -> mx.array:
        self._require_ready()
        if not block_ids_by_group:
            raise RuntimeError("Qwen MTP request has no target scheduler blocks")
        return self._require_boundary_cache().read(
            block_ids_by_group[0],
            token_position,
        )

    def _project_mtp_hidden(self, hidden: mx.array) -> mx.array:
        """Project only MTP rows whose vocabulary logits are consumed."""
        args = getattr(self.model, "args", None)
        inner_model = getattr(self.model, "model", None)
        embed_tokens = getattr(inner_model, "embed_tokens", None)
        if bool(getattr(args, "tie_word_embeddings", False)):
            if embed_tokens is None or not callable(
                getattr(embed_tokens, "as_linear", None)
            ):
                raise RuntimeError("Qwen MTP tied output projection is unavailable")
            return embed_tokens.as_linear(hidden)
        lm_head = getattr(self.model, "lm_head", None)
        if lm_head is None:
            raise RuntimeError("Qwen MTP model has no output projection")
        return lm_head(hidden)

    def run_pairs_batch(
        self,
        *,
        hidden_rows_batch: Sequence[mx.array],
        next_token_ids_batch: Sequence[Sequence[int]],
        block_ids_by_group_batch: Sequence[Sequence[Sequence[int]]],
        start_positions: Sequence[int],
        draft_request_indices: Sequence[int] | None = None,
    ) -> list[int]:
        """Advance independent MTP cache segments in one packed forward.

        Varlen scheduler metadata keeps requests isolated. Only segment ends
        that need a speculative token pay the shared vocabulary projection;
        prefix-maintenance rows update MTP KV without materializing logits.
        """
        self._require_ready()
        num_requests = len(hidden_rows_batch)
        if not (
            num_requests
            == len(next_token_ids_batch)
            == len(block_ids_by_group_batch)
            == len(start_positions)
        ):
            raise RuntimeError("Qwen MTP batched request metadata length mismatch")
        if num_requests == 0:
            return []

        if draft_request_indices is None:
            draft_indices = list(range(num_requests))
        else:
            draft_indices = [int(index) for index in draft_request_indices]
        if len(set(draft_indices)) != len(draft_indices) or any(
            index < 0 or index >= num_requests for index in draft_indices
        ):
            raise RuntimeError("Qwen MTP draft request index is out of range")

        ordinal = self.mtp_group_ordinal
        assert self._mtp_block_size is not None
        prefill_requests: list[tuple[list[int], int, int]] = []
        hidden_parts: list[mx.array] = []
        flat_tokens: list[int] = []
        segment_end_rows: list[int] = []
        cursor = 0

        for hidden_rows, next_token_ids, block_ids_by_group, start_pos in zip(
            hidden_rows_batch,
            next_token_ids_batch,
            block_ids_by_group_batch,
            start_positions,
            strict=True,
        ):
            token_ids = [int(token) for token in next_token_ids]
            if hidden_rows.shape[0] != len(token_ids):
                raise RuntimeError(
                    "Qwen MTP hidden/token pair count mismatch: "
                    f"{hidden_rows.shape[0]} != {len(token_ids)}"
                )
            if not token_ids:
                raise RuntimeError("Qwen MTP forward requires at least one pair")
            if ordinal >= len(block_ids_by_group):
                raise RuntimeError("Qwen MTP request is missing its scheduler KV group")
            mtp_blocks = list(block_ids_by_group[ordinal])
            last_pos = int(start_pos) + len(token_ids) - 1
            if last_pos // self._mtp_block_size >= len(mtp_blocks):
                raise RuntimeError("Qwen MTP scheduler block table is too short")

            prefill_requests.append((mtp_blocks, len(token_ids), int(start_pos)))
            hidden_parts.append(hidden_rows)
            flat_tokens.extend(token_ids)
            cursor += len(token_ids)
            segment_end_rows.append(cursor - 1)

        prepare_unified(
            decode_requests=[],
            prefill_requests=prefill_requests,
            block_size=self._mtp_block_size,
        )
        cache = self._require_mtp_cache()
        try:
            packed_hidden = (
                hidden_parts[0]
                if len(hidden_parts) == 1
                else mx.concatenate(hidden_parts, axis=0)
            )
            mtp_hidden = self.mtp_module(
                packed_hidden[None],
                mx.array([flat_tokens], dtype=mx.uint32),
                self.model.model.embed_tokens,
                [None] * self.num_layers,
            )
            if draft_indices:
                end_rows = mx.array(
                    [segment_end_rows[index] for index in draft_indices],
                    dtype=mx.int32,
                )
                selected_hidden = mtp_hidden[0, end_rows]
                selected_logits = self._project_mtp_hidden(selected_hidden)
                draft_ids = mx.argmax(selected_logits, axis=-1)
                mx.eval(
                    draft_ids,
                    *cache.key_caches,
                    *cache.value_caches,
                )
                return [int(token) for token in draft_ids.tolist()]

            mx.eval(*cache.key_caches, *cache.value_caches)
            return []
        finally:
            clear_context()

    def run_pairs(
        self,
        *,
        hidden_rows: mx.array,
        next_token_ids: Sequence[int],
        block_ids_by_group: Sequence[Sequence[int]],
        start_pos: int,
    ) -> int:
        drafts = self.run_pairs_batch(
            hidden_rows_batch=[hidden_rows],
            next_token_ids_batch=[next_token_ids],
            block_ids_by_group_batch=[block_ids_by_group],
            start_positions=[start_pos],
        )
        if len(drafts) != 1:
            raise RuntimeError("Qwen MTP single-request draft result is missing")
        return drafts[0]

    def copy_blocks(self, block_copies: Sequence[tuple[int, int]]) -> None:
        if not self.ready:
            return
        self._require_mtp_cache().copy_blocks(block_copies)
        self._require_boundary_cache().copy_blocks(block_copies)

    def extend_forward_eval_outputs(self, outputs: list[mx.array]) -> None:
        if self.boundary_cache is not None:
            outputs.extend([self.boundary_cache.cache, self.boundary_cache.valid])

    def _require_ready(self) -> None:
        if not self.ready:
            raise RuntimeError("Qwen MTP paged state is not scheduler-configured")

    def _require_mtp_cache(self) -> MetalPagedKVCache:
        if self.mtp_cache is None:
            raise RuntimeError("Qwen MTP paged KV accessed before initialize()")
        return self.mtp_cache

    def _require_boundary_cache(self) -> QwenMTPBoundaryHiddenCache:
        if self.boundary_cache is None:
            raise RuntimeError("Qwen MTP boundary cache accessed before initialize()")
        return self.boundary_cache
