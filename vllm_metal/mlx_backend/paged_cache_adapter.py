# SPDX-License-Identifier: Apache-2.0
"""Paged KV Cache Adapters for mlx_lm model compatibility.

These adapters wrap PagedKVCache to match mlx_lm's KVCache interface,
enabling paged attention with existing mlx_lm models without modification.

The mlx_lm KVCache interface requires:
- update_and_fetch(keys, values) -> (cached_keys, cached_values)
- state property -> (keys, values) tuple
- offset property -> current sequence length
"""

from __future__ import annotations

import mlx.core as mx
from vllm.logger import init_logger

from vllm_metal.mlx_backend.cache import PagedKVCache

logger = init_logger(__name__)


class PagedKVCacheAdapter:
    """Single-layer adapter matching mlx_lm KVCache interface.

    Wraps a PagedKVCache layer to provide the update_and_fetch() interface
    expected by mlx_lm models. Stores K/V in blocks and gathers them into
    contiguous tensors when accessed.

    Supports two modes:
    1. External block IDs (from vLLM scheduler) - use block_ids parameter
    2. Internal allocation (legacy) - allocates blocks dynamically
    """

    def __init__(
        self,
        paged_cache: PagedKVCache,
        seq_id: int,
        layer_idx: int,
        block_size: int,
        block_ids: list[int] | None = None,
    ):
        """Initialize adapter for a single layer.

        Args:
            paged_cache: Shared PagedKVCache instance
            seq_id: Sequence identifier for block allocation
            layer_idx: Transformer layer index
            block_size: Number of tokens per block
            block_ids: Optional external block IDs from scheduler.
                       If provided, uses these instead of internal allocation.
        """
        self.paged_cache = paged_cache
        self.seq_id = seq_id
        self.layer_idx = layer_idx
        self.block_size = block_size
        self._offset = 0  # Current position in sequence
        # External block IDs from scheduler (None = use internal allocation)
        self._external_block_ids: list[int] | None = block_ids
        # Mutable copy for appending new blocks during decode
        self._block_ids: list[int] = list(block_ids) if block_ids else []

        if layer_idx == 0:  # Log once per sequence, not per layer
            logger.debug(
                f"PagedKVCacheAdapter: seq_id={seq_id}, "
                f"external_blocks={block_ids is not None}, "
                f"num_blocks={len(self._block_ids)}"
            )

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Update cache with new K/V and return all cached K/V.

        This matches the mlx_lm KVCache.update_and_fetch() interface.

        Args:
            keys: Shape (batch, num_kv_heads, new_tokens, head_dim)
            values: Shape (batch, num_kv_heads, new_tokens, head_dim)

        Returns:
            Tuple of (all_keys, all_values) accumulated so far,
            each of shape (batch, num_kv_heads, total_tokens, head_dim)
        """
        # Extract dimensions - mlx_lm uses (batch, heads, seq, dim)
        batch_size = keys.shape[0]
        num_kv_heads = keys.shape[1]
        head_dim = keys.shape[3]

        # Transpose for block storage: (new_tokens, num_kv_heads, head_dim)
        # From (batch, heads, seq, dim) -> (seq, heads, dim)
        keys_t = keys[0].transpose(1, 0, 2)
        values_t = values[0].transpose(1, 0, 2)

        # Store tokens into blocks
        self._store_tokens(keys_t, values_t)

        # Gather all cached KV as contiguous tensor
        total_len = self._offset
        if total_len == 0:
            # Return empty tensors with correct shape
            empty_k = mx.zeros(
                (batch_size, num_kv_heads, 0, head_dim),
                dtype=keys.dtype,
            )
            empty_v = mx.zeros(
                (batch_size, num_kv_heads, 0, head_dim),
                dtype=values.dtype,
            )
            return empty_k, empty_v

        # Get KV using block IDs (external or from allocator)
        block_ids = self.get_block_ids()
        cached_k, cached_v = self.paged_cache.get_kv_from_blocks(
            block_ids=block_ids,
            layer_idx=self.layer_idx,
            seq_len=total_len,
        )

        # Transpose back: (seq, heads, dim) -> (batch, heads, seq, dim)
        cached_k = cached_k.transpose(1, 0, 2)[None, ...]
        cached_v = cached_v.transpose(1, 0, 2)[None, ...]

        return cached_k, cached_v

    def _store_tokens(self, keys: mx.array, values: mx.array) -> None:
        """Store tokens into blocks, using scheduler's block IDs or allocating.

        Args:
            keys: Shape (num_tokens, num_kv_heads, head_dim)
            values: Shape (num_tokens, num_kv_heads, head_dim)
        """
        num_tokens = keys.shape[0]
        remaining = num_tokens
        src_offset = 0

        while remaining > 0:
            # Get current block index within sequence
            block_idx_in_seq = self._offset // self.block_size
            slot_in_block = self._offset % self.block_size

            # Get block ID - either from external list or internal allocator
            if self._external_block_ids is not None:
                # Use scheduler-provided block IDs
                if block_idx_in_seq >= len(self._block_ids):
                    logger.error(
                        f"Block allocation failed: seq_id={self.seq_id}, "
                        f"layer={self.layer_idx}, offset={self._offset}, "
                        f"need_block_idx={block_idx_in_seq}, "
                        f"have_blocks={len(self._block_ids)}, "
                        f"block_ids={self._block_ids}"
                    )
                    raise RuntimeError(
                        f"Not enough blocks allocated by scheduler: "
                        f"need block {block_idx_in_seq}, have {len(self._block_ids)}"
                    )
                block_idx = self._block_ids[block_idx_in_seq]
            else:
                # Legacy: use internal allocator
                seq_blocks = self.paged_cache._allocator.get_sequence_blocks(
                    self.seq_id
                )
                while block_idx_in_seq >= len(seq_blocks):
                    self.paged_cache.allocate_blocks(self.seq_id, 1)
                    seq_blocks = self.paged_cache._allocator.get_sequence_blocks(
                        self.seq_id
                    )
                block_idx = seq_blocks[block_idx_in_seq]

            # How many tokens fit in current block
            space_in_block = self.block_size - slot_in_block
            tokens_to_write = min(remaining, space_in_block)

            # Write to block
            self.paged_cache.update_block(
                block_idx=block_idx,
                layer_idx=self.layer_idx,
                key=keys[src_offset : src_offset + tokens_to_write],
                value=values[src_offset : src_offset + tokens_to_write],
                slot_offset=slot_in_block,
            )

            self._offset += tokens_to_write
            src_offset += tokens_to_write
            remaining -= tokens_to_write

    def append_block_ids(self, new_block_ids: list[int]) -> None:
        """Append new block IDs from scheduler during decode.

        Args:
            new_block_ids: Additional block IDs allocated by scheduler
        """
        if new_block_ids:
            logger.debug(
                f"Appending blocks: seq_id={self.seq_id}, "
                f"new_blocks={new_block_ids}, "
                f"total_blocks={len(self._block_ids) + len(new_block_ids)}"
            )
        self._block_ids.extend(new_block_ids)

    def get_block_ids(self) -> list[int]:
        """Return current block IDs for this sequence."""
        if self._external_block_ids is not None:
            return self._block_ids
        return self.paged_cache._allocator.get_sequence_blocks(self.seq_id)

    @property
    def state(self) -> tuple[mx.array, mx.array]:
        """Return cache state as (keys, values) tuple.

        This matches the mlx_lm KVCache.state property for mx.eval().
        """
        if self._offset == 0:
            return (
                mx.zeros(
                    (0, self.paged_cache.num_kv_heads, self.paged_cache.head_dim),
                    dtype=self.paged_cache.dtype,
                ),
                mx.zeros(
                    (0, self.paged_cache.num_kv_heads, self.paged_cache.head_dim),
                    dtype=self.paged_cache.dtype,
                ),
            )
        block_ids = self.get_block_ids()
        return self.paged_cache.get_kv_from_blocks(
            block_ids=block_ids,
            layer_idx=self.layer_idx,
            seq_len=self._offset,
        )

    @state.setter
    def state(self, value: tuple[mx.array, mx.array]) -> None:
        """Set cache state from (keys, values) tuple.

        Note: This is a no-op for paged cache since state is managed
        through blocks. The setter exists for interface compatibility.
        """
        # For paged cache, state is managed through blocks
        # Setting state directly is not supported
        pass

    @property
    def offset(self) -> int:
        """Return current sequence length."""
        return self._offset


class PagedKVCacheAdapterList:
    """List of per-layer adapters for a single sequence.

    Replaces the result of make_prompt_cache() with paged equivalents.
    Implements the list-like interface expected by mlx_lm models.

    Supports two modes:
    1. External block IDs (from vLLM scheduler) - use block_ids parameter
    2. Internal allocation (legacy) - allocates blocks dynamically
    """

    def __init__(
        self,
        paged_cache: PagedKVCache,
        seq_id: int,
        num_layers: int,
        block_size: int,
        block_ids: list[int] | None = None,
    ):
        """Initialize adapters for all layers.

        Args:
            paged_cache: Shared PagedKVCache instance
            seq_id: Sequence identifier
            num_layers: Number of transformer layers
            block_size: Number of tokens per block
            block_ids: Optional external block IDs from scheduler.
                       If provided, all layers share the same block IDs.
        """
        self.paged_cache = paged_cache
        self.seq_id = seq_id
        self._use_external_blocks = block_ids is not None
        self._adapters = [
            PagedKVCacheAdapter(
                paged_cache, seq_id, layer_idx, block_size, block_ids=block_ids
            )
            for layer_idx in range(num_layers)
        ]

    def __getitem__(self, idx: int) -> PagedKVCacheAdapter:
        """Get adapter for a specific layer."""
        return self._adapters[idx]

    def __len__(self) -> int:
        """Return number of layers."""
        return len(self._adapters)

    def __iter__(self):
        """Iterate over layer adapters."""
        return iter(self._adapters)

    @property
    def state(self) -> tuple[mx.array, ...]:
        """Aggregate state for mx.eval().

        Returns states from all layer adapters.
        """
        states = []
        for adapter in self._adapters:
            k, v = adapter.state
            states.extend([k, v])
        return tuple(states)

    @property
    def offset(self) -> int:
        """Return current sequence length (same for all layers)."""
        if self._adapters:
            return self._adapters[0].offset
        return 0

    def append_block_ids(self, new_block_ids: list[int]) -> None:
        """Append new block IDs from scheduler during decode.

        Args:
            new_block_ids: Additional block IDs allocated by scheduler
        """
        for adapter in self._adapters:
            adapter.append_block_ids(new_block_ids)

    def get_block_ids(self) -> list[int]:
        """Return current block IDs for this sequence."""
        if self._adapters:
            return self._adapters[0].get_block_ids()
        return []

    def free(self) -> None:
        """Release all blocks for this sequence back to the pool.

        For internal allocation, frees blocks via the allocator.
        For external block IDs, the scheduler manages block lifecycle,
        but we still need to clean up internal state.
        """
        offset = self._adapters[0].offset if self._adapters else 0
        num_blocks = len(self._adapters[0]._block_ids) if self._adapters else 0

        if not self._use_external_blocks:
            logger.debug(
                f"Freeing sequence: seq_id={self.seq_id}, "
                f"offset={offset}, blocks={num_blocks}"
            )
            self.paged_cache.free_sequence(self.seq_id)
        else:
            # External blocks are managed by scheduler, but log for debugging
            logger.debug(
                f"Sequence complete (external blocks): seq_id={self.seq_id}, "
                f"offset={offset}, blocks={num_blocks}"
            )


class PagedBatchKVCacheAdapterLayer:
    """Per-layer view into batched paged cache."""

    def __init__(
        self,
        adapters: list[PagedKVCacheAdapterList],
        layer_idx: int,
    ):
        """Initialize batched layer adapter.

        Args:
            adapters: List of per-sequence adapter lists
            layer_idx: Transformer layer index
        """
        self.adapters = adapters
        self.layer_idx = layer_idx
        self.paged_cache: PagedKVCache | None = (
            adapters[0].paged_cache if adapters else None
        )

    def update_and_fetch(
        self,
        keys: mx.array,
        values: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Batched update and fetch.

        Args:
            keys: Shape (batch_size, num_kv_heads, 1, head_dim) - single token per seq
            values: Same shape as keys

        Returns:
            Batched (keys, values) with left-padding applied for variable lengths
        """
        batch_size = keys.shape[0]
        num_kv_heads = keys.shape[1]
        head_dim = keys.shape[3]

        # Update each sequence's cache
        for i in range(batch_size):
            adapter = self.adapters[i][self.layer_idx]
            k_i = keys[i : i + 1]  # Keep batch dim
            v_i = values[i : i + 1]
            # Store tokens - transpose from (1, heads, 1, dim) to (1, heads, dim)
            adapter._store_tokens(
                k_i[0].transpose(1, 0, 2),
                v_i[0].transpose(1, 0, 2),
            )

        # Gather all cached KV with left-padding for attention
        seq_lens = [a[self.layer_idx].offset for a in self.adapters]
        max_len = max(seq_lens) if seq_lens else 0

        if max_len == 0:
            empty_k = mx.zeros(
                (batch_size, num_kv_heads, 0, head_dim),
                dtype=keys.dtype,
            )
            empty_v = mx.zeros(
                (batch_size, num_kv_heads, 0, head_dim),
                dtype=values.dtype,
            )
            return empty_k, empty_v

        all_keys = []
        all_values = []

        for adapter_list in self.adapters:
            adapter = adapter_list[self.layer_idx]
            seq_len = adapter.offset

            # Get this sequence's KV using adapter's block IDs
            # (not the internal allocator, which doesn't know about scheduler-provided blocks)
            if self.paged_cache is None:
                raise RuntimeError("Paged cache not initialized")
            cached_k, cached_v = self.paged_cache.get_kv_from_blocks(
                block_ids=adapter.get_block_ids(),
                layer_idx=self.layer_idx,
                seq_len=seq_len,
            )

            # cached_k/v shape: (seq_len, num_kv_heads, head_dim)
            # Left-pad to max_len
            pad_len = max_len - seq_len
            if pad_len > 0:
                pad_k = mx.zeros(
                    (pad_len, num_kv_heads, head_dim),
                    dtype=cached_k.dtype,
                )
                pad_v = mx.zeros(
                    (pad_len, num_kv_heads, head_dim),
                    dtype=cached_v.dtype,
                )
                cached_k = mx.concatenate([pad_k, cached_k], axis=0)
                cached_v = mx.concatenate([pad_v, cached_v], axis=0)

            # Transpose: (seq_len, heads, dim) -> (heads, seq_len, dim)
            all_keys.append(cached_k.transpose(1, 0, 2))
            all_values.append(cached_v.transpose(1, 0, 2))

        # Stack into batch: (batch, heads, seq, dim)
        batched_k = mx.stack(all_keys, axis=0)
        batched_v = mx.stack(all_values, axis=0)

        return batched_k, batched_v

    @property
    def state(self) -> tuple[mx.array, ...]:
        """Return states for mx.eval()."""
        states = []
        for adapter_list in self.adapters:
            k, v = adapter_list[self.layer_idx].state
            states.extend([k, v])
        return tuple(states)

    @property
    def offset(self) -> mx.array:
        """Per-sequence offsets for attention masking."""
        return mx.array([a[self.layer_idx].offset for a in self.adapters])


class PagedBatchKVCacheAdapter:
    """Batched adapter for multiple sequences sharing the same PagedKVCache.

    Since all sequences share the same underlying block pool, "merging"
    just tracks which sequences are batched together.
    """

    def __init__(self, adapters: list[PagedKVCacheAdapterList]):
        """Initialize batched adapter.

        Args:
            adapters: List of per-sequence adapter lists
        """
        self.adapters = adapters
        self.paged_cache = adapters[0].paged_cache if adapters else None
        self.num_layers = len(adapters[0]) if adapters else 0

    @classmethod
    def merge(
        cls, adapters: list[PagedKVCacheAdapterList]
    ) -> list[PagedBatchKVCacheAdapterLayer]:
        """Merge individual caches into batched per-layer caches.

        This is the equivalent of BatchKVCache.merge() for paged caches.

        Args:
            adapters: List of per-sequence adapter lists

        Returns:
            List of per-layer batch cache adapters
        """
        if not adapters:
            return []

        batch = cls(adapters)
        seq_lens = [a.offset for a in adapters]
        logger.debug(
            f"Merging batch: size={len(adapters)}, "
            f"seq_lens={seq_lens}, layers={batch.num_layers}"
        )
        return [
            PagedBatchKVCacheAdapterLayer(adapters, layer_idx)
            for layer_idx in range(batch.num_layers)
        ]

    def extract(self, idx: int) -> PagedKVCacheAdapterList:
        """Extract a single sequence's cache from the batch.

        Since adapters share the same PagedKVCache, this just returns
        the original adapter list.

        Args:
            idx: Index of the sequence in the batch

        Returns:
            The original adapter list for that sequence
        """
        return self.adapters[idx]
