# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx


class MLAPagedLatentCache:
    """Per-layer latent views for MLA paged attention.

    Each token's cache entry is a combined latent vector [kv_norm || k_pe]:
      - kv_norm = kv_a_layernorm(compressed_kv) — the normalised KV latent
      - k_pe    = rope(k_pe_raw)                — RoPE-encoded position key

    Layout per layer: [num_blocks, block_size, latent_dim].

    The count-based path (draft-model speculation) allocates its own arrays;
    otherwise MLA binds views of the scheduler-owned shared cache storage.
    """

    @classmethod
    def from_upstream(cls, storage, names) -> MLAPagedLatentCache:
        """Bind latent views from vLLM's resolved cache layout."""
        if not names:
            raise ValueError("MLA cache requires at least one layer")

        specs = [storage.specs[name] for name in names]
        first = specs[0]
        if first.num_kv_heads != 1:
            raise ValueError(
                f"MLA cache layer {names[0]!r} has num_kv_heads="
                f"{first.num_kv_heads}; expected 1"
            )
        expected = (1, first.head_size, first.block_size)
        for name, spec in zip(names[1:], specs[1:], strict=True):
            actual = (spec.num_kv_heads, spec.head_size, spec.block_size)
            if actual != expected:
                raise ValueError(
                    f"MLA cache layer {name!r} has "
                    "(num_kv_heads, head_size, block_size)="
                    f"{actual}; expected {expected} from {names[0]!r}"
                )

        cache = cls(
            num_layers=len(names),
            latent_dim=first.head_size,
            num_blocks=storage.config.num_blocks,
            block_size=first.block_size,
            _allocate=False,
        )
        latent_tensors = []
        for name in names:
            latent = storage.tensors[name].transpose(1, 2)
            if latent.shape[2] != 1:
                raise ValueError("MLA cache requires exactly one latent head")
            latent_tensors.append(latent.squeeze(2))
        cache.latent_caches = storage.views(latent_tensors)
        cache.dtype = cache.latent_caches[0].dtype
        cache.has_dense_pages = all(tensor.is_contiguous() for tensor in latent_tensors)
        cache._storage = storage
        return cache

    def write_slots(self, layer_idx: int, slot_ids: mx.array, values: mx.array) -> None:
        """Scatter latent rows, preserving upstream storage aliases."""
        if self._storage is None:
            flat = self.latent_caches[layer_idx].reshape(-1, self.latent_dim)
            flat[slot_ids] = values
            updated = flat.reshape(self.num_blocks, self.block_size, self.latent_dim)
        else:
            from vllm_metal.metal import get_ops

            # Flattening padded pages would copy and detach the shared backing.
            updated = get_ops().gdn_state_scatter(
                self.latent_caches[layer_idx],
                values.astype(self.dtype),
                slot_ids.astype(mx.int32),
                paged=True,
            )
        self.latent_caches[layer_idx] = updated

    def __init__(
        self,
        num_layers: int,
        latent_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: mx.Dtype = mx.float16,
        *,
        _allocate: bool = True,
    ) -> None:
        if dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError(f"Unsupported dtype for MLA paged cache: {dtype}")

        self.num_layers = num_layers
        self.latent_dim = latent_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype
        self.has_dense_pages = True
        self._storage = None

        self.latent_caches: list[mx.array] = []
        if _allocate:
            self.latent_caches = [
                mx.zeros((num_blocks, block_size, latent_dim), dtype=dtype)
                for _ in range(num_layers)
            ]
            # Force allocation so Metal buffers exist before use
            mx.eval(*self.latent_caches)

    def copy_blocks(self, block_copies: Sequence[tuple[int, int]]) -> None:
        """Apply scheduler copy-on-write operations to latent-cache blocks."""
        if self._storage is not None:
            # An index-assign on a view would never reach the shared backing.
            self._storage.copy_blocks(block_copies)
            return
        if not block_copies:
            return

        src_ids, dst_ids = zip(*block_copies, strict=True)
        if any(
            block_id < 0 or block_id >= self.num_blocks
            for block_id in (*src_ids, *dst_ids)
        ):
            raise RuntimeError("paged MLA block copy contains an out-of-range block id")
        src = mx.array(src_ids, dtype=mx.int32)
        dst = mx.array(dst_ids, dtype=mx.int32)
        for cache in self.latent_caches:
            cache[dst] = cache[src]
