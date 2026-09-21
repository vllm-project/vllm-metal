# SPDX-License-Identifier: Apache-2.0
"""Key/value views consumed by native Metal paged attention.

Target attention binds vLLM's allocation through ``from_upstream``.
Model-based speculative decoding still uses the count-based constructor.
Native writes return aliased handles whose dependencies are retained by the
cache owner. Scheduler block tables determine which pages each request uses.
"""

from __future__ import annotations

from collections.abc import Sequence

import mlx.core as mx
import torch
from vllm.logger import init_logger

from vllm_metal.attention.caches.turboquant import (
    BLOCK_SIZE,
    FWHT_SUPPORTED_HEAD_DIMS,
    QUANT_PARAMS,
    V_QUANT_PARAMS,
    packed_dim,
)

logger = init_logger(__name__)


class MetalPagedKVCache:
    """Native attention cache views, with allocation delegated to vLLM.

    The count-based constructor remains for model-based speculative decoding.
    """

    @classmethod
    def from_upstream(cls, storage, names, *, dtype=mx.float16):
        """Bind attention views from vLLM's resolved cache layout."""
        specs = [storage.specs[name] for name in names]
        turboquant = hasattr(specs[0], "k_quant")
        cache = cls(
            num_layers=len(names),
            num_kv_heads=specs[0].num_kv_heads,
            head_dim=specs[0].head_size,
            num_blocks=storage.config.num_blocks,
            block_size=specs[0].block_size,
            dtype=dtype,
            turboquant=turboquant,
            k_quant=specs[0].k_quant if turboquant else None,
            v_quant=specs[0].v_quant if turboquant else None,
            _allocate=False,
        )
        cache.kv_heads_per_layer = [s.num_kv_heads for s in specs]
        cache.head_dim_per_layer = [s.head_size for s in specs]
        cache.sliding_window_per_layer = [
            getattr(s, "sliding_window", None) or -1 for s in specs
        ]
        groups = [
            group
            for group in storage.config.kv_cache_groups
            if any(name in names for name in group.layer_names)
        ]
        cache._upstream_group_indices = [
            next(i for i, group in enumerate(groups) if name in group.layer_names)
            for name in names
        ]
        cache._upstream_block_sizes = [spec.block_size for spec in specs]
        keys, values, key_scales, value_scales, zeros = [], [], [], [], []
        for name, spec in zip(names, specs, strict=True):
            kv = storage.tensors[name].transpose(1, 2)
            if turboquant:
                # K, V and all scales occupy the upstream content dimension.
                offset = 0
                components = []
                for width, dtype in (
                    (
                        cache.k_packed_dim,
                        torch.int8 if cache.k_size == mx.int8 else torch.uint8,
                    ),
                    (cache.v_packed_dim, torch.uint8),
                    (spec.head_size // BLOCK_SIZE, torch.float16),
                    (spec.head_size // BLOCK_SIZE, torch.float16),
                    (spec.head_size // BLOCK_SIZE, torch.float16),
                ):
                    size = width * dtype.itemsize
                    components.append(kv[..., offset : offset + size].view(dtype))
                    offset += size
                key, value, ks, vs, zp = components
                key_scales.append(ks)
                value_scales.append(vs)
                zeros.append(zp)
            else:
                key, value = kv.split((spec.head_size, spec.head_size_v), dim=-1)
            keys.append(key)
            values.append(value)
        cache.key_caches = storage.views(keys)
        cache.value_caches = storage.views(values)
        if turboquant:
            cache.key_scale_caches = storage.views(key_scales)
            cache.value_scale_caches = storage.views(value_scales)
            cache.key_zero_caches = storage.views(zeros)
        else:
            cache.dtype = cache.key_caches[0].dtype
        cache._storage = storage
        return cache

    def __init__(
        self,
        num_layers: int,
        num_kv_heads: int,
        head_dim: int,
        num_blocks: int,
        block_size: int,
        dtype: mx.Dtype = mx.float16,
        turboquant: bool = False,
        k_quant: str | None = None,
        v_quant: str | None = None,
        *,
        kv_heads_per_layer: list[int] | None = None,
        head_dim_per_layer: list[int] | None = None,
        sliding_window_per_layer: list[int] | None = None,
        _allocate: bool = True,
    ) -> None:
        self.num_layers = num_layers
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim
        self.num_blocks = num_blocks
        self.block_size = block_size
        self.dtype = dtype
        self.turboquant = turboquant
        self.k_quant = k_quant
        self.v_quant = v_quant

        if turboquant:
            if k_quant is None or k_quant not in QUANT_PARAMS:
                available = ", ".join(sorted(QUANT_PARAMS.keys()))
                raise ValueError(
                    f"turboquant requires valid k_quant, got {k_quant!r}. "
                    f"Available: {available}"
                )
            # Default v_quant to "q3_0" (3-bit Lloyd-Max)
            if v_quant is None:
                v_quant = "q3_0"
                self.v_quant = v_quant
            if v_quant not in V_QUANT_PARAMS:
                available = ", ".join(sorted(V_QUANT_PARAMS.keys()))
                raise ValueError(
                    f"turboquant requires valid v_quant, got {v_quant!r}. "
                    f"Available: {available}"
                )
            if head_dim % 32 != 0:
                raise ValueError(
                    f"TurboQuant requires head_dim divisible by 32, got {head_dim}"
                )
            if head_dim not in FWHT_SUPPORTED_HEAD_DIMS:
                supported_head_dims = ", ".join(
                    str(dim) for dim in FWHT_SUPPORTED_HEAD_DIMS
                )
                raise ValueError(
                    "TurboQuant V FWHT only supports "
                    f"head_dim in ({supported_head_dims}), got {head_dim}"
                )

        self.k_size = QUANT_PARAMS[k_quant]["dtype"] if turboquant else None

        # Bit-packed dimensions for sub-8-bit quant types
        if turboquant:
            k_bits = QUANT_PARAMS[k_quant]["bits"]
            v_bits = V_QUANT_PARAMS[v_quant]["bits"]
            self.k_bits = k_bits
            self.v_bits = v_bits
            self.k_packed_dim = packed_dim(head_dim, k_bits)
            self.v_packed_dim = packed_dim(head_dim, v_bits)
        else:
            self.k_bits = 0
            self.v_bits = 0
            self.k_packed_dim = head_dim
            self.v_packed_dim = head_dim

        self.kv_heads_per_layer = kv_heads_per_layer or [num_kv_heads] * num_layers
        self.head_dim_per_layer = head_dim_per_layer or [head_dim] * num_layers
        self.sliding_window_per_layer = sliding_window_per_layer or [-1] * num_layers

        if len(self.kv_heads_per_layer) != num_layers:
            raise ValueError(
                f"kv_heads_per_layer length {len(self.kv_heads_per_layer)} "
                f"!= num_layers {num_layers}"
            )
        if len(self.head_dim_per_layer) != num_layers:
            raise ValueError(
                f"head_dim_per_layer length {len(self.head_dim_per_layer)} "
                f"!= num_layers {num_layers}"
            )
        if len(self.sliding_window_per_layer) != num_layers:
            raise ValueError(
                f"sliding_window_per_layer length "
                f"{len(self.sliding_window_per_layer)} != num_layers {num_layers}"
            )

        if dtype not in (mx.float16, mx.bfloat16, mx.float32):
            raise ValueError(f"Unsupported dtype for paged KV cache: {dtype}")

        # Per-layer caches — each layer sized by its own (kv_heads, head_dim)
        self.key_caches: list[mx.array] = []
        self.value_caches: list[mx.array] = []
        self.key_scale_caches: list[mx.array] = []
        self.value_scale_caches: list[mx.array] = []
        self.key_zero_caches: list[mx.array] = []  # asymmetric K zero_point
        if not _allocate:
            return
        if not turboquant:
            self._allocate_dense_caches(dtype)
            self._log_dense_cache()
        else:
            for _ in range(num_layers):
                self.key_caches.append(
                    mx.zeros(
                        (num_blocks, block_size, num_kv_heads, self.k_packed_dim),
                        dtype=self.k_size,
                    )
                )
                self.value_caches.append(
                    mx.zeros(
                        (num_blocks, block_size, num_kv_heads, self.v_packed_dim),
                        dtype=mx.uint8,
                    )
                )
                self.key_scale_caches.append(
                    mx.zeros(
                        (num_blocks, block_size, num_kv_heads, head_dim // BLOCK_SIZE),
                        dtype=mx.float16,
                    )
                )
                self.value_scale_caches.append(
                    mx.zeros(
                        (num_blocks, block_size, num_kv_heads, head_dim // BLOCK_SIZE),
                        dtype=mx.float16,
                    )
                )
                self.key_zero_caches.append(
                    mx.zeros(
                        (num_blocks, block_size, num_kv_heads, head_dim // BLOCK_SIZE),
                        dtype=mx.float16,
                    )
                )
            mx.eval(
                *self.key_caches,
                *self.value_caches,
                *self.key_scale_caches,
                *self.value_scale_caches,
                *self.key_zero_caches,
            )

            # Log TurboQuant KV cache memory usage with comparison.
            from vllm_metal.v1.cache_policy import turboquant_page_size_bytes

            per_block_bytes = turboquant_page_size_bytes(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                k_quant=k_quant,
                v_quant=v_quant,
            )
            tq_total = num_layers * num_blocks * per_block_bytes
            fp16_equivalent = (
                num_layers * num_blocks * block_size * num_kv_heads * head_dim * 2 * 2
            )  # fp16 K+V
            compression = fp16_equivalent / tq_total if tq_total > 0 else float("inf")
            logger.info(
                f"TurboQuant KV cache (packed): {tq_total / 1e6:.1f} MB "
                f"(K: {self.k_bits}b->{self.k_packed_dim}d, V: {self.v_bits}b->{self.v_packed_dim}d, "
                f"vs {fp16_equivalent / 1e6:.1f} MB fp16, {compression:.2f}x compression)"
            )

    def _allocate_dense_caches(self, dtype: mx.Dtype) -> None:
        """Allocate one independent K/V pair for each logical layer."""
        for i in range(self.num_layers):
            shape = (
                self.num_blocks,
                self.block_size,
                self.kv_heads_per_layer[i],
                self.head_dim_per_layer[i],
            )
            self.key_caches.append(mx.zeros(shape, dtype=dtype))
            self.value_caches.append(mx.zeros(shape, dtype=dtype))
        mx.eval(*self.key_caches, *self.value_caches)

    def _log_dense_cache(self) -> None:
        # Sum the allocated arrays instead of multiplying the scalar
        # ``num_kv_heads``/``head_dim``: heterogeneous layers are sized from
        # ``kv_heads_per_layer``/``head_dim_per_layer`` while the scalars carry
        # the widened dispatch shape, so the product over-bills narrow layers.
        kv_bytes = sum(cache.nbytes for cache in self.key_caches) + sum(
            cache.nbytes for cache in self.value_caches
        )
        logger.info(
            f"KV cache: {kv_bytes / 1e6:.1f} MB "
            f"({self.num_layers} layers, {self.num_blocks} blocks, "
            f"{self.block_size} tokens/block)"
        )

    def group_index_for_layer(self, layer_idx: int) -> int:
        """Return the vLLM cache-group index for ``layer_idx``."""
        if hasattr(self, "_storage"):
            return self._upstream_group_indices[layer_idx]
        return 0

    def block_size_for_layer(self, layer_idx: int) -> int:
        """Return the vLLM page size for ``layer_idx``."""
        if hasattr(self, "_storage"):
            return self._upstream_block_sizes[layer_idx]
        return self.block_size

    def replace_layer_cache(
        self, layer_idx: int, key_cache: mx.array, value_cache: mx.array
    ) -> None:
        """Rebind a native primitive result and every layer sharing its slot."""
        self.key_caches[layer_idx] = key_cache
        self.value_caches[layer_idx] = value_cache

    def copy_blocks(self, block_copies: Sequence[tuple[int, int]]) -> None:
        """Apply scheduler copy-on-write operations to physical KV blocks."""
        if hasattr(self, "_storage"):
            self._storage.copy_blocks(block_copies)
            return
        if not block_copies:
            return

        src_ids, dst_ids = zip(*block_copies, strict=True)
        if any(
            block_id < 0 or block_id >= self.num_blocks
            for block_id in (*src_ids, *dst_ids)
        ):
            raise RuntimeError("paged KV block copy contains an out-of-range block id")

        src = mx.array(src_ids, dtype=mx.int32)
        dst = mx.array(dst_ids, dtype=mx.int32)
        arrays = [*self.key_caches, *self.value_caches]
        if self.turboquant:
            arrays.extend(self.key_scale_caches)
            arrays.extend(self.value_scale_caches)
            arrays.extend(self.key_zero_caches)

        for array in arrays:
            array[dst] = array[src]

    _DTYPE_SIZES = {
        mx.float16: 2,
        mx.bfloat16: 2,
        mx.float32: 4,
        mx.int8: 1,
        mx.uint8: 1,
    }

    @staticmethod
    def _dtype_size(dtype: mx.Dtype) -> int:
        """Return size in bytes for an MLX dtype."""
        size = MetalPagedKVCache._DTYPE_SIZES.get(dtype)
        if size is None:
            raise ValueError(
                f"Unknown dtype {dtype} in _dtype_size. "
                f"Supported: {list(MetalPagedKVCache._DTYPE_SIZES.keys())}"
            )
        return size
