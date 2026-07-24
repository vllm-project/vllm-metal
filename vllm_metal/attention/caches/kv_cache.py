# SPDX-License-Identifier: Apache-2.0
"""MLX-backed paged KV cache for native Metal paged attention.

Stores per-layer key/value caches as MLX arrays.  KV is written via
MLX-native scatter (Python, donation-friendly) and read by the v2
paged-attention kernels:

- key_cache:   [num_blocks, block_size, num_kv_heads, head_dim]
- value_cache: [num_blocks, block_size, num_kv_heads, head_dim]

Both caches use the same token-contiguous layout where each token's
KV vector is stored contiguously.  This simplifies indexing and is
compatible with variable-length / FlashInfer-style paged attention.

Block allocation is managed externally by the scheduler's KV cache manager.
"""

from __future__ import annotations

import mlx.core as mx
from vllm.logger import init_logger

from vllm_metal.attention.caches.mha_layout import MHAKVCacheLayout
from vllm_metal.attention.caches.turboquant import (
    BLOCK_SIZE,
    FWHT_SUPPORTED_HEAD_DIMS,
    QUANT_PARAMS,
    V_QUANT_PARAMS,
    packed_dim,
)

logger = init_logger(__name__)


class MetalPagedKVCache:
    """Per-layer MLX arrays for native Metal paged attention.

    Supports heterogeneous per-layer shapes: when ``kv_heads_per_layer``
    and ``head_dim_per_layer`` are provided, each cache layer is allocated
    with its own ``(num_kv_heads, head_dim)`` pair.  When omitted, all
    layers share the scalar ``num_kv_heads`` / ``head_dim`` (backward
    compat for MLA, Hybrid, and uniform MHA models).
    """

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
        layout: MHAKVCacheLayout | None = None,
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
        self._layout = layout

        if layout is not None and turboquant:
            raise ValueError("layout-backed KV caches do not support TurboQuant")

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
        self._key_slots: list[mx.array] = []
        self._value_slots: list[mx.array] = []
        if not turboquant:
            if layout is None:
                self._allocate_dense_caches(dtype)
                self._log_dense_cache(dtype)
            else:
                self._allocate_layout_caches(layout, dtype)
                self._log_layout_cache(layout)
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

    def _allocate_layout_caches(
        self, layout: MHAKVCacheLayout, dtype: mx.Dtype
    ) -> None:
        """Allocate shared physical K/V slots and per-layer logical views."""
        for slot_layers in layout.slot_layers:
            shape = layout.layers[slot_layers[0]].cache_shape(self.num_blocks)
            self._key_slots.append(mx.zeros(shape, dtype=dtype))
            self._value_slots.append(mx.zeros(shape, dtype=dtype))
        mx.eval(*self._key_slots, *self._value_slots)

        for layer in layout.layers:
            self.key_caches.append(
                self._key_slots[layer.tensor_index].reshape(
                    layer.cache_shape(self.num_blocks)
                )
            )
            self.value_caches.append(
                self._value_slots[layer.tensor_index].reshape(
                    layer.cache_shape(self.num_blocks)
                )
            )

    def _log_dense_cache(self, dtype: mx.Dtype) -> None:
        kv_bytes = (
            self.num_layers
            * self.num_blocks
            * self.block_size
            * self.num_kv_heads
            * self.head_dim
            * 2
            * self._dtype_size(dtype)
        )
        logger.info(
            f"KV cache: {kv_bytes / 1e6:.1f} MB "
            f"({self.num_layers} layers, {self.num_blocks} blocks, "
            f"{self.block_size} tokens/block)"
        )

    def _log_layout_cache(self, layout: MHAKVCacheLayout) -> None:
        logger.info(
            f"KV cache: {layout.total_bytes / 1e6:.1f} MB "
            f"({len(layout.slot_layers)} physical slots across "
            f"{len(layout.group_block_sizes)} groups, {self.num_blocks} blocks)"
        )

    @classmethod
    def from_layout(
        cls, layout: MHAKVCacheLayout, dtype: mx.Dtype
    ) -> MetalPagedKVCache:
        """Allocate one physical K/V pair for every upstream tensor slot."""
        first_layer = layout.layers[0]
        return cls(
            num_layers=len(layout.layers),
            num_kv_heads=first_layer.num_kv_heads,
            head_dim=first_layer.head_dim,
            num_blocks=layout.num_blocks,
            block_size=first_layer.block_size,
            dtype=dtype,
            kv_heads_per_layer=[layer.num_kv_heads for layer in layout.layers],
            head_dim_per_layer=[layer.head_dim for layer in layout.layers],
            sliding_window_per_layer=[layer.sliding_window for layer in layout.layers],
            layout=layout,
        )

    def group_index_for_layer(self, layer_idx: int) -> int:
        """Return the vLLM cache-group index for ``layer_idx``."""
        return 0 if self._layout is None else self._layout.layers[layer_idx].group_index

    def block_size_for_layer(self, layer_idx: int) -> int:
        """Return the vLLM page size for ``layer_idx``."""
        return (
            self.block_size
            if self._layout is None
            else self._layout.layers[layer_idx].block_size
        )

    def replace_layer_cache(
        self, layer_idx: int, key_cache: mx.array, value_cache: mx.array
    ) -> None:
        """Rebind a native primitive result and every layer sharing its slot."""
        if self._layout is None:
            self.key_caches[layer_idx] = key_cache
            self.value_caches[layer_idx] = value_cache
            return

        slot = self._layout.layers[layer_idx].tensor_index
        self._key_slots[slot] = key_cache
        self._value_slots[slot] = value_cache
        for shared_layer in self._layout.slot_layers[slot]:
            shape = self._layout.layers[shared_layer].cache_shape(self.num_blocks)
            self.key_caches[shared_layer] = key_cache.reshape(shape)
            self.value_caches[shared_layer] = value_cache.reshape(shape)

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
