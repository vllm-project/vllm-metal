# SPDX-License-Identifier: Apache-2.0
"""Native PyTorch SDPA attention for the experimental MPS model path."""

from __future__ import annotations

from typing import ClassVar

import torch
from torch.nn import functional
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionMetadataBuilder,
    AttentionType,
    CommonAttentionMetadata,
    MultipleOf,
)

try:
    from vllm.v1.kv_cache_layout import KVCacheLayout
except ImportError:  # vLLM 0.28 and older
    KVCacheLayout = None  # type: ignore[assignment,misc]


class TorchAttentionMetadataBuilder(AttentionMetadataBuilder[CommonAttentionMetadata]):
    """Pass through the common scheduler metadata needed by torch SDPA."""

    def __init__(self, kv_cache_spec, layer_names, vllm_config, device) -> None:
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(
        self,
        common_prefix_len: int,
        common_attn_metadata: CommonAttentionMetadata,
        fast_build: bool = False,
    ) -> CommonAttentionMetadata:
        del common_prefix_len, fast_build
        if common_attn_metadata.causal is not True:
            raise NotImplementedError(
                "PyTorch SDPA attention currently supports causal decoder attention only."
            )
        return common_attn_metadata


class TorchAttentionBackend(AttentionBackend):
    """Portable MPS/CPU backend using only native PyTorch operations."""

    forward_includes_kv_cache_update = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
        torch.float32,
    ]
    supported_kv_cache_dtypes = ["auto", "float16", "bfloat16"]

    @staticmethod
    def get_name() -> str:
        return "CUSTOM"

    @staticmethod
    def get_impl_cls() -> type[TorchAttentionImpl]:
        return TorchAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TorchAttentionMetadataBuilder]:
        return TorchAttentionMetadataBuilder

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(1)]

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        del cache_dtype_str
        return 2, num_blocks, block_size, num_kv_heads, head_size

    @classmethod
    def supported_kv_cache_layouts(cls):
        """Prefer main's head-major packed per-layer cache view."""
        if KVCacheLayout is None:
            return None
        return (KVCacheLayout.LBHNC,)

    @classmethod
    def supports_non_causal(cls) -> bool:
        return False

    @classmethod
    def supports_sliding_window(cls) -> bool:
        return True

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False


class TorchAttentionImpl(AttentionImpl):
    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int | None = None,
        alibi_slopes: list[float] | None = None,
        sliding_window: int | None = None,
        kv_cache_dtype: str = "auto",
        logits_soft_cap: float | None = None,
        attn_type: str = AttentionType.DECODER,
        kv_sharing_target_layer_name: str | None = None,
        sinks: torch.Tensor | None = None,
    ) -> None:
        if attn_type != AttentionType.DECODER:
            raise NotImplementedError("Only causal decoder attention is supported.")
        if alibi_slopes is not None:
            raise NotImplementedError(
                "ALiBi is not supported by PyTorch SDPA attention."
            )
        if logits_soft_cap is not None:
            raise NotImplementedError(
                "Logits soft-cap is not supported by PyTorch SDPA attention."
            )
        if sinks is not None:
            raise NotImplementedError(
                "Attention sinks are not supported by PyTorch SDPA attention."
            )
        if kv_sharing_target_layer_name is not None:
            raise NotImplementedError(
                "KV-cache sharing is not supported by PyTorch SDPA attention."
            )
        if kv_cache_dtype not in ("auto", "float16", "bfloat16"):
            raise NotImplementedError(
                f"Quantized KV cache {kv_cache_dtype!r} is not supported."
            )
        num_kv_heads = num_heads if num_kv_heads is None else num_kv_heads
        if num_heads % num_kv_heads:
            raise ValueError("num_heads must be divisible by num_kv_heads.")
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.sliding_window = sliding_window
        self.kv_cache_dtype = kv_cache_dtype

    @staticmethod
    def _cache_views(
        kv_cache: torch.Tensor,
        head_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logical ``[block, token, head, dim]`` K/V views.

        vLLM 0.28 gives backends a 5D split cache. Current main allocates a
        logical ``[block, head, token, 2 * dim]`` per-layer view whose strides
        encode the selected physical layout. Slicing and permuting the latter
        preserves those strides, so cache updates reach the shared allocation.
        """
        if kv_cache.ndim == 5:
            if kv_cache.shape[0] != 2:
                raise ValueError(f"Unexpected split KV cache shape: {kv_cache.shape}")
            return kv_cache[0], kv_cache[1]
        if kv_cache.ndim == 4 and kv_cache.shape[-1] == 2 * head_size:
            key_cache = kv_cache[..., :head_size].permute(0, 2, 1, 3)
            value_cache = kv_cache[..., head_size:].permute(0, 2, 1, 3)
            return key_cache, value_cache
        raise ValueError(f"Unexpected KV cache shape: {kv_cache.shape}")

    @staticmethod
    def _write_cache(
        key: torch.Tensor,
        value: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        num_tokens: int,
    ) -> None:
        block_size = key_cache.shape[1]
        slots = slot_mapping[:num_tokens].to(device=key.device, dtype=torch.long)
        valid = slots >= 0
        slots = slots[valid]
        if slots.numel() == 0:
            return
        blocks = torch.div(slots, block_size, rounding_mode="floor")
        offsets = torch.remainder(slots, block_size)
        key_cache[blocks, offsets] = key[:num_tokens][valid]
        value_cache[blocks, offsets] = value[:num_tokens][valid]

    @staticmethod
    def _gather_cache(
        cache: torch.Tensor,
        block_table_row: torch.Tensor,
        start: int,
        seq_len: int,
    ) -> torch.Tensor:
        block_size = cache.shape[1]
        first_block = start // block_size
        end_block = (seq_len + block_size - 1) // block_size
        blocks = block_table_row[first_block:end_block].to(
            device=cache.device, dtype=torch.long
        )
        offset = start % block_size
        return cache.index_select(0, blocks).flatten(0, 1)[
            offset : offset + seq_len - start
        ]

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: CommonAttentionMetadata | None,
        output: torch.Tensor,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del layer
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("Fused output quantization is not supported.")
        if attn_metadata is None:
            return output

        num_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = self._cache_views(kv_cache, self.head_size)
        self._write_cache(
            key,
            value,
            key_cache,
            value_cache,
            attn_metadata.slot_mapping,
            num_tokens,
        )
        query_starts = attn_metadata.query_start_loc.tolist()
        seq_lens = attn_metadata.seq_lens.tolist()
        repeats = self.num_heads // self.num_kv_heads

        for req_idx, seq_len in enumerate(seq_lens):
            q_start, q_end = query_starts[req_idx : req_idx + 2]
            if q_start == q_end:
                continue
            query_len = q_end - q_start
            key_start = 0
            if self.sliding_window is not None:
                key_start = max(0, seq_len - query_len - self.sliding_window + 1)
            keys = self._gather_cache(
                key_cache,
                attn_metadata.block_table_tensor[req_idx],
                key_start,
                seq_len,
            )
            values = self._gather_cache(
                value_cache,
                attn_metadata.block_table_tensor[req_idx],
                key_start,
                seq_len,
            )
            if repeats != 1:
                keys = keys.repeat_interleave(repeats, dim=1)
                values = values.repeat_interleave(repeats, dim=1)

            query_positions = torch.arange(
                seq_len - query_len, seq_len, device=query.device
            )
            key_positions = torch.arange(key_start, seq_len, device=query.device)
            mask = key_positions[None, :] <= query_positions[:, None]
            if self.sliding_window is not None:
                mask &= key_positions[None, :] > (
                    query_positions[:, None] - self.sliding_window
                )

            result = functional.scaled_dot_product_attention(
                query[q_start:q_end].transpose(0, 1).unsqueeze(0),
                keys.transpose(0, 1).unsqueeze(0),
                values.transpose(0, 1).unsqueeze(0),
                attn_mask=mask[None, None],
                dropout_p=0.0,
                scale=self.scale,
            )
            output_slice = output[q_start:q_end]
            output_slice.copy_(
                result.squeeze(0).transpose(0, 1).reshape_as(output_slice)
            )
        return output
