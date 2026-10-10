# SPDX-License-Identifier: Apache-2.0
"""vLLM attention backend for the experimental PyTorch MPS runner."""

from dataclasses import dataclass

import torch
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionMetadata,
    AttentionMetadataBuilder,
    AttentionType,
)

from vllm_metal.pytorch_backend.mps_ops import get_mps_ops


@dataclass
class MPSAttentionMetadata(AttentionMetadata):
    slot_mapping: torch.Tensor
    block_tables: torch.Tensor
    seq_lens: torch.Tensor
    cu_seqlens: torch.Tensor
    max_seq_len: int
    window_seqlen_q: int = 1


class MPSAttentionBackend(AttentionBackend):
    @staticmethod
    def get_name():
        return "CUSTOM"

    @staticmethod
    def get_supported_kernel_block_sizes(kv_cache_spec=None):
        return [8, 16, 32]

    @staticmethod
    def get_supported_head_sizes():
        return [64, 96, 128, 256, 512]

    @classmethod
    def supports_sliding_window(cls):
        return True

    @staticmethod
    def get_impl_cls():
        return MPSAttentionImpl

    @staticmethod
    def get_builder_cls():
        return MPSAttentionMetadataBuilder


class MPSAttentionMetadataBuilder(AttentionMetadataBuilder[MPSAttentionMetadata]):
    def __init__(self, kv_cache_spec, layer_names, vllm_config, device):
        super().__init__(kv_cache_spec, layer_names, vllm_config, device)

    def build(self, common_prefix_len, common_attn_metadata, **kwargs):
        m = common_attn_metadata
        # MRv2 supplies these flags on CPU. As on MLX, any true prefill keeps
        # the batch on the prefill route; only decode/verification uses windows.
        window_seqlen_q = 1
        if (
            m.max_query_len > 1
            and m.is_prefilling is not None
            and not m.is_prefilling.any()
        ):
            window_seqlen_q = m.max_query_len
        return MPSAttentionMetadata(
            m.slot_mapping,
            m.block_table_tensor,
            m.seq_lens,
            m.query_start_loc,
            m.max_seq_len,
            window_seqlen_q,
        )


class MPSAttentionImpl(AttentionImpl[MPSAttentionMetadata]):
    def __init__(
        self,
        num_heads,
        head_size,
        scale,
        num_kv_heads=None,
        alibi_slopes=None,
        sliding_window=None,
        kv_cache_dtype="auto",
        logits_soft_cap=None,
        attn_type=AttentionType.DECODER,
        kv_sharing_target_layer_name=None,
        **kwargs,
    ):
        if (
            head_size not in MPSAttentionBackend.get_supported_head_sizes()
            or alibi_slopes is not None
            or attn_type != AttentionType.DECODER
            or kv_sharing_target_layer_name is not None
        ):
            raise NotImplementedError(
                "MPS requires causal attention with a supported head size, "
                "without ALiBi or cross-layer KV sharing"
            )
        self.num_heads = num_heads
        self.head_size = head_size
        self.num_kv_heads = num_kv_heads or num_heads
        self.scale = scale
        self.sliding_window = sliding_window if sliding_window is not None else -1
        self.logits_soft_cap = logits_soft_cap or 0.0
        self.kv_cache_dtype = kv_cache_dtype
        self.ops = get_mps_ops()

    def forward(
        self,
        layer,
        query,
        key,
        value,
        kv_cache,
        attn_metadata,
        output,
        output_scale=None,
        output_block_scale=None,
    ):
        if output_scale is not None or output_block_scale is not None:
            raise NotImplementedError("MPS output quantization is unsupported")
        m = attn_metadata
        if m is None:
            return output.zero_()
        # Upstream's LBNHC cache is [block, head, token, K+V].
        kv_cache = kv_cache.transpose(1, 2)
        self.ops.forward(
            query.contiguous(),
            key,
            value,
            kv_cache[..., : self.head_size],
            kv_cache[..., self.head_size :],
            m.slot_mapping,
            m.block_tables,
            m.seq_lens,
            m.cu_seqlens,
            output,
            m.max_seq_len,
            self.scale,
            self.sliding_window,
            self.logits_soft_cap,
            m.window_seqlen_q,
        )
        return output
