# SPDX-License-Identifier: Apache-2.0
"""Cache-policy ownership for the v1 Metal runtime."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import mlx.core as mx
import torch
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec

from vllm_metal.paged_attention_backend.hybrid import (
    HybridPagedAttentionBackend,
    _build_linear_layer_spec,
)
from vllm_metal.paged_attention_backend.mha import MHAPagedAttentionBackend
from vllm_metal.paged_attention_backend.mla import MLAPagedAttentionBackend
from vllm_metal.paged_attention_backend.protocol import PagedAttentionBackend
from vllm_metal.pytorch_backend.tensor_bridge import MLX_TO_TORCH_DTYPE
from vllm_metal.stt.policy import STT_SCHED_BLOCK_BYTES
from vllm_metal.v1.model_adapter import ModelAdapter

if TYPE_CHECKING:
    from vllm_metal.v1.model_runner import MetalModelRunner

logger = init_logger(__name__)


class ModelCachePolicy:
    """Cache shape, size, and backend-selection policy for one runner."""

    def __init__(self, runner: MetalModelRunner, model_adapter: ModelAdapter) -> None:
        self._runner = runner
        self._model_adapter = model_adapter

    def should_setup_paged_attention(self) -> bool:
        """Whether worker-side paged-attention setup should run."""
        return not self._runner._is_stt

    def validate_paged_attention_support(self) -> None:
        """Validate that the loaded model can run on the paged-attention path."""
        self._model_adapter.require_uniform_kv_heads(
            self._runner.model_args,
            self._runner.num_kv_heads,
        )

    def scheduler_memory_reporting_mode(
        self, *, paged_attention_enabled: bool
    ) -> Literal["stt_nominal", "paged_attention_capacity", "single_sequence_estimate"]:
        """Return which scheduler memory-reporting mode worker should use."""
        if self._runner._is_stt:
            return "stt_nominal"
        if paged_attention_enabled:
            return "paged_attention_capacity"
        return "single_sequence_estimate"

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Build the scheduler-visible KV cache specification."""
        runner = self._runner
        if runner._is_stt:
            return {
                "layers.0.self_attn": FullAttentionSpec(
                    block_size=runner.metal_config.block_size,
                    num_kv_heads=1,
                    head_size=64,
                    dtype=torch.float16,
                ),
            }

        block_size = runner.cache_config.block_size
        if runner.kv_cache_dtype is None:
            raise RuntimeError("KV cache dtype not initialized; load_model() first")

        torch_dtype = MLX_TO_TORCH_DTYPE[runner.kv_cache_dtype]
        specs: dict[str, KVCacheSpec] = {}
        for layer_idx in range(runner.num_layers):
            if runner.is_hybrid and layer_idx not in runner.sdpa_layer_indices:
                layer_name = f"layers.{layer_idx}.linear_attn"
                specs[layer_name] = _build_linear_layer_spec(
                    conv_kernel_dim=runner.linear_conv_kernel_dim,
                    conv_dim=runner.linear_conv_dim,
                    num_v_heads=runner.linear_num_v_heads,
                    value_head_dim=runner.linear_value_head_dim,
                    key_head_dim=runner.linear_key_head_dim,
                    torch_dtype=torch_dtype,
                    page_size_padded=runner.cache_config.mamba_page_size_padded,
                    block_size=block_size,
                )
            else:
                layer_name = f"layers.{layer_idx}.self_attn"
                specs[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=runner.num_kv_heads,
                    head_size=runner.head_dim,
                    dtype=torch_dtype,
                )

        return specs

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Accept engine KV cache config for API compatibility."""
        logger.info(
            "KV cache config received: %d blocks (MLX manages cache internally)",
            kv_cache_config.num_blocks,
        )

    def get_cache_block_size_bytes(self) -> int:
        """Return the byte size of one cache block."""
        runner = self._runner
        if runner._is_stt:
            return STT_SCHED_BLOCK_BYTES

        if runner.kv_cache_dtype is None:
            raise RuntimeError("KV cache dtype not initialized; load_model() first")

        block_size = runner.cache_config.block_size
        dtype_size = runner.kv_cache_dtype.size
        num_kv_layers = (
            runner.num_sdpa_layers if runner.is_hybrid else runner.num_kv_cache_layers
        )
        kv_factor = 1 if runner.is_mla else 2
        return (
            kv_factor
            * num_kv_layers
            * block_size
            * runner.num_kv_heads
            * runner.head_dim
            * dtype_size
        )

    def linear_cache_bytes_per_slot(self) -> int:
        """Return bytes for one request's linear-attention state."""
        runner = self._runner
        if not runner.is_hybrid:
            raise RuntimeError("linear_cache_bytes_per_slot() requires a hybrid model")
        if runner.kv_cache_dtype is None:
            raise RuntimeError("KV cache dtype not initialized; load_model() first")

        dtype_size = runner.kv_cache_dtype.size
        recurrent_dtype_size = mx.float32.size
        conv_bytes = (
            (runner.linear_conv_kernel_dim - 1) * runner.linear_conv_dim * dtype_size
        )
        recurrent_bytes = (
            runner.linear_num_v_heads
            * runner.linear_value_head_dim
            * runner.linear_key_head_dim
            * recurrent_dtype_size
        )
        return runner.num_linear_layers * (conv_bytes + recurrent_bytes)

    def build_paged_attention_backend(
        self, *, block_size: int
    ) -> PagedAttentionBackend:
        """Create the paged-attention backend for the loaded model."""
        runner = self._runner
        if runner.is_hybrid:
            return HybridPagedAttentionBackend(
                num_layers=runner.num_layers,
                full_attention_interval=runner.full_attention_interval,
                max_num_seqs=runner.scheduler_config.max_num_seqs,
                num_kv_heads=runner.num_kv_heads,
                head_dim=runner.head_dim,
                linear_num_v_heads=runner.linear_num_v_heads,
                linear_key_head_dim=runner.linear_key_head_dim,
                linear_value_head_dim=runner.linear_value_head_dim,
                linear_conv_kernel_dim=runner.linear_conv_kernel_dim,
                linear_conv_dim=runner.linear_conv_dim,
                block_size=block_size,
                dtype=runner.kv_cache_dtype,
            )

        if runner.is_mla:
            return MLAPagedAttentionBackend(
                num_layers=runner.num_layers,
                latent_dim=runner.mla_latent_dim,
                block_size=block_size,
                dtype=runner.kv_cache_dtype,
            )

        yoco = runner._yoco_cache_mapping
        if yoco is not None:
            num_cache_layers, cache_idx_map = yoco
            logger.info(
                "YOCO KV sharing: %d unique cache layers (reduced from %d total)",
                num_cache_layers,
                runner.num_layers,
            )
        else:
            num_cache_layers = runner.num_kv_cache_layers
            cache_idx_map = None

        return MHAPagedAttentionBackend(
            num_layers=num_cache_layers,
            num_kv_heads=runner.num_kv_heads,
            head_dim=runner.head_dim,
            block_size=block_size,
            dtype=runner.kv_cache_dtype,
            cache_idx_map=cache_idx_map,
        )

    def estimate_one_sequence_kv_bytes(
        self, *, max_model_len: int, block_size: int
    ) -> int:
        """Estimate bytes for one max-length sequence of cache state."""
        runner = self._runner
        if runner.kv_cache_dtype is None:
            raise RuntimeError("KV cache dtype not initialized; load_model() first")

        dtype_size = runner.kv_cache_dtype.size
        aligned_tokens = -(-max_model_len // block_size) * block_size
        num_kv_layers = (
            runner.num_sdpa_layers if runner.is_hybrid else runner.num_kv_cache_layers
        )
        kv_factor = 1 if runner.is_mla else 2
        sdpa_kv_bytes = (
            kv_factor
            * num_kv_layers
            * aligned_tokens
            * runner.num_kv_heads
            * runner.head_dim
            * dtype_size
        )
        if runner.is_hybrid:
            return sdpa_kv_bytes + self.linear_cache_bytes_per_slot()
        return sdpa_kv_bytes
