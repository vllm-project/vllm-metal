# SPDX-License-Identifier: Apache-2.0
"""Cache-policy ownership for the v1 Metal runtime."""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mlx.core as mx
import torch
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec

from vllm_metal.config import (
    PAGED_ATTENTION_DEFAULT_MEMORY_FRACTION,
    PAGED_ATTENTION_MIN_BLOCKS,
)
from vllm_metal.paged_attention_backend.hybrid import (
    HybridPagedAttentionBackend,
    _build_linear_layer_spec,
)
from vllm_metal.paged_attention_backend.mha import MHAPagedAttentionBackend
from vllm_metal.paged_attention_backend.mla import MLAPagedAttentionBackend
from vllm_metal.paged_attention_backend.protocol import PagedAttentionBackend
from vllm_metal.pytorch_backend.tensor_bridge import MLX_TO_TORCH_DTYPE
from vllm_metal.stt.policy import (
    STT_SCHED_AVAILABLE_BYTES,
    STT_SCHED_BLOCK_BYTES,
    STT_SCHED_NOMINAL_HEAD_SIZE,
)
from vllm_metal.v1.model_adapter import ModelAdapter

if TYPE_CHECKING:
    from vllm_metal.v1.model_runner import MetalModelRunner
    from vllm_metal.v1.worker import MetalWorker

logger = init_logger(__name__)


@dataclass(frozen=True)
class _PagedAttentionPlan:
    block_size: int
    fraction: float
    metal_limit: int
    usable_metal: int
    model_memory: int
    per_block_bytes: int
    kv_budget: int
    num_blocks: int


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
        if self._runner._is_stt:
            return {
                "layers.0.self_attn": FullAttentionSpec(
                    block_size=self._runner.metal_config.block_size,
                    num_kv_heads=1,
                    head_size=STT_SCHED_NOMINAL_HEAD_SIZE,
                    dtype=torch.float16,
                ),
            }

        block_size = self._runner.cache_config.block_size
        torch_dtype = MLX_TO_TORCH_DTYPE[self._require_kv_cache_dtype()]
        kv_heads_list = self._runner.kv_heads_per_layer
        head_dim_list = self._runner.head_dim_per_layer
        has_per_layer = kv_heads_list is not None and head_dim_list is not None
        specs: dict[str, KVCacheSpec] = {}
        for layer_idx in range(self._runner.num_layers):
            if (
                self._runner.is_hybrid
                and layer_idx not in self._runner.sdpa_layer_indices
            ):
                layer_name = f"layers.{layer_idx}.linear_attn"
                specs[layer_name] = _build_linear_layer_spec(
                    conv_kernel_dim=self._runner.linear_conv_kernel_dim,
                    conv_dim=self._runner.linear_conv_dim,
                    num_v_heads=self._runner.linear_num_v_heads,
                    value_head_dim=self._runner.linear_value_head_dim,
                    key_head_dim=self._runner.linear_key_head_dim,
                    torch_dtype=torch_dtype,
                    page_size_padded=self._runner.cache_config.mamba_page_size_padded,
                    block_size=block_size,
                )
            else:
                kv_h = (
                    kv_heads_list[layer_idx]
                    if has_per_layer
                    else self._runner.num_kv_heads
                )
                hd = (
                    head_dim_list[layer_idx] if has_per_layer else self._runner.head_dim
                )
                layer_name = f"layers.{layer_idx}.self_attn"
                specs[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=kv_h,
                    head_size=hd,
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
        """Return the byte size of one cache block.

        For per-layer shapes, sums each layer's contribution individually.
        For uniform shapes, reduces to the existing product formula.
        """
        if self._runner._is_stt:
            return STT_SCHED_BLOCK_BYTES

        block_size = self._runner.cache_config.block_size
        dtype_size = self._require_kv_cache_dtype().size
        kv_factor = 1 if self._runner.is_mla else 2
        return kv_factor * block_size * dtype_size * self._kv_layer_size_sum()

    def linear_cache_bytes_per_slot(self) -> int:
        """Return bytes for one request's linear-attention state."""
        if not self._runner.is_hybrid:
            raise RuntimeError("linear_cache_bytes_per_slot() requires a hybrid model")
        dtype_size = self._require_kv_cache_dtype().size
        recurrent_dtype_size = mx.float32.size
        conv_bytes = (
            (self._runner.linear_conv_kernel_dim - 1)
            * self._runner.linear_conv_dim
            * dtype_size
        )
        recurrent_bytes = (
            self._runner.linear_num_v_heads
            * self._runner.linear_value_head_dim
            * self._runner.linear_key_head_dim
            * recurrent_dtype_size
        )
        return self._runner.num_linear_layers * (conv_bytes + recurrent_bytes)

    def build_paged_attention_backend(
        self, *, block_size: int
    ) -> PagedAttentionBackend:
        """Create the paged-attention backend for the loaded model."""
        if self._runner.is_hybrid:
            return self._build_hybrid_backend(block_size)
        if self._runner.is_mla:
            return self._build_mla_backend(block_size)
        return self._build_mha_backend(block_size)

    def estimate_one_sequence_kv_bytes(
        self, *, max_model_len: int, block_size: int
    ) -> int:
        """Estimate bytes for one max-length sequence of cache state."""
        dtype_size = self._require_kv_cache_dtype().size
        aligned_tokens = -(-max_model_len // block_size) * block_size
        kv_factor = 1 if self._runner.is_mla else 2
        sdpa_kv_bytes = (
            kv_factor * aligned_tokens * dtype_size * self._kv_layer_size_sum()
        )
        if self._runner.is_hybrid:
            return sdpa_kv_bytes + self.linear_cache_bytes_per_slot()
        return sdpa_kv_bytes

    def _build_hybrid_backend(self, block_size: int) -> HybridPagedAttentionBackend:
        return HybridPagedAttentionBackend(
            num_layers=self._runner.num_layers,
            full_attention_interval=self._runner.full_attention_interval,
            max_num_seqs=self._runner.scheduler_config.max_num_seqs,
            num_kv_heads=self._runner.num_kv_heads,
            head_dim=self._runner.head_dim,
            linear_num_v_heads=self._runner.linear_num_v_heads,
            linear_key_head_dim=self._runner.linear_key_head_dim,
            linear_value_head_dim=self._runner.linear_value_head_dim,
            linear_conv_kernel_dim=self._runner.linear_conv_kernel_dim,
            linear_conv_dim=self._runner.linear_conv_dim,
            block_size=block_size,
            dtype=self._require_kv_cache_dtype(),
        )

    def _build_mla_backend(self, block_size: int) -> MLAPagedAttentionBackend:
        return MLAPagedAttentionBackend(
            num_layers=self._runner.num_layers,
            latent_dim=self._runner.mla_latent_dim,
            block_size=block_size,
            dtype=self._require_kv_cache_dtype(),
        )

    def _build_mha_backend(self, block_size: int) -> MHAPagedAttentionBackend:
        num_layers, cache_idx_map = self._mha_cache_layout()
        kv_heads, head_dims = self._cache_layer_shapes(num_layers)
        return MHAPagedAttentionBackend(
            num_layers=num_layers,
            num_kv_heads=self._runner.num_kv_heads,
            head_dim=self._runner.head_dim,
            block_size=block_size,
            dtype=self._require_kv_cache_dtype(),
            cache_idx_map=cache_idx_map,
            kv_heads_per_layer=kv_heads,
            head_dim_per_layer=head_dims,
        )

    def _cache_layer_shapes(self, num_cache_layers: int) -> tuple[list[int], list[int]]:
        """Build per-cache-layer ``(kv_heads, head_dim)`` lists.

        When the runner has per-layer shape lists (populated by the model
        adapter in a later PR), extract the first ``num_cache_layers`` entries
        (which correspond to the unique layers for YOCO models).  Otherwise
        replicate the scalar values for backward-compat uniform allocation.
        """
        kv_heads = self._runner.kv_heads_per_layer
        head_dims = self._runner.head_dim_per_layer
        if kv_heads is not None and head_dims is not None:
            return kv_heads[:num_cache_layers], head_dims[:num_cache_layers]
        return (
            [self._runner.num_kv_heads] * num_cache_layers,
            [self._runner.head_dim] * num_cache_layers,
        )

    def _kv_layer_size_sum(self) -> int:
        """Sum of ``kv_heads × head_dim`` across KV cache layers.

        For uniform models this equals ``num_kv_layers × kv_heads × head_dim``.
        """
        num_kv_layers = (
            self._runner.num_sdpa_layers
            if self._runner.is_hybrid
            else self._runner.num_kv_cache_layers
        )
        kv_heads = self._runner.kv_heads_per_layer
        head_dims = self._runner.head_dim_per_layer
        if kv_heads is not None and head_dims is not None:
            return sum(kv_heads[i] * head_dims[i] for i in range(num_kv_layers))
        return num_kv_layers * self._runner.num_kv_heads * self._runner.head_dim

    def _mha_cache_layout(self) -> tuple[int, list[int] | None]:
        if self._runner._yoco_cache_mapping is None:
            return self._runner.num_kv_cache_layers, None

        num_cache_layers, cache_idx_map = self._runner._yoco_cache_mapping
        logger.info(
            "YOCO KV sharing: %d unique cache layers (reduced from %d total)",
            num_cache_layers,
            self._runner.num_layers,
        )
        return num_cache_layers, cache_idx_map

    def _require_kv_cache_dtype(self) -> mx.Dtype:
        if self._runner.kv_cache_dtype is None:
            raise RuntimeError("KV cache dtype not initialized; load_model() first")
        return self._runner.kv_cache_dtype


class WorkerCachePlanner:
    """Worker-owned cache budgeting and paged-attention setup."""

    def __init__(self, worker: MetalWorker) -> None:
        self._worker = worker

    @staticmethod
    def kv_budget_bytes(
        metal_limit: int,
        model_memory: int,
        fraction: float,
        overhead: int,
    ) -> int:
        """Return Metal-memory budget available for paged KV cache."""
        return int(metal_limit * fraction) - model_memory - overhead

    def setup_paged_attention(self, *, overhead: int) -> None:
        """Allocate paged KV cache and patch the loaded model."""
        self._worker.model_runner.validate_paged_attention_support()
        plan = self._paged_attention_plan(overhead=overhead)
        logger.info(
            "Paged attention memory breakdown: "
            "metal_limit=%.2fGB, fraction=%.2f, usable_metal=%.2fGB, "
            "model_memory=%.2fGB, overhead=%.2fGB, "
            "kv_budget=%.2fGB, per_block_bytes=%d, "
            "num_blocks=%d, max_tokens_cached=%d",
            plan.metal_limit / 1e9,
            plan.fraction,
            plan.usable_metal / 1e9,
            plan.model_memory / 1e9,
            overhead / 1e9,
            plan.kv_budget / 1e9,
            plan.per_block_bytes,
            plan.num_blocks,
            plan.num_blocks * plan.block_size,
        )

        backend = self._worker.model_runner.build_paged_attention_backend(
            block_size=plan.block_size
        )
        backend.initialize(plan.num_blocks)
        n_patched = backend.patch_model(self._worker.model_runner.model)
        logger.info(
            "Paged attention enabled: %d layers patched, "
            "%d blocks allocated (block_size=%d, mla=%s)",
            n_patched,
            plan.num_blocks,
            plan.block_size,
            self._worker.model_runner.is_mla,
        )

        self._worker.model_runner._paged_attention_backend = backend
        self._worker.model_runner._paged_block_size = plan.block_size

    def get_model_memory_usage(self) -> int:
        """Return current model memory usage in bytes."""
        mx.eval(mx.array([0]))
        return mx.get_active_memory()

    def determine_available_memory(self) -> int:
        """Return scheduler-visible available cache memory."""
        mode = self._worker.model_runner.scheduler_memory_reporting_mode(
            paged_attention_enabled=self._worker.metal_config.use_paged_attention
        )

        if mode == "stt_nominal":
            logger.info("STT model: reporting nominal memory for scheduler")
            return STT_SCHED_AVAILABLE_BYTES

        if mode == "paged_attention_capacity":
            overhead = self._worker.model_runner.profile_run()
            self._worker._setup_paged_attention(overhead=overhead)
            backend = self._worker.model_runner._paged_attention_backend
            if backend is None:
                raise RuntimeError(
                    "Paged attention backend not initialized for capacity reporting"
                )
            block_size_bytes = self._worker.get_cache_block_size_bytes()
            available = backend.num_blocks() * block_size_bytes
            logger.info(
                "Paged attention: reporting MPS cache capacity "
                "(%d blocks × %d bytes = %.2f GB)",
                backend.num_blocks(),
                block_size_bytes,
                available / 1e9,
            )
            return available

        available = self._worker._one_sequence_kv_bytes()
        logger.info(
            "MLX path: reporting %.2f GB for scheduler admission control "
            "(one max-length sequence, max_model_len=%d)",
            available / 1e9,
            self._worker.model_config.max_model_len,
        )
        return available

    def _paged_attention_plan(self, *, overhead: int) -> _PagedAttentionPlan:
        block_size = self._worker.vllm_config.cache_config.block_size
        fraction = self._memory_fraction()
        metal_limit = self._metal_limit_bytes()
        model_memory = self.get_model_memory_usage()
        per_block_bytes = self._worker.get_cache_block_size_bytes()
        usable_metal = int(metal_limit * fraction)
        kv_budget = self.kv_budget_bytes(
            metal_limit,
            model_memory,
            fraction,
            overhead,
        )

        if self._worker.model_runner.is_hybrid:
            kv_budget -= (
                self._worker.model_runner.linear_cache_bytes_per_slot()
                * self._worker.model_runner.scheduler_config.max_num_seqs
            )

        if kv_budget <= 0:
            raise ValueError(
                "Paged attention: not enough Metal memory for KV cache. "
                f"metal_limit={metal_limit / 1e9:.2f}GB, "
                f"fraction={fraction}, "
                f"usable_metal={usable_metal / 1e9:.2f}GB, "
                f"model_memory={model_memory / 1e9:.2f}GB, "
                f"overhead={overhead / 1e9:.2f}GB, "
                f"kv_budget={kv_budget / 1e9:.2f}GB. "
                "Mitigations: increase VLLM_METAL_MEMORY_FRACTION, "
                "use a smaller or more quantized model."
            )

        num_blocks = kv_budget // per_block_bytes
        if num_blocks < PAGED_ATTENTION_MIN_BLOCKS:
            raise ValueError(
                "Paged attention: computed num_blocks too low "
                f"({num_blocks} < minimum {PAGED_ATTENTION_MIN_BLOCKS}). "
                f"metal_limit={metal_limit / 1e9:.2f}GB, "
                f"fraction={fraction}, "
                f"usable_metal={usable_metal / 1e9:.2f}GB, "
                f"model_memory={model_memory / 1e9:.2f}GB, "
                f"overhead={overhead / 1e9:.2f}GB, "
                f"kv_budget={kv_budget / 1e9:.2f}GB, "
                f"per_block_bytes={per_block_bytes}. "
                "Mitigations: increase VLLM_METAL_MEMORY_FRACTION, "
                "use a smaller or more quantized model."
            )

        return _PagedAttentionPlan(
            block_size=block_size,
            fraction=fraction,
            metal_limit=metal_limit,
            usable_metal=usable_metal,
            model_memory=model_memory,
            per_block_bytes=per_block_bytes,
            kv_budget=kv_budget,
            num_blocks=num_blocks,
        )

    def _memory_fraction(self) -> float:
        if self._worker.metal_config.is_auto_memory:
            logger.info(
                "Paged attention: VLLM_METAL_MEMORY_FRACTION=auto, "
                "defaulting to %.2f for paged path",
                PAGED_ATTENTION_DEFAULT_MEMORY_FRACTION,
            )
            return PAGED_ATTENTION_DEFAULT_MEMORY_FRACTION
        return self._worker.metal_config.memory_fraction

    def _metal_limit_bytes(self) -> int:
        device_info = mx.device_info()
        metal_limit = int(device_info.get("max_recommended_working_set_size", 0))
        if metal_limit <= 0:
            raise RuntimeError(
                "Paged attention: mx.device_info() did not return "
                "max_recommended_working_set_size. "
                "Ensure MLX is up to date and running on Apple Silicon. "
                f"Reported device_info keys: {list(device_info.keys())}"
            )
        return metal_limit
