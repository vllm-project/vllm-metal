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
    get_config,
)
from vllm_metal.metal_kernel_backend.turboquant import (
    BLOCK_SIZE as TQ_BLOCK_SIZE,
)
from vllm_metal.metal_kernel_backend.turboquant import (
    QUANT_PARAMS,
    V_QUANT_PARAMS,
    packed_dim,
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


def _turboquant_page_size_bytes(
    block_size: int, num_kv_heads: int, head_dim: int, k_quant: str, v_quant: str
) -> int:
    """Calculate TurboQuant-compressed page size for one layer."""
    k_bits = QUANT_PARAMS[k_quant]["bits"]
    v_bits = V_QUANT_PARAMS[v_quant]["bits"]
    k_packed = packed_dim(head_dim, k_bits)
    v_packed = packed_dim(head_dim, v_bits)
    kv_bytes = block_size * num_kv_heads * (k_packed + v_packed)
    scale_groups = head_dim // TQ_BLOCK_SIZE
    scale_bytes = 3 * block_size * num_kv_heads * scale_groups * 2
    return kv_bytes + scale_bytes


@dataclass(frozen=True, kw_only=True)
class TurboQuantAttentionSpec(FullAttentionSpec):
    """FullAttentionSpec for TurboQuant-compressed KV cache.

    Reports the true packed byte count per page via an override of
    ``real_page_size_bytes`` so vLLM's scheduler can budget more blocks
    than the FP16 formula would allow — without lying about ``head_size``
    (the ``head_size_v`` reverse-engineering trick the previous version
    used produced negative values for aggressive 2-bit configs).

    Mirrors the upstream pattern of :class:`MLAAttentionSpec` which
    overrides ``real_page_size_bytes`` for its ``fp8_ds_mla`` cache layout.
    """

    k_quant: str
    v_quant: str

    @property
    def real_page_size_bytes(self) -> int:
        return _turboquant_page_size_bytes(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
            k_quant=self.k_quant,
            v_quant=self.v_quant,
        )

    @classmethod
    def merge(cls, specs):
        assert all(isinstance(s, TurboQuantAttentionSpec) for s in specs), (
            "All attention layers in the same KV cache group must be "
            "TurboQuantAttentionSpec."
        )
        k_set = {s.k_quant for s in specs}
        v_set = {s.v_quant for s in specs}
        assert len(k_set) == 1 and len(v_set) == 1, (
            "All TurboQuant layers in the same cache group must share the "
            "same (k_quant, v_quant); mixed-quant groups are not supported."
        )
        return cls(
            block_size=specs[0].block_size,
            num_kv_heads=specs[0].num_kv_heads,
            head_size=specs[0].head_size,
            head_size_v=specs[0].head_size_v,
            dtype=specs[0].dtype,
            page_size_padded=specs[0].page_size_padded,
            sliding_window=cls.merge_window_sizes(
                {s.sliding_window for s in specs if s.sliding_window is not None}
            ),
            attention_chunk_size=cls.merge_window_sizes(
                {
                    s.attention_chunk_size
                    for s in specs
                    if s.attention_chunk_size is not None
                }
            ),
            k_quant=k_set.pop(),
            v_quant=v_set.pop(),
        )


def _build_turboquant_attention_spec(
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    k_quant: str,
    v_quant: str,
) -> TurboQuantAttentionSpec:
    """Build a TurboQuantAttentionSpec for a single attention layer.

    Reports the real compressed page size via ``real_page_size_bytes``
    override, so the scheduler allocates the right number of blocks and
    ``head_size`` stays equal to the model's real head_dim.
    """
    return TurboQuantAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=torch.int8,
        k_quant=k_quant,
        v_quant=v_quant,
    )


def _register_turboquant_spec_manager() -> None:
    """Register ``TurboQuantAttentionSpec`` in vLLM's spec→manager map.

    vLLM's ``get_manager_for_kv_cache_spec`` uses strict-type lookup
    (``spec_manager_map[type(spec)]``), not ``isinstance``, so the
    ``FullAttentionSpec`` entry does not cover subclasses.  We reuse
    ``FullAttentionManager`` because a TurboQuant cache is accessed
    like a regular KV page from the scheduler's POV — block indexing,
    no special slot math (per-element byte layout is handled entirely
    inside the Metal kernel).

    Mirrors the upstream registration for ``MLAAttentionSpec`` (which
    vLLM also maps to ``FullAttentionManager``).
    """
    try:
        from vllm.v1.core.single_type_kv_cache_manager import (
            FullAttentionManager,
            spec_manager_map,
        )
    except ImportError:
        # vLLM shape changed; let the scheduler raise its own clearer error.
        return
    spec_manager_map.setdefault(TurboQuantAttentionSpec, FullAttentionManager)


_register_turboquant_spec_manager()


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
        self._require_supported_per_layer_shapes()
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

        self._require_supported_per_layer_shapes()
        block_size = self._runner.cache_config.block_size
        torch_dtype = MLX_TO_TORCH_DTYPE[self._require_kv_cache_dtype()]
        config = get_config()
        use_turboquant = (
            config.turboquant and not self._runner.is_hybrid and not self._runner.is_mla
        )

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
            elif use_turboquant:
                layer_name = f"layers.{layer_idx}.self_attn"
                specs[layer_name] = _build_turboquant_attention_spec(
                    block_size=block_size,
                    num_kv_heads=self._runner.num_kv_heads,
                    head_dim=self._runner.head_dim,
                    k_quant=config.k_quant,
                    v_quant=config.v_quant,
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

        self._require_supported_per_layer_shapes()
        block_size = self._runner.cache_config.block_size
        dtype_size = self._require_kv_cache_dtype().size
        num_kv_layers = (
            self._runner.num_sdpa_layers
            if self._runner.is_hybrid
            else self._runner.num_kv_cache_layers
        )

        # TurboQuant uses quantized KV cache with different byte layout
        config = get_config()
        if config.turboquant and not self._runner.is_hybrid and not self._runner.is_mla:
            return num_kv_layers * _turboquant_page_size_bytes(
                block_size=block_size,
                num_kv_heads=self._runner.num_kv_heads,
                head_dim=self._runner.head_dim,
                k_quant=config.k_quant,
                v_quant=config.v_quant,
            )

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
        self._require_supported_per_layer_shapes()
        if self._runner.is_hybrid:
            return self._build_hybrid_backend(block_size)
        if self._runner.is_mla:
            return self._build_mla_backend(block_size)
        return self._build_mha_backend(block_size)

    def estimate_one_sequence_kv_bytes(
        self, *, max_model_len: int, block_size: int
    ) -> int:
        """Estimate bytes for one max-length sequence of cache state."""
        self._require_supported_per_layer_shapes()
        dtype_size = self._require_kv_cache_dtype().size
        aligned_tokens = -(-max_model_len // block_size) * block_size
        num_kv_layers = (
            self._runner.num_sdpa_layers
            if self._runner.is_hybrid
            else self._runner.num_kv_cache_layers
        )

        # TurboQuant uses quantized KV cache with different byte layout
        config = get_config()
        if config.turboquant and not self._runner.is_hybrid and not self._runner.is_mla:
            # _turboquant_page_size_bytes is parameterised by tokens (block_size);
            # pass aligned_tokens to get the per-sequence byte total directly.
            return num_kv_layers * _turboquant_page_size_bytes(
                block_size=aligned_tokens,
                num_kv_heads=self._runner.num_kv_heads,
                head_dim=self._runner.head_dim,
                k_quant=config.k_quant,
                v_quant=config.v_quant,
            )

        kv_factor = 1 if self._runner.is_mla else 2
        sdpa_kv_bytes = (
            kv_factor * aligned_tokens * dtype_size * self._kv_layer_size_sum()
        )
        if self._runner.is_hybrid:
            return sdpa_kv_bytes + self.linear_cache_bytes_per_slot()
        return sdpa_kv_bytes

    def _build_hybrid_backend(self, block_size: int) -> HybridPagedAttentionBackend:
        config = get_config()
        if config.turboquant:
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
                turboquant=True,
                k_quant=config.k_quant,
                v_quant=config.v_quant,
            )
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
            turboquant=False,
            k_quant=None,
            v_quant=None,
        )

    def _build_mla_backend(self, block_size: int) -> MLAPagedAttentionBackend:
        config = get_config()
        if config.turboquant:
            raise NotImplementedError(
                "TurboQuant is not supported for MLA models. "
                "Disable `turboquant` in --additional-config or select a "
                "non-MLA model."
            )
        return MLAPagedAttentionBackend(
            num_layers=self._runner.num_layers,
            latent_dim=self._runner.mla_latent_dim,
            block_size=block_size,
            dtype=self._require_kv_cache_dtype(),
        )

    def _build_mha_backend(self, block_size: int) -> MHAPagedAttentionBackend:
        num_layers, cache_idx_map = self._mha_cache_layout()
        config = get_config()
        kv_heads, head_dims = self._cache_layer_shapes(num_layers)
        return MHAPagedAttentionBackend(
            num_layers=num_layers,
            num_kv_heads=self._runner.num_kv_heads,
            head_dim=self._runner.head_dim,
            block_size=block_size,
            dtype=self._require_kv_cache_dtype(),
            turboquant=config.turboquant,
            k_quant=config.k_quant if config.turboquant else None,
            v_quant=config.v_quant if config.turboquant else None,
            cache_idx_map=cache_idx_map,
            kv_heads_per_layer=kv_heads,
            head_dim_per_layer=head_dims,
        )

    def _cache_layer_shapes(self, num_cache_layers: int) -> tuple[list[int], list[int]]:
        """Build per-cache-layer ``(kv_heads, head_dim)`` lists.

        When the runner has per-layer shape lists, extract the first
        ``num_cache_layers`` entries (which correspond to the unique
        layers for YOCO models).  Otherwise replicate the scalar values
        for backward-compat uniform allocation.
        """
        kv_heads = self._runner.kv_heads_per_layer
        head_dims = self._runner.head_dim_per_layer
        if kv_heads is not None and head_dims is not None:
            return kv_heads[:num_cache_layers], head_dims[:num_cache_layers]
        return (
            [self._runner.num_kv_heads] * num_cache_layers,
            [self._runner.head_dim] * num_cache_layers,
        )

    def _require_supported_per_layer_shapes(self) -> None:
        """Reject unsupported per-layer KV shape combinations early."""
        kv_heads = self._runner.kv_heads_per_layer
        head_dims = self._runner.head_dim_per_layer
        if (kv_heads is None) != (head_dims is None):
            raise ValueError(
                "kv_heads_per_layer and head_dim_per_layer must be set together."
            )
        if kv_heads is None:
            return
        if self._runner.is_hybrid:
            raise NotImplementedError(
                "Per-layer KV shapes with hybrid models require "
                "SDPA-layer index remapping, which is not yet implemented."
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
        config = get_config()
        logger.info(
            "Paged attention enabled: %d layers patched, "
            "%d blocks allocated (block_size=%d, mla=%s, turboquant=%s, k_quant=%s)",
            n_patched,
            plan.num_blocks,
            plan.block_size,
            self._worker.model_runner.is_mla,
            config.turboquant,
            config.k_quant if config.turboquant else "N/A",
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
