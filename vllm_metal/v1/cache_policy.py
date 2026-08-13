# SPDX-License-Identifier: Apache-2.0
"""Cache-policy ownership for the v1 Metal runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import mlx.core as mx
import torch
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheSpec,
    MambaSpec,
    MLAAttentionSpec,
    SlidingWindowSpec,
)

from vllm_metal.attention.caches.mha_layout import MHAKVCacheLayout
from vllm_metal.attention.caches.turboquant import (
    BLOCK_SIZE as TQ_BLOCK_SIZE,
)
from vllm_metal.attention.caches.turboquant import (
    QUANT_PARAMS,
    V_QUANT_PARAMS,
    packed_dim,
)
from vllm_metal.attention.runtime.hybrid import (
    HybridPagedAttentionRuntime,
    _build_linear_layer_spec,
)
from vllm_metal.attention.runtime.mha import MHAPagedAttentionRuntime
from vllm_metal.attention.runtime.mla import MLAPagedAttentionRuntime
from vllm_metal.attention.runtime.protocol import PagedAttentionRuntime
from vllm_metal.attention.yoco import try_enable_gemma4_yoco_fast_prefill
from vllm_metal.config import (
    PAGED_ATTENTION_MIN_BLOCKS,
    MetalConfig,
    get_config,
)
from vllm_metal.pytorch_backend.tensor_bridge import MLX_TO_TORCH_DTYPE
from vllm_metal.stt.policy import STT_SCHED_AVAILABLE_BYTES
from vllm_metal.v1.gemma4_mtp import Gemma4MTPTargetMetadata
from vllm_metal.v1.model_adapter import ModelAdapter

if TYPE_CHECKING:
    from vllm_metal.v1.model_runner import MetalModelRunner
    from vllm_metal.v1.worker import MetalWorker

logger = init_logger(__name__)


def _align_state_pool_count(num_linear_layers: int, num_sdpa_layers: int) -> int:
    """Physical GDN state pools under align mode, as the memory plan sees it.

    Striping yields mamba groups the size of the attention group and one
    pool per within-group position, hence ``num_sdpa_layers`` pools; layouts
    that do not divide evenly fall back to one pool per layer so the plan
    never under-budgets.  Validated against the adopted layout at install.
    """
    if num_sdpa_layers > 0 and num_linear_layers % num_sdpa_layers == 0:
        return num_sdpa_layers
    return num_linear_layers


HYBRID_GDN_GROWTH_CUSHION_SLOTS = 2


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
        return turboquant_page_size_bytes(
            block_size=self.block_size,
            num_kv_heads=self.num_kv_heads,
            head_dim=self.head_size,
            k_quant=self.k_quant,
            v_quant=self.v_quant,
        )

    @classmethod
    def merge(cls, specs: Sequence[FullAttentionSpec]) -> TurboQuantAttentionSpec:
        # vLLM's uniformity probe treats AssertionError as "mixed spec type".
        turbo_specs: list[TurboQuantAttentionSpec] = []
        for spec in specs:
            if not isinstance(spec, TurboQuantAttentionSpec):
                raise AssertionError(
                    "All attention layers in the same KV cache group must be "
                    "TurboQuantAttentionSpec."
                )
            turbo_specs.append(spec)
        if not turbo_specs:
            raise ValueError("TurboQuantAttentionSpec.merge() requires specs")

        k_set = {s.k_quant for s in turbo_specs}
        v_set = {s.v_quant for s in turbo_specs}
        if len(k_set) != 1 or len(v_set) != 1:
            raise ValueError(
                "All TurboQuant layers in the same cache group must share the "
                "same (k_quant, v_quant); mixed-quant groups are not supported."
            )
        first = turbo_specs[0]
        return cls(
            block_size=first.block_size,
            num_kv_heads=first.num_kv_heads,
            head_size=first.head_size,
            head_size_v=first.head_size_v,
            dtype=first.dtype,
            page_size_padded=first.page_size_padded,
            sliding_window=cls.merge_window_sizes(
                {s.sliding_window for s in turbo_specs if s.sliding_window is not None}
            ),
            attention_chunk_size=cls.merge_window_sizes(
                {
                    s.attention_chunk_size
                    for s in turbo_specs
                    if s.attention_chunk_size is not None
                }
            ),
            k_quant=k_set.pop(),
            v_quant=v_set.pop(),
        )


def turboquant_page_size_bytes(
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


@dataclass(frozen=True)
class _HybridGDNReservation:
    bytes_per_slot: int = 0
    reserved_slots: int = 0
    max_num_seqs: int = 0

    @property
    def total_bytes(self) -> int:
        return self.bytes_per_slot * self.reserved_slots

    @property
    def enabled(self) -> bool:
        return self.total_bytes > 0

    @property
    def is_hybrid(self) -> bool:
        return self.bytes_per_slot > 0 and self.max_num_seqs > 0


@dataclass(frozen=True)
class _PagedAttentionPlan:
    block_size: int
    fraction: float
    metal_limit: int
    usable_metal: int
    model_memory: int
    overhead: int
    per_block_bytes: int
    base_kv_budget: int
    hybrid_gdn_reservation: _HybridGDNReservation
    kv_budget: int
    num_blocks: int

    def format_breakdown(self) -> str:
        parts = [
            f"metal_limit={self.metal_limit / 1e9:.2f}GB",
            f"fraction={self.fraction}",
            f"usable_metal={self.usable_metal / 1e9:.2f}GB",
            f"model_memory={self.model_memory / 1e9:.2f}GB",
            f"overhead={self.overhead / 1e9:.2f}GB",
        ]
        if self.hybrid_gdn_reservation.enabled:
            parts.append(f"kv_budget_before_hybrid={self.base_kv_budget / 1e9:.2f}GB")
        if self.hybrid_gdn_reservation.is_hybrid:
            parts.append(self._hybrid_gdn_detail())
        parts.append(f"kv_budget={self.kv_budget / 1e9:.2f}GB")
        return ", ".join(parts)

    def format_mitigations(self) -> str:
        mitigations = [
            "increase VLLM_METAL_MEMORY_FRACTION",
            "use a smaller or more quantized model",
        ]
        reservation = self.hybrid_gdn_reservation
        if reservation.enabled and reservation.max_num_seqs > 1:
            seq_mitigation = (
                "lower --max-num-seqs (for single-user serving, try --max-num-seqs 1)"
            )
            if self.base_kv_budget > 0:
                mitigations.insert(0, seq_mitigation)
            else:
                mitigations.append(seq_mitigation)
        return "Mitigations: " + "; ".join(mitigations) + "."

    def _hybrid_gdn_detail(self) -> str:
        reservation = self.hybrid_gdn_reservation
        if not reservation.is_hybrid:
            return ""
        return (
            "hybrid_gdn_state=lazy "
            f"(growth_peak_reserve={reservation.total_bytes / 1e9:.2f}GB, "
            f"{reservation.bytes_per_slot / 1e6:.1f}MB/seq * "
            f"peak_slots={reservation.reserved_slots}/"
            f"max_num_seqs={reservation.max_num_seqs})"
        )


class ModelCachePolicy:
    """Cache shape, size, and backend-selection policy for one runner."""

    def __init__(self, runner: MetalModelRunner, model_adapter: ModelAdapter) -> None:
        self._runner = runner
        self._model_adapter = model_adapter

    def validate_paged_attention_support(self) -> None:
        """Validate that the loaded model can run on the paged-attention path."""
        self._require_supported_per_layer_shapes()
        # ``require_uniform_kv_heads`` is the fail-fast for configs whose
        # ``num_global_key_value_heads`` differs from ``num_key_value_heads``
        # and which would silently fall back to the scalar uniform cache
        # path with wrong sizing.  Adapters that populate
        # ``kv_heads_per_layer`` via ``build_per_layer_kv_shapes`` (Gemma4
        # 26B/31B) have already opted into the heterogeneous cache and
        # handle mismatched KV counts layer-by-layer, so the uniform
        # guarantee does not apply and the check is skipped for them.
        if self._runner.kv_heads_per_layer is None:
            self._model_adapter.require_uniform_kv_heads(
                self._runner.model_args,
                self._runner.num_kv_heads,
            )

    def scheduler_memory_reporting_mode(
        self, *, paged_attention_enabled: bool
    ) -> Literal[
        "paged_attention_capacity",
        "paged_attention_mha_layout_budget",
        "pooling_no_kv",
        "single_sequence_estimate",
    ]:
        """Return which scheduler memory-reporting mode worker should use."""
        pooling_backend = self._runner._pooling_backend
        if (
            pooling_backend is not None
            and not pooling_backend.capabilities.uses_kv_cache
        ):
            return "pooling_no_kv"
        if paged_attention_enabled:
            if self._uses_deferred_mha_layout():
                return "paged_attention_mha_layout_budget"
            return "paged_attention_capacity"
        return "single_sequence_estimate"

    def _uses_deferred_mha_layout(self) -> bool:
        """Return whether vLLM's grouped MHA config must own allocation."""
        kv_heads = self._runner.kv_heads_per_layer
        head_dims = self._runner.head_dim_per_layer
        sliding_windows = self._runner.sliding_window_per_layer
        vllm_config = self._runner.vllm_config
        return (
            kv_heads is not None
            and head_dims is not None
            and sliding_windows is not None
            and self._runner._yoco_cache_mapping is None
            and not self._runner.is_hybrid
            and not self._runner.is_mla
            and not self._use_turboquant(get_config())
            and self._runner.vllm_config.speculative_config is None
            and self._runner._gemma4_mtp_assistant is None
            and not vllm_config.scheduler_config.disable_hybrid_kv_cache_manager
            and vllm_config.cache_config.num_gpu_blocks_override is None
            and any(window >= 0 for window in sliding_windows)
            and any(window < 0 for window in sliding_windows)
        )

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Build the scheduler-visible KV cache specification."""
        pooling_backend = self._runner._pooling_backend
        if (
            pooling_backend is not None
            and not pooling_backend.capabilities.uses_kv_cache
        ):
            return {}

        self._require_supported_per_layer_shapes()
        block_size = self._runner.cache_config.block_size
        torch_dtype = MLX_TO_TORCH_DTYPE[self._require_kv_cache_dtype()]
        config = get_config()
        use_turboquant = self._use_turboquant(config)

        kv_heads, head_dims = self._cache_layer_shapes(self._runner.num_layers)
        # Under YOCO KV sharing only the leading ``num_cache_layers`` layers own
        # a cache; the trailing ones reuse it (see ``_mha_cache_layout``, whose
        # mapping assigns the owners first).  Emit no spec for the sharers, so
        # the engine sizes against the layers that were actually allocated --
        # the same way upstream expresses sharing by omission in
        # ``GPUModelRunner.get_kv_cache_spec``.
        num_spec_layers = self._runner.num_layers
        if self._runner._yoco_cache_mapping is not None:
            num_spec_layers, _ = self._runner._yoco_cache_mapping
        specs: dict[str, KVCacheSpec] = {}
        use_deferred_mha_layout = self._uses_deferred_mha_layout()
        for layer_idx in range(num_spec_layers):
            if (
                self._runner.is_hybrid
                and layer_idx not in self._runner.sdpa_layer_indices
            ):
                layer_name = f"layers.{layer_idx}.linear_attn"
                cache_config = self._runner.cache_config
                mamba_block_size = cache_config.mamba_block_size
                # Upstream resolves this during config setup and asserts it here.
                assert mamba_block_size is not None
                specs[layer_name] = _build_linear_layer_spec(
                    conv_kernel_dim=self._runner.linear_conv_kernel_dim,
                    conv_dim=self._runner.linear_conv_dim,
                    num_v_heads=self._runner.linear_num_v_heads,
                    value_head_dim=self._runner.linear_value_head_dim,
                    key_head_dim=self._runner.linear_key_head_dim,
                    torch_dtype=torch_dtype,
                    page_size_padded=cache_config.mamba_page_size_padded,
                    mamba_block_size=mamba_block_size,
                    mamba_cache_mode=cache_config.mamba_cache_mode,
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
                layer_name = f"layers.{layer_idx}.self_attn"
                # MLA caches a single latent tensor per layer, not separate K
                # and V (see ``MLAPagedLatentCache``), which is why
                # ``_kv_factor`` bills it at 1.  ``FullAttentionSpec`` hardcodes
                # the 2x K/V factor in ``real_page_size_bytes``, so describing
                # MLA with it makes the engine halve the block count it plans
                # against relative to the pool actually allocated.
                if self._runner.is_mla:
                    specs[layer_name] = MLAAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=kv_heads[layer_idx],
                        head_size=head_dims[layer_idx],
                        dtype=torch_dtype,
                    )
                elif use_deferred_mha_layout:
                    specs[layer_name] = self._build_mha_attention_spec(
                        layer_idx=layer_idx,
                        block_size=block_size,
                        num_kv_heads=kv_heads[layer_idx],
                        head_dim=head_dims[layer_idx],
                        torch_dtype=torch_dtype,
                    )
                else:
                    specs[layer_name] = FullAttentionSpec(
                        block_size=block_size,
                        num_kv_heads=kv_heads[layer_idx],
                        head_size=head_dims[layer_idx],
                        dtype=torch_dtype,
                    )

        return specs

    def _build_mha_attention_spec(
        self,
        layer_idx: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        torch_dtype: torch.dtype,
    ) -> FullAttentionSpec | SlidingWindowSpec:
        """Build the scheduler spec for one standard MHA cache layer."""
        sliding_windows = self._runner.sliding_window_per_layer
        if sliding_windows is not None and sliding_windows[layer_idx] >= 0:
            return SlidingWindowSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_dim,
                dtype=torch_dtype,
                sliding_window=sliding_windows[layer_idx],
            )
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            dtype=torch_dtype,
        )

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Accept engine KV cache config for API compatibility.

        MLX owns the paged pool, which was already allocated from the
        profiled capacity in ``setup_paged_attention``.  The engine's block
        count normally round-trips back to that same number, but
        ``--num-gpu-blocks-override`` replaces it after the fact, so verify
        the engine is not planning against more blocks than exist.  The full
        upstream config is also the source of truth for scheduler KV groups, so
        adopt its grouping before serving.
        """
        runtime = self._runner.paged_attention_runtime
        pooling_backend = self._runner._pooling_backend
        if (
            pooling_backend is not None
            and not pooling_backend.capabilities.uses_kv_cache
        ):
            if kv_cache_config.kv_cache_groups or kv_cache_config.kv_cache_tensors:
                raise ValueError(
                    "Metal encoder pooling does not use KV cache, but vLLM "
                    "returned a non-empty KV cache config."
                )
            logger.info("Encoder pooling: no KV cache initialized.")
            return

        if self._uses_deferred_mha_layout():
            if runtime is not None:
                raise RuntimeError(
                    "deferred MHA layout path must not preallocate paged KV cache"
                )
            self._initialize_deferred_mha_layout(kv_cache_config)
            logger.info(
                "KV cache config received: %d grouped blocks "
                "(MLX layout initialized from vLLM config)",
                kv_cache_config.num_blocks,
            )
            return

        if runtime is not None and kv_cache_config.num_blocks > runtime.num_blocks():
            raise ValueError(
                f"Engine KV cache config requests {kv_cache_config.num_blocks} "
                f"blocks but the Metal paged pool was allocated with "
                f"{runtime.num_blocks()}. vllm-metal sizes its pool from "
                "available Metal memory and cannot grow it afterwards. If "
                "--num-gpu-blocks-override is set, lower or remove it; "
                "otherwise this is a capacity-accounting bug, please report it."
            )
        if runtime is not None:
            self._adopt_scheduler_groups(runtime, kv_cache_config)
        logger.info(
            "KV cache config received: %d blocks (MLX manages cache internally)",
            kv_cache_config.num_blocks,
        )

    def _initialize_deferred_mha_layout(self, kv_cache_config: KVCacheConfig) -> None:
        model_layer_names = self._mha_model_layer_names()
        layout = MHAKVCacheLayout.from_config(kv_cache_config, model_layer_names)
        runtime = self._build_mha_backend(block_size=layout.group_block_sizes[0])
        runtime.adopt_layout(layout)
        runtime.patch_model(self._runner.model)
        self._runner.install_paged_attention_runtime(
            runtime,
            block_size=layout.group_block_sizes[0],
        )

    def _adopt_scheduler_groups(
        self,
        runtime: PagedAttentionRuntime,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        if self._runner.is_mla:
            return
        if self._runner.is_hybrid:
            self._adopt_hybrid_scheduler_group(runtime, kv_cache_config)
            return
        self._adopt_mha_layout(runtime, kv_cache_config)

    def _adopt_mha_layout(
        self,
        runtime: PagedAttentionRuntime,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        if not isinstance(runtime, MHAPagedAttentionRuntime):
            raise RuntimeError("MHA cache config requires MHAPagedAttentionRuntime")

        model_layer_names = self._mha_model_layer_names()
        group_indices = self._scheduler_group_indices_for_layers(
            kv_cache_config,
            model_layer_names,
        )
        if get_config().turboquant:
            if group_indices != (0,):
                raise NotImplementedError(
                    "TurboQuant MHA currently supports one scheduler KV group"
                )
            return
        if len(group_indices) == 1:
            return

        layout = MHAKVCacheLayout.from_config(kv_cache_config, model_layer_names)
        runtime.adopt_layout(layout)
        runtime.patch_model(self._runner.model)
        self.install_gemma4_mtp_kv_sharing(
            runtime,
            block_size=layout.group_block_sizes[0],
        )
        self._runner.install_paged_attention_runtime(
            runtime,
            block_size=layout.group_block_sizes[0],
        )

    def _adopt_hybrid_scheduler_group(
        self,
        runtime: PagedAttentionRuntime,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        if not isinstance(runtime, HybridPagedAttentionRuntime):
            raise RuntimeError(
                "hybrid cache config requires HybridPagedAttentionRuntime"
            )

        group_indices = self._scheduler_group_indices_for_layers(
            kv_cache_config,
            tuple(
                f"layers.{layer_idx}.self_attn"
                for layer_idx in sorted(self._runner.sdpa_layer_indices)
            ),
        )
        if len(group_indices) != 1:
            raise NotImplementedError(
                "hybrid paged attention requires all SDPA layers to share one "
                "scheduler KV group"
            )
        group_index = group_indices[0]
        block_size = kv_cache_config.kv_cache_groups[
            group_index
        ].kv_cache_spec.block_size
        # Align mode keys GDN state slabs by scheduler block id.  The engine
        # stripes same-spec linear layers across several mamba cache groups
        # (each group hands every request one block-table row), so the runtime
        # needs all those groups plus each linear layer's group ordinal.  None
        # mode keeps a private per-request slot pool and ignores the
        # scheduler's mamba groups.
        state_group_indices: tuple[int, ...] = ()
        layer_group_ordinals: list[int] | None = None
        layer_pool_ordinals: list[int] | None = None
        if self._runner.cache_config.mamba_cache_mode == "align":
            cache_idx_by_name = {
                f"layers.{layer_idx}.linear_attn": cache_idx
                for cache_idx, layer_idx in enumerate(
                    layer_idx
                    for layer_idx in range(self._runner.num_layers)
                    if layer_idx not in self._runner.sdpa_layer_indices
                )
            }
            mamba_group_ids = [
                index
                for index, group in enumerate(kv_cache_config.kv_cache_groups)
                if isinstance(group.kv_cache_spec, MambaSpec)
            ]
            state_group_indices = tuple(mamba_group_ids)
            layer_group_ordinals = [-1] * len(cache_idx_by_name)
            for ordinal, mamba_group_id in enumerate(mamba_group_ids):
                group = kv_cache_config.kv_cache_groups[mamba_group_id]
                for layer_name in group.layer_names:
                    cache_idx = cache_idx_by_name.get(layer_name)
                    if cache_idx is None:
                        raise RuntimeError(
                            f"mamba cache group {mamba_group_id} holds "
                            f"{layer_name!r}, which is not one of the "
                            "runner's linear-attention layers"
                        )
                    layer_group_ordinals[cache_idx] = ordinal
            if -1 in layer_group_ordinals:
                raise RuntimeError(
                    "scheduler mamba cache groups do not cover every "
                    "linear-attention layer"
                )
            # Physical pools follow the engine's tensor sharing: each
            # kv_cache_tensor is shared by one layer from each cache group, so
            # linear layers sharing a tensor share one state pool (their
            # groups own disjoint block ids and never collide).
            layer_pool_ordinals = [-1] * len(cache_idx_by_name)
            pools_used = 0
            for tensor in kv_cache_config.kv_cache_tensors:
                members = [
                    cache_idx_by_name[name]
                    for name in tensor.shared_by
                    if name in cache_idx_by_name
                ]
                if not members:
                    continue
                for cache_idx in members:
                    if layer_pool_ordinals[cache_idx] != -1:
                        raise RuntimeError(
                            "a linear-attention layer appears in two "
                            "kv_cache_tensors; cannot derive state pools"
                        )
                    layer_pool_ordinals[cache_idx] = pools_used
                pools_used += 1
            if -1 in layer_pool_ordinals:
                raise RuntimeError(
                    "kv_cache_tensors do not cover every linear-attention "
                    "layer; cannot derive state pools"
                )
            budgeted = _align_state_pool_count(
                len(cache_idx_by_name), len(self._runner.sdpa_layer_indices)
            )
            if pools_used > budgeted:
                raise RuntimeError(
                    f"engine layout needs {pools_used} GDN state pools but "
                    f"the memory plan budgeted {budgeted}; refusing to "
                    "exceed the paged memory budget"
                )
        runtime.adopt_scheduler_group(
            group_index,
            block_size,
            state_group_indices=state_group_indices,
            layer_group_ordinals=layer_group_ordinals,
            layer_pool_ordinals=layer_pool_ordinals,
        )
        self._runner.install_paged_attention_runtime(runtime, block_size=block_size)

    def _scheduler_group_indices_for_layers(
        self,
        kv_cache_config: KVCacheConfig,
        layer_names: tuple[str, ...],
    ) -> tuple[int, ...]:
        layer_set = set(layer_names)
        layer_to_group: dict[str, int] = {}
        for group_index, group in enumerate(kv_cache_config.kv_cache_groups):
            for layer_name in group.layer_names:
                if layer_name in layer_set:
                    layer_to_group[layer_name] = group_index

        missing = layer_set - set(layer_to_group)
        if missing:
            raise ValueError(
                "KV cache config is missing scheduler groups for layers: "
                f"{', '.join(sorted(missing))}"
            )
        return tuple(dict.fromkeys(layer_to_group[name] for name in layer_names))

    def _mha_model_layer_names(self) -> tuple[str, ...]:
        num_layers, _ = self._mha_cache_layout()
        return tuple(f"layers.{layer_idx}.self_attn" for layer_idx in range(num_layers))

    def get_cache_block_size_bytes(self) -> int:
        """Return the byte size of one cache block.

        For per-layer shapes, sums each layer's contribution individually.
        For uniform shapes, reduces to the existing product formula.
        """
        self._require_supported_per_layer_shapes()
        block_size = self._runner.cache_config.block_size
        dtype_size = self._require_kv_cache_dtype().size
        num_kv_layers = self._num_kv_cache_layers()

        # TurboQuant uses quantized KV cache with different byte layout
        config = get_config()
        if self._use_turboquant(config):
            return num_kv_layers * turboquant_page_size_bytes(
                block_size=block_size,
                num_kv_heads=self._runner.num_kv_heads,
                head_dim=self._runner.head_dim,
                k_quant=config.k_quant,
                v_quant=config.v_quant,
            )

        return self._kv_factor() * block_size * dtype_size * self._kv_layer_size_sum()

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

    def build_paged_attention_runtime(
        self, *, block_size: int
    ) -> PagedAttentionRuntime:
        """Create the paged-attention backend for the loaded model."""
        self._require_supported_per_layer_shapes()
        if self._runner.is_hybrid:
            return self._build_hybrid_backend(block_size)
        if self._runner.is_mla:
            return self._build_mla_backend(block_size)
        return self._build_mha_backend(block_size)

    def install_gemma4_mtp_kv_sharing(
        self,
        backend: PagedAttentionRuntime,
        *,
        block_size: int,
    ) -> None:
        """Wire Gemma4 MTP assistant layers to the target paged KV cache."""
        assistant = self._runner._gemma4_mtp_assistant
        if assistant is None:
            return
        if not isinstance(backend, MHAPagedAttentionRuntime):
            raise NotImplementedError(
                "Gemma4 MTP assistant KV sharing requires the MHA paged "
                "attention backend on Metal."
            )
        target_metadata = Gemma4MTPTargetMetadata.from_model_args(
            self._runner.model_args
        )
        self._runner._gemma4_mtp_assistant = assistant.with_target_kv_sharing(
            target_metadata=target_metadata,
            target_kv_cache=backend.kv_cache,
            block_size=block_size,
            group_block_sizes=backend.kv_group_block_sizes(),
        )

    def estimate_one_sequence_kv_bytes(
        self, *, max_model_len: int, block_size: int
    ) -> int:
        """Estimate bytes for one max-length sequence of cache state."""
        self._require_supported_per_layer_shapes()
        dtype_size = self._require_kv_cache_dtype().size
        aligned_tokens = -(-max_model_len // block_size) * block_size
        num_kv_layers = self._num_kv_cache_layers()

        # TurboQuant uses quantized KV cache with different byte layout
        config = get_config()
        if self._use_turboquant(config):
            return num_kv_layers * turboquant_page_size_bytes(
                block_size=aligned_tokens,
                num_kv_heads=self._runner.num_kv_heads,
                head_dim=self._runner.head_dim,
                k_quant=config.k_quant,
                v_quant=config.v_quant,
            )

        sdpa_kv_bytes = (
            self._kv_factor() * aligned_tokens * dtype_size * self._kv_layer_size_sum()
        )
        if self._runner.is_hybrid:
            return sdpa_kv_bytes + self._linear_spec_bytes_per_slot()
        return sdpa_kv_bytes

    def _linear_spec_bytes_per_slot(self) -> int:
        """Per-slot linear-state bytes as the reported MambaSpec charges them.

        vLLM admits against the specs this worker reports, and the linear
        MambaSpec carries ``mamba_page_size_padded`` — an unpadded estimate
        falls short of that requirement by the padding margin.
        """
        # Mirrors MambaSpec.max_memory_usage_bytes with zero speculative
        # blocks and mamba_cache_mode "none"; if vLLM's defaults change,
        # this mirror must follow.
        padded = self._runner.cache_config.mamba_page_size_padded
        if padded is not None:
            return self._runner.num_linear_layers * padded
        return self.linear_cache_bytes_per_slot()

    def _build_hybrid_backend(self, block_size: int) -> HybridPagedAttentionRuntime:
        config = get_config()
        return HybridPagedAttentionRuntime(
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
            mamba_cache_mode=self._runner.cache_config.mamba_cache_mode,
            turboquant=config.turboquant,
            k_quant=config.k_quant if config.turboquant else None,
            v_quant=config.v_quant if config.turboquant else None,
        )

    def _build_mla_backend(self, block_size: int) -> MLAPagedAttentionRuntime:
        config = get_config()
        if config.turboquant:
            raise NotImplementedError(
                "TurboQuant is not supported for MLA models. "
                "Disable `turboquant` in --additional-config or select a "
                "non-MLA model."
            )
        return MLAPagedAttentionRuntime(
            num_layers=self._runner.num_layers,
            latent_dim=self._runner.mla_latent_dim,
            block_size=block_size,
            dtype=self._require_kv_cache_dtype(),
        )

    def _build_mha_backend(self, block_size: int) -> MHAPagedAttentionRuntime:
        num_layers, cache_idx_map = self._mha_cache_layout()
        config = get_config()
        kv_heads, head_dims = self._cache_layer_shapes(num_layers)
        # YOCO's ``build_yoco_cache_mapping`` assigns the first
        # ``num_cache_layers`` model layers identity-style (``mapping[i] = i``),
        # so slicing the first ``num_cache_layers`` entries of the full
        # per-model-layer list yields the correct window for each cache slot.
        # Shared layers then retrieve the right window via ``cache_idx_map``,
        # which points back to a same-type unique layer by construction.
        sw = self._runner.sliding_window_per_layer
        sw_list = sw[:num_layers] if sw is not None else None
        return MHAPagedAttentionRuntime(
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
            sliding_window_per_layer=sw_list,
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
        if get_config().turboquant:
            raise NotImplementedError(
                "TurboQuant with per-layer KV shapes is not yet supported."
            )
        if self._runner.is_hybrid:
            raise NotImplementedError(
                "Per-layer KV shapes with hybrid models require "
                "SDPA-layer index remapping, which is not yet implemented."
            )

    def _kv_layer_size_sum(self) -> int:
        """Sum of ``kv_heads × head_dim`` across KV cache layers.

        For uniform models this equals ``num_kv_layers × kv_heads × head_dim``.
        """
        num_kv_layers = self._num_kv_cache_layers()
        kv_heads = self._runner.kv_heads_per_layer
        head_dims = self._runner.head_dim_per_layer
        if kv_heads is not None and head_dims is not None:
            return sum(kv_heads[i] * head_dims[i] for i in range(num_kv_layers))
        return num_kv_layers * self._runner.num_kv_heads * self._runner.head_dim

    def _num_kv_cache_layers(self) -> int:
        if self._runner.is_hybrid:
            return self._runner.num_sdpa_layers
        return self._runner.num_kv_cache_layers

    def _use_turboquant(self, config: MetalConfig) -> bool:
        # Hybrid models compress their SDPA layers too (see
        # ``_build_hybrid_backend``), so they must not be excluded here:
        # every scheduler-visible sizing path (specs, per-block bytes,
        # one-sequence estimates) has to agree with the runtime layout.
        return bool(config.turboquant and not self._runner.is_mla)

    def _kv_factor(self) -> int:
        return 1 if self._runner.is_mla else 2

    def _mha_cache_layout(self) -> tuple[int, dict[int, int] | None]:
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

    def setup_paged_attention(self, *, overhead: int) -> None:
        """Allocate paged KV cache and patch the loaded model."""
        self._worker.model_runner.validate_paged_attention_support()
        plan = self._paged_attention_plan(overhead=overhead)
        logger.info(
            "Paged attention memory breakdown: "
            "%s, per_block_bytes=%d, "
            "num_blocks=%d, max_tokens_cached=%d",
            plan.format_breakdown(),
            plan.per_block_bytes,
            plan.num_blocks,
            plan.num_blocks * plan.block_size,
        )

        backend = self._worker.model_runner.build_paged_attention_runtime(
            block_size=plan.block_size
        )
        backend.initialize(plan.num_blocks)
        self._worker.model_runner.install_gemma4_mtp_kv_sharing(
            backend,
            block_size=plan.block_size,
        )
        n_patched = backend.patch_model(self._worker.model_runner.model)
        self._worker.model_runner.install_drafter(
            num_blocks=plan.num_blocks,
            block_size=plan.block_size,
        )
        config = get_config()

        try_enable_gemma4_yoco_fast_prefill(
            self._worker.model_runner.model,
            self._worker.model_runner.model_args,
            num_paged_layers=n_patched,
        )
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

        self._worker.model_runner.install_paged_attention_runtime(
            backend,
            block_size=plan.block_size,
        )

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
            self.setup_paged_attention(overhead=overhead)
            backend = self._worker.model_runner.paged_attention_runtime
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

        if mode == "paged_attention_mha_layout_budget":
            overhead = self._worker.model_runner.profile_run()
            plan = self._paged_attention_plan(
                overhead=overhead,
                require_min_blocks=False,
            )
            logger.info(
                "Mixed MHA paged attention: reporting %.2f GB KV budget; "
                "runtime allocation deferred until vLLM KVCacheConfig",
                plan.kv_budget / 1e9,
            )
            return plan.kv_budget

        if mode == "pooling_no_kv":
            self._worker.model_runner.profile_run()
            logger.info("Encoder pooling: reporting zero KV-cache bytes")
            return 0

        available = self._worker._one_sequence_kv_bytes()
        logger.info(
            "MLX path: reporting %.2f GB for scheduler admission control "
            "(one max-length sequence, max_model_len=%d)",
            available / 1e9,
            self._worker.model_config.max_model_len,
        )
        return available

    @staticmethod
    def base_kv_budget_bytes(
        metal_limit: int,
        model_memory: int,
        fraction: float,
        overhead: int,
    ) -> int:
        """Return Metal-memory budget before hybrid GDN reservation."""
        return int(metal_limit * fraction) - model_memory - overhead

    def _paged_attention_plan(
        self, *, overhead: int, require_min_blocks: bool = True
    ) -> _PagedAttentionPlan:
        block_size = self._worker.vllm_config.cache_config.block_size
        fraction = self._memory_fraction()
        metal_limit = self._metal_limit_bytes()
        model_memory = self.get_model_memory_usage()
        per_block_bytes = self._worker.get_cache_block_size_bytes()
        # Align-mode hybrid caching addresses GDN state by scheduler block id:
        # any pool block can become a mamba state slab, so every planned block
        # carries the linear-state bytes alongside its SDPA pages (the pool is
        # fungible; the per-request reservation below stays zero instead).
        # Growing the lazy state cache holds one old physical pool while its
        # replacement materializes, so budget that one-pool overlap too.
        per_block_bytes += self._hybrid_align_state_bytes_per_block()
        per_block_bytes += self._hybrid_align_growth_bytes_per_block()
        usable_metal = int(metal_limit * fraction)
        base_kv_budget = self.base_kv_budget_bytes(
            metal_limit,
            model_memory,
            fraction,
            overhead,
        )
        reservation = self._hybrid_gdn_reservation()
        kv_budget = base_kv_budget - reservation.total_bytes
        plan = _PagedAttentionPlan(
            block_size=block_size,
            fraction=fraction,
            metal_limit=metal_limit,
            usable_metal=usable_metal,
            model_memory=model_memory,
            overhead=overhead,
            per_block_bytes=per_block_bytes,
            base_kv_budget=base_kv_budget,
            hybrid_gdn_reservation=reservation,
            kv_budget=kv_budget,
            num_blocks=max(0, kv_budget // per_block_bytes),
        )
        self._validate_paged_attention_plan(
            plan,
            require_min_blocks=require_min_blocks,
        )
        return plan

    def _validate_paged_attention_plan(
        self, plan: _PagedAttentionPlan, *, require_min_blocks: bool
    ) -> None:
        if plan.kv_budget <= 0:
            raise ValueError(
                "Paged attention: not enough Metal memory for KV cache. "
                f"{plan.format_breakdown()}. {plan.format_mitigations()}"
            )

        if require_min_blocks and plan.num_blocks < PAGED_ATTENTION_MIN_BLOCKS:
            raise ValueError(
                "Paged attention: computed num_blocks too low "
                f"({plan.num_blocks} < minimum {PAGED_ATTENTION_MIN_BLOCKS}). "
                f"{plan.format_breakdown()}, "
                f"per_block_bytes={plan.per_block_bytes}. "
                f"{plan.format_mitigations()}"
            )

    def _hybrid_align_state_bytes_per_block(self) -> int:
        """Per-pool-block linear-state bytes under align-mode prefix caching."""
        runner = self._worker.model_runner
        if not runner.is_hybrid:
            return 0
        if runner.cache_config.mamba_cache_mode != "align":
            return 0
        num_linear = runner.num_layers - len(runner.sdpa_layer_indices)
        pools = _align_state_pool_count(num_linear, len(runner.sdpa_layer_indices))
        return runner.linear_cache_bytes_per_slot() * pools // num_linear

    def _hybrid_align_growth_bytes_per_block(self) -> int:
        """One old physical state pool retained during align-cache growth."""
        runner = self._worker.model_runner
        if not runner.is_hybrid:
            return 0
        if runner.cache_config.mamba_cache_mode != "align":
            return 0
        num_linear = runner.num_layers - len(runner.sdpa_layer_indices)
        return runner.linear_cache_bytes_per_slot() // num_linear

    def _hybrid_gdn_reservation(self) -> _HybridGDNReservation:
        """Return lazy GDN headroom reserved outside the paged KV pool."""
        runner = self._worker.model_runner
        if not runner.is_hybrid:
            return _HybridGDNReservation()
        if runner.cache_config.mamba_cache_mode == "align":
            # Align mode folds the state pool into per-block sizing above.
            return _HybridGDNReservation()
        max_num_seqs = runner.scheduler_config.max_num_seqs
        if max_num_seqs <= 0:
            return _HybridGDNReservation()
        cushion_slots = min(max_num_seqs, HYBRID_GDN_GROWTH_CUSHION_SLOTS)
        return _HybridGDNReservation(
            bytes_per_slot=runner.linear_cache_bytes_per_slot(),
            # ``ensure_capacity`` grows by allocating a larger state pool and
            # copying the old pool into it. Reserve a bounded growth cushion
            # instead of the full scheduler cap so large max_num_seqs values
            # still benefit from lazy GDN allocation.
            reserved_slots=(2 * cushion_slots) - 1,
            max_num_seqs=max_num_seqs,
        )

    def _memory_fraction(self) -> float:
        """Resolve the paged KV memory fraction.

        Precedence:
        1. Numeric VLLM_METAL_MEMORY_FRACTION, for example 0.6, wins.
        2. Otherwise, VLLM_METAL_MEMORY_FRACTION=auto uses the user-provided
           --gpu-memory-utilization value.
        3. If the user did not provide --gpu-memory-utilization, vLLM 0.27.1
           supplies its default value, 0.92.
        """
        metal_config = self._worker.metal_config

        if not metal_config.is_auto_memory:
            metal_memory_fraction = metal_config.memory_fraction
            logger.info(
                "Paged attention: using VLLM_METAL_MEMORY_FRACTION=%.2f",
                metal_memory_fraction,
            )
            return metal_memory_fraction

        vllm_memory_fraction = (
            self._worker.vllm_config.cache_config.gpu_memory_utilization
        )
        logger.info(
            "Paged attention: VLLM_METAL_MEMORY_FRACTION=auto, "
            "using --gpu-memory-utilization=%.2f",
            vllm_memory_fraction,
        )
        return vllm_memory_fraction

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
