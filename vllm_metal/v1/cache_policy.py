# SPDX-License-Identifier: Apache-2.0
"""Cache-policy ownership for the v1 Metal runtime."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace
from functools import cached_property
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

import vllm_metal.envs as envs
from vllm_metal.attention.caches.turboquant import (
    BLOCK_SIZE as TQ_BLOCK_SIZE,
)
from vllm_metal.attention.caches.turboquant import (
    QUANT_PARAMS,
    V_QUANT_PARAMS,
    packed_dim,
    prefill_workspace_bytes,
)
from vllm_metal.attention.impls.turboquant_prefill import (
    dtype_head_reason,
    workspace_upper_bound,
)
from vllm_metal.attention.runtime.hybrid import HybridPagedAttentionRuntime
from vllm_metal.attention.runtime.hybrid_plan import HybridRuntimePlan
from vllm_metal.attention.runtime.mla import MLAPagedAttentionRuntime
from vllm_metal.attention.runtime.protocol import PagedAttentionRuntime
from vllm_metal.attention.runtime.sdpa import SDPAPagedAttentionRuntime
from vllm_metal.attention.yoco import try_enable_gemma4_yoco_fast_prefill
from vllm_metal.config import (
    PAGED_ATTENTION_MIN_BLOCKS,
    MetalConfig,
    get_config,
)
from vllm_metal.pytorch_backend.tensor_bridge import MLX_TO_TORCH_DTYPE
from vllm_metal.stt.policy import STT_SCHED_AVAILABLE_BYTES
from vllm_metal.utils import CommitProbe, probe_commit
from vllm_metal.v1.gemma4_mtp import Gemma4MTPTargetMetadata
from vllm_metal.v1.kv_offload.config import AUTO_POOL_KEY
from vllm_metal.v1.model_adapter import ModelAdapter

if TYPE_CHECKING:
    from vllm.config import VllmConfig

    from vllm_metal.v1.model_runner import MetalModelRunner
    from vllm_metal.v1.worker import MetalWorker

logger = init_logger(__name__)


@dataclass(frozen=True, kw_only=True)
class TurboQuantAttentionSpec(FullAttentionSpec):
    """FullAttentionSpec for TurboQuant-compressed KV cache.

    Publishes the packed per-(head, token) byte count through the base
    spec's ``state_content_bytes`` field so vLLM's scheduler budgets
    blocks from the true compressed page size — without lying about
    ``head_size``. Since vLLM 0.28.0 the scheduler derives ``page_size_bytes``
    from that field and ``real_page_size_bytes`` is only an alias, so
    overriding the latter would not reach it; publishing the field is the
    same mechanism upstream's ``TurboQuantAttentionBackend.customize_spec``
    uses for its packed layout.
    """

    k_quant: str
    v_quant: str

    def __post_init__(self) -> None:
        # Derive the packed per-cell size here, not in the builder, so a
        # bare construction can never fall back to the dense int8 formula
        # (the same in-class derivation pattern as the base head_size_v).
        super().__post_init__()
        if self.state_content_bytes is None:
            object.__setattr__(
                self,
                "state_content_bytes",
                turboquant_state_content_bytes(
                    self.head_size, self.k_quant, self.v_quant
                ),
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


def turboquant_state_content_bytes(head_dim: int, k_quant: str, v_quant: str) -> int:
    """Packed bytes for one (head, token) cell: K + V payload plus scales.

    The scale term is 3 fp16 tensors (k_scale, v_scale, v_bias) per
    ``TQ_BLOCK_SIZE``-wide group, hence ``3 * scale_groups * 2`` bytes.
    """
    k_bits = QUANT_PARAMS[k_quant]["bits"]
    v_bits = V_QUANT_PARAMS[v_quant]["bits"]
    k_packed = packed_dim(head_dim, k_bits)
    v_packed = packed_dim(head_dim, v_bits)
    scale_groups = head_dim // TQ_BLOCK_SIZE
    return k_packed + v_packed + 3 * scale_groups * 2


def turboquant_page_size_bytes(
    block_size: int, num_kv_heads: int, head_dim: int, k_quant: str, v_quant: str
) -> int:
    """Calculate TurboQuant-compressed page size for one layer."""
    return (
        block_size
        * num_kv_heads
        * turboquant_state_content_bytes(head_dim, k_quant, v_quant)
    )


def _build_turboquant_attention_spec(
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    k_quant: str,
    v_quant: str,
) -> TurboQuantAttentionSpec:
    """Build a TurboQuantAttentionSpec for a single attention layer.

    The spec derives its packed ``state_content_bytes`` itself, so the
    scheduler allocates the right number of blocks and ``head_size``
    stays equal to the model's real head_dim.
    """
    return TurboQuantAttentionSpec(
        block_size=block_size,
        num_kv_heads=num_kv_heads,
        head_size=head_dim,
        dtype=torch.int8,
        k_quant=k_quant,
        v_quant=v_quant,
    )


# Max share of the KV budget for an automatic host pool. See #1037: pool size
# did not change TTFT.
_AUTO_POOL_MAX_SHARE = 0.25


def _offload_chunk_bytes(
    extra: dict, per_block_bytes: int, block_size: int | None
) -> int:
    """Bytes of one aligned host-pool chunk, as vLLM's offloading spec sizes it.

    Approximate for non-dense layouts, where vLLM's per-block bytes can differ
    from Metal's estimate; a pool below one real chunk still fails at startup.
    """
    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

    blocks_per_chunk = max(1, int(extra.get("blocks_per_chunk") or 1))
    if extra.get("block_size") and block_size:
        blocks_per_chunk = max(1, int(extra["block_size"]) // block_size)
    align = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    return -(-per_block_bytes * blocks_per_chunk // align) * align


def _bounded_pool(pool: int, cap: int, min_pool: int) -> int:
    """Clamp an automatic pool to between one host chunk and max(cap, one chunk)."""
    return min(max(pool, min_pool), max(cap, min_pool))


def _no_room_message(kv_budget: int, request: int, min_pool: int) -> str:
    needs = f"the smallest host pool ({min_pool / 2**20:.1f} MiB)"
    if request:
        needs = f"one --max-model-len request ({request / 2**20:.1f} MiB) and {needs}"
    return (
        f"KV offloading on Metal: the KV budget ({kv_budget / 2**20:.1f} MiB) "
        f"cannot hold {needs}. Raise --gpu-memory-utilization"
        + (" or lower --max-model-len." if request else ".")
    )


def _paging_message(room: int, request: int, min_pool: int) -> str:
    """Startup error when a paging machine has no room for one request and
    one host chunk. Never advises --kv-offloading-size, which the user did
    not set."""
    left = (
        "KV offloading on Metal: the machine is paging, and the memory left "
        f"for KV cache and host pool ({room / 2**20:.1f} MiB) cannot hold one "
        f"--max-model-len request ({request / 2**20:.1f} MiB)"
    )
    if request > room:
        # Not even one request fits, with or without offloading.
        return f"{left}. Free memory on the machine or lower --max-model-len."
    return (
        f"{left} and the smallest host pool ({min_pool / 2**20:.1f} MiB). Free "
        "memory on the machine, lower --max-model-len, or start without KV "
        "offloading."
    )


def uses_metal_offloading(vllm_config: VllmConfig) -> bool:
    """True when the Metal offloading connector is configured.

    The platform hook sets it. Other connectors keep upstream behaviour.
    """
    kv_transfer_config = getattr(vllm_config, "kv_transfer_config", None)
    return (
        kv_transfer_config is not None
        and kv_transfer_config.kv_connector == "MetalOffloadingConnector"
    )


@dataclass(frozen=True)
class _PagedAttentionPlan:
    block_size: int
    fraction: float
    metal_limit: int
    usable_metal: int
    model_memory: int
    overhead: int
    per_block_bytes: int
    kv_budget: int
    num_blocks: int
    # KV offload host pool. Pageable, but the same physical RAM as the wired
    # cache on unified memory, so it comes out of the same budget.
    kv_offload_pool: int = 0

    def format_breakdown(self) -> str:
        parts = [
            f"metal_limit={self.metal_limit / 1e9:.2f}GB",
            f"fraction={self.fraction}",
            f"usable_metal={self.usable_metal / 1e9:.2f}GB",
            f"model_memory={self.model_memory / 1e9:.2f}GB",
            f"overhead={self.overhead / 1e9:.2f}GB",
        ]
        if self.kv_offload_pool:
            parts.append(f"kv_offload_pool={self.kv_offload_pool / 1e9:.2f}GB")
        parts.append(f"kv_budget={self.kv_budget / 1e9:.2f}GB")
        return ", ".join(parts)

    def format_mitigations(self) -> str:
        mitigations = [
            f"increase --gpu-memory-utilization (currently {self.fraction})",
            "use a smaller or more quantized model",
        ]
        if self.kv_offload_pool:
            mitigations.insert(0, "lower --kv-offloading-size")
        return "Mitigations: " + "; ".join(mitigations) + "."


# The commit probe (``vllm_metal.utils.probe_commit``) touches a bounded sample
# of the pool: enough to make the machine fault pages in and to notice it paging
# out to swap, small enough that the pages it forces resident (and then drops)
# never amount to the multi-GB pool the lazy allocation exists to avoid.
KV_COMMIT_SAMPLE_BYTES = 512 << 20

# Free memory the pool has to leave for everything else on the machine. Scaled
# off Metal's recommended working set, with a floor so a small machine keeps a
# small margin.
KV_COMMIT_RESERVE_FLOOR_BYTES = 1 << 30

# Memory the kernel may compress or swap out before the probe calls it
# pressure. A touch can stir a few MB of background paging out of other
# processes; a real shortfall moves a noticeable part of the sample. The floor
# keeps a small probe (where the fraction rounds to nothing) from treating that
# noise as pressure.
KV_COMMIT_SWAP_TOLERANCE_DIVISOR = 8
KV_COMMIT_SWAP_TOLERANCE_FLOOR_BYTES = 1 << 20


def kv_swap_tolerance_bytes(probed_bytes: int) -> int:
    """Swap a probe of ``probed_bytes`` tolerates before it means pressure."""
    return max(
        KV_COMMIT_SWAP_TOLERANCE_FLOOR_BYTES,
        probed_bytes // KV_COMMIT_SWAP_TOLERANCE_DIVISOR,
    )


def kv_pool_bytes_after_probe(
    plan_bytes: int,
    probe: CommitProbe,
    *,
    reserve_bytes: int,
    future_reserved_bytes: int,
    swap_tolerance_bytes: int,
) -> int:
    """Pool bytes to allocate after a commit probe.

    The probe answers two different questions and only one of them is a reason
    to give capacity back. If the kernel had to move memory out of the way, into
    the compressor or out to swap, to fault the sample in, the machine has no
    headroom *now*: a pool it cannot back is served from swap, so size it to
    what is free, less ``reserve_bytes`` and less ``future_reserved_bytes``.

    ``future_reserved_bytes`` is memory the engine will allocate later out of
    the same free pool -- the TurboQuant prefill workspace, the KV offload host
    pool -- so it is *not* reflected in ``probe.available_before`` yet and
    subtracting it is not double counting. Memory already allocated (the
    weights, the profiling buffers) is inside that measurement already and must
    not be subtracted again.

    If the probe merely ran on a machine with less free memory than the plan
    asks for, that is reported rather than acted on: the plan is a cap, not a
    commitment. The lazy pool backs blocks as requests use them, and giving
    back capacity an idle pool never needed is the regression the lazy
    allocation exists to avoid.
    """
    if probe.displaced_bytes <= swap_tolerance_bytes:
        return plan_bytes
    cap = probe.available_before - reserve_bytes - future_reserved_bytes
    return min(plan_bytes, max(0, cap))


class ModelCachePolicy:
    """Cache shape, size, and backend-selection policy for one runner."""

    def __init__(self, runner: MetalModelRunner, model_adapter: ModelAdapter) -> None:
        self._runner = runner
        self._model_adapter = model_adapter

    @cached_property
    def tq_prefill_workspace_bytes(self) -> int:
        """Resolve one reservation before KV sizing and reuse it during serving."""
        config = get_config()
        if not self._use_turboquant(config):
            return 0
        runner = self._runner
        cap = None
        # Speculation expands segments/lookahead beyond the ordinary scheduler
        # bounds. Preserve its allowance until those bounds are accounted for.
        # vLLM resolves max_model_len before workers start. Its later auto-fit
        # may shorten the context, but does not reclaim this fixed reservation.
        if runner.vllm_config.speculative_config is None:
            if reason := dtype_head_reason(
                self._require_kv_cache_dtype(), runner.head_dim
            ):
                cap = 0
                logger.info_once(
                    "Metal: TurboQuant prefill stays compressed (%s).", reason
                )
            else:
                cap = workspace_upper_bound(
                    max_model_len=runner.model_config.max_model_len,
                    max_num_seqs=runner.scheduler_config.max_num_seqs,
                    max_num_batched_tokens=runner.scheduler_config.max_num_batched_tokens,
                    num_query_heads=runner.model_config.get_num_attention_heads(
                        runner.vllm_config.parallel_config
                    ),
                    num_kv_heads=runner.num_kv_heads,
                    head_dim=runner.head_dim,
                    block_size=runner.cache_config.block_size,
                    key_quant_type=config.k_quant,
                    value_bits=V_QUANT_PARAMS[config.v_quant]["bits"],
                )
        return prefill_workspace_bytes(max_bytes=cap)

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
        self,
    ) -> Literal[
        "paged_attention_capacity",
        "paged_attention_layout_budget",
        "pooling_no_kv",
    ]:
        """Return which scheduler memory-reporting mode worker should use."""
        pooling_backend = self._runner._pooling_backend
        if (
            pooling_backend is not None
            and not pooling_backend.capabilities.uses_kv_cache
        ):
            return "pooling_no_kv"
        if self._uses_upstream_storage():
            return "paged_attention_layout_budget"
        return "paged_attention_capacity"

    def _hybrid_plan(self) -> HybridRuntimePlan:
        """Return the resolved hybrid plan, failing fast if lifecycle skipped it."""
        plan = self._runner.hybrid_runtime_plan
        if plan is None:
            raise RuntimeError(
                "hybrid model has no resolved hybrid_runtime_plan; "
                "ModelLifecycle.resolve_model_dims must run before cache sizing"
            )
        return plan

    def _uses_upstream_storage(self) -> bool:
        return self._runner.is_hybrid or (
            not self._runner.is_mla and self._runner._draft_dims is None
        )

    def _uses_grouped_attention(self) -> bool:
        """Return whether the scheduler keeps distinct attention-window groups."""
        if self._runner.is_hybrid:
            return True
        sliding_windows = self._runner.sliding_window_per_layer
        vllm_config = self._runner.vllm_config
        return (
            # Mixed attention windows need grouped allocation even when all
            # layers have the same head geometry (for example, OLMo 3).
            sliding_windows is not None
            and self._runner._yoco_cache_mapping is None
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
        # a cache; the trailing ones reuse it (see ``_cache_layer_mapping``, whose
        # mapping assigns the owners first).  Emit no spec for the sharers, so
        # the engine sizes against the layers that were actually allocated --
        # the same way upstream expresses sharing by omission in
        # ``GPUModelRunner.get_kv_cache_spec``.
        num_spec_layers = self._runner.num_layers
        if self._runner._yoco_cache_mapping is not None:
            num_spec_layers, _ = self._runner._yoco_cache_mapping
        specs: dict[str, KVCacheSpec] = {}
        use_grouped_attention = self._uses_grouped_attention()

        def attention_spec(layer_idx: int) -> KVCacheSpec:
            return self._attention_layer_spec(
                layer_idx,
                block_size=block_size,
                num_kv_heads=kv_heads[layer_idx],
                head_dim=head_dims[layer_idx],
                torch_dtype=torch_dtype,
                use_turboquant=use_turboquant,
                config=config,
                use_grouped_attention=use_grouped_attention,
            )

        if self._runner.is_hybrid:
            hybrid_plan = self._hybrid_plan()
            state_spec = self._state_layer_spec(hybrid_plan)
            for layer_idx in range(num_spec_layers):
                if hybrid_plan.layers.is_state_layer(layer_idx):
                    specs[f"layers.{layer_idx}.{hybrid_plan.family.layer_name}"] = (
                        state_spec
                    )
                elif hybrid_plan.layers.is_attention_layer(layer_idx):
                    specs[f"layers.{layer_idx}.self_attn"] = attention_spec(layer_idx)
        else:
            for layer_idx in range(num_spec_layers):
                specs[f"layers.{layer_idx}.self_attn"] = attention_spec(layer_idx)

        specs.update(
            self._draft_layer_specs(block_size=block_size, torch_dtype=torch_dtype)
        )
        if self._runner._drafter is not None:
            specs.update(self._runner._drafter.kv_specs(block_size))
        return specs

    def _state_layer_spec(self, hybrid_plan: HybridRuntimePlan) -> MambaSpec:
        """Build the scheduler-visible spec shared by every state layer."""
        cache_config = self._runner.cache_config
        mamba_block_size = cache_config.mamba_block_size
        # Upstream resolves this during config setup and asserts it here.
        assert mamba_block_size is not None
        return hybrid_plan.state_cache_spec(
            mamba_block_size=mamba_block_size,
            page_size_padded=cache_config.mamba_page_size_padded,
            mamba_cache_mode=cache_config.mamba_cache_mode,
        )

    def _attention_layer_spec(
        self,
        layer_idx: int,
        *,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        torch_dtype: torch.dtype,
        use_turboquant: bool,
        config: MetalConfig,
        use_grouped_attention: bool,
    ) -> KVCacheSpec:
        """Build the scheduler-visible spec for one attention layer."""
        if use_turboquant:
            return _build_turboquant_attention_spec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                k_quant=config.k_quant,
                v_quant=config.v_quant,
            )
        # MLA caches a single latent tensor per layer, not separate K and V
        # (see ``MLAPagedLatentCache``), which is why ``_kv_factor`` bills it
        # at 1.  ``FullAttentionSpec`` bakes the 2x K/V factor into
        # ``page_size_bytes`` (head_size + head_size_v), so describing MLA with
        # it makes the engine halve the block count it plans against relative
        # to the pool actually allocated.
        if self._runner.is_mla:
            return MLAAttentionSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_dim,
                dtype=torch_dtype,
            )
        if use_grouped_attention:
            return self._build_attention_spec(
                layer_idx=layer_idx,
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_dim=head_dim,
                torch_dtype=torch_dtype,
            )
        return FullAttentionSpec(
            block_size=block_size,
            num_kv_heads=num_kv_heads,
            head_size=head_dim,
            dtype=torch_dtype,
        )

    def _draft_layer_specs(
        self, *, block_size: int, torch_dtype: torch.dtype
    ) -> dict[str, KVCacheSpec]:
        """Scheduler-visible spec for the draft model's KV-cache group.

        Draft models must be plain transformers (no sliding window / MLA /
        hybrid) -- enforced at startup by ``resolve_draft_dims`` -- so a
        uniform ``FullAttentionSpec`` per layer under distinct synthetic names
        is enough to let the scheduler size the draft's KV cache.
        """
        draft_dims = self._runner._draft_dims
        if draft_dims is None:
            return {}
        return {
            f"draft_layers.{layer_idx}.self_attn": FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=draft_dims.num_kv_heads,
                head_size=draft_dims.head_dim,
                dtype=torch_dtype,
            )
            for layer_idx in range(draft_dims.num_layers)
        }

    def _build_attention_spec(
        self,
        layer_idx: int,
        block_size: int,
        num_kv_heads: int,
        head_dim: int,
        torch_dtype: torch.dtype,
    ) -> FullAttentionSpec | SlidingWindowSpec:
        """Build the scheduler spec for one standard attention cache layer."""
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
        """Bind layout-driven caches from the final engine configuration.

        Count-initialized speculative/MLA runtimes adopt the engine's groups;
        other runtimes bind the engine's shared backing here.
        """
        self._adopt_draft_scheduler_group(kv_cache_config)
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

        if self._uses_upstream_storage():
            if runtime is not None:
                raise RuntimeError(
                    "upstream storage must be initialized after cache planning"
                )
            self._initialize_upstream_storage(kv_cache_config)
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

    def _initialize_upstream_storage(self, kv_cache_config: KVCacheConfig) -> None:
        self.validate_paged_attention_support()
        if self._runner.is_hybrid:
            runtime = self._build_hybrid_backend()
            runtime.initialize_from_config(kv_cache_config)
            runtime.patch_model(self._runner.model)
            self._runner.install_paged_attention_runtime(
                runtime, block_size=self._runner.cache_config.block_size
            )
            self._runner.install_drafter(
                num_blocks=kv_cache_config.num_blocks,
                block_size=self._runner.cache_config.block_size,
            )
            return
        model_layer_names = self._attention_layer_names()
        runtime = self._build_sdpa_backend(
            block_size=self._runner.cache_config.block_size
        )
        runtime.initialize_from_config(kv_cache_config, model_layer_names)
        n_patched = runtime.patch_model(self._runner.model)
        block_size = runtime.kv_group_block_sizes()[0]
        self.install_gemma4_mtp_kv_sharing(runtime, block_size=block_size)
        self._runner.install_paged_attention_runtime(runtime, block_size=block_size)
        drafter = self._runner._drafter
        if drafter is not None and (drafter_specs := drafter.kv_specs(block_size)):
            groups = self._scheduler_group_indices_for_layers(
                kv_cache_config, tuple(drafter_specs)
            )
            if len(groups) != 1:
                raise NotImplementedError(
                    f"{type(drafter).__name__} layers must share one scheduler KV group"
                )
            drafter.bind_cache(
                runtime.storage,
                group_index=groups[0],
                max_model_len=self._runner.model_config.max_model_len,
            )
        self._runner.install_drafter(
            num_blocks=kv_cache_config.num_blocks, block_size=block_size
        )
        try_enable_gemma4_yoco_fast_prefill(
            self._runner.model,
            self._runner.model_args,
            num_paged_layers=n_patched,
        )

    def _adopt_scheduler_groups(
        self,
        runtime: PagedAttentionRuntime,
        kv_cache_config: KVCacheConfig,
    ) -> None:
        if isinstance(runtime, SDPAPagedAttentionRuntime):
            runtime.adopt_scheduler_groups(
                kv_cache_config, self._attention_layer_names()
            )
            self._runner.install_paged_attention_runtime(
                runtime, block_size=runtime.kv_group_block_sizes()[0]
            )

    def _adopt_draft_scheduler_group(self, kv_cache_config: KVCacheConfig) -> None:
        """Pass the scheduler KV group and final context limit to the drafter.

        The draft model's own physical backend is already built by this
        point (``install_drafter``, called from ``determine_available_memory``
        -- before the engine has computed ``kv_cache_config``, so it cannot
        know its group index at construction time). This runs after, once
        ``kv_cache_config.kv_cache_groups`` exists, and resolves which group
        the synthetic ``draft_layers.*`` names from
        ``ModelCachePolicy._draft_layer_specs`` landed in -- mirroring
        ``_adopt_scheduler_groups``'s resolution for the target. No-op without a
        draft model configured.
        """
        draft_dims = self._runner._draft_dims
        if draft_dims is None:
            return
        layer_names = tuple(
            f"draft_layers.{layer_idx}.self_attn"
            for layer_idx in range(draft_dims.num_layers)
        )
        group_indices = self._scheduler_group_indices_for_layers(
            kv_cache_config, layer_names
        )
        if len(group_indices) != 1:
            raise NotImplementedError(
                "draft-model speculative decoding requires all draft layers "
                "to share one scheduler KV cache group"
            )

        from vllm_metal.v1.draft_model_proposer import DraftModelProposer

        drafter = self._runner._drafter
        if not isinstance(drafter, DraftModelProposer):
            raise RuntimeError(
                "draft KV-cache spec registered but no DraftModelProposer is "
                f"installed (got {type(drafter).__name__})"
            )
        drafter.adopt_scheduler_group(
            group_indices[0], self._runner.model_config.max_model_len
        )

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

    def _attention_layer_names(self) -> tuple[str, ...]:
        num_layers, _ = self._cache_layer_mapping()
        return tuple(f"layers.{layer_idx}.self_attn" for layer_idx in range(num_layers))

    def get_cache_block_size_bytes(self) -> int:
        """Return the byte size of one cache block.

        For per-layer shapes, sums each layer's contribution individually.
        For uniform shapes, reduces to the existing product formula. Adds the
        draft model's own per-block bytes when one is configured (see
        ``_draft_cache_block_size_bytes``), so every caller of this method --
        scheduler capacity reporting and the local budget-to-num_blocks
        division alike -- sizes against the true combined cost of one block
        index, which now has real storage in both the target's and the
        draft's KV-cache groups.
        """
        self._require_supported_per_layer_shapes()
        block_size = self._runner.cache_config.block_size
        dtype_size = self._require_kv_cache_dtype().size
        num_kv_layers = self._num_kv_cache_layers()

        # TurboQuant uses quantized KV cache with different byte layout
        config = get_config()
        if self._use_turboquant(config):
            return (
                num_kv_layers
                * turboquant_page_size_bytes(
                    block_size=block_size,
                    num_kv_heads=self._runner.num_kv_heads,
                    head_dim=self._runner.head_dim,
                    k_quant=config.k_quant,
                    v_quant=config.v_quant,
                )
                + self._draft_cache_block_size_bytes()
            )

        return (
            self._kv_factor() * block_size * dtype_size * self._kv_layer_size_sum()
            + self._draft_cache_block_size_bytes()
        )

    def _draft_cache_block_size_bytes(self) -> int:
        """Byte size of one draft-model cache block, or 0 without a draft.

        Derived from the same ``FullAttentionSpec`` objects
        ``_draft_layer_specs`` registers with the scheduler
        (``page_size_bytes``), rather than a parallel hand-rolled
        formula, so the two cannot drift apart. Naturally 0 when no draft is
        configured, since ``_draft_layer_specs`` returns ``{}`` in that case.
        """
        block_size = self._runner.cache_config.block_size
        torch_dtype = MLX_TO_TORCH_DTYPE[self._require_kv_cache_dtype()]
        specs = self._draft_layer_specs(block_size=block_size, torch_dtype=torch_dtype)
        return sum(spec.page_size_bytes for spec in specs.values())

    def build_paged_attention_runtime(
        self, *, block_size: int
    ) -> PagedAttentionRuntime:
        """Create the paged-attention backend for the loaded model."""
        self._require_supported_per_layer_shapes()
        if self._runner.is_hybrid:
            return self._build_hybrid_backend()
        if self._runner.is_mla:
            return self._build_mla_backend(block_size)
        return self._build_sdpa_backend(block_size)

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
        if not isinstance(backend, SDPAPagedAttentionRuntime):
            raise NotImplementedError(
                "Gemma4 MTP assistant KV sharing requires the SDPA paged "
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

    def _build_hybrid_backend(self) -> HybridPagedAttentionRuntime:
        self._reject_turboquant_for_mla()
        return HybridPagedAttentionRuntime(
            hybrid_plan=self._hybrid_plan(),
            dtype=self._require_kv_cache_dtype(),
            mamba_cache_mode=self._runner.cache_config.mamba_cache_mode,
        )

    def _build_mla_backend(self, block_size: int) -> MLAPagedAttentionRuntime:
        self._reject_turboquant_for_mla()
        return MLAPagedAttentionRuntime(
            num_layers=self._runner.num_layers,
            latent_dim=self._runner.mla_latent_dim,
            block_size=block_size,
            dtype=self._require_kv_cache_dtype(),
        )

    def _reject_turboquant_for_mla(self) -> None:
        if self._runner.is_mla and get_config().turboquant:
            raise NotImplementedError(
                "TurboQuant is not supported for MLA models. "
                "Disable `turboquant` in --additional-config or select a "
                "non-MLA model."
            )

    def _build_sdpa_backend(self, block_size: int) -> SDPAPagedAttentionRuntime:
        num_layers, cache_idx_map = self._cache_layer_mapping()
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
        return SDPAPagedAttentionRuntime(
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
        layers for YOCO models). Otherwise use the model's uniform shape
        for every cache layer.
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
            return self._hybrid_plan().layers.num_attention
        return self._runner.num_kv_cache_layers

    def _use_turboquant(self, config: MetalConfig) -> bool:
        # Hybrid models with standard attention compress their SDPA layers too,
        # so every sizing path must agree with the runtime layout. Hybrid MLA
        # is excluded here and rejected when its backend is built.
        return bool(config.turboquant and not self._runner.is_mla)

    def _kv_factor(self) -> int:
        return 1 if self._runner.is_mla else 2

    def _cache_layer_mapping(self) -> tuple[int, dict[int, int] | None]:
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
        self._validate_paged_attention_plan(plan, require_min_blocks=True)
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
        # Hybrid models always size their cache from vLLM's KV cache config
        # (``ModelCachePolicy._uses_upstream_storage``), so only the SDPA and
        # MLA runtimes, which own ``initialize``, reach this path.
        if not isinstance(
            backend, (SDPAPagedAttentionRuntime, MLAPagedAttentionRuntime)
        ):
            raise RuntimeError(
                "Paged attention: the capacity path initializes only the SDPA "
                f"and MLA runtimes; {type(backend).__name__} sizes its cache "
                "from vLLM's KV cache config"
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
        mode = self._worker.model_runner.scheduler_memory_reporting_mode()

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

        if mode == "paged_attention_layout_budget":
            overhead = self._worker.model_runner.profile_run()
            plan = self._paged_attention_plan(overhead=overhead)
            if self._worker.vllm_config.cache_config.num_gpu_blocks_override is None:
                self._validate_paged_attention_plan(
                    plan,
                    require_min_blocks=False,
                )
            budget = plan.kv_budget
            logger.info(
                "Upstream cache layout: reporting %.2f GB KV budget; "
                "runtime allocation deferred until vLLM KVCacheConfig",
                budget / 1e9,
            )
            return budget

        if mode == "pooling_no_kv":
            self._worker.model_runner.profile_run()
            logger.info("Encoder pooling: reporting zero KV-cache bytes")
            return 0

        raise AssertionError(f"Unknown scheduler memory reporting mode: {mode}")

    @staticmethod
    def base_kv_budget_bytes(
        metal_limit: int,
        model_memory: int,
        fraction: float,
        overhead: int,
    ) -> int:
        """Return cache bytes after model weights and execution overhead."""
        return int(metal_limit * fraction) - model_memory - overhead

    def _paged_attention_plan(self, *, overhead: int) -> _PagedAttentionPlan:
        """Build the memory plan without applying caller-specific validation."""
        block_size = self._worker.vllm_config.cache_config.block_size
        fraction = self._memory_fraction()
        metal_limit = self._metal_limit_bytes()
        model_memory = self.get_model_memory_usage()
        per_block_bytes = self._worker.get_cache_block_size_bytes()
        # Memory the engine will allocate later out of the same free pool: the
        # TurboQuant prefill workspace is reserved before KV sizing and held
        # while serving, so the probe's free-memory reading does not include it.
        future_reserved_bytes = 0
        if get_config().turboquant:
            # Profiling precedes paged-cache binding, so it cannot observe
            # materialized TQ histories. Reserve the admission limit once,
            # inside gpu_memory_utilization, before upstream allocates KV.
            workspace = self._worker.model_runner.tq_prefill_workspace_bytes
            overhead += workspace
            future_reserved_bytes = workspace
            if workspace:
                logger.info_once(
                    "TurboQuant prefill: reserving %.2f MiB within the Metal "
                    "memory budget before KV sizing.",
                    workspace / 2**20,
                )
        usable_metal = int(metal_limit * fraction)
        base_kv_budget = self.base_kv_budget_bytes(
            metal_limit,
            model_memory,
            fraction,
            overhead,
        )
        kv_offload_pool = self._resolve_kv_offload_pool(base_kv_budget, per_block_bytes)
        kv_budget = base_kv_budget - kv_offload_pool
        plan, room = self._probe_committed_pool(
            _PagedAttentionPlan(
                block_size=block_size,
                fraction=fraction,
                metal_limit=metal_limit,
                usable_metal=usable_metal,
                model_memory=model_memory,
                overhead=overhead,
                per_block_bytes=per_block_bytes,
                kv_budget=kv_budget,
                num_blocks=max(0, kv_budget // per_block_bytes),
                kv_offload_pool=kv_offload_pool,
            ),
            future_reserved_bytes=future_reserved_bytes,
        )
        if room is not None and plan.kv_offload_pool:
            # The probe shrank only the KV cache. Re-cap an automatic pool
            # against the room the probe sees for KV cache and pool.
            pool = self._recap_kv_offload_pool(
                room, plan.kv_offload_pool, per_block_bytes
            )
            if pool < plan.kv_offload_pool:
                plan = replace(
                    plan,
                    kv_budget=room - pool,
                    num_blocks=(room - pool) // per_block_bytes,
                    kv_offload_pool=pool,
                )
        return plan

    def _auto_pool_limits(
        self, kv_budget: int, per_block_bytes: int
    ) -> tuple[int, int, int, dict] | None:
        """Cap, floor, request bytes and connector config for an automatic pool.

        None for a user-set size, which is never capped.
        """
        vllm_config = self._worker.vllm_config
        if not uses_metal_offloading(vllm_config):
            return None
        assert vllm_config.kv_transfer_config is not None
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        if extra.get(AUTO_POOL_KEY) is not True:
            return None
        cache_config = vllm_config.cache_config
        block_size = cache_config.block_size
        cap = int(kv_budget * _AUTO_POOL_MAX_SHARE)
        # Leave room for one full-length request, plus the null block vLLM's
        # pool holds back. Not when vLLM auto-fits the length later
        # (--max-model-len -1), or sizes the cache from a block override.
        request = 0
        model_config = vllm_config.model_config
        if (
            getattr(model_config, "original_max_model_len", None) != -1
            and cache_config.num_gpu_blocks_override is None
        ):
            request_blocks = -(-model_config.max_model_len // block_size) + 1
            request = request_blocks * per_block_bytes
            cap = min(cap, kv_budget - request)
        # The host pool holds whole chunks, each aligned in the shared region.
        min_pool = _offload_chunk_bytes(extra, per_block_bytes, block_size)
        return cap, min_pool, request, extra

    def _resolve_kv_offload_pool(self, kv_budget: int, per_block_bytes: int) -> int:
        """Host pool bytes of the Metal offloading connector, else 0.

        The pool is the same physical RAM as the wired cache, so it comes out
        of the same budget. An automatic pool is at most a quarter of the
        budget, or one host chunk if that is larger, and leaves room for one
        full-length request. Writes a changed size back to cpu_bytes_to_use.
        Offloading uses the uni executor, so the connector and scheduler share
        this config and read it after planning.
        """
        vllm_config = self._worker.vllm_config
        if not uses_metal_offloading(vllm_config):
            return 0
        assert vllm_config.kv_transfer_config is not None
        extra = vllm_config.kv_transfer_config.kv_connector_extra_config or {}
        pool = max(0, int(extra.get("cpu_bytes_to_use", 0)))
        limits = self._auto_pool_limits(kv_budget, per_block_bytes)
        if limits is None:
            return pool
        if kv_budget <= 0:
            # No room even before the pool: plan validation reports it with
            # the full breakdown, so leave the automatic pool out of it.
            return 0
        cap, min_pool, request, extra = limits
        new_pool = _bounded_pool(pool, cap, min_pool)
        # Fail only where offloading is the cause. A request larger than the
        # whole budget fails without it too, and vLLM's own check reports that.
        if kv_budget - new_pool < request <= kv_budget:
            raise ValueError(_no_room_message(kv_budget, request, min_pool))
        if new_pool != pool:
            logger.info(
                "KV offloading host pool set to %.2f GiB (default %.2f GiB) to "
                "fit the KV budget and host chunk size; pass --kv-offloading-size "
                "to set it",
                new_pool / 2**30,
                pool / 2**30,
            )
            extra["cpu_bytes_to_use"] = new_pool
        return new_pool

    def _recap_kv_offload_pool(
        self, total: int, pool: int, per_block_bytes: int
    ) -> int:
        """Cap an automatic pool again after the commit probe shrank the budget.

        ``total`` is the room the probe sees for KV cache and pool together.
        The pool only shrinks. If one request and the smallest pool do not
        fit, startup fails and says the machine is paging.
        """
        limits = self._auto_pool_limits(total, per_block_bytes)
        if limits is None:
            return pool
        cap, min_pool, request, extra = limits
        new_pool = min(pool, _bounded_pool(pool, cap, min_pool))
        if request and total - new_pool < request:
            raise ValueError(_paging_message(total, request, min_pool))
        if total < new_pool:
            # No request reserved (--max-model-len -1 or a block override) and
            # not even one chunk fits. Plan validation, or vLLM with the
            # override, decides as without the re-cap.
            return pool
        if new_pool < pool:
            logger.warning(
                "KV offloading host pool cut from %.2f to %.2f GB after the "
                "commit probe; the KV cache gets the difference, %.2f GB, "
                "in place of the size logged above",
                pool / 1e9,
                new_pool / 1e9,
                (total - new_pool) / 1e9,
            )
            extra["cpu_bytes_to_use"] = new_pool
        return new_pool

    def _probe_committed_pool(
        self, plan: _PagedAttentionPlan, *, future_reserved_bytes: int = 0
    ) -> tuple[_PagedAttentionPlan, int | None]:
        """Check the plan against the machine before serving starts.

        The budget follows ``--gpu-memory-utilization`` against Metal's
        recommended working set, which knows nothing about what else is using
        the machine, and the pool is allocated lazily, so a plan the machine
        cannot hold stays invisible until a request writes a block -- by which
        point it is a swap storm or a jetsam kill mid-generation. Forcing a
        sample resident here makes the machine answer at load time, where the
        answer is still cheap to act on, and it costs no RSS because
        ``probe_commit`` drops the sample again.

        Only paging is acted on. Free memory below the plan is reported: the
        pool backs blocks as requests use them, so an idle pool never needs the
        capacity the plan allows, and shrinking to whatever happens to be free
        today would hand back the capacity the lazy allocation exists to keep.

        Also returns the room the probe sees for KV cache and pool together
        when it shrank the plan, else None.
        """
        if not envs.VLLM_METAL_KV_COMMIT_PROBE or plan.kv_budget <= 0:
            return plan, None

        # The offload host pool is allocated later, is pageable, and shares the
        # same physical RAM, so it is held back from the cap exactly as the KV
        # budget holds it back from the Metal budget.
        other_reserved_bytes = future_reserved_bytes
        future_reserved_bytes += plan.kv_offload_pool

        probe = probe_commit(min(plan.kv_budget, KV_COMMIT_SAMPLE_BYTES))
        reserve_bytes = max(
            KV_COMMIT_RESERVE_FLOOR_BYTES,
            plan.metal_limit // 16,
        )
        logger.info(
            "KV commit probe: %s; holding back %.2f GB for the rest of the machine%s",
            probe.describe(),
            reserve_bytes / 1e9,
            (
                f" and {future_reserved_bytes / 1e9:.2f} GB the engine holds later"
                if future_reserved_bytes
                else ""
            ),
        )

        fit = kv_pool_bytes_after_probe(
            plan.kv_budget,
            probe,
            reserve_bytes=reserve_bytes,
            future_reserved_bytes=future_reserved_bytes,
            swap_tolerance_bytes=kv_swap_tolerance_bytes(probe.probed_bytes),
        )
        if fit >= plan.kv_budget:
            if plan.kv_budget > probe.available_before:
                logger.warning(
                    "KV commit probe: the pool (%.2f GB) is larger than this "
                    "machine's free memory (%.2f GB). An idle pool does not "
                    "need those blocks; blocks a request writes will page. %s",
                    plan.kv_budget / 1e9,
                    probe.available_before / 1e9,
                    probe.describe(),
                )
            return plan, None

        # Keep the unrounded fit as the budget: vLLM picks its own grouped
        # layout from the bytes it is handed, so rounding here to the dense
        # per-block estimate would throw away capacity the layout can use.
        # ``num_blocks`` is the count-based path's allocation size.
        num_blocks = fit // plan.per_block_bytes
        logger.warning(
            "Paged attention: KV cache sized down from %.2f GB to %.2f GB "
            "(%d blocks): the machine paged memory out to back a sample, so it "
            "cannot hold what --gpu-memory-utilization=%.2f asks for. %s",
            plan.kv_budget / 1e9,
            fit / 1e9,
            num_blocks,
            plan.fraction,
            probe.describe(),
        )
        # The room for KV cache and pool together. Unlike ``fit`` it is not
        # floored by the pool, so an automatic pool can be re-capped even when
        # the pool alone fills it.
        room = probe.available_before - reserve_bytes - other_reserved_bytes
        return replace(plan, kv_budget=fit, num_blocks=num_blocks), max(0, room)

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

    def _memory_fraction(self) -> float:
        """Resolve the paged KV memory fraction from ``--gpu-memory-utilization``."""
        fraction = self._worker.vllm_config.cache_config.gpu_memory_utilization
        logger.info(
            "Paged attention: using --gpu-memory-utilization=%.2f",
            fraction,
        )
        return fraction

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
