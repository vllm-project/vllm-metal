# SPDX-License-Identifier: Apache-2.0
"""
Metal vLLM v1 model runner.

Orchestration only: coordinates scheduling, dispatch, and output assembly.
Model-specific behavior belongs in adapters; backend-specific kernels live in
backend modules. Keep this file thin and stable.

Key contracts:
- execute_model()/sample_tokens() handoff remains unchanged.
- Outputs align with scheduler expectations for paged and non-paged paths.
"""

from dataclasses import dataclass, field
from typing import Any, Literal, NamedTuple, TypeAlias

import mlx.core as mx
import numpy as np
import torch
from mlx_lm import stream_generate
from mlx_lm.models.cache import make_prompt_cache
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.lora.request import LoRARequest
from vllm.pooling_params import PoolingParams
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import (
    EMPTY_MODEL_RUNNER_OUTPUT,
    DraftTokenIds,
    LogprobsLists,
    ModelRunnerOutput,
)
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_metal.attention.context import (
    OffsetCache,
    clear_context,
    get_context,
    prepare_unified,
)
from vllm_metal.attention.impls.mla import MLA_DEFAULT_QK_ROPE_HEAD_DIM
from vllm_metal.attention.runtime.protocol import PagedAttentionRuntime
from vllm_metal.config import get_config
from vllm_metal.distributed import (
    PipelinedModel,
    PipelineGroup,
    is_non_last_stage,
    pipeline_send,
)
from vllm_metal.multimodal import merge_multimodal_embeddings
from vllm_metal.multimodal.feature_spec import MultiModalFeatureSpec
from vllm_metal.v1.cache_policy import ModelCachePolicy
from vllm_metal.v1.contiguous_cache import (
    _MIN_BATCH_SIZE_FOR_BATCHING,
    AnyCache,
    KVCache,
    _extract_kv_cache,
    _merge_kv_caches,
)
from vllm_metal.v1.gemma4_mtp import (
    Gemma4MTPAssistantRuntime,
    Gemma4MTPAssistantSource,
)
from vllm_metal.v1.lora import MetalLoRARuntime
from vllm_metal.v1.mm import EncoderCache
from vllm_metal.v1.model_adapter import (
    DefaultModelAdapter,
    ModelAdapter,
    MultimodalRuntimeAdapter,
    TargetModelForwardOutput,
)
from vllm_metal.v1.model_lifecycle import ModelLifecycle
from vllm_metal.v1.pooling import (
    finish_paged_pooling_batch,
    forward_sequence_hidden_states,
    has_paged_pooling_work,
    pooling_dummy_forward_outputs,
    supported_pooling_tasks,
    validate_pooling_request,
)
from vllm_metal.v1.proposer import (
    Gemma4MTPProposer,
    MetalProposer,
    ProposeContext,
)
from vllm_metal.v1.sampling_batch import (
    GREEDY_TEMPERATURE_EPS,
    SamplingBatch,
    _SamplingResult,
    sample_decode_tokens,
    sample_from_logits,
    sample_prefill_tokens,
)
from vllm_metal.v1.spec_decode import (
    PagedDecodeSegment,
    SpeculativeDecodeController,
)
from vllm_metal.v1.structured_output import MetalStructuredOutputApplier

logger = init_logger(__name__)


SchedulerMemoryReportingMode: TypeAlias = Literal[
    "stt_nominal",
    "paged_attention_capacity",
    "single_sequence_estimate",
]


def _lora_id_from_request_data(new_req: NewRequestData) -> int | None:
    """Pull the int LoRA ID off a `NewRequestData` record."""
    if new_req.lora_request is None:
        return None
    return int(new_req.lora_request.lora_int_id)


def _create_request_generator(
    device: torch.device,
    sampling_params: SamplingParams,
) -> torch.Generator | None:
    """Create a per-request generator for seeded sampling.

    vLLM uses a per-request generator only when an explicit seed is provided.
    For unseeded sampling, vLLM relies on the global RNG state.
    """
    if sampling_params.seed is None:
        return None
    if sampling_params.temperature < GREEDY_TEMPERATURE_EPS:
        return None
    generator = torch.Generator(device=device)
    generator.manual_seed(sampling_params.seed)
    return generator


@dataclass
class RequestState:
    """State for an ongoing request with KV cache."""

    token_ids: list[int]
    # Length of the original prompt (prefix) within `token_ids`.
    # vLLM applies repetition penalties to both prompt+output tokens, but applies
    # presence/frequency penalties only to generated (output) tokens.
    prompt_len: int
    cache: list[AnyCache]  # Per-layer caches (KVCache, RotatingKVCache, or ArraysCache)
    sampling_params: SamplingParams  # Sampling parameters for this request
    pooling_params: PoolingParams | None = None
    generator: torch.Generator | None = None
    generated_tokens: int = 0
    block_ids: list[int] = field(
        default_factory=list
    )  # Scheduler-assigned paged KV blocks
    lora_id: int | None = None
    # Decode reconstructs M-RoPE positions as
    # ``len(token_ids) - 1 + mrope_position_delta``; ``None`` for text-only.
    mrope_position_delta: int | None = None


class PrefillRequest(NamedTuple):
    """Packed prefill request passed to ``_start_paged_forward``."""

    req_id: str
    token_ids: list[int]  # suffix slice forwarded through the model
    sampling_params: SamplingParams
    block_ids: list[int]
    generator: torch.Generator | None
    prompt_len: int | None  # full prompt length (None for intermediate chunks)
    start_pos: int  # RoPE / slot offset (0 = fresh, >0 = continuation)
    full_prompt_token_ids: list[int] | None  # full prompt for sampling metadata
    lora_id: int | None = None  # None = no LoRA adapter for this request
    pooling_params: PoolingParams | None = None


@dataclass
class _PendingPrefillEntry:
    """Paged prefill work plus the metadata needed for post-processing."""

    output_idx: int
    prefill: PrefillRequest
    result_mode: Literal["intermediate", "new_final", "cached_final"]


@dataclass
class _ExecutionBatch:
    """Typed accumulator for one ``execute_model()`` call."""

    req_ids: list[str] = field(default_factory=list)
    req_id_to_index: dict[str, int] = field(default_factory=dict)
    sampled_tokens: list[list[int]] = field(default_factory=list)
    sample_logprobs: list[LogprobsLists | None] = field(default_factory=list)
    pooler_outputs: list[torch.Tensor | None] = field(default_factory=list)
    new_reqs_by_id: dict[str, NewRequestData] = field(default_factory=dict)
    paged_prefill_entries: list[_PendingPrefillEntry] = field(default_factory=list)
    paged_decode_reqs: list[tuple[str, RequestState]] = field(default_factory=list)
    scheduled_cached_req_ids: list[str] = field(default_factory=list)
    valid_decode_reqs: list[tuple[str, RequestState]] = field(default_factory=list)

    def add_output(
        self,
        req_id: str,
        token_ids: list[int],
        logprobs: LogprobsLists | None = None,
        pooler_output: torch.Tensor | None = None,
    ) -> int:
        """Append one output slot and return its index."""
        self.req_ids.append(req_id)
        output_idx = len(self.req_ids) - 1
        self.req_id_to_index[req_id] = output_idx
        self.sampled_tokens.append(token_ids)
        self.sample_logprobs.append(logprobs)
        self.pooler_outputs.append(pooler_output)
        return output_idx

    def set_output(
        self,
        output_idx: int,
        token_ids: list[int],
        logprobs: LogprobsLists | None = None,
        pooler_output: torch.Tensor | None = None,
    ) -> None:
        """Set tokens and logprobs for an existing output slot."""
        self.sampled_tokens[output_idx] = token_ids
        self.sample_logprobs[output_idx] = logprobs
        self.pooler_outputs[output_idx] = pooler_output

    def merged_logprobs(self) -> LogprobsLists | None:
        """Merge per-output-slot logprobs for ``ModelRunnerOutput``."""
        return SamplingBatch.merge_logprobs_rows(self.sample_logprobs)

    def has_paged_work(self) -> bool:
        """Return whether this step has any paged execution work."""
        return bool(self.paged_prefill_entries or self.paged_decode_reqs)


class _PagedForwardState(NamedTuple):
    """State stashed by ``_start_paged_forward`` for ``_sample_paged_batch``."""

    batch: _ExecutionBatch
    prefill_reqs: list[PrefillRequest]
    decode_reqs: list[tuple[str, RequestState]]
    scheduler_output: SchedulerOutput
    logits: mx.array | None
    target_hidden_states: mx.array | None
    cu_seqlens: list[int]
    decode_segments: tuple[PagedDecodeSegment, ...]
    num_decode_tokens: int
    # ``{req_id: mrope_position_delta}`` from paged mm prefill;
    # ``_sample_paged_batch`` stashes each onto ``RequestState``.
    mm_prefill_deltas: dict[str, int]
    pooling_hidden_states: mx.array | None = None


class MetalModelRunner:
    """Model runner for MLX-based inference on Metal.

    Implements the vLLM v1 model runner interface for Apple Silicon.
    Uses true batched decode with BatchKVCache for efficient parallel processing.
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        device: torch.device,
    ):
        """Initialize model runner.

        Args:
            vllm_config: vLLM configuration
            device: PyTorch device (CPU for Metal interop)
        """
        self.vllm_config = vllm_config
        self.model_config = vllm_config.model_config
        self.cache_config = vllm_config.cache_config
        self.scheduler_config = vllm_config.scheduler_config
        self.use_async_scheduling = bool(self.scheduler_config.async_scheduling)
        self.device = device
        self.metal_config = get_config()
        self._model_adapter: ModelAdapter = DefaultModelAdapter()
        self._cache_policy = ModelCachePolicy(self, self._model_adapter)
        self._model_lifecycle = ModelLifecycle(self, self._model_adapter)
        self._lora = MetalLoRARuntime()
        self._spec_decode_controller = SpeculativeDecodeController()

        self.model: Any = None
        self.tokenizer: Any = None
        self.model_args: dict[str, Any] = {}
        self._is_vlm: bool = False  # Will be set during model loading
        self._is_pooling: bool = (
            getattr(self.model_config, "runner_type", None) == "pooling"
        )
        self._multimodal_adapter: MultimodalRuntimeAdapter | None = None
        self._gemma4_mtp_assistant: Gemma4MTPAssistantRuntime | None = None
        self._drafter: MetalProposer | None = None
        self.encoder_cache: EncoderCache | None = None

        # Request state cache for incremental decoding
        self._request_states: dict[str, RequestState] = {}

        # vLLM Sampler for token sampling with temperature, top_k, top_p support
        self._sampler = Sampler()

        # Build logits processors (includes custom plugins from entry-points)
        pin_memory = is_pin_memory_available()
        custom_lp = vllm_config.model_config.logits_processors
        custom_logitsprocs = tuple(custom_lp) if custom_lp is not None else ()
        self._logitsprocs = build_logitsprocs(
            vllm_config,
            device,
            pin_memory,
            self._is_pooling,
            custom_logitsprocs,
        )

        # vLLM v1 async scheduling calls sample_tokens after execute_model.
        # Keep the latest execution output so sample_tokens can return it.
        self._pending_output: ModelRunnerOutput | None = None
        self._draft_token_ids: DraftTokenIds | None = None

        # Paged attention state (set by worker when enabled)
        self._paged_attention_runtime: PagedAttentionRuntime | None = None
        self._paged_block_size: int = 0
        self._paged_request_seq_lens: dict[str, int] = {}  # req_id → seq_len
        self.kv_cache_dtype: mx.Dtype | None = None

        # Layer counts derived from the model config by ModelLifecycle after
        # load (declared here so the type is fixed; apply_pipeline_split resets
        # both to this stage's local slice under pipeline parallelism).
        self.num_layers: int = 0
        self.num_kv_cache_layers: int = 0

        # Per-layer KV cache shapes (None = uniform across layers)
        self.kv_heads_per_layer: list[int] | None = None
        self.head_dim_per_layer: list[int] | None = None
        # Per-layer attention metadata (None = enforcement disabled)
        self.sliding_window_per_layer: list[int] | None = None

        # Async forward state: stashed by execute_model, consumed by
        # sample_tokens (mirrors upstream's execute_model_state pattern).
        self._execute_model_state: _PagedForwardState | None = None

        # Pipeline-parallel group (set by the worker when pipeline_parallel_size
        # > 1). None means single-stage: the forward and sampling paths run
        # exactly as before, with no cross-stage send/recv.
        self.pp: PipelineGroup | None = None
        # PP-aware forward wrapper for this stage, built in apply_pipeline_split
        # when pp.size > 1; stays None on the single-stage path.
        self._pp_model: PipelinedModel | None = None

        # Structured-output bitmask applier for the paged path.
        self._structured_output_applier = MetalStructuredOutputApplier()

    @property
    def is_mla(self) -> bool:
        """Whether the model uses Multi-head Latent Attention (MLA).

        MLA models (GLM/DeepSeek lineage) have no q_proj/k_proj/v_proj and
        cannot use the standard Metal kernel. Worker uses this to select the
        appropriate paged attention backend for PR2.
        """
        return "kv_lora_rank" in self.model_args

    @property
    def is_hybrid(self) -> bool:
        """Whether the model mixes SDPA and linear attention layers.

        Hybrid models (Qwen3.5) have ``full_attention_interval`` in their
        config: every N-th layer uses SDPA, the rest use GDN linear attention.
        """
        fai = self.model_args.get("full_attention_interval", 0)
        return isinstance(fai, int) and fai > 0

    @property
    def _forward_model(self) -> Any:
        """The model object to use for forward passes.

        For VLMs loaded via mlx-vlm, the top-level ``Model.__call__`` requires
        ``pixel_values`` and ``mask`` arguments that are absent in text-only
        requests.  Routing through ``model.language_model`` bypasses the vision
        encoder and uses the standard ``(input_ids, cache=...)`` signature.

        NOTE: scheduled multimodal encoder inputs fail fast until the runner
        wires decomposed encode → feature-fusion → forward execution.
        """
        if self._is_vlm:
            if self._multimodal_adapter is not None:
                return self._multimodal_adapter.text_model()
            return self._model_adapter.text_model(self.model)
        return self.model

    @property
    def mla_latent_dim(self) -> int:
        """Combined latent dimension for MLA cache: kv_lora_rank + qk_rope_head_dim.

        Only valid when is_mla is True. Derived directly from model_args so
        callers do not depend on resolved runtime head_dim overrides.
        """
        if not self.is_mla:
            raise AttributeError("mla_latent_dim is only valid for MLA models")
        return int(self.model_args["kv_lora_rank"]) + int(
            self.model_args.get("qk_rope_head_dim", MLA_DEFAULT_QK_ROPE_HEAD_DIM)
        )

    def validate_paged_attention_support(self) -> None:
        """Validate that the loaded model can run on the paged-attention path."""
        self._cache_policy.validate_paged_attention_support()

    def scheduler_memory_reporting_mode(
        self, *, paged_attention_enabled: bool
    ) -> SchedulerMemoryReportingMode:
        """Return which scheduler memory-reporting mode worker should use.

        Worker delegates this decision to the runner so STT-specific policy is
        not open-coded in `worker.py`.
        """
        return self._cache_policy.scheduler_memory_reporting_mode(
            paged_attention_enabled=paged_attention_enabled
        )

    def supported_worker_tasks(self) -> tuple[SupportedTask, ...]:
        """Return worker task capabilities for the loaded model."""
        if self._is_pooling:
            if self._paged_attention_runtime is None:
                return ()
            return supported_pooling_tasks(
                self._forward_model, self.model_config, self.tokenizer
            )
        return ("generate",)

    def load_model(self) -> None:
        """Load the configured model and derive runtime metadata."""
        self._model_lifecycle.load()
        text_config = getattr(self.model_config.hf_config, "get_text_config", None)
        max_position_embeddings = None
        if callable(text_config):
            cfg = text_config()
            max_position_embeddings = getattr(cfg, "max_position_embeddings", None)
        self._lora.setup(
            model=self._forward_model,
            lora_config=getattr(self.vllm_config, "lora_config", None),
            is_stt=False,
            paged_attention_enabled=self.metal_config.use_paged_attention,
            speculative_decode_enabled=self.vllm_config.speculative_config is not None,
            max_num_seqs=self.scheduler_config.max_num_seqs,
            max_num_batched_tokens=self.scheduler_config.max_num_batched_tokens,
            dtype=self.kv_cache_dtype or mx.float16,
            max_position_embeddings=max_position_embeddings,
        )

    def add_lora(self, lora_request: LoRARequest) -> bool:
        return self._lora.add_adapter(lora_request)

    def remove_lora(self, lora_id: int) -> bool:
        return self._lora.remove_adapter(lora_id)

    def pin_lora(self, lora_id: int) -> bool:
        return self._lora.pin_adapter(lora_id)

    def list_loras(self) -> set[int]:
        return self._lora.list_adapters()

    def _paged_lora_routing(
        self,
        decode_reqs: list[tuple[str, RequestState]],
        prefill_pack: list[PrefillRequest],
    ) -> list[tuple[int | None, int]]:
        entries: list[tuple[int | None, int]] = []
        for _, state in decode_reqs:
            entries.append((state.lora_id, 1))
        for pr in prefill_pack:
            entries.append((pr.lora_id, len(pr.token_ids)))
        return entries

    def apply_pipeline_split(self, pp: PipelineGroup) -> None:
        """Slice the loaded model to this pipeline stage and fix layer counts.

        Called by the worker after ``load_model`` (Phase 0 load-then-slice).
        Delegates the in-place backbone slice to the adapter, then resets the
        runner's layer accounting to the LOCAL slice so per-layer offset caches
        (``_start_paged_forward``) and the KV-cache spec size only this stage's
        layers — not the full model.

        Only the validated path (uniform MHA, e.g. Qwen3) is supported under
        pipeline parallelism: fail loud on YOCO / hybrid / MLA models whose
        KV-cache layer accounting does not map cleanly onto a contiguous layer
        slice yet.
        """
        if pp.size == 1:
            return

        unsupported = (
            self._yoco_cache_mapping is not None
            or self.is_hybrid
            or self.is_mla
            or self._is_pooling
            or self._is_vlm
            or not self.metal_config.use_paged_attention
        )
        if unsupported:
            raise NotImplementedError(
                "Pipeline parallelism on Metal is validated only for uniform-"
                "attention generation on the paged path (e.g. Qwen3 with "
                "VLLM_METAL_USE_PAGED_ATTENTION=1); YOCO / hybrid (GDN) / MLA / "
                "pooling / VLM (multimodal) / non-paged configs are not "
                "supported under PP yet."
            )

        self._model_adapter.apply_pipeline_split(self.model, pp)
        # Mirror the sliced backbone's local layer count onto the runner so
        # offset_caches and KV-cache sizing cover only this stage's layers.
        local_num_layers = len(self._forward_model.model.layers)
        self.num_layers = local_num_layers
        self.num_kv_cache_layers = local_num_layers

        # Install the PP-aware forward wrapper for this stage. It owns the stage
        # forward (recv -> local layers -> final norm + head on the last stage);
        # the runner owns the downstream send (see the PP branch in the forward).
        self._pp_model = PipelinedModel(self._forward_model, pp)

    def _submit_paged_forward_outputs(
        self,
        *outputs: mx.array,
    ) -> None:
        """Submit caller outputs first, then runtime-owned side effects."""
        eval_outputs = list(outputs)
        runtime = self._paged_attention_runtime
        if runtime is not None:
            runtime.extend_forward_eval_outputs(eval_outputs)
        mx.async_eval(*eval_outputs)

    def _extract_logits(self, model_output: Any) -> mx.array:
        """Extract logits from model output.

        Handles both mlx-lm (returns array directly) and mlx-vlm
        (returns LanguageModelOutput with .logits attribute).

        Args:
            model_output: Output from model forward pass

        Returns:
            Logits array
        """
        return self._model_adapter.extract_logits(model_output)

    def _target_forward(
        self,
        input_ids: mx.array,
        *,
        cache: Any | None = None,
        collect_hidden_states: bool = False,
    ) -> TargetModelForwardOutput:
        return self._model_adapter.target_forward(
            self._forward_model,
            input_ids,
            cache=cache,
            collect_hidden_states=collect_hidden_states,
        )

    def _target_input_embeddings(self, input_ids: mx.array) -> mx.array:
        return self._model_adapter.target_input_embeddings(
            self._forward_model, input_ids
        )

    def take_draft_token_ids(self) -> DraftTokenIds | None:
        """Return and clear draft tokens generated by the last sampled step."""
        draft_token_ids = self._draft_token_ids
        self._draft_token_ids = None
        return draft_token_ids

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get KV cache specification.

        Returns:
            Dictionary mapping attention layer names to KV cache specs
        """
        return self._cache_policy.get_kv_cache_spec()

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Accept KV cache config from engine (no-op for MLX path).

        MLX manages its own KV cache via make_prompt_cache().
        This method exists to satisfy the engine's initialization protocol.
        """
        self._cache_policy.initialize_kv_cache(kv_cache_config)

    def reset_mm_cache(self) -> None:
        """Reset profiling-time multimodal cache state when present."""
        if self.encoder_cache is not None:
            self.encoder_cache.reset_mm_cache()

    def reset_encoder_cache(self) -> None:
        """Clear cached multimodal encoder outputs when present."""
        if self.encoder_cache is not None:
            self.encoder_cache.reset_encoder_cache()

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of a single cache block in bytes.

        Returns:
            Block size in bytes
        """
        return self._cache_policy.get_cache_block_size_bytes()

    def linear_cache_bytes_per_slot(self) -> int:
        """Bytes for one request's linear attention state across all GDN layers."""
        return self._cache_policy.linear_cache_bytes_per_slot()

    def profile_run(self) -> int:
        """Measure MLX buffer-cache footprint of one forward pass and cap the allocator.

        Called from ``MetalWorker.determine_available_memory`` before KV cache
        sizing so the measured overhead replaces the historical 800 MB
        placeholder. ``mx.set_cache_limit`` prevents unbounded buffer-cache
        growth during serving (issue #234).
        """
        warmup_len = self.scheduler_config.max_num_batched_tokens
        mx.clear_cache()
        cache_before = mx.get_cache_memory()
        dummy_tokens = mx.zeros((1, warmup_len), dtype=mx.int32)
        mx.eval(*self._dummy_forward_outputs(dummy_tokens))
        overhead = mx.get_cache_memory() - cache_before
        mx.set_cache_limit(overhead)
        return overhead

    def _dummy_forward_outputs(self, input_ids: mx.array) -> list[mx.array]:
        if self._is_pooling:
            return pooling_dummy_forward_outputs(
                self._forward_model,
                input_ids,
                model_config=self.model_config,
            )

        output = self._forward_model(input_ids)
        logits = self._extract_logits(output)
        return [logits]

    def build_paged_attention_runtime(
        self, *, block_size: int
    ) -> PagedAttentionRuntime:
        """Build the paged-attention backend for the loaded model."""
        return self._cache_policy.build_paged_attention_runtime(block_size=block_size)

    @property
    def paged_attention_runtime(self) -> PagedAttentionRuntime | None:
        """Return the installed paged-attention backend, if any."""
        return self._paged_attention_runtime

    def install_paged_attention_runtime(
        self,
        backend: PagedAttentionRuntime,
        *,
        block_size: int,
    ) -> None:
        """Record the initialized paged-attention backend owned by this runner."""
        self._paged_attention_runtime = backend
        self._paged_block_size = block_size

    def install_gemma4_mtp_kv_sharing(
        self,
        backend: PagedAttentionRuntime,
        *,
        block_size: int,
    ) -> None:
        """Wire Gemma4 MTP assistant sharing after paged cache initialization."""
        self._cache_policy.install_gemma4_mtp_kv_sharing(
            backend,
            block_size=block_size,
        )

    def install_drafter(self, *, num_blocks: int, block_size: int) -> None:
        """Construct the polymorphic drafter once the paged cache is ready.

        One factory for both speculative methods, keyed on the speculative
        method. Gemma4 MTP uses the in-model assistant loaded in
        ``ModelLifecycle`` (read lazily by the proposer); draft-model SD loads
        its own model + a paged cache sized to the target's ``num_blocks`` —
        which is why this runs after the paged backend exists. A configured but
        unsupported method fails loud rather than silently degrading to plain
        decode (which would look like a drafter that never accepts anything).
        """
        spec = self.vllm_config.speculative_config
        if spec is None:
            return
        if Gemma4MTPAssistantSource.is_gemma4_mtp(spec):
            self._drafter = Gemma4MTPProposer(self)
        elif spec.uses_draft_model():
            from vllm_metal.v1.draft_model_proposer import DraftModelProposer

            self._drafter = DraftModelProposer.build(
                speculative_config=spec,
                controller=self._spec_decode_controller,
                extract_logits=self._model_adapter.extract_logits,
                num_blocks=num_blocks,
                block_size=block_size,
                dtype=self.kv_cache_dtype,
            )
            max_num_seqs = self.scheduler_config.max_num_seqs
            extra_per_req = (spec.num_speculative_tokens + block_size - 1) // block_size
            if num_blocks < max_num_seqs * extra_per_req:
                raise ValueError(
                    f"Draft KV cache too small: {num_blocks} blocks cannot "
                    f"support {max_num_seqs} concurrent requests each needing "
                    f"{extra_per_req} extra block(s) for "
                    f"{spec.num_speculative_tokens} speculative tokens. "
                    "Raise VLLM_METAL_MEMORY_FRACTION or lower --max-num-seqs."
                )
        elif spec.method == "ngram":
            from vllm_metal.v1.ngram_proposer import NgramProposer

            # N-gram drafts from token history alone — no model, no KV cache, so
            # num_blocks/block_size are unused here.
            self._drafter = NgramProposer.build(
                vllm_config=self.vllm_config,
                controller=self._spec_decode_controller,
            )
        else:
            raise NotImplementedError(
                f"Speculative method {spec.method!r} is not supported on Metal "
                "(supported: Gemma4 MTP, draft_model, ngram)."
            )

    def estimate_one_sequence_kv_bytes(
        self, *, max_model_len: int, block_size: int
    ) -> int:
        """Estimate bytes for one max-length sequence of cache state."""
        return self._cache_policy.estimate_one_sequence_kv_bytes(
            max_model_len=max_model_len,
            block_size=block_size,
        )

    def warm_up(self) -> None:
        """Warm up the model with a dummy forward pass.

        Paged-attention Metal/MLX kernels JIT-compile lazily on first use,
        so the paged backend's ``warm_up`` is a no-op; this method only runs
        a small dummy forward pass.
        """
        if self.model is None:
            logger.warning("Model not loaded, skipping warm-up")
            return

        logger.info("Warming up model...")

        # Run a small dummy inference (standard MLX path)
        try:
            dummy_tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
            mx.eval(*self._dummy_forward_outputs(dummy_tokens))
            logger.info("Model warm-up complete")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

        if self._paged_attention_runtime is not None:
            self._paged_attention_runtime.warm_up()

    def _make_sampling_metadata(
        self,
        sampling_params_list: list[SamplingParams],
        prompt_token_id_lists: list[list[int]],
        output_token_id_lists: list[list[int]],
        generators: dict[int, torch.Generator] | None = None,
    ) -> SamplingMetadata:
        """Create SamplingMetadata from per-request SamplingParams."""
        return SamplingBatch(
            sampling_params_list,
            prompt_token_id_lists,
            output_token_id_lists,
            vocab_size=self._vocab_size,
            device=self.device,
            logitsprocs=self._logitsprocs,
            generators=generators,
        ).make_sampling_metadata()

    def _prefill_single(
        self,
        token_ids: list[int],
        sampling_params: SamplingParams,
        generator: torch.Generator | None = None,
    ) -> tuple[int, list[KVCache], LogprobsLists | None]:
        """Process a single prefill request.

        Args:
            token_ids: Prompt token IDs
            sampling_params: Sampling parameters for this request

        Returns:
            Tuple of (next_token, cache)
        """
        cache: list[KVCache] = make_prompt_cache(self._forward_model)

        input_ids = mx.array([token_ids], dtype=mx.int32)
        model_output = self._forward_model(input_ids, cache=cache)

        logits = self._extract_logits(model_output)

        # Extract last token logits
        last_logits = logits[:, -1, :]

        vocab_size = self._vocab_size
        generators = {} if generator is None else {0: generator}
        batch = SamplingBatch(
            [sampling_params],
            [token_ids],
            [[]],
            vocab_size=vocab_size,
            device=self.device,
            logitsprocs=self._logitsprocs,
            generators=generators,
        )
        result = sample_from_logits(last_logits, batch, self._sampler, self.device)
        [next_token] = result.token_ids
        mx.eval(*[c.state for c in cache])

        return next_token, cache, result.logprobs

    def _batched_decode(
        self, decode_reqs: list[tuple[str, RequestState]]
    ) -> _SamplingResult:
        """Process multiple decode requests in a single batched forward pass.

        Uses BatchKVCache to merge individual caches, run ONE forward pass,
        then extract updated caches back.

        Args:
            decode_reqs: List of (req_id, state) tuples

        Returns:
            Sampled token IDs and optional logprobs for each request.
        """
        last_tokens = [
            state.token_ids[-1] if state.token_ids else 0 for _, state in decode_reqs
        ]

        # Collect individual caches for merging
        caches_list = [state.cache for _, state in decode_reqs]

        # Merge individual KV caches into batched cache (one per layer)
        batch_cache = _merge_kv_caches(caches_list)

        # Create batched input: shape (batch_size, 1) for single-token decode
        batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]

        # === SINGLE FORWARD PASS FOR ALL REQUESTS ===
        model_output = self._forward_model(batched_input, cache=batch_cache)
        logits = self._extract_logits(model_output)

        # Extract next token logits
        next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)

        vocab_size = self._vocab_size
        sampling_params_list = [state.sampling_params for _, state in decode_reqs]
        prompt_token_ids_list = [
            state.token_ids[: state.prompt_len] for _, state in decode_reqs
        ]
        output_tokens_list = [
            state.token_ids[state.prompt_len :] for _, state in decode_reqs
        ]
        generators = {
            i: state.generator
            for i, (_, state) in enumerate(decode_reqs)
            if state.generator is not None
        }
        batch = SamplingBatch(
            sampling_params_list,
            prompt_token_ids_list,
            output_tokens_list,
            vocab_size=vocab_size,
            device=self.device,
            logitsprocs=self._logitsprocs,
            generators=generators,
        )
        result = sample_from_logits(
            next_token_logits, batch, self._sampler, self.device
        )
        next_tokens = result.token_ids

        # Extract updated caches back to individual requests
        for i, (_req_id, state) in enumerate(decode_reqs):
            state.cache = _extract_kv_cache(batch_cache, i)
            state.token_ids.append(next_tokens[i])
            state.generated_tokens += 1

        return result

    def _sequential_decode(
        self, decode_reqs: list[tuple[str, RequestState]]
    ) -> _SamplingResult:
        """Fallback: process decode requests sequentially.

        Used when batch size is 1 (no benefit from batching).

        Args:
            decode_reqs: List of (req_id, state) tuples

        Returns:
            Sampled token IDs and optional logprobs for each request.
        """
        next_tokens = []
        logprobs_rows: list[LogprobsLists | None] = []

        for _req_id, state in decode_reqs:
            last_token = state.token_ids[-1] if state.token_ids else 0
            input_ids = mx.array([[last_token]], dtype=mx.int32)

            model_output = self._forward_model(input_ids, cache=state.cache)
            logits = self._extract_logits(model_output)
            last_logits = logits[:, -1, :]

            vocab_size = self._vocab_size
            generators = {} if state.generator is None else {0: state.generator}
            batch = SamplingBatch(
                [state.sampling_params],
                [state.token_ids[: state.prompt_len]],
                [state.token_ids[state.prompt_len :]],
                vocab_size=vocab_size,
                device=self.device,
                logitsprocs=self._logitsprocs,
                generators=generators,
            )
            result = sample_from_logits(last_logits, batch, self._sampler, self.device)
            [next_token] = result.token_ids

            next_tokens.append(next_token)
            logprobs_rows.append(result.logprobs)

            # Update state
            state.token_ids.append(next_token)
            state.generated_tokens += 1

        return _SamplingResult(
            next_tokens,
            SamplingBatch.merge_logprobs_rows(logprobs_rows),
        )

    # ------------------------------------------------------------------
    # Unified prefill + decode (single forward pass)
    # ------------------------------------------------------------------

    def _start_paged_forward(
        self,
        batch: _ExecutionBatch,
        prefill_reqs: list[PrefillRequest],
        decode_reqs: list[tuple[str, RequestState]],
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Build graph and submit forward pass to GPU (async).

        Stashes all state needed by ``sample_tokens`` in
        ``_execute_model_state`` (mirrors upstream's pattern).
        """
        decode_segments = self._spec_decode_controller.build_decode_segments(
            decode_reqs,
            scheduler_output.scheduled_spec_decode_tokens,
            self._paged_request_seq_lens,
        )
        num_decode_tokens = sum(segment.num_query_tokens for segment in decode_segments)
        has_pooling_work = has_paged_pooling_work(prefill_reqs, decode_reqs)

        # prompt_len=None marks an intermediate prefill chunk; only final
        # prefill rows can seed the next Gemma4 MTP draft step. Pooling batches
        # do not sample or draft tokens, so they never request target hidden
        # states here.
        collect_target_hidden_states = (
            not has_pooling_work
            and self._drafter is not None
            and self._drafter.needs_target_hidden_states(
                decode_segments,
                has_final_prefill=any(pr.prompt_len is not None for pr in prefill_reqs),
            )
        )

        # Fail fast on mm requests reaching the paged path without a
        # forward-ready adapter: continuation chunks whose vision features
        # were encoded earlier have no scheduled encoder input and would
        # otherwise slip through to the text path.
        has_mm_prefill = any(self._is_mm_request(pr.req_id) for pr in prefill_reqs)
        has_mm_decode = any(
            state.mrope_position_delta is not None for _, state in decode_reqs
        )
        has_mm = has_mm_prefill or has_mm_decode
        adapter = self._multimodal_adapter
        if has_mm and (adapter is None or not adapter.forward_ready):
            raise RuntimeError(
                "Paged forward saw a multimodal request but the adapter is "
                "not forward_ready; this indicates a misconfigured adapter "
                "or a bookkeeping bug — only forward-ready adapters should "
                "let mm requests reach paged forward."
            )

        # Some VLM LMs derive RoPE from model-level position state.  On the
        # text path they would re-derive positions against zero-offset paged
        # caches, corrupting decode/packed/chunked text batches.  Adapters flag
        # ``requires_explicit_positions`` so text-only batches also run the mm
        # forward, which always passes position_ids.
        use_mm_forward = has_mm or (
            adapter is not None
            and adapter.forward_ready
            and adapter.requires_explicit_positions
        )

        # ---- build unified token sequence: decode first, then prefill ----
        all_token_ids: list[int] = []

        # Decode: last token plus any scheduled draft tokens per request.
        for segment in decode_segments:
            all_token_ids.extend(segment.input_token_ids)

        # Prefill: tokens per request
        for pr in prefill_reqs:
            all_token_ids.extend(pr.token_ids)

        # ---- build metadata for prepare_unified ----
        decode_info: list[tuple[list[int], int, int]] = []
        for segment in decode_segments:
            decode_info.append(
                (
                    list(segment.block_ids),
                    segment.cache_start_pos,
                    segment.num_query_tokens,
                )
            )

        prefill_info: list[tuple[list[int], int, int]] = []
        for pr in prefill_reqs:
            prefill_info.append((pr.block_ids, len(pr.token_ids), pr.start_pos))

        logits: mx.array | None = None
        target_hidden_states: mx.array | None = None
        pooling_hidden_states: mx.array | None = None
        mm_prefill_deltas: dict[str, int] = {}
        # Lazy send op for the non-last pipeline stage (None otherwise).
        pp_send_handle: mx.array | None = None

        prepare_unified(decode_info, prefill_info, self._paged_block_size)
        try:
            ctx = get_context()
            runtime = self._paged_attention_runtime
            if ctx is not None and runtime is not None and runtime.needs_step_context():
                step_req_ids = [req_id for req_id, _ in decode_reqs]
                step_req_ids.extend(pr.req_id for pr in prefill_reqs)
                runtime.populate_step_context(req_ids=step_req_ids, ctx=ctx)

            # ---- forward (lazy graph + async submit) ----
            offset_caches = [OffsetCache(0) for _ in range(self.num_layers)]
            input_ids = mx.array([all_token_ids], dtype=mx.int32)
            if has_pooling_work:
                pooling_hidden_states = forward_sequence_hidden_states(
                    self._forward_model,
                    input_ids,
                    cache=offset_caches,
                    model_config=self.model_config,
                )
            elif use_mm_forward:
                model_output, mm_prefill_deltas = self._run_mm_paged_forward(
                    input_ids,
                    offset_caches,
                    prefill_reqs,
                    decode_segments,
                )
                logits = self._extract_logits(model_output)
                target_hidden_states = None
                del model_output
            elif self.pp is not None and self.pp.size > 1:
                # Pipeline-parallel stage. The PP-aware wrapper owns the forward
                # (recv the upstream hidden state if not first -> this stage's
                # layer slice -> final norm + head on the last stage). The runner
                # owns the downstream send: only the last stage has logits; every
                # other stage hands its raw hidden state to pipeline_send, whose
                # lazy op is forced by the async_eval below.
                assert self._pp_model is not None
                stage_output = self._pp_model(input_ids, cache=offset_caches)
                if self.pp.is_last:
                    logits = self._extract_logits(stage_output)
                else:
                    logits = None
                    pp_send_handle = pipeline_send(stage_output, self.pp)
                target_hidden_states = None
            else:
                target_output = self._target_forward(
                    input_ids,
                    cache=offset_caches,
                    collect_hidden_states=collect_target_hidden_states,
                )
                logits = target_output.logits
                target_hidden_states = target_output.hidden_states
                del target_output
        finally:
            clear_context()

        # Submit to GPU — returns immediately, GPU runs in background.
        if has_pooling_work:
            assert pooling_hidden_states is not None
            self._submit_paged_forward_outputs(pooling_hidden_states)
        elif pp_send_handle is not None:
            # Non-last pipeline stage: no logits, just push the hidden state to
            # the next stage, plus any runtime-owned forward side effects.
            self._submit_paged_forward_outputs(pp_send_handle)
        else:
            assert logits is not None
            # Runtime-owned forward side effects may not be forced by
            # evaluating logits alone, so submit the complete output set.
            forward_outputs = [logits]
            if target_hidden_states is not None:
                forward_outputs.append(target_hidden_states)
            self._submit_paged_forward_outputs(*forward_outputs)

        # ---- build cu_seqlens for logit extraction ----
        cu_seqlens: list[int] = [0]
        for segment in decode_segments:
            cu_seqlens.append(cu_seqlens[-1] + segment.num_query_tokens)
        for pr in prefill_reqs:
            cu_seqlens.append(cu_seqlens[-1] + len(pr.token_ids))

        self._execute_model_state = _PagedForwardState(
            batch=batch,
            prefill_reqs=prefill_reqs,
            decode_reqs=decode_reqs,
            scheduler_output=scheduler_output,
            logits=logits,
            target_hidden_states=target_hidden_states,
            pooling_hidden_states=pooling_hidden_states,
            cu_seqlens=cu_seqlens,
            decode_segments=decode_segments,
            num_decode_tokens=num_decode_tokens,
            mm_prefill_deltas=mm_prefill_deltas,
        )

    def _sample_paged_batch(
        self,
        grammar_output: GrammarOutput | None = None,
    ) -> tuple[_ExecutionBatch, SchedulerOutput]:
        """Eval logits, sample tokens, and postprocess paged batch.

        Consumes state stashed by ``_start_paged_forward``.
        Returns ``(batch, scheduler_output)`` for the caller to finalize.
        """
        paged_state = self._execute_model_state
        assert paged_state is not None
        self._execute_model_state = None
        batch = paged_state.batch
        prefill_reqs = paged_state.prefill_reqs
        decode_reqs = paged_state.decode_reqs
        scheduler_output = paged_state.scheduler_output
        logits = paged_state.logits
        target_hidden_states = paged_state.target_hidden_states
        pooling_hidden_states = paged_state.pooling_hidden_states
        cu_seqlens = paged_state.cu_seqlens
        decode_segments = paged_state.decode_segments
        num_decode_segments = len(decode_segments)
        num_decode_tokens = paged_state.num_decode_tokens
        mm_prefill_deltas = paged_state.mm_prefill_deltas
        has_scheduled_drafts = any(
            segment.draft_token_ids for segment in decode_segments
        )
        self._draft_token_ids = None

        if pooling_hidden_states is not None:
            finish_paged_pooling_batch(
                batch,
                pooling_hidden_states,
                cu_seqlens=cu_seqlens,
                num_decode_segments=num_decode_segments,
                model=self._forward_model,
                tokenizer=self.tokenizer,
                model_config=self.model_config,
            )
            return batch, scheduler_output

        assert logits is not None

        # ---- wait for MLX forward to complete ----
        if target_hidden_states is not None:
            # The Gemma4 MTP assistant drafter will consume these rows after
            # sampling; evaluate them with logits so the retained state is ready.
            mx.eval(logits, target_hidden_states)
        else:
            mx.eval(logits)

        # ---- apply structured output bitmask if present ----
        if grammar_output is not None:
            logits = self._structured_output_applier.apply_paged(
                scheduler_output,
                grammar_output,
                decode_reqs,
                prefill_reqs,
                cu_seqlens,
                num_decode_segments,
                logits,
                decode_segments=decode_segments,
            )

        # ---- sample tokens ----
        vocab_size = self._vocab_size
        logitsprocs = self._logitsprocs
        decode_token_ids: list[list[int]] = [[] for _ in decode_reqs]
        decode_logprobs_rows: list[LogprobsLists | None] = [None for _ in decode_reqs]
        if has_scheduled_drafts:
            spec_items = [
                (i, req, segment)
                for i, (req, segment) in enumerate(
                    zip(decode_reqs, decode_segments, strict=True)
                )
                if segment.draft_token_ids
            ]
            if spec_items:
                spec_token_ids = self._spec_decode_controller.verify_greedy(
                    logits,
                    [req for _, req, _ in spec_items],
                    [segment for _, _, segment in spec_items],
                    logitsprocs=logitsprocs,
                )
                for (decode_index, _, _), sampled_ids in zip(
                    spec_items,
                    spec_token_ids,
                    strict=True,
                ):
                    decode_token_ids[decode_index] = sampled_ids

            plain_items = [
                (i, req, segment)
                for i, (req, segment) in enumerate(
                    zip(decode_reqs, decode_segments, strict=True)
                )
                if not segment.draft_token_ids
            ]
            if plain_items:
                plain_logits = mx.stack(
                    [logits[0, segment.start_row, :] for _, _, segment in plain_items]
                )
                plain_reqs = [req for _, req, _ in plain_items]
                sampling_params_list = [
                    state.sampling_params for _, state in plain_reqs
                ]
                prompt_token_ids_list = [
                    state.token_ids[: state.prompt_len] for _, state in plain_reqs
                ]
                output_tokens_list = [
                    state.token_ids[state.prompt_len :] for _, state in plain_reqs
                ]
                generators = {
                    i: state.generator
                    for i, (_, state) in enumerate(plain_reqs)
                    if state.generator is not None
                }
                plain_batch = SamplingBatch(
                    sampling_params_list,
                    prompt_token_ids_list,
                    output_tokens_list,
                    vocab_size=vocab_size,
                    device=self.device,
                    logitsprocs=logitsprocs,
                    generators=generators,
                )
                plain_result = sample_from_logits(
                    plain_logits,
                    plain_batch,
                    self._sampler,
                    self.device,
                )
                for plain_index, (decode_index, _, _) in enumerate(plain_items):
                    decode_token_ids[decode_index] = [
                        plain_result.token_ids[plain_index]
                    ]
                    if plain_result.logprobs is not None:
                        decode_logprobs_rows[decode_index] = (
                            plain_result.logprobs.slice_request(plain_index, 1)
                        )
            decode_logprobs = SamplingBatch.merge_logprobs_rows(decode_logprobs_rows)
        else:
            decode_result = sample_decode_tokens(
                logits,
                decode_reqs,
                num_decode_tokens,
                self._sampler,
                self.device,
                vocab_size=vocab_size,
                logitsprocs=logitsprocs,
            )
            decode_token_ids = [[token_id] for token_id in decode_result.token_ids]
            decode_logprobs = decode_result.logprobs
        prefill_result = sample_prefill_tokens(
            logits,
            prefill_reqs,
            cu_seqlens,
            num_decode_segments,
            self._sampler,
            self.device,
            vocab_size=vocab_size,
            logitsprocs=logitsprocs,
        )

        # ---- update decode state ----
        for i, (req_id, state) in enumerate(decode_reqs):
            sampled_ids = decode_token_ids[i]
            state.token_ids.extend(sampled_ids)
            state.generated_tokens += len(sampled_ids)
            self._paged_request_seq_lens[req_id] = self._paged_request_seq_lens.get(
                req_id,
                len(state.token_ids) - len(sampled_ids) - 1,
            ) + len(sampled_ids)

        # ---- update prefill seq lens ----
        for pr in prefill_reqs:
            self._paged_request_seq_lens[pr.req_id] = pr.start_pos + len(pr.token_ids)

        # ---- postprocess: write results back into batch ----
        for i, entry in enumerate(batch.paged_prefill_entries):
            next_token = prefill_result.token_ids[i]
            logprobs = (
                prefill_result.logprobs.slice_request(i, 1)
                if prefill_result.logprobs is not None
                else None
            )
            prefill = prefill_reqs[i]

            if entry.result_mode == "intermediate":
                batch.set_output(entry.output_idx, [], logprobs)
                continue

            batch.set_output(entry.output_idx, [next_token], logprobs)
            mm_delta = mm_prefill_deltas.get(prefill.req_id)
            if entry.result_mode == "new_final":
                prompt_len = prefill.prompt_len
                assert prompt_len is not None
                full_prompt = (
                    prefill.full_prompt_token_ids
                    if prefill.full_prompt_token_ids is not None
                    else prefill.token_ids
                )
                self._request_states[prefill.req_id] = RequestState(
                    token_ids=full_prompt + [next_token],
                    prompt_len=prompt_len,
                    cache=[],
                    sampling_params=prefill.sampling_params,
                    pooling_params=prefill.pooling_params,
                    generator=prefill.generator,
                    generated_tokens=1,
                    block_ids=prefill.block_ids,
                    lora_id=prefill.lora_id,
                    mrope_position_delta=mm_delta,
                )
                continue

            req_state = self._request_states[prefill.req_id]
            req_state.token_ids.append(next_token)
            req_state.generated_tokens = len(req_state.token_ids) - req_state.prompt_len
            if mm_delta is not None:
                # Stash the freshly computed delta so the next decode round
                # routes through the mm path.
                req_state.mrope_position_delta = mm_delta

        for i, (req_id, _) in enumerate(batch.paged_decode_reqs):
            logprobs = (
                decode_logprobs.slice_request(i, 1)
                if decode_logprobs is not None
                else None
            )
            batch.add_output(req_id, decode_token_ids[i], logprobs)

        draft_ctx = ProposeContext(
            target_hidden_states=target_hidden_states,
            decode_reqs=decode_reqs,
            decode_segments=decode_segments,
            decode_token_ids=decode_token_ids,
            prefill_reqs=prefill_reqs,
            prefill_token_ids=prefill_result.token_ids,
            prefill_result_modes=[
                entry.result_mode for entry in batch.paged_prefill_entries
            ],
            request_states=self._request_states,
            cu_seqlens=cu_seqlens,
            num_decode_segments=num_decode_segments,
            logitsprocs=self._logitsprocs,
        )
        self._draft_token_ids = (
            self._drafter.propose(draft_ctx) if self._drafter is not None else None
        )

        return batch, scheduler_output

    def _register_new_request_mm_features(
        self, req_id: str, new_req: NewRequestData
    ) -> None:
        """Store scheduler-provided multimodal features for future encoder use."""
        if self.encoder_cache is None:
            return
        self.encoder_cache.remove_request(req_id)
        self.encoder_cache.add_request(req_id, new_req.mm_features)

    def _pre_register_new_request_mm_features(
        self, new_reqs: list[NewRequestData]
    ) -> None:
        """Register mm_features for new requests before encoder dispatch.

        The vLLM scheduler can place a brand-new multimodal request and its
        first ``scheduled_encoder_inputs`` in the same ``SchedulerOutput``.
        Encoder dispatch looks the request's mm_features up by ``req_id``,
        so registration must happen before
        :meth:`_reject_scheduled_encoder_inputs`, not later inside
        :meth:`_handle_new_requests`.  Only the lightweight mm-features
        bookkeeping is moved up; per-request prefill scheduling remains in
        ``_handle_new_requests`` so the fail-fast checks still guard the
        real model work.
        """
        if self.encoder_cache is None:
            return
        for new_req in new_reqs:
            self._register_new_request_mm_features(new_req.req_id, new_req)

    def _remove_request_mm_features(self, req_id: str) -> None:
        """Drop request-scoped multimodal feature metadata."""
        if self.encoder_cache is not None:
            self.encoder_cache.remove_request(req_id)

    def _free_encoder_outputs(self, mm_hashes: list[str]) -> None:
        """Drop encoder outputs released by the scheduler."""
        if self.encoder_cache is None:
            return
        for mm_hash in mm_hashes:
            self.encoder_cache.free_encoder_cache(mm_hash)

    @staticmethod
    def _finished_req_ids(
        scheduler_output: SchedulerOutput,
    ) -> set[str]:
        """Return request ids whose runner-owned state should be evicted."""
        return scheduler_output.finished_req_ids

    def _reject_scheduled_encoder_inputs(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
    ) -> None:
        """Dispatch to vision encoders or fail fast based on adapter state.

        When the active adapter signals ``forward_ready`` *and* the paged
        attention backend is active, scheduled encoder inputs are routed to
        :meth:`_run_vision_encoders`.  Otherwise the gate raises so that mm
        requests never reach a misconfigured adapter or the non-paged path —
        only the paged path splices encoded image embeddings, so the legacy
        path would run the language model on raw image placeholder tokens.
        """
        if not scheduled_encoder_inputs:
            return
        adapter = self._multimodal_adapter
        if adapter is None or not adapter.forward_ready:
            raise RuntimeError(
                "Multimodal encoder dispatch requested but adapter is missing "
                "or not forward_ready; mm requests cannot run on this "
                "configuration."
            )
        if self._paged_attention_runtime is None:
            raise NotImplementedError(
                "Multimodal requests require the paged attention backend. "
                "Set VLLM_METAL_USE_PAGED_ATTENTION=1: only the paged path "
                "splices encoded image embeddings via _run_mm_paged_forward; "
                "the non-paged legacy path would run the language model on raw "
                "image placeholder tokens (RFC #319 hard rule 4: multimodal "
                "is paged-only)."
            )
        self._run_vision_encoders(scheduled_encoder_inputs)

    def _spec_decode_preflight_reqs(
        self,
        scheduler_output: SchedulerOutput,
    ) -> tuple[tuple[str, RequestState], ...]:
        """Return current decode requests without mutating runner state."""
        if self._paged_attention_runtime is None:
            return ()

        decode_reqs: list[tuple[str, RequestState]] = []
        for req_id in scheduler_output.scheduled_cached_reqs.req_ids:
            state = self._request_states.get(req_id)
            if state is not None and state.generated_tokens > 0:
                decode_reqs.append((req_id, state))
        return tuple(decode_reqs)

    def _validate_spec_decode_supported(
        self,
        scheduler_output: SchedulerOutput,
    ) -> None:
        self._spec_decode_controller.validate_supported(
            scheduler_output,
            self._spec_decode_preflight_reqs(scheduler_output),
            paged_attention_enabled=self._paged_attention_runtime is not None,
            is_hybrid=self.is_hybrid,
            logitsprocs=self._logitsprocs,
            use_async_scheduling=self.use_async_scheduling,
            speculative_config=self.vllm_config.speculative_config,
        )

    def _run_vision_encoders(
        self,
        scheduled_encoder_inputs: dict[str, list[int]],
    ) -> None:
        """Run the vision encoder for scheduled features, stash by identifier.

        Skips features whose ``identifier`` already lives in
        ``encoder_cache.encoder_outputs`` (cache hit; the scheduler may
        re-list a feature across chunks).  Uncached features are batched into
        one ``adapter.encode_multimodal`` call so adapters can handle the
        scheduled encoder work for the whole step.
        """
        adapter = self._multimodal_adapter
        cache = self.encoder_cache
        if adapter is None or cache is None:
            return

        features_to_encode: list[MultiModalFeatureSpec] = []
        identifiers_to_encode: set[str] = set()
        for req_id, feature_indices in scheduled_encoder_inputs.items():
            mm_features = cache.mm_features.get(req_id)
            if mm_features is None:
                raise RuntimeError(
                    f"Scheduled encoder input for unregistered request "
                    f"{req_id!r}; encoder cache mm_features missing."
                )
            for idx in feature_indices:
                if idx < 0 or idx >= len(mm_features):
                    raise IndexError(
                        f"Encoder feature index {idx} out of range for "
                        f"request {req_id!r} with {len(mm_features)} features."
                    )
                feature = mm_features[idx]
                if (
                    feature.identifier in cache.encoder_outputs
                    or feature.identifier in identifiers_to_encode
                ):
                    continue
                features_to_encode.append(feature)
                identifiers_to_encode.add(feature.identifier)

        if not features_to_encode:
            return

        outputs = adapter.encode_multimodal(features_to_encode)
        if len(outputs) != len(features_to_encode):
            raise RuntimeError(
                f"encode_multimodal returned {len(outputs)} outputs for "
                f"{len(features_to_encode)} features; adapter must return one "
                "result per feature."
            )
        for feature, output in zip(features_to_encode, outputs, strict=True):
            cache.encoder_outputs[feature.identifier] = output

    def _run_mm_paged_forward(
        self,
        input_ids: mx.array,
        offset_caches: list[OffsetCache],
        prefill_reqs: list[PrefillRequest],
        decode_segments: tuple[PagedDecodeSegment, ...],
    ) -> tuple[Any, dict[str, int]]:
        """Run paged forward through ``adapter.call_lm`` with packed splice.

        Builds per-segment M-RoPE positions (sliced out of the full-prompt
        positions for mm prefill chunks, computed as ``cache_start_pos +
        delta + arange(num_query_tokens)`` for mm decode), splices vision
        embeds into the packed text embeds at placeholder positions
        (chunk-aware: each feature only contributes the slice that lands
        in *this* chunk), and concatenates per-layer deepstack residual
        arrays across all mm prefill segments in packed order.

        Sets ``ctx.segment_positions`` so ``apply_packed_rope`` reads
        caller-supplied positions on mm segments and falls back to the
        int-offset arange path on text segments.

        Any speculative-decode segment in the batch is rejected up
        front: a decode request with ``num_query_tokens > 1`` makes
        ``prepare_unified`` append one ``cu_seqlens`` entry per query
        token, but ``ctx.segment_positions`` carries one entry per
        ``PagedDecodeSegment``.  The mismatch corrupts M-RoPE positions
        for every segment packed after the draft — including text-only
        spec decode that happens to share the batch with an mm prefill,
        since the whole batch still routes through this method.
        Lifting the restriction is tracked as a follow-up to RFC #319.
        """
        adapter = self._multimodal_adapter
        assert adapter is not None and adapter.forward_ready
        encoder_cache = self.encoder_cache
        assert encoder_cache is not None

        for segment in decode_segments:
            if segment.num_query_tokens > 1:
                raise NotImplementedError(
                    "Speculative decode is not supported on the multimodal "
                    "paged path yet: prepare_unified() expands a decode "
                    "segment with num_query_tokens > 1 into one cu_seqlens "
                    "span per query token, but ctx.segment_positions stores "
                    "one entry per PagedDecodeSegment — cu_seqlens and "
                    "segment_positions would misalign for every segment "
                    "packed after the draft, including text-only spec decode "
                    "that shares the batch with an mm prefill.  Tracked as "
                    "a follow-up to RFC #319."
                )

        # Full-prompt M-RoPE positions per mm prefill request.
        mm_request_meta: dict[
            str, tuple[mx.array, int, list[MultiModalFeatureSpec]]
        ] = {}
        for pr in prefill_reqs:
            if not self._is_mm_request(pr.req_id):
                continue
            full_prompt = pr.full_prompt_token_ids
            if full_prompt is None:
                raise RuntimeError(
                    f"mm prefill request {pr.req_id!r} reached paged forward "
                    f"without full_prompt_token_ids; _build_prefill_pack bug."
                )
            mm_features = encoder_cache.mm_features.get(pr.req_id, [])
            sorted_features = sorted(mm_features, key=lambda f: f.mm_position.offset)
            full_positions, delta = adapter.get_mrope_input_positions(
                full_prompt, sorted_features
            )
            mm_request_meta[pr.req_id] = (full_positions, delta, sorted_features)

        total_len = int(input_ids.shape[1])
        visual_pos_masks_np = np.zeros(total_len, dtype=bool)
        mm_embeds_parts: list[mx.array] = []
        deepstack_per_layer: list[list[mx.array]] = []
        deepstack_present: bool | None = None
        ctx_segment_positions: list[Any] = []
        position_ids_parts: list[mx.array] = []
        cursor = 0

        for segment in decode_segments:
            n = segment.num_query_tokens
            state = self._request_states.get(segment.req_id)
            is_mm_decode = state is not None and state.mrope_position_delta is not None
            if is_mm_decode:
                assert state is not None and state.mrope_position_delta is not None
                offset_arr = np.arange(
                    segment.cache_start_pos,
                    segment.cache_start_pos + n,
                    dtype=np.int32,
                )
                offset_arr = offset_arr + state.mrope_position_delta
            else:
                offset_arr = np.arange(
                    segment.cache_start_pos,
                    segment.cache_start_pos + n,
                    dtype=np.int32,
                )
            seg_positions = mx.broadcast_to(
                mx.array(offset_arr)[None, None, :], (3, 1, n)
            )
            ctx_segment_positions.append(seg_positions if is_mm_decode else None)
            position_ids_parts.append(seg_positions)
            cursor += n

        for pr in prefill_reqs:
            n = len(pr.token_ids)
            if pr.req_id in mm_request_meta:
                full_positions, _delta, sorted_features = mm_request_meta[pr.req_id]
                seg_positions = full_positions[:, :, pr.start_pos : pr.start_pos + n]
                ctx_segment_positions.append(seg_positions)
                position_ids_parts.append(seg_positions)

                for feature in sorted_features:
                    f_start = feature.mm_position.offset
                    f_end = f_start + feature.mm_position.length
                    chunk_start = pr.start_pos
                    chunk_end = pr.start_pos + n
                    inter_start = max(f_start, chunk_start)
                    inter_end = min(f_end, chunk_end)
                    if inter_start >= inter_end:
                        continue  # feature doesn't overlap this chunk
                    length = inter_end - inter_start
                    chunk_local = inter_start - chunk_start
                    feature_local = inter_start - f_start

                    result = encoder_cache.encoder_outputs.get(feature.identifier)
                    if result is None:
                        raise RuntimeError(
                            f"Encoder output for feature {feature.identifier!r} "
                            f"of request {pr.req_id!r} is missing; the encoder "
                            f"gate should have populated it before prefill."
                        )

                    packed_start = cursor + chunk_local
                    visual_pos_masks_np[packed_start : packed_start + length] = True

                    mm_embeds_parts.append(
                        result.hidden_states[feature_local : feature_local + length]
                    )

                    # Deepstack: same chunk-aware slice per layer.
                    layers = result.deepstack_visual_embeds
                    this_has = layers is not None
                    if deepstack_present is None:
                        deepstack_present = this_has
                    elif deepstack_present != this_has:
                        raise RuntimeError(
                            f"Mixed deepstack presence across features for "
                            f"request {pr.req_id!r}: either all features must "
                            f"carry ``deepstack_visual_embeds`` or none.  "
                            f"Partial deepstack would leave the LM with mask "
                            f"positions that out-number the concatenated "
                            f"residual rows."
                        )
                    if not this_has:
                        continue
                    assert layers is not None
                    if not deepstack_per_layer:
                        deepstack_per_layer = [
                            [layer[feature_local : feature_local + length]]
                            for layer in layers
                        ]
                    elif len(deepstack_per_layer) != len(layers):
                        raise RuntimeError(
                            f"Inconsistent deepstack layer count across "
                            f"features for request {pr.req_id!r}: expected "
                            f"{len(deepstack_per_layer)}, got {len(layers)}."
                        )
                    else:
                        for layer_idx, layer in enumerate(layers):
                            deepstack_per_layer[layer_idx].append(
                                layer[feature_local : feature_local + length]
                            )
            else:
                # text prefill: mark segment as None so attention uses the
                # int-offset arange path.
                offset_arr = np.arange(pr.start_pos, pr.start_pos + n, dtype=np.int32)
                seg_positions = mx.broadcast_to(
                    mx.array(offset_arr)[None, None, :], (3, 1, n)
                )
                ctx_segment_positions.append(None)
                position_ids_parts.append(seg_positions)
            cursor += n

        inputs_embeds_text = adapter.embed_tokens(input_ids)
        visual_pos_masks = mx.array(visual_pos_masks_np)[None, :]
        if mm_embeds_parts:
            inputs_embeds = merge_multimodal_embeddings(
                inputs_embeds_text, mm_embeds_parts, visual_pos_masks
            )
        else:
            inputs_embeds = inputs_embeds_text

        deepstack_visual_embeds: Any | None = None
        if deepstack_per_layer:
            deepstack_visual_embeds = [
                mx.concatenate(layer_parts, axis=0)
                for layer_parts in deepstack_per_layer
            ]

        position_ids = mx.concatenate(position_ids_parts, axis=2)

        # Hand per-segment positions to ``apply_packed_rope`` via the
        # paged context, overriding the sequential-arange policy.
        ctx = get_context()
        if ctx is not None:
            ctx.segment_positions = ctx_segment_positions

        mm_prefill_deltas = {
            req_id: int(meta[1]) for req_id, meta in mm_request_meta.items()
        }

        model_output = adapter.call_lm(
            input_ids,
            inputs_embeds,
            offset_caches,
            position_ids,
            visual_pos_masks=visual_pos_masks,
            deepstack_visual_embeds=deepstack_visual_embeds,
        )
        return model_output, mm_prefill_deltas

    def _handle_new_requests(
        self,
        batch: _ExecutionBatch,
        new_reqs: list[NewRequestData],
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Register new requests and execute any required per-request prefill."""
        batch.new_reqs_by_id = {req.req_id: req for req in new_reqs}

        for new_req in new_reqs:
            req_id = new_req.req_id
            pooling_params = new_req.pooling_params
            validate_pooling_request(
                new_req,
                self.model_config,
                paged_attention_enabled=self._paged_attention_runtime is not None,
            )

            # mm_features were pre-registered before encoder dispatch in
            # ``execute_model``; no further bookkeeping needed here.
            token_ids = new_req.prompt_token_ids or []
            sampling_params = new_req.sampling_params or SamplingParams()
            lora_id = _lora_id_from_request_data(new_req)
            if new_req.lora_request is not None:
                self._lora.add_adapter(new_req.lora_request)

            if not token_ids:
                batch.add_output(req_id, [0])
                continue

            generator = _create_request_generator(self.device, sampling_params)

            if self._paged_attention_runtime is not None:
                sched_block_ids = list(new_req.block_ids[0])
                scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                computed_tokens = new_req.num_computed_tokens
                prompt_len = len(token_ids)
                cur_len = computed_tokens + scheduled_tokens
                is_intermediate = cur_len < prompt_len

                output_idx = batch.add_output(req_id, [])

                batch.paged_prefill_entries.append(
                    _PendingPrefillEntry(
                        output_idx=output_idx,
                        prefill=PrefillRequest(
                            req_id=req_id,
                            token_ids=token_ids[computed_tokens:cur_len],
                            sampling_params=sampling_params,
                            pooling_params=pooling_params,
                            block_ids=sched_block_ids,
                            generator=generator,
                            prompt_len=prompt_len if not is_intermediate else None,
                            start_pos=computed_tokens,
                            full_prompt_token_ids=None,
                            lora_id=lora_id,
                        ),
                        result_mode="intermediate" if is_intermediate else "new_final",
                    )
                )

                # Intermediate chunks need RequestState immediately so a cached
                # continuation in the next step can find the request.
                if is_intermediate:
                    self._request_states[req_id] = RequestState(
                        token_ids=list(token_ids),
                        prompt_len=prompt_len,
                        cache=[],
                        sampling_params=sampling_params,
                        pooling_params=pooling_params,
                        generator=generator,
                        generated_tokens=0,
                        block_ids=sched_block_ids,
                        lora_id=lora_id,
                    )
                continue

            next_token, cache, logprobs = self._prefill_single(
                token_ids,
                sampling_params,
                generator=generator,
            )
            batch.add_output(req_id, [next_token], logprobs)
            self._request_states[req_id] = RequestState(
                token_ids=list(token_ids) + [next_token],
                prompt_len=len(token_ids),
                cache=cache,
                sampling_params=sampling_params,
                pooling_params=None,
                generator=generator,
                generated_tokens=1,
                block_ids=[],
                lora_id=lora_id,
            )

    def _update_pp_stage_states(self, scheduler_output: SchedulerOutput) -> None:
        """Maintain ``_request_states`` on a non-last pipeline stage.

        Only the last stage samples, and that is where the normal request-state
        lifecycle is built (``_sample_paged_batch`` needs the sampled token). A
        non-last stage would otherwise have no state and emit a placeholder for
        every cached step in :meth:`_collect_cached_requests`, skipping the
        forward + ``pipeline_send`` the downstream stage is blocked on (its ring
        ``recv`` then trips the Metal command-buffer watchdog and aborts).

        Mirror upstream vLLM's ``_update_states`` (``gpu_model_runner.py``, the
        ``if not is_last_rank`` branch): seed state from the scheduler's
        ``NewRequestData`` and advance it with the sampled tokens the scheduler
        broadcasts in ``CachedRequestData.new_token_ids`` — the first stage never
        samples, it reconstructs the token stream from the scheduler. Block ids
        are left to :meth:`_update_cached_request_blocks`, which runs next and
        already guards a missing state.
        """
        for new_req in scheduler_output.scheduled_new_reqs:
            token_ids = list(new_req.prompt_token_ids or [])
            self._request_states[new_req.req_id] = RequestState(
                token_ids=token_ids,
                prompt_len=len(token_ids),
                cache=[],
                sampling_params=new_req.sampling_params or SamplingParams(),
                pooling_params=new_req.pooling_params,
                block_ids=list(new_req.block_ids[0]) if new_req.block_ids else [],
            )

        cached = scheduler_output.scheduled_cached_reqs
        if not cached.new_token_ids:
            return
        for i, req_id in enumerate(cached.req_ids):
            state = self._request_states.get(req_id)
            if state is None:
                continue
            new_token_ids = cached.new_token_ids[i]
            # Append only the tokens not already reflected in token_ids (mirrors
            # upstream's num_new_tokens; tolerates >1 sampled token per step).
            num_new = (
                cached.num_computed_tokens[i]
                + len(new_token_ids)
                - len(state.token_ids)
            )
            if num_new > 0:
                state.token_ids.extend(new_token_ids[-num_new:])
                state.generated_tokens = len(state.token_ids) - state.prompt_len

    def _update_cached_request_blocks(
        self,
        cached_reqs: CachedRequestData,
    ) -> None:
        """Apply scheduler-provided block updates for paged cached requests."""
        if self._paged_attention_runtime is None:
            return

        for i, req_id in enumerate(cached_reqs.req_ids):
            state = self._request_states.get(req_id)
            if state is None:
                continue

            new_block_ids = cached_reqs.new_block_ids[i]
            resumed = req_id in cached_reqs.resumed_req_ids
            if not resumed:
                if new_block_ids is not None:
                    state.block_ids.extend(new_block_ids[0])
                continue

            assert new_block_ids is not None
            state.block_ids = list(new_block_ids[0])
            state.generated_tokens = 0
            self._paged_request_seq_lens.pop(req_id, None)

    def _collect_cached_requests(
        self,
        batch: _ExecutionBatch,
        cached_reqs: CachedRequestData,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Classify cached requests into prefill continuation or decode work."""
        if not cached_reqs.req_ids:
            return

        if self._paged_attention_runtime is None:
            batch.scheduled_cached_req_ids.extend(cached_reqs.req_ids)
            for req_id in cached_reqs.req_ids:
                state = self._request_states.get(req_id)
                if state is not None:
                    batch.valid_decode_reqs.append((req_id, state))
            return

        for idx, req_id in enumerate(cached_reqs.req_ids):
            state = self._request_states.get(req_id)
            if state is None:
                logger.warning(
                    "Paged cached request %s has no RequestState; "
                    "emitting placeholder token. This indicates scheduler/runner "
                    "state desync.",
                    req_id,
                )
                batch.add_output(req_id, [0])
                continue

            if state.generated_tokens == 0:
                computed_tokens = cached_reqs.num_computed_tokens[idx]
                scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                target_len = computed_tokens + scheduled_tokens
                is_intermediate = target_len < len(state.token_ids)

                output_idx = batch.add_output(req_id, [])

                batch.paged_prefill_entries.append(
                    _PendingPrefillEntry(
                        output_idx=output_idx,
                        prefill=PrefillRequest(
                            req_id=req_id,
                            token_ids=state.token_ids[computed_tokens:target_len],
                            sampling_params=state.sampling_params,
                            pooling_params=state.pooling_params,
                            block_ids=state.block_ids,
                            generator=state.generator,
                            prompt_len=(
                                state.prompt_len if not is_intermediate else None
                            ),
                            start_pos=computed_tokens,
                            full_prompt_token_ids=None,
                            lora_id=state.lora_id,
                        ),
                        result_mode=(
                            "intermediate" if is_intermediate else "cached_final"
                        ),
                    )
                )
                continue

            batch.paged_decode_reqs.append((req_id, state))

    def _is_mm_request(self, req_id: str) -> bool:
        """Whether the request has multimodal features registered."""
        if self.encoder_cache is None:
            return False
        return bool(self.encoder_cache.mm_features.get(req_id))

    def _build_prefill_pack(
        self,
        batch: _ExecutionBatch,
    ) -> list[PrefillRequest]:
        """Reconstruct full prompt context for paged prefill requests.

        Continuation chunks (``start_pos > 0``) need the full prompt so
        sampling metadata reflects the whole prefix, not just this chunk.
        Multimodal requests need it at every chunk including the first —
        ``adapter.get_mrope_input_positions`` must see the whole prompt
        to compute correct M-RoPE positions for image placeholders, then
        a later commit slices the chunk-relevant range.  Both conditions
        share the same two-source resolution (RequestState first, new_req
        fallback) and the same contract-bug raises.
        """
        prefill_pack: list[PrefillRequest] = []
        for entry in batch.paged_prefill_entries:
            prefill = entry.prefill
            full_prompt = None

            needs_full_prompt = prefill.start_pos > 0 or self._is_mm_request(
                prefill.req_id
            )
            if needs_full_prompt:
                state = self._request_states.get(prefill.req_id)
                if state is not None:
                    full_prompt = state.token_ids[: state.prompt_len]
                else:
                    new_req = batch.new_reqs_by_id.get(prefill.req_id)
                    if new_req is None:
                        raise RuntimeError(
                            f"Need full prompt (start_pos={prefill.start_pos}, "
                            f"mm={self._is_mm_request(prefill.req_id)}) for "
                            f"request {prefill.req_id!r} but it has no "
                            f"RequestState and is not in new_reqs. This is a "
                            f"state tracking bug."
                        )
                    prompt_token_ids = new_req.prompt_token_ids
                    if prompt_token_ids is None:
                        raise RuntimeError(
                            f"Need full prompt (start_pos={prefill.start_pos}, "
                            f"mm={self._is_mm_request(prefill.req_id)}) for "
                            f"request {prefill.req_id!r} but prompt_token_ids "
                            f"is missing. This is a scheduler contract bug."
                        )
                    full_prompt = list(prompt_token_ids)

            prefill_pack.append(
                PrefillRequest(
                    req_id=prefill.req_id,
                    token_ids=prefill.token_ids,
                    sampling_params=prefill.sampling_params,
                    pooling_params=prefill.pooling_params,
                    block_ids=prefill.block_ids,
                    generator=prefill.generator,
                    prompt_len=prefill.prompt_len,
                    start_pos=prefill.start_pos,
                    full_prompt_token_ids=full_prompt,
                    lora_id=prefill.lora_id,
                )
            )

        return prefill_pack

    @staticmethod
    def _build_output(batch: _ExecutionBatch) -> ModelRunnerOutput:
        """Build ``ModelRunnerOutput`` from a completed batch."""
        return ModelRunnerOutput(
            req_ids=batch.req_ids,
            req_id_to_index=batch.req_id_to_index,
            sampled_token_ids=batch.sampled_tokens,
            logprobs=batch.merged_logprobs(),
            prompt_logprobs_dict={},
            pooler_output=batch.pooler_outputs,
        )

    def _run_non_paged_decode_batch(
        self,
        batch: _ExecutionBatch,
    ) -> None:
        """Run non-paged decode work and append placeholder outputs as needed."""
        if batch.valid_decode_reqs:
            if len(batch.valid_decode_reqs) >= _MIN_BATCH_SIZE_FOR_BATCHING:
                decode_result = self._batched_decode(batch.valid_decode_reqs)
            else:
                decode_result = self._sequential_decode(batch.valid_decode_reqs)

            for i, (req_id, _) in enumerate(batch.valid_decode_reqs):
                logprobs = (
                    decode_result.logprobs.slice_request(i, 1)
                    if decode_result.logprobs is not None
                    else None
                )
                batch.add_output(req_id, [decode_result.token_ids[i]], logprobs)

        for req_id in batch.scheduled_cached_req_ids:
            if req_id not in batch.req_id_to_index:
                batch.add_output(req_id, [0])

    def _validate_scheduled_outputs(
        self,
        batch: _ExecutionBatch,
        scheduler_output: SchedulerOutput,
    ) -> None:
        """Check that every scheduled request has a valid output slot."""
        if scheduler_output.total_num_scheduled_tokens <= 0:
            return

        missing_req_ids: list[str] = []
        unexpected_empty_req_ids: list[str] = []
        for req_id in scheduler_output.num_scheduled_tokens:
            output_idx = batch.req_id_to_index.get(req_id)
            if output_idx is None:
                missing_req_ids.append(req_id)
                continue

            if (
                batch.sampled_tokens[output_idx]
                or batch.pooler_outputs[output_idx] is not None
            ):
                continue

            state = self._request_states.get(req_id)
            is_intermediate_ctx = state is not None and state.generated_tokens == 0
            if not is_intermediate_ctx:
                new_req = batch.new_reqs_by_id.get(req_id)
                if new_req is not None:
                    prompt_len = len(new_req.prompt_token_ids or [])
                    computed_tokens = new_req.num_computed_tokens
                    scheduled_tokens = scheduler_output.num_scheduled_tokens[req_id]
                    is_intermediate_ctx = (
                        computed_tokens + scheduled_tokens < prompt_len
                    )

            if not is_intermediate_ctx:
                unexpected_empty_req_ids.append(req_id)

        if missing_req_ids or unexpected_empty_req_ids:
            logger.error(
                "ModelRunner scheduled/output mismatch: scheduled=%d emitted=%d "
                "missing=%d unexpected_empty=%d",
                len(scheduler_output.num_scheduled_tokens),
                len(batch.req_ids),
                len(missing_req_ids),
                len(unexpected_empty_req_ids),
            )
            if missing_req_ids:
                logger.error("Missing scheduled req ids: %s", missing_req_ids[:16])
            if unexpected_empty_req_ids:
                logger.error(
                    "Unexpected empty outputs for req ids: %s",
                    unexpected_empty_req_ids[:16],
                )

    def _cleanup_finished_requests(
        self,
        evicted_req_ids: set[str],
        *,
        materialize_runtime_state: bool = True,
    ) -> None:
        """Evict runner-owned state for finished requests."""
        runtime = self._paged_attention_runtime
        if not evicted_req_ids:
            if materialize_runtime_state and runtime is not None:
                runtime.materialize_pending_state()
            return

        for req_id in evicted_req_ids:
            state = self._request_states.pop(req_id, None)
            if state is not None:
                if state.cache:
                    del state.cache
                del state

            self._remove_request_mm_features(req_id)

            # Block freeing is handled by the scheduler's kv_cache_manager.
            self._paged_request_seq_lens.pop(req_id, None)

        if runtime is not None:
            runtime.release_requests(evicted_req_ids)
            if materialize_runtime_state:
                runtime.materialize_pending_state()

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        """Execute model forward pass and submit to GPU.

        For the paged attention path, the forward pass is submitted
        asynchronously — sampling and postprocessing are deferred to
        ``sample_tokens`` so the scheduler can run while the GPU computes.
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        self._free_encoder_outputs(scheduler_output.free_encoder_mm_hashes)
        evicted_req_ids = self._finished_req_ids(scheduler_output)
        has_scheduled_encoder_inputs = bool(scheduler_output.scheduled_encoder_inputs)
        spec_decode_error: Exception | None = None
        try:
            self._validate_spec_decode_supported(scheduler_output)
        except (NotImplementedError, ValueError) as exc:
            spec_decode_error = exc
        has_unsupported_non_paged_structured_output = (
            self._paged_attention_runtime is None
            and scheduler_output.has_structured_output_requests
        )
        will_fail_fast_before_model_work = (
            has_scheduled_encoder_inputs
            or spec_decode_error is not None
            or has_unsupported_non_paged_structured_output
        )

        # Scheduler cleanup is independent of whether this step's work is
        # supported. If the next check raises, old request state must still be
        # evicted and any pending GDN release must be materialized now.
        self._cleanup_finished_requests(
            evicted_req_ids,
            materialize_runtime_state=will_fail_fast_before_model_work,
        )
        # Pre-register mm_features so a new request whose first encoder input
        # lands in the same SchedulerOutput is already known to the encoder
        # cache when dispatch runs.
        self._pre_register_new_request_mm_features(scheduler_output.scheduled_new_reqs)
        self._reject_scheduled_encoder_inputs(scheduler_output.scheduled_encoder_inputs)
        if spec_decode_error is not None:
            raise spec_decode_error

        # Fail fast before any model work runs.  On the non-paged path,
        # _handle_new_requests immediately calls _prefill_single for new
        # requests, so the guard must come before it — not after.
        if has_unsupported_non_paged_structured_output:
            raise NotImplementedError(
                "Grammar/structured-output constraints are not supported on "
                "the non-paged (legacy) Metal path. "
                "Enable paged attention (VLLM_METAL_USE_PAGED_ATTENTION=1) "
                "to use structured output."
            )

        batch = _ExecutionBatch()
        self._handle_new_requests(
            batch, scheduler_output.scheduled_new_reqs, scheduler_output
        )

        # Non-last PP stages have no sampler, so the request-state lifecycle that
        # _sample_paged_batch builds from the sampled token never runs for them.
        # Seed/advance their state from the scheduler's broadcast instead, so the
        # forward + pipeline_send have live state to work from. No-op on the last
        # stage and the single-process path.
        if is_non_last_stage(self.pp):
            self._update_pp_stage_states(scheduler_output)

        cached_reqs = scheduler_output.scheduled_cached_reqs
        self._update_cached_request_blocks(cached_reqs)
        self._collect_cached_requests(batch, cached_reqs, scheduler_output)

        if self._paged_attention_runtime is not None and batch.has_paged_work():
            prefill_pack = self._build_prefill_pack(batch)
            self._lora.prepare_step(
                self._paged_lora_routing(batch.paged_decode_reqs, prefill_pack)
            )
            self._start_paged_forward(
                batch,
                prefill_pack,
                batch.paged_decode_reqs,
                scheduler_output,
            )
            if self._is_pooling:
                batch, scheduler_output = self._sample_paged_batch(None)
                runtime = self._paged_attention_runtime
                if runtime is not None:
                    runtime.materialize_pending_state()
                self._validate_scheduled_outputs(batch, scheduler_output)
                return self._build_output(batch)
            return None

        # Defensive invariant: the vLLM scheduler sets has_structured_output_requests
        # only when at least one SO request is present in the *current* scheduled
        # batch (not the global queue). Any such request on the paged path must
        # contribute a paged decode or prefill entry, so has_paged_work() must be
        # True. If this fires, a scheduler change broke that contract and the
        # bitmask would have been silently skipped on the synchronous tail.
        if (
            self._paged_attention_runtime is not None
            and scheduler_output.has_structured_output_requests
        ):
            raise RuntimeError(
                "Structured-output request present but no paged work was scheduled — "
                "invariant violated."
            )

        if self._paged_attention_runtime is None:
            self._run_non_paged_decode_batch(batch)

        # Non-paged path: complete synchronously
        runtime = self._paged_attention_runtime
        if runtime is not None:
            runtime.materialize_pending_state()
        self._validate_scheduled_outputs(batch, scheduler_output)
        if not batch.req_ids:
            return self._build_output(batch)
        output = self._build_output(batch)
        if self._is_pooling:
            return output
        self._pending_output = output
        return None

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | None:
        """Wait for GPU forward, sample tokens, and postprocess.

        Called by the vLLM v1 engine after ``execute_model`` returns ``None``.
        For the paged path, this is where the actual GPU synchronization,
        token sampling, and request state updates happen — allowing the
        scheduler to run while the GPU was computing the forward pass.
        """
        # Paged path: wait for MLX forward, apply grammar bitmask, sample tokens.
        if self._execute_model_state is not None:
            # Pipeline parallelism: only the last stage holds logits and samples.
            # Non-last stages produced no logits (they piped their hidden state
            # downstream), so clear the stash and return an empty output — the
            # engine collects results from the last stage only.
            if is_non_last_stage(self.pp):
                self._execute_model_state = None
                runtime = self._paged_attention_runtime
                if runtime is not None:
                    runtime.materialize_pending_state()
                return EMPTY_MODEL_RUNNER_OUTPUT
            batch, scheduler_output = self._sample_paged_batch(grammar_output)
            runtime = self._paged_attention_runtime
            if runtime is not None:
                runtime.materialize_pending_state()
            self._validate_scheduled_outputs(batch, scheduler_output)
            return self._build_output(batch)

        # Non-paged path: return output built by execute_model
        if self._pending_output is not None:
            output = self._pending_output
            self._pending_output = None
            return output

        # Async scheduling: execute_model may have failed; return None so
        # vLLM can surface the original exception.
        logger.error(
            "sample_tokens called with no pending state — "
            "neither _execute_model_state nor _pending_output was set."
        )
        return None

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
    ) -> str:
        """Generate text from a prompt.

        This is a simplified interface for direct text generation.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            raise RuntimeError("Model and tokenizer must be loaded")

        segments: list[str] = []

        # Create sampler based on temperature (mlx_lm 0.29+ uses sampler param)
        def sampler(logits: mx.array) -> mx.array:
            if temperature < GREEDY_TEMPERATURE_EPS:
                return mx.argmax(logits, axis=-1)
            return mx.random.categorical(logits / temperature)

        for response in stream_generate(
            self._forward_model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            segments.append(response.text)

        return "".join(segments)
