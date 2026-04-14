# SPDX-License-Identifier: Apache-2.0
"""
Metal vLLM v1 model runner.

Orchestration only: coordinates scheduling, dispatch, and output assembly.
Model-specific behavior belongs in adapters; backend-specific kernels live in
backend modules. Keep this file thin and stable.

Key contracts:
- execute_model()/sample_tokens() handoff remains unchanged.
- Outputs align with scheduler expectations for paged and non-paged paths.
- Prefix-cache hits reconstruct full prompts for sampling metadata.
"""

import time
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal, NamedTuple, TypeAlias

import mlx.core as mx
import torch
from mlx_lm import load as mlx_lm_load
from mlx_lm import stream_generate

# mlx_vlm for vision-language models
from mlx_vlm import load as mlx_vlm_load
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.tasks import SupportedTask
from vllm.utils.platform_utils import is_pin_memory_available
from vllm.v1.core.sched.output import (
    CachedRequestData,
    GrammarOutput,
    NewRequestData,
    SchedulerOutput,
)
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.logits_processor import build_logitsprocs
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_metal.config import get_config
from vllm_metal.paged_attention_backend.hybrid import (
    HybridPagedAttentionBackend,
    _build_linear_layer_spec,
)
from vllm_metal.paged_attention_backend.mla import MLA_DEFAULT_QK_ROPE_HEAD_DIM
from vllm_metal.paged_attention_backend.protocol import PagedAttentionBackend
from vllm_metal.paged_attention_common import (
    OffsetCache,
    clear_context,
    prepare_unified,
)
from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx
from vllm_metal.stt.detection import is_stt_model
from vllm_metal.stt.policy import STT_SCHED_BLOCK_BYTES
from vllm_metal.stt.runtime import STTRuntimeAdapter
from vllm_metal.stt.serve import VLLMSTTRequestAdapter
from vllm_metal.utils import get_model_download_path
from vllm_metal.v1 import contiguous_cache
from vllm_metal.v1.contiguous_cache import (
    _MIN_BATCH_SIZE_FOR_BATCHING,
    _PREFIX_CACHE_ENABLED,
    AnyCache,
    KVCache,
    PrefixCacheManager,
    _extract_kv_cache,
    _merge_kv_caches,
)
from vllm_metal.v1.model_adapter import DefaultModelAdapter, ModelAdapter
from vllm_metal.v1.sampling_batch import (
    GREEDY_TEMPERATURE_EPS,
    SamplingBatch,
    sample_decode_tokens,
    sample_from_logits,
    sample_prefill_tokens,
)

logger = init_logger(__name__)

# Global model cache for fast repeated loads
_model_cache: dict[str, tuple[Any, Any]] = {}  # model_name -> (model, tokenizer)
_model_cache_lock = Lock()


# Performance tuning
_CACHE_CLEAR_INTERVAL = 50  # Clear cache every N finished requests

SchedulerMemoryReportingMode: TypeAlias = Literal[
    "stt_nominal",
    "paged_attention_capacity",
    "single_sequence_estimate",
]


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
    generator: torch.Generator | None = None
    generated_tokens: int = 0
    block_ids: list[int] = field(
        default_factory=list
    )  # Scheduler-assigned paged KV blocks


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
    new_reqs_by_id: dict[str, NewRequestData] = field(default_factory=dict)
    paged_prefill_entries: list[_PendingPrefillEntry] = field(default_factory=list)
    paged_decode_reqs: list[tuple[str, RequestState]] = field(default_factory=list)
    scheduled_cached_req_ids: list[str] = field(default_factory=list)
    valid_decode_reqs: list[tuple[str, RequestState]] = field(default_factory=list)

    def add_output(self, req_id: str, token_ids: list[int]) -> int:
        """Append one output slot and return its index."""
        self.req_ids.append(req_id)
        output_idx = len(self.req_ids) - 1
        self.req_id_to_index[req_id] = output_idx
        self.sampled_tokens.append(token_ids)
        return output_idx

    def has_paged_work(self) -> bool:
        """Return whether this step has any paged execution work."""
        return bool(self.paged_prefill_entries or self.paged_decode_reqs)


class _PagedForwardState(NamedTuple):
    """State stashed by ``_start_paged_forward`` for ``_sample_paged_batch``."""

    batch: _ExecutionBatch
    prefill_reqs: list[PrefillRequest]
    decode_reqs: list[tuple[str, RequestState]]
    scheduler_output: SchedulerOutput
    logits: mx.array
    cu_seqlens: list[int]
    num_decode: int


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

        self.model: Any = None
        self.tokenizer: Any = None
        self.model_args: dict[str, Any] = {}
        self._is_vlm: bool = False  # Will be set during model loading
        self._is_stt: bool = False  # Will be set during model loading
        self._stt_runtime_adapter: STTRuntimeAdapter | None = (
            None  # Set during STT loading
        )

        # Request state cache for incremental decoding
        self._request_states: dict[str, RequestState] = {}

        # GDN slot allocator: stable request_id → slot mapping for hybrid
        # models so recurrent state survives request reordering/preemption.
        self._gdn_req_to_slot: dict[str, int] = {}
        self._gdn_free_slots: list[int] = []
        self._gdn_needs_materialize = False

        # vLLM Sampler for token sampling with temperature, top_k, top_p support
        self._sampler = Sampler()

        # Build logits processors (includes custom plugins from entry-points)
        is_pooling_model = getattr(self.model_config, "runner_type", None) == "pooling"
        pin_memory = is_pin_memory_available()
        custom_lp = vllm_config.model_config.logits_processors
        custom_logitsprocs = tuple(custom_lp) if custom_lp is not None else ()
        self._logitsprocs = build_logitsprocs(
            vllm_config,
            device,
            pin_memory,
            is_pooling_model,
            custom_logitsprocs,
        )

        # Track finished requests for lazy cache clearing
        self._finished_request_count = 0

        # vLLM v1 async scheduling calls sample_tokens after execute_model.
        # Keep the latest execution output so sample_tokens can return it.
        self._pending_output: ModelRunnerOutput | None = None

        # Prefix cache for shared prompt reuse
        self._prefix_cache: PrefixCacheManager | None = None
        if _PREFIX_CACHE_ENABLED:
            self._prefix_cache = PrefixCacheManager(model_adapter=self._model_adapter)

        # Paged attention state (set by worker when enabled)
        self._paged_attention_backend: PagedAttentionBackend | None = None
        self._paged_block_size: int = 0
        self._paged_request_seq_lens: dict[str, int] = {}  # req_id → seq_len
        self.kv_cache_dtype: mx.Dtype | None = None

        # Async forward state: stashed by execute_model, consumed by
        # sample_tokens (mirrors upstream's execute_model_state pattern).
        self._execute_model_state: _PagedForwardState | None = None

    @property
    def is_stt(self) -> bool:
        """Whether the loaded model is a Speech-to-Text model."""
        return self._is_stt

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

        NOTE: This means multimodal (image) inputs are not supported — the
        vision head is intentionally skipped.  Full multimodal inference would
        require a decomposed encode → feature-fusion → forward pass, following
        the upstream pattern, and is a separate future effort.
        """
        if self._is_vlm:
            return self._model_adapter.text_model(self.model)
        return self.model

    @property
    def mla_latent_dim(self) -> int:
        """Combined latent dimension for MLA cache: kv_lora_rank + qk_rope_head_dim.

        Only valid when is_mla is True. Derived directly from model_args so
        callers do not depend on the _resolve_model_dims head_dim override.
        """
        if not self.is_mla:
            raise AttributeError("mla_latent_dim is only valid for MLA models")
        return int(self.model_args["kv_lora_rank"]) + int(
            self.model_args.get("qk_rope_head_dim", MLA_DEFAULT_QK_ROPE_HEAD_DIM)
        )

    def should_setup_paged_attention(self) -> bool:
        """Whether worker-side paged-attention setup should run.

        STT models own their runtime path and do not use the paged-attention
        cache path that the text/VLM runner uses.
        """
        return not self._is_stt

    def validate_paged_attention_support(self) -> None:
        """Validate that the loaded model can run on the paged-attention path."""
        self._model_adapter.require_uniform_kv_heads(self.model_args, self.num_kv_heads)

    def scheduler_memory_reporting_mode(
        self, *, paged_attention_enabled: bool
    ) -> SchedulerMemoryReportingMode:
        """Return which scheduler memory-reporting mode worker should use.

        Worker delegates this decision to the runner so STT-specific policy is
        not open-coded in `worker.py`.
        """
        if self._is_stt:
            return "stt_nominal"
        if paged_attention_enabled and self._paged_attention_backend is not None:
            return "paged_attention_capacity"
        return "single_sequence_estimate"

    def supported_worker_tasks(self) -> tuple[SupportedTask, ...]:
        """Return worker task capabilities for the loaded model."""
        if self._is_stt:
            return ("transcription",)
        return ("generate",)

    def _is_vlm_model(self) -> bool:
        """Check if the model is a vision-language model (VLM).

        Returns:
            True if the model is multimodal/VLM, False otherwise
        """
        hf_config = getattr(self.model_config, "hf_config", None)
        if hf_config is not None and self._model_adapter.should_force_text_backbone(
            hf_config
        ):
            return False
        if hasattr(self.model_config, "is_multimodal_model"):
            return self.model_config.is_multimodal_model
        return False

    def load_model(self) -> None:
        """Load the model using MLX with caching for fast repeated loads.

        Uses mlx_vlm for vision-language models and mlx_lm for text-only models.
        """
        model_name = get_model_download_path(self.model_config.model)

        # STT models use their own loading path — skip VLM/LM logic entirely.
        if is_stt_model(model_name):
            self._load_stt_model(model_name)
            return

        is_vlm = self._is_vlm_model()

        logger.info(f"Loading model: {model_name} (VLM: {is_vlm})")
        start_time = time.time()

        # Check global cache first for fast repeated loads
        with _model_cache_lock:
            if model_name in _model_cache:
                self.model, self.tokenizer = _model_cache[model_name]
                self._is_vlm = is_vlm
                load_time = time.time() - start_time
                logger.info(
                    f"Model loaded from cache in {load_time:.3f}s: {model_name}"
                )
                self._extract_model_args()
                self._resolve_model_dims()
                self._initialize_kv_cache_dtype()
                return

        # Load model using appropriate backend
        if is_vlm:
            logger.info("Using mlx-vlm for vision-language model")
            # NOTE: Only text-only (language-model) inference is supported.
            # Image inputs are not processed — the vision encoder is bypassed
            # by routing all forward passes through model.language_model.
            # Full multimodal inference (encode → fuse → forward) would follow
            # the upstream decomposed encode/forward pattern and is a separate
            # future effort.
            logger.warning(
                "VLM loaded in text-only mode: multimodal (image) inputs are "
                "not yet supported. Vision encoder will be bypassed."
            )
            self.model, self.tokenizer = mlx_vlm_load(model_name)
            self._is_vlm = True
        else:
            # Load model and tokenizer using mlx_lm for text-only models
            self.model, self.tokenizer = mlx_lm_load(
                model_name,
                tokenizer_config={
                    "trust_remote_code": self.model_config.trust_remote_code
                },
            )
            self._is_vlm = False

        # Cache for future loads
        with _model_cache_lock:
            _model_cache[model_name] = (self.model, self.tokenizer)

        self._extract_model_args()
        self._resolve_model_dims()
        self._initialize_kv_cache_dtype()
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s: {model_name}")

    def _load_stt_model(self, model_name: str) -> None:
        """Load a Speech-to-Text model (e.g. Whisper) with caching."""
        start_time = time.time()

        with _model_cache_lock:
            if model_name in _model_cache:
                self.model, _ = _model_cache[model_name]
                load_time = time.time() - start_time
                logger.info(
                    f"STT model loaded from cache in {load_time:.3f}s: {model_name}"
                )
                self.tokenizer = None  # Whisper manages its own tokenizer
                self._is_stt = True
                self._stt_runtime_adapter = self.model.create_runtime_adapter(
                    model_name
                )
                return

        # Local import: keep non-STT startup/import path light.
        from vllm_metal.stt.loader import load_model as stt_load_model

        logger.info(f"Loading STT model: {model_name}")
        self.model = stt_load_model(model_name)
        self.tokenizer = None  # Whisper manages its own tokenizer
        self._is_stt = True
        self._stt_runtime_adapter = self.model.create_runtime_adapter(model_name)

        with _model_cache_lock:
            _model_cache[model_name] = (self.model, None)

        load_time = time.time() - start_time
        logger.info(f"STT model loaded in {load_time:.2f}s: {model_name}")

    def _initialize_kv_cache_dtype(self) -> None:
        """Resolve the KV cache element dtype from model_config.dtype.

        model_config.dtype is the authoritative compute dtype, set from
        config.json torch_dtype at engine startup — the same source upstream
        vLLM uses for kv_cache_dtype. Quantization changes weight storage
        format but not compute precision, so this is correct for all model
        families (dense, MoE, MLA) and quantisation levels.

        torch_to_mlx on a zero-element probe tensor maps the torch.dtype to
        its MLX equivalent without allocating memory.
        """
        self.kv_cache_dtype = torch_to_mlx(
            torch.empty(0, dtype=self.model_config.dtype)
        ).dtype

    def _extract_model_args(self) -> None:
        """Extract model configuration from loaded model.

        Handles both text-only models and VLMs (which have nested text_config).
        """
        if hasattr(self.model, "args"):
            # mlx-lm models (Qwen, Llama, etc.)
            self.model_args = vars(self.model.args)
        elif hasattr(self.model, "config"):
            config = self.model.config
            if self._is_vlm and hasattr(config, "text_config"):
                # VLMs with nested text config (LLaVA, Pixtral via mlx-vlm)
                text_config = config.text_config
                if hasattr(text_config, "to_dict"):
                    self.model_args = text_config.to_dict()
                else:
                    self.model_args = {
                        k: getattr(text_config, k)
                        for k in dir(text_config)
                        if not k.startswith("_")
                        and not callable(getattr(text_config, k))
                    }
            elif hasattr(config, "to_dict"):
                # Standard HuggingFace config objects
                self.model_args = config.to_dict()
            else:
                self.model_args = vars(config)
        else:
            raise ValueError(
                "Cannot extract model config: model has neither .args nor "
                ".config attribute."
            )
        # Merge nested text_config (Qwen3.5, VLMs) into top-level args.
        # setdefault preserves any top-level keys that already exist.
        tc = self.model_args.get("text_config")
        if tc is not None:
            if isinstance(tc, dict):
                tc_dict = tc
            elif hasattr(tc, "to_dict"):
                tc_dict = tc.to_dict()
            else:
                tc_dict = vars(tc) if hasattr(tc, "__dict__") else {}
            for k, v in tc_dict.items():
                self.model_args.setdefault(k, v)

        self._vocab_size: int = self.model_args["vocab_size"]

        if self.metal_config.debug:
            logger.info(f"Model args: {self.model_args}")

    def _resolve_model_dims(self) -> None:
        """Extract and validate model dimensions from ``self.model_args``.

        Must be called after ``_extract_model_args()``.  Stores validated
        dimensions as instance attributes so that every consumer reads from
        one canonical source instead of repeating fallback chains.

        Raises:
            ValueError: If any critical dimension cannot be determined.
        """
        args = self.model_args

        num_layers = args.get("num_hidden_layers") or args.get("n_layers")
        num_attention_heads = args.get("num_attention_heads")
        num_kv_heads = (
            args.get("num_key_value_heads")
            or args.get("n_kv_heads")
            or num_attention_heads
        )
        hidden_size = args.get("hidden_size")
        head_dim = args.get("head_dim") or (
            hidden_size // num_attention_heads
            if hidden_size and num_attention_heads
            else None
        )
        head_dim = self._model_adapter.resolve_max_head_dim(args, head_dim)

        # Fail fast if critical dims are missing
        missing = []
        if not num_layers:
            missing.append("num_layers (num_hidden_layers / n_layers)")
        if not num_kv_heads:
            missing.append("num_kv_heads (num_key_value_heads / n_kv_heads)")
        if not head_dim:
            missing.append("head_dim")
        if missing:
            raise ValueError(
                f"Cannot resolve model dimensions: {', '.join(missing)}. "
                f"Available keys: {sorted(args.keys())}"
            )

        self.num_layers: int = int(num_layers)
        self.num_attention_heads = num_attention_heads
        self.num_kv_heads: int = int(num_kv_heads)
        self.hidden_size = hidden_size
        self.head_dim: int = int(head_dim)

        # MLA (GLM/DeepSeek lineage): cache stores a joint latent vector per
        # layer, not per-head K/V. Use one virtual head sized kv_lora_rank +
        # qk_rope_head_dim so shared sizing paths can reuse head_dim/num_kv_heads
        # while get_cache_block_size_bytes() applies an MLA-specific factor.
        if self.is_mla:
            self.num_kv_heads = 1
            self.head_dim = int(args["kv_lora_rank"]) + int(
                args.get("qk_rope_head_dim", MLA_DEFAULT_QK_ROPE_HEAD_DIM)
            )

        # Hybrid (Qwen3.5): mix of SDPA and GDN linear attention layers.
        # Store per-type layer counts and GDN dimensions for cache allocation.
        if self.is_hybrid:
            fai = int(args["full_attention_interval"])
            self.full_attention_interval: int = fai
            self.sdpa_layer_indices: frozenset[int] = frozenset(
                i for i in range(self.num_layers) if (i + 1) % fai == 0
            )
            self.num_sdpa_layers = len(self.sdpa_layer_indices)
            self.num_linear_layers = self.num_layers - self.num_sdpa_layers
            self.linear_num_k_heads: int = int(args["linear_num_key_heads"])
            self.linear_num_v_heads: int = int(args["linear_num_value_heads"])
            self.linear_key_head_dim: int = int(args["linear_key_head_dim"])
            self.linear_value_head_dim: int = int(args["linear_value_head_dim"])
            self.linear_conv_kernel_dim: int = int(args["linear_conv_kernel_dim"])
            # Derived: total conv1d channel width (key_dim*2 + value_dim)
            self.linear_conv_dim: int = (
                self.linear_num_k_heads * self.linear_key_head_dim * 2
                + self.linear_num_v_heads * self.linear_value_head_dim
            )

    def _gdn_alloc_slot(self, req_id: str) -> int:
        """Allocate a stable GDN state pool slot for a request."""
        if req_id in self._gdn_req_to_slot:
            return self._gdn_req_to_slot[req_id]
        reused = False
        if self._gdn_free_slots:
            slot = self._gdn_free_slots.pop()
            reused = True
        else:
            slot = len(self._gdn_req_to_slot)
        self._gdn_req_to_slot[req_id] = slot
        # Zero state for reused slots so the new request starts clean.
        # Done at alloc time (inside the forward-pass graph) rather than
        # at free time to avoid mx.eval synchronisation issues.
        if reused:
            backend = self._paged_attention_backend
            if not isinstance(backend, HybridPagedAttentionBackend):
                raise RuntimeError("GDN slot allocation requires hybrid paged backend")
            sc = backend._state_cache
            if sc is None:
                raise RuntimeError("GDN state cache is not initialized")
            for layer_idx in range(sc.num_layers):
                conv = sc.conv_states[layer_idx]
                conv[slot] = mx.zeros_like(conv[slot])
                sc.conv_states[layer_idx] = conv
                rec = sc.recurrent_states[layer_idx]
                rec[slot] = mx.zeros_like(rec[slot])
                sc.recurrent_states[layer_idx] = rec
        return slot

    def _gdn_materialize_state_cache(self) -> None:
        """Detach GDN state arrays from the lazy graph to prevent growth."""
        backend = self._paged_attention_backend
        if not isinstance(backend, HybridPagedAttentionBackend):
            raise RuntimeError("GDN state cache requires hybrid paged backend")
        sc = backend._state_cache
        if sc is None:
            raise RuntimeError("GDN state cache is not initialized")
        mx.eval(*sc.conv_states, *sc.recurrent_states)

    def _gdn_release_slots(self, req_ids: set[str]) -> None:
        """Release finished GDN slots and defer state materialization."""
        freed_slots: list[int] = []
        for req_id in req_ids:
            slot = self._gdn_req_to_slot.pop(req_id, None)
            if slot is not None:
                freed_slots.append(slot)

        if not freed_slots:
            return

        self._gdn_needs_materialize = True
        self._gdn_free_slots.extend(freed_slots)

    def _gdn_materialize_pending_state_cache(self) -> None:
        """Materialize GDN state after slot recycling if the step needs it."""
        if not self._gdn_needs_materialize:
            return
        self._gdn_materialize_state_cache()
        self._gdn_needs_materialize = False

    def _extract_logits(self, model_output: Any) -> mx.array:
        """Extract logits from model output.

        Handles both mlx-lm (returns array directly) and mlx-vlm
        (returns LanguageModelOutput with .logits attribute).

        Args:
            model_output: Output from model forward pass

        Returns:
            Logits array
        """
        if hasattr(model_output, "logits"):
            # mlx-vlm returns LanguageModelOutput
            return model_output.logits
        # mlx-lm returns logits directly
        return model_output

    def get_kv_cache_spec(self) -> dict[str, KVCacheSpec]:
        """Get KV cache specification.

        Returns:
            Dictionary mapping attention layer names to KV cache specs
        """
        if self._is_stt:
            # STT models manage their own KV cache internally.
            # vLLM requires a non-empty spec for scheduler initialization,
            # so we return a single minimal entry.
            return {
                "layers.0.self_attn": FullAttentionSpec(
                    block_size=self.metal_config.block_size,
                    num_kv_heads=1,
                    head_size=64,
                    dtype=torch.float16,
                ),
            }

        # Use cache_config.block_size (not metal_config.block_size) because
        # vLLM's hybrid alignment may have adjusted it to unify page sizes
        # across SDPA and Mamba/GDN layers.
        block_size = self.cache_config.block_size
        if self.kv_cache_dtype is None:
            raise RuntimeError("KV cache dtype not initialized; load_model() first")

        # FullAttentionSpec (upstream vLLM) expects torch.dtype
        from vllm_metal.pytorch_backend.tensor_bridge import MLX_TO_TORCH_DTYPE

        torch_dtype = MLX_TO_TORCH_DTYPE[self.kv_cache_dtype]

        specs: dict[str, KVCacheSpec] = {}
        for layer_idx in range(self.num_layers):
            if self.is_hybrid and layer_idx not in self.sdpa_layer_indices:
                layer_name = f"layers.{layer_idx}.linear_attn"
                specs[layer_name] = _build_linear_layer_spec(
                    conv_kernel_dim=self.linear_conv_kernel_dim,
                    conv_dim=self.linear_conv_dim,
                    num_v_heads=self.linear_num_v_heads,
                    value_head_dim=self.linear_value_head_dim,
                    key_head_dim=self.linear_key_head_dim,
                    torch_dtype=torch_dtype,
                    page_size_padded=self.cache_config.mamba_page_size_padded,
                    block_size=block_size,
                )
            else:
                layer_name = f"layers.{layer_idx}.self_attn"
                specs[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=self.num_kv_heads,
                    head_size=self.head_dim,
                    dtype=torch_dtype,
                )

        return specs

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Accept KV cache config from engine (no-op for MLX path).

        MLX manages its own KV cache via make_prompt_cache().
        This method exists to satisfy the engine's initialization protocol.
        """
        logger.info(
            "KV cache config received: %d blocks (MLX manages cache internally)",
            kv_cache_config.num_blocks,
        )

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of a single cache block in bytes.

        Returns:
            Block size in bytes
        """
        if self._is_stt:
            return STT_SCHED_BLOCK_BYTES

        # Use cache_config.block_size (not metal_config) because vLLM's
        # hybrid alignment may have adjusted it to match mamba page size.
        block_size = self.cache_config.block_size

        # Each block stores key and value for SDPA layers only.
        # Hybrid models (Qwen3.5) have linear attention layers that use
        # fixed-size recurrent state, not paged KV — exclude them.
        if self.kv_cache_dtype is None:
            raise RuntimeError("KV cache dtype not initialized; load_model() first")
        dtype_size = self.kv_cache_dtype.size
        num_kv_layers = self.num_sdpa_layers if self.is_hybrid else self.num_layers
        kv_factor = 1 if self.is_mla else 2
        return (
            kv_factor
            * num_kv_layers
            * block_size
            * self.num_kv_heads
            * self.head_dim
            * dtype_size
        )

    def linear_cache_bytes_per_slot(self) -> int:
        """Bytes for one request's linear attention state across all GDN layers."""
        if not self.is_hybrid:
            raise RuntimeError("linear_cache_bytes_per_slot() requires a hybrid model")
        if self.kv_cache_dtype is None:
            raise RuntimeError("KV cache dtype not initialized; load_model() first")
        dtype_size = self.kv_cache_dtype.size
        # GDN recurrent state is always float32 (see GDNPagedStateCache).
        recurrent_dtype_size = mx.float32.size
        conv_bytes = (
            (self.linear_conv_kernel_dim - 1) * self.linear_conv_dim * dtype_size
        )
        recurrent_bytes = (
            self.linear_num_v_heads
            * self.linear_value_head_dim
            * self.linear_key_head_dim
            * recurrent_dtype_size
        )
        return self.num_linear_layers * (conv_bytes + recurrent_bytes)

    def warm_up(self) -> None:
        """Warm up the model with a dummy forward pass.

        When paged attention is enabled, also loads the HF Metal kernel and
        runs a tiny ``reshape_and_cache`` to force Metal library creation.
        This catches Metal language-version incompatibilities at startup
        rather than during the first real inference request.
        """
        if self.model is None:
            logger.warning("Model not loaded, skipping warm-up")
            return

        if self._is_stt:
            assert self._stt_runtime_adapter is not None
            logger.info("Warming up STT model...")
            self._stt_runtime_adapter.warm_up()
            logger.info("STT model warm-up complete")
            return

        logger.info("Warming up model...")

        # Run a small dummy inference (standard MLX path)
        try:
            dummy_tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
            output = self._forward_model(dummy_tokens)
            logits = self._extract_logits(output)
            mx.eval(logits)
            logger.info("Model warm-up complete")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

        if self._paged_attention_backend is not None:
            self._paged_attention_backend.warm_up()

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
        req_id: str,
        token_ids: list[int],
        sampling_params: SamplingParams,
        generator: torch.Generator | None = None,
    ) -> tuple[int, list[KVCache]]:
        """Process a single prefill request.

        Args:
            req_id: Request ID
            token_ids: Prompt token IDs
            sampling_params: Sampling parameters for this request

        Returns:
            Tuple of (next_token, cache)
        """
        cache: list[KVCache]
        cached_prefix_len = 0

        # Prefix caching: cache KV for tokens[:-1], always process last token
        prefix = token_ids[:-1] if len(token_ids) > 1 else []

        # Create cache to check if model supports prefix caching
        cache = contiguous_cache.make_prompt_cache(self._forward_model)
        # Prefix caching only safe for pure KVCache models (not Mamba/hybrid)
        supports_prefix_cache = all(isinstance(c, KVCache) for c in cache)

        # Try to reuse cached prefix
        if supports_prefix_cache and self._prefix_cache is not None and len(prefix) > 0:
            cached = self._prefix_cache.lookup(prefix)
            if cached is not None:
                # Cache hit: restore KV for prefix, process only last token
                cache = self._prefix_cache.restore_cache(
                    cached, self.model, self._is_vlm
                )
                cached_prefix_len = len(cached.token_ids)
            else:
                # Cache miss: process prefix first, cache it, then last token
                prefix_ids = mx.array([prefix], dtype=mx.int32)
                _ = self._forward_model(prefix_ids, cache=cache)
                self._prefix_cache.insert(prefix, cache)
                cached_prefix_len = len(prefix)

        # Prefill: process remaining tokens (always at least the last token)
        tokens_to_process = token_ids[cached_prefix_len:]
        input_ids = mx.array([tokens_to_process], dtype=mx.int32)
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
        [next_token] = sample_from_logits(
            last_logits, batch, self._sampler, self.device
        )
        mx.eval(*[c.state for c in cache])

        return next_token, cache

    def _batched_decode(self, decode_reqs: list[tuple[str, RequestState]]) -> list[int]:
        """Process multiple decode requests in a single batched forward pass.

        Uses BatchKVCache to merge individual caches, run ONE forward pass,
        then extract updated caches back.

        Args:
            decode_reqs: List of (req_id, state) tuples

        Returns:
            List of next tokens for each request
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
        next_tokens = sample_from_logits(
            next_token_logits, batch, self._sampler, self.device
        )

        # Extract updated caches back to individual requests
        for i, (_req_id, state) in enumerate(decode_reqs):
            state.cache = _extract_kv_cache(batch_cache, i)
            state.token_ids.append(next_tokens[i])
            state.generated_tokens += 1

        return next_tokens

    def _sequential_decode(
        self, decode_reqs: list[tuple[str, RequestState]]
    ) -> list[int]:
        """Fallback: process decode requests sequentially.

        Used when batch size is 1 (no benefit from batching).

        Args:
            decode_reqs: List of (req_id, state) tuples

        Returns:
            List of next tokens for each request
        """
        next_tokens = []

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
            [next_token] = sample_from_logits(
                last_logits, batch, self._sampler, self.device
            )

            next_tokens.append(next_token)

            # Update state
            state.token_ids.append(next_token)
            state.generated_tokens += 1

        return next_tokens

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
        num_decode = len(decode_reqs)

        # ---- build unified token sequence: decode first, then prefill ----
        all_token_ids: list[int] = []

        # Decode: last token per request
        last_tokens = [
            state.token_ids[-1] if state.token_ids else 0 for _, state in decode_reqs
        ]
        all_token_ids.extend(last_tokens)

        # Prefill: tokens per request
        for pr in prefill_reqs:
            all_token_ids.extend(pr.token_ids)

        # ---- build metadata for prepare_unified ----
        decode_info: list[tuple[list[int], int]] = []
        for req_id, state in decode_reqs:
            seq_len = self._paged_request_seq_lens.get(req_id, len(state.token_ids) - 1)
            decode_info.append((state.block_ids, seq_len))

        prefill_info: list[tuple[list[int], int, int]] = []
        for pr in prefill_reqs:
            prefill_info.append((pr.block_ids, len(pr.token_ids), pr.start_pos))

        prepare_unified(decode_info, prefill_info, self._paged_block_size)

        # ---- GDN slot mapping (hybrid models) ----
        if self.is_hybrid:
            from vllm_metal.paged_attention_common import get_context

            ctx = get_context()
            if ctx is not None:
                gdn_slots = []
                # Decode requests come first, then prefill
                for req_id, _ in decode_reqs:
                    gdn_slots.append(self._gdn_alloc_slot(req_id))
                for pr in prefill_reqs:
                    gdn_slots.append(self._gdn_alloc_slot(pr.req_id))
                ctx.gdn_slot_mapping = gdn_slots

        # ---- forward (lazy graph + async submit) ----
        offset_caches = [OffsetCache(0) for _ in range(self.num_layers)]
        input_ids = mx.array([all_token_ids], dtype=mx.int32)
        try:
            model_output = self._forward_model(input_ids, cache=offset_caches)
            logits = self._extract_logits(model_output)
            # MLX uses lazy evaluation — model_output holds the entire
            # computation graph.  Dropping it before mx.eval lets MLX
            # free intermediate buffers (per-layer Q/K/V, MLP outputs)
            # as the graph evaluates, rather than pinning them all.
            del model_output
        finally:
            clear_context()

        # Submit to GPU — returns immediately, GPU runs in background
        mx.async_eval(logits)

        # ---- build cu_seqlens for logit extraction ----
        cu_seqlens: list[int] = [0]
        for _ in decode_reqs:
            cu_seqlens.append(cu_seqlens[-1] + 1)
        for pr in prefill_reqs:
            cu_seqlens.append(cu_seqlens[-1] + len(pr.token_ids))

        self._execute_model_state = _PagedForwardState(
            batch=batch,
            prefill_reqs=prefill_reqs,
            decode_reqs=decode_reqs,
            scheduler_output=scheduler_output,
            logits=logits,
            cu_seqlens=cu_seqlens,
            num_decode=num_decode,
        )

    def _sample_paged_batch(self) -> tuple[_ExecutionBatch, SchedulerOutput]:
        """Eval logits, sample tokens, and postprocess paged batch.

        Consumes state stashed by ``_start_paged_forward``.
        Returns ``(batch, scheduler_output)`` for the caller to finalize.
        """
        state = self._execute_model_state
        assert state is not None
        self._execute_model_state = None
        batch = state.batch
        prefill_reqs = state.prefill_reqs
        decode_reqs = state.decode_reqs
        scheduler_output = state.scheduler_output
        logits = state.logits
        cu_seqlens = state.cu_seqlens
        num_decode = state.num_decode

        # ---- wait for GPU forward to complete ----
        mx.eval(logits)

        # ---- sample tokens ----
        vocab_size = self._vocab_size
        logitsprocs = self._logitsprocs
        decode_next_tokens = sample_decode_tokens(
            logits,
            decode_reqs,
            num_decode,
            self._sampler,
            self.device,
            vocab_size=vocab_size,
            logitsprocs=logitsprocs,
        )
        prefill_next_tokens = sample_prefill_tokens(
            logits,
            prefill_reqs,
            cu_seqlens,
            num_decode,
            self._sampler,
            self.device,
            vocab_size=vocab_size,
            logitsprocs=logitsprocs,
        )

        # ---- update decode state ----
        for i, (req_id, state) in enumerate(decode_reqs):
            state.token_ids.append(decode_next_tokens[i])
            state.generated_tokens += 1
            self._paged_request_seq_lens[req_id] = (
                self._paged_request_seq_lens.get(req_id, len(state.token_ids) - 2) + 1
            )

        # ---- update prefill seq lens ----
        for pr in prefill_reqs:
            self._paged_request_seq_lens[pr.req_id] = pr.start_pos + len(pr.token_ids)

        # ---- postprocess: write results back into batch ----
        for i, entry in enumerate(batch.paged_prefill_entries):
            next_token = prefill_next_tokens[i]
            prefill = prefill_reqs[i]

            if entry.result_mode == "intermediate":
                batch.sampled_tokens[entry.output_idx] = []
                continue

            batch.sampled_tokens[entry.output_idx] = [next_token]
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
                    generator=prefill.generator,
                    generated_tokens=1,
                    block_ids=prefill.block_ids,
                )
                continue

            req_state = self._request_states[prefill.req_id]
            req_state.token_ids.append(next_token)
            req_state.generated_tokens = len(req_state.token_ids) - req_state.prompt_len

        for i, (req_id, _) in enumerate(batch.paged_decode_reqs):
            batch.add_output(req_id, [decode_next_tokens[i]])

        return batch, scheduler_output

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
            token_ids = new_req.prompt_token_ids or []
            sampling_params = new_req.sampling_params or SamplingParams()

            if not token_ids:
                batch.add_output(req_id, [0])
                continue

            generator = _create_request_generator(self.device, sampling_params)

            if self._paged_attention_backend is not None:
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
                            block_ids=sched_block_ids,
                            generator=generator,
                            prompt_len=prompt_len if not is_intermediate else None,
                            start_pos=computed_tokens,
                            full_prompt_token_ids=None,
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
                        generator=generator,
                        generated_tokens=0,
                        block_ids=sched_block_ids,
                    )
                continue

            next_token, cache = self._prefill_single(
                req_id,
                token_ids,
                sampling_params,
                generator=generator,
            )
            batch.add_output(req_id, [next_token])
            self._request_states[req_id] = RequestState(
                token_ids=list(token_ids) + [next_token],
                prompt_len=len(token_ids),
                cache=cache,
                sampling_params=sampling_params,
                generator=generator,
                generated_tokens=1,
                block_ids=[],
            )

    def _update_cached_request_blocks(
        self,
        cached_reqs: CachedRequestData,
    ) -> None:
        """Apply scheduler-provided block updates for paged cached requests."""
        if self._paged_attention_backend is None:
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

        if self._paged_attention_backend is None:
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
                            block_ids=state.block_ids,
                            generator=state.generator,
                            prompt_len=(
                                state.prompt_len if not is_intermediate else None
                            ),
                            start_pos=computed_tokens,
                            full_prompt_token_ids=None,
                        ),
                        result_mode=(
                            "intermediate" if is_intermediate else "cached_final"
                        ),
                    )
                )
                continue

            batch.paged_decode_reqs.append((req_id, state))

    def _build_prefill_pack(
        self,
        batch: _ExecutionBatch,
    ) -> list[PrefillRequest]:
        """Reconstruct full prompt context for paged prefill requests."""
        prefill_pack: list[PrefillRequest] = []
        for entry in batch.paged_prefill_entries:
            prefill = entry.prefill
            full_prompt = None

            if prefill.start_pos > 0:
                state = self._request_states.get(prefill.req_id)
                if state is not None:
                    full_prompt = state.token_ids[: state.prompt_len]
                else:
                    new_req = batch.new_reqs_by_id.get(prefill.req_id)
                    if new_req is None:
                        raise RuntimeError(
                            f"Prefix cache hit (start_pos={prefill.start_pos}) for "
                            f"request {prefill.req_id!r} but it has no RequestState "
                            "and is not in new_reqs. This is a state tracking bug."
                        )
                    prompt_token_ids = new_req.prompt_token_ids
                    if prompt_token_ids is None:
                        raise RuntimeError(
                            f"Prefix cache hit (start_pos={prefill.start_pos}) for "
                            f"request {prefill.req_id!r} but prompt_token_ids is "
                            "missing. This is a scheduler contract bug."
                        )
                    full_prompt = list(prompt_token_ids)

            prefill_pack.append(
                PrefillRequest(
                    req_id=prefill.req_id,
                    token_ids=prefill.token_ids,
                    sampling_params=prefill.sampling_params,
                    block_ids=prefill.block_ids,
                    generator=prefill.generator,
                    prompt_len=prefill.prompt_len,
                    start_pos=prefill.start_pos,
                    full_prompt_token_ids=full_prompt,
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
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None] * len(batch.req_ids),
        )

    def _run_non_paged_decode_batch(
        self,
        batch: _ExecutionBatch,
    ) -> None:
        """Run non-paged decode work and append placeholder outputs as needed."""
        if batch.valid_decode_reqs:
            if len(batch.valid_decode_reqs) >= _MIN_BATCH_SIZE_FOR_BATCHING:
                decode_tokens = self._batched_decode(batch.valid_decode_reqs)
            else:
                decode_tokens = self._sequential_decode(batch.valid_decode_reqs)

            for i, (req_id, _) in enumerate(batch.valid_decode_reqs):
                batch.add_output(req_id, [decode_tokens[i]])

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

            if batch.sampled_tokens[output_idx]:
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
        finished_req_ids: set[str],
    ) -> None:
        """Evict finished request state and periodically clear MLX cache."""
        if not finished_req_ids:
            self._gdn_materialize_pending_state_cache()
            return

        for req_id in finished_req_ids:
            state = self._request_states.pop(req_id, None)
            if state is not None:
                if state.cache:
                    del state.cache
                del state

            # Block freeing is handled by the scheduler's kv_cache_manager.
            self._paged_request_seq_lens.pop(req_id, None)

        self._gdn_release_slots(finished_req_ids)
        self._gdn_materialize_pending_state_cache()

        self._finished_request_count += len(finished_req_ids)
        if self._finished_request_count < _CACHE_CLEAR_INTERVAL:
            return

        mx.clear_cache()
        self._finished_request_count = 0

        if self._prefix_cache is None:
            return

        stats = self._prefix_cache.get_stats()
        logger.info(
            "Prefix cache: %.1f%% hit rate "
            "(hits=%d, misses=%d, cached=%d, "
            "%.1fMB/%.1fMB)",
            stats["hit_rate"] * 100,
            stats["hits"],
            stats["misses"],
            stats["cached_entries"],
            stats["current_bytes"] / (1024 * 1024),
            stats["max_bytes"] / (1024 * 1024),
        )

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

        if self._is_stt:
            return self._execute_stt(scheduler_output)

        batch = _ExecutionBatch()
        self._handle_new_requests(
            batch, scheduler_output.scheduled_new_reqs, scheduler_output
        )

        cached_reqs = scheduler_output.scheduled_cached_reqs
        self._update_cached_request_blocks(cached_reqs)
        self._collect_cached_requests(batch, cached_reqs, scheduler_output)

        if self._paged_attention_backend is not None and batch.has_paged_work():
            # Free GDN slots for finished requests BEFORE allocating new
            # ones, so slots can be reused within the same scheduling step.
            if self.is_hybrid and scheduler_output.finished_req_ids:
                self._gdn_release_slots(scheduler_output.finished_req_ids)

            prefill_pack = self._build_prefill_pack(batch)
            self._start_paged_forward(
                batch,
                prefill_pack,
                batch.paged_decode_reqs,
                scheduler_output,
            )
            return None

        if self._paged_attention_backend is None:
            self._run_non_paged_decode_batch(batch)

        # Non-paged path: complete synchronously
        self._validate_scheduled_outputs(batch, scheduler_output)
        self._cleanup_finished_requests(scheduler_output.finished_req_ids)
        if not batch.req_ids:
            return self._build_output(batch)
        self._pending_output = self._build_output(batch)
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
        del grammar_output

        # Paged path: eval + sample + postprocess
        if self._execute_model_state is not None:
            batch, scheduler_output = self._sample_paged_batch()
            self._validate_scheduled_outputs(batch, scheduler_output)
            self._cleanup_finished_requests(scheduler_output.finished_req_ids)
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

    # ------------------------------------------------------------------
    # STT (Speech-to-Text) helpers
    # ------------------------------------------------------------------

    def _execute_stt(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        """Execute STT inference for all new requests in the batch.

        Raises:
            ValueError: If a request uses non-greedy sampling params.
        """
        assert self._stt_runtime_adapter is not None

        req_ids: list[str] = []
        req_id_to_index: dict[str, int] = {}
        sampled_tokens: list[list[int]] = []

        eot_token = self._stt_runtime_adapter.eot_token

        for new_req in scheduler_output.scheduled_new_reqs:
            stt_request = VLLMSTTRequestAdapter.from_vllm_request(new_req)
            sampling_params = new_req.sampling_params or SamplingParams()

            # Only greedy decoding is supported for STT
            if sampling_params.temperature > 0:
                raise ValueError(
                    "STT models only support greedy decoding (temperature=0). "
                    f"Got temperature={sampling_params.temperature}"
                )

            audio_features = self._stt_runtime_adapter.extract_audio_features(
                stt_request.input_features
            )
            tokens = self._stt_runtime_adapter.decode_tokens(
                audio_features, list(stt_request.prompt_token_ids)
            )

            req_ids.append(stt_request.req_id)
            req_id_to_index[stt_request.req_id] = len(req_ids) - 1
            sampled_tokens.append(tokens)

        # Handle cached requests: STT processes everything in one shot,
        # so any "cached" (decode-phase) request just gets an EOT to finish.
        cached_req_ids = list(scheduler_output.scheduled_cached_reqs.req_ids)
        for req_id in cached_req_ids:
            req_ids.append(req_id)
            req_id_to_index[req_id] = len(req_ids) - 1
            sampled_tokens.append([eot_token])

        # Clean up finished requests
        if scheduler_output.finished_req_ids:
            for req_id in scheduler_output.finished_req_ids:
                self._request_states.pop(req_id, None)

        if not req_ids:
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )

        self._pending_output = ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_tokens,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None] * len(req_ids),
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
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            segments.append(response.text)

        return "".join(segments)
