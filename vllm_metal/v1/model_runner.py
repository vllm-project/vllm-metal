# SPDX-License-Identifier: Apache-2.0
"""Metal Model Runner for vLLM v1 engine.

Optimized for performance with:
- True batched decode using BatchKVCache for O(1) forward passes per batch
- Async evaluation pipeline for pipelined computation
- Pre-allocated input buffers to reduce allocation overhead
- Global model cache for fast repeated loads
- Content hash prefix caching for shared prompt reuse
"""

import hashlib
import math
import os
import time
from array import array
from dataclasses import dataclass, field
from threading import Lock
from typing import Any, Literal, NamedTuple, TypeAlias

import mlx.core as mx
import torch
from mlx_lm import load as mlx_lm_load
from mlx_lm import stream_generate
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    BatchRotatingKVCache,
    KVCache,
    RotatingKVCache,
    make_prompt_cache,
)

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
from vllm_metal.paged_attention_backend.hybrid import _build_linear_layer_spec
from vllm_metal.paged_attention_backend.mla import MLA_DEFAULT_QK_ROPE_HEAD_DIM
from vllm_metal.paged_attention_backend.protocol import PagedAttentionBackend
from vllm_metal.paged_attention_common import (
    OffsetCache,
    clear_context,
    prepare_unified,
)
from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch, torch_to_mlx
from vllm_metal.stt.detection import is_stt_model
from vllm_metal.stt.policy import STT_SCHED_BLOCK_BYTES
from vllm_metal.stt.runtime import STTRuntimeAdapter
from vllm_metal.stt.serve import VLLMSTTRequestAdapter
from vllm_metal.utils import get_model_download_path
from vllm_metal.v1.sampling_batch import (
    DEFAULT_VOCAB_SIZE,
    GREEDY_TEMPERATURE_EPS,
    SamplingBatch,
)

logger = init_logger(__name__)

# Global model cache for fast repeated loads
_model_cache: dict[str, tuple[Any, Any]] = {}  # model_name -> (model, tokenizer)
_model_cache_lock = Lock()


# Configuration for batched operations
_MIN_BATCH_SIZE_FOR_BATCHING = 2  # Minimum requests to use BatchKVCache
_MAX_BATCH_SIZE = 64  # Maximum batch size for decode


# Performance tuning
_CACHE_CLEAR_INTERVAL = 50  # Clear cache every N finished requests

# Prefix cache configuration — enabled by setting VLLM_METAL_PREFIX_CACHE
# in the environment (any value; unset to disable).


def _prefix_cache_enabled() -> bool:
    """Check whether prefix caching is enabled via environment variable."""
    return "VLLM_METAL_PREFIX_CACHE" in os.environ


_PREFIX_CACHE_ENABLED = _prefix_cache_enabled()
_PREFIX_CACHE_DEFAULT_FRACTION = 0.05  # 5% of MLX working set


def _get_prefix_cache_max_bytes() -> int:
    """Get prefix cache memory limit based on MLX recommended working set."""
    fraction_str = os.environ.get("VLLM_METAL_PREFIX_CACHE_FRACTION", "")
    if fraction_str:
        try:
            fraction = float(fraction_str)
            if not math.isfinite(fraction) or fraction <= 0 or fraction > 1:
                logger.warning(
                    "VLLM_METAL_PREFIX_CACHE_FRACTION=%r out of range (0, 1], "
                    "using default %.2f",
                    fraction_str,
                    _PREFIX_CACHE_DEFAULT_FRACTION,
                )
                fraction = _PREFIX_CACHE_DEFAULT_FRACTION
        except ValueError:
            logger.warning(
                "Invalid VLLM_METAL_PREFIX_CACHE_FRACTION=%r, using default %.2f",
                fraction_str,
                _PREFIX_CACHE_DEFAULT_FRACTION,
            )
            fraction = _PREFIX_CACHE_DEFAULT_FRACTION
    else:
        fraction = _PREFIX_CACHE_DEFAULT_FRACTION

    fallback_bytes = 8 * 1024 * 1024 * 1024  # 8 GB
    try:
        device_info = mx.metal.device_info()
        total = int(device_info.get("max_recommended_working_set_size", 0))
    except (AttributeError, RuntimeError):
        total = 0

    if total == 0:
        total = fallback_bytes
        logger.warning("Could not get MLX working set size, using 8GB fallback")

    max_bytes = int(total * fraction)
    logger.info(
        "Prefix cache: %.1fGB limit (%.1f%% of %.1fGB MLX working set)",
        max_bytes / (1024 * 1024 * 1024),
        fraction * 100,
        total / (1024 * 1024 * 1024),
    )
    return max_bytes


def _compute_prefix_hash(token_ids: list[int]) -> bytes:
    """Compute content hash for a token sequence."""
    h = hashlib.sha256()
    h.update(array("I", token_ids).tobytes())
    return h.digest()


def _compute_entry_bytes(cache_state: list[tuple[mx.array, mx.array] | None]) -> int:
    """Compute memory usage of a cache entry in bytes."""
    total = 0
    for pair in cache_state:
        if pair is not None:
            total += pair[0].nbytes + pair[1].nbytes
    return total


@dataclass
class CachedPrefix:
    """Cached KV state for a token prefix.

    cache_state contains (k, v) tuples for KVCache layers, or None for
    ArraysCache layers in hybrid models.
    """

    token_ids: list[int]
    cache_state: list[tuple[mx.array, mx.array] | None]
    size_bytes: int = 0
    ref_count: int = 0


class PrefixCacheManager:
    """Manager for prefix KV cache reuse with memory-based eviction."""

    def __init__(self, max_bytes: int | None = None):
        self._cache: dict[bytes, CachedPrefix] = {}
        self._max_bytes = (
            max_bytes if max_bytes is not None else _get_prefix_cache_max_bytes()
        )
        self._current_bytes = 0
        self._hits = 0
        self._misses = 0

    def lookup(self, token_ids: list[int]) -> CachedPrefix | None:
        """Look up cached prefix by token IDs."""
        prefix_hash = _compute_prefix_hash(token_ids)
        cached = self._cache.get(prefix_hash)
        if cached is not None:
            self._hits += 1
            cached.ref_count += 1
            logger.debug(
                "Prefix cache HIT: %d hits, %d misses, rate=%.1f%%",
                self._hits,
                self._misses,
                self.hit_rate * 100,
            )
            return cached
        self._misses += 1
        logger.debug(
            "Prefix cache MISS: %d hits, %d misses, rate=%.1f%%",
            self._hits,
            self._misses,
            self.hit_rate * 100,
        )
        return None

    def _evict_until_fits(self, needed_bytes: int) -> None:
        """Evict entries until we have room for needed_bytes."""
        while self._current_bytes + needed_bytes > self._max_bytes and self._cache:
            min_hash, min_entry = min(self._cache.items(), key=lambda x: x[1].ref_count)
            self._current_bytes -= min_entry.size_bytes
            del self._cache[min_hash]
            logger.debug(
                "Prefix cache eviction: freed %.1fMB",
                min_entry.size_bytes / (1024 * 1024),
            )

    def insert(self, token_ids: list[int], cache: list[KVCache]) -> None:
        """Insert a prefix cache entry with memory-based eviction.

        Only KVCache layers are cached. ArraysCache layers are skipped (stored as
        None) for hybrid model compatibility.
        """
        prefix_hash = _compute_prefix_hash(token_ids)
        if prefix_hash in self._cache:
            return

        cache_state = []
        for layer_cache in cache:
            if isinstance(layer_cache, KVCache):
                k = layer_cache.state[0]
                v = layer_cache.state[1]
                cache_state.append((mx.array(k), mx.array(v)))
            else:
                cache_state.append(None)

        entry_bytes = _compute_entry_bytes(cache_state)

        # Skip if single entry exceeds memory limit
        if entry_bytes > self._max_bytes:
            logger.debug(
                "Prefix cache skip: entry %.1fMB exceeds limit %.1fGB",
                entry_bytes / (1024 * 1024),
                self._max_bytes / (1024 * 1024 * 1024),
            )
            return

        self._evict_until_fits(entry_bytes)

        self._cache[prefix_hash] = CachedPrefix(
            token_ids=list(token_ids),
            cache_state=cache_state,
            size_bytes=entry_bytes,
            ref_count=1,
        )
        self._current_bytes += entry_bytes

    def restore_cache(
        self, cached: CachedPrefix, model: Any, is_vlm: bool
    ) -> list["AnyCache"]:
        """Restore a cached prefix to a fresh KVCache.

        Only KVCache layers are restored. RotatingKVCache / ArraysCache layers
        remain in their fresh state.
        """
        cache_model = (
            model.language_model
            if is_vlm and hasattr(model, "language_model")
            else model
        )
        cache = make_prompt_cache(cache_model)
        for i, layer_cache in enumerate(cache):
            if i < len(cached.cache_state) and cached.cache_state[i] is not None:
                if isinstance(layer_cache, KVCache):
                    k, v = cached.cache_state[i]
                    layer_cache.state = [mx.array(k), mx.array(v)]
                    # Keep RoPE position correct even if KVCache.state setter
                    # behavior changes in future mlx-lm versions.
                    layer_cache.offset = int(k.shape[2])
        return cache

    @property
    def hit_rate(self) -> float:
        """Return prefix cache hit rate."""
        total = self._hits + self._misses
        return self._hits / total if total > 0 else 0.0

    def get_stats(self) -> dict:
        """Return prefix cache statistics."""
        return {
            "hits": self._hits,
            "misses": self._misses,
            "hit_rate": self.hit_rate,
            "cached_entries": len(self._cache),
            "current_bytes": self._current_bytes,
            "max_bytes": self._max_bytes,
        }


# Type alias for any per-layer cache type supported by the model.
#
# Notes:
# - Some models (e.g. gpt_oss) use `RotatingKVCache` for sliding-window attention.
# - Hybrid models use `ArraysCache` for non-attention state.
AnyCache: TypeAlias = KVCache | RotatingKVCache | ArraysCache
SchedulerMemoryReportingMode: TypeAlias = Literal[
    "stt_nominal",
    "paged_attention_capacity",
    "single_sequence_estimate",
]


def _merge_arrays_caches(caches: list[ArraysCache]) -> ArraysCache:
    """Merge per-request ArraysCache objects into a single batched ArraysCache.

    This mirrors the behavior of `mlx_lm.models.cache.ArraysCache.merge` but is
    implemented here for compatibility with older mlx-lm versions that do not
    provide `merge()` / `extract()`.
    """
    if not caches:
        raise ValueError("caches must be non-empty")

    num_entries = len(caches[0].state)
    batch_size = len(caches)

    merged = ArraysCache(num_entries)
    for entry_idx in range(num_entries):
        values = [cache.state[entry_idx] for cache in caches]
        template = next((value for value in values if value is not None), None)
        if template is None:
            continue

        shape = list(template.shape)
        shape[0] = batch_size
        merged_state = mx.zeros(tuple(shape), template.dtype)
        for batch_idx, value in enumerate(values):
            if value is None:
                continue
            merged_state[batch_idx : batch_idx + 1] = value

        merged[entry_idx] = merged_state

    return merged


def _extract_arrays_cache(batch_cache: ArraysCache, idx: int) -> ArraysCache:
    """Extract a single request's ArraysCache from a batched ArraysCache."""
    state = batch_cache.state
    extracted = ArraysCache(len(state))
    extracted.state = [
        None if value is None else value[idx : idx + 1] for value in state
    ]
    return extracted


def _merge_rotating_kv_caches(
    caches: list[RotatingKVCache],
) -> BatchRotatingKVCache:
    """Merge per-request RotatingKVCache objects into a single BatchRotatingKVCache.

    This mirrors ``BatchRotatingKVCache.merge`` but pre-computes the temporal-ordered
    keys/values, trims them to ``len(cache)`` (the effective sliding-window length),
    and uses that length for the copy width.  The upstream implementation in
    mlx-lm <= 0.29.1 uses ``c.offset`` which can exceed the underlying array size
    after the cache has rotated, causing a broadcast shape error.

    This workaround can be removed once vllm-metal can depend on an mlx-lm version
    that includes the upstream fix (ml-explore/mlx-lm#738) and has been verified
    to work with gpt-oss models end-to-end.
    """
    if not caches:
        raise ValueError("caches must be non-empty")

    if any(c.keys is None or c.values is None for c in caches):
        raise ValueError(
            "Cannot merge unpopulated RotatingKVCache (keys/values is None)"
        )

    if not all(c.max_size == caches[0].max_size for c in caches):
        raise ValueError(
            "BatchRotatingKVCache can only merge caches with the same maximum size"
        )

    # Pre-compute temporal-ordered keys/values and trim to the effective
    # sliding-window length.  ``_temporal_order`` may return an array larger
    # than ``len(cache)`` when the internal buffer has not been trimmed yet
    # (e.g. after a large prefill), so we trim via ``_trim`` to preserve
    # the ``keep`` prefix semantics used by RotatingKVCache internally.
    ordered: list[tuple[mx.array, mx.array]] = []
    for c in caches:
        effective_len = c.size() if hasattr(c, "size") else len(c)
        ordered_keys = c._temporal_order(c.keys)
        ordered_values = c._temporal_order(c.values)
        if ordered_keys.shape[2] > effective_len:
            trim_size = ordered_keys.shape[2] - effective_len
            ordered_keys = c._trim(trim_size, ordered_keys)
            ordered_values = c._trim(trim_size, ordered_values)
        else:
            ordered_keys = ordered_keys[..., :effective_len, :]
            ordered_values = ordered_values[..., :effective_len, :]
        ordered.append((ordered_keys, ordered_values))

    lengths = [k.shape[2] for k, _ in ordered]
    max_length = max(lengths)
    padding = [max_length - length for length in lengths]
    batch_size = len(caches)
    n_heads = max(k.shape[1] for k, _ in ordered)
    k_dim = max(k.shape[3] for k, _ in ordered)
    v_dim = max(v.shape[3] for _, v in ordered)
    dtype = next(iter(k.dtype for k, _ in ordered))

    keys = mx.zeros((batch_size, n_heads, max_length, k_dim), dtype=dtype)
    values = mx.zeros((batch_size, n_heads, max_length, v_dim), dtype=dtype)
    for i, (pad, (k, v)) in enumerate(zip(padding, ordered, strict=True)):
        n = k.shape[2]
        keys[i : i + 1, :, pad : pad + n] = k
        values[i : i + 1, :, pad : pad + n] = v

    cache = BatchRotatingKVCache(caches[0].max_size, padding)
    cache.keys = keys
    cache.values = values
    cache.offset = mx.array([c.offset for c in caches])
    cache._idx = keys.shape[2]
    cache._offset = keys.shape[2]

    return cache


def _mlx_greedy_sample(logits: mx.array) -> mx.array:
    """Native MLX greedy sampling - avoids PyTorch round-trip.

    Args:
        logits: Logits tensor of shape (batch_size, vocab_size)

    Returns:
        Token IDs of shape (batch_size,)
    """
    return mx.argmax(logits, axis=-1)


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
class SamplerOutput:
    """Output from the sampler."""

    token_ids: list[int]
    logprobs: list[float] | None = None


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
    """Packed prefill request passed to ``_unified_prefill_decode_paged``."""

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


def _merge_kv_caches(
    caches_list: list[list[AnyCache]],
) -> list[BatchKVCache | BatchRotatingKVCache | ArraysCache]:
    """Merge multiple per-request caches into batched caches.

    Args:
        caches_list: List of per-request caches, each is a list of per-layer caches

    Returns:
        List of batched caches, one per layer
    """
    if not caches_list:
        return []

    num_layers = len(caches_list[0])
    merged: list[BatchKVCache | BatchRotatingKVCache | ArraysCache] = []

    for layer_idx in range(num_layers):
        layer_caches = [caches[layer_idx] for caches in caches_list]
        if isinstance(layer_caches[0], ArraysCache):
            arrays_caches: list[ArraysCache] = []
            for cache in layer_caches:
                if not isinstance(cache, ArraysCache):
                    raise TypeError(
                        "Mixed cache types in a single layer: expected ArraysCache"
                    )
                arrays_caches.append(cache)
            batch_cache = _merge_arrays_caches(arrays_caches)
        elif isinstance(layer_caches[0], RotatingKVCache):
            rotating_caches: list[RotatingKVCache] = []
            for cache in layer_caches:
                if not isinstance(cache, RotatingKVCache):
                    raise TypeError(
                        "Mixed cache types in a single layer: expected RotatingKVCache"
                    )
                rotating_caches.append(cache)
            batch_cache = _merge_rotating_kv_caches(rotating_caches)
        elif isinstance(layer_caches[0], KVCache):
            kv_caches: list[KVCache] = []
            for cache in layer_caches:
                if not isinstance(cache, KVCache):
                    raise TypeError(
                        "Mixed cache types in a single layer: expected KVCache"
                    )
                kv_caches.append(cache)
            batch_cache = BatchKVCache.merge(kv_caches)
        else:
            cache_type = type(layer_caches[0]).__name__
            raise TypeError(f"Unsupported cache type for batching: {cache_type}")
        merged.append(batch_cache)

    return merged


def _extract_kv_cache(
    batch_caches: list[BatchKVCache | BatchRotatingKVCache | ArraysCache], idx: int
) -> list[AnyCache]:
    """Extract a single request's cache from batched caches.

    Args:
        batch_caches: List of batched caches, one per layer
        idx: Index of the request in the batch

    Returns:
        List of caches for the request, one per layer
    """
    extracted: list[AnyCache] = []
    for cache in batch_caches:
        if isinstance(cache, ArraysCache):
            extracted.append(_extract_arrays_cache(cache, idx))
        else:
            c = cache.extract(idx)
            # After extract, RotatingKVCache may have offset > max_size but
            # keys.shape[2] < max_size (buffer was sliced).  Pad the buffer
            # back to max_size so _update_in_place won't try to grow it
            # (which would compute a negative new_size).  The padded region
            # is dead space that will be overwritten on the next rotation.
            if (
                isinstance(c, RotatingKVCache)
                and c.keys is not None
                and c.offset > c.max_size
                and c.keys.shape[2] < c.max_size
            ):
                pad = c.max_size - c.keys.shape[2]
                z_k = mx.zeros(
                    (1, c.keys.shape[1], pad, c.keys.shape[3]),
                    dtype=c.keys.dtype,
                )
                z_v = mx.zeros(
                    (1, c.values.shape[1], pad, c.values.shape[3]),
                    dtype=c.values.dtype,
                )
                c.keys = mx.concatenate([c.keys, z_k], axis=2)
                c.values = mx.concatenate([c.values, z_v], axis=2)
            extracted.append(c)
    return extracted


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

        # Pre-allocated buffer for decode input tokens
        self._max_batch_size = _MAX_BATCH_SIZE

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
            self._prefix_cache = PrefixCacheManager()

        # Paged attention state (set by worker when enabled)
        self._paged_attention_backend: PagedAttentionBackend | None = None
        self._paged_block_size: int = 0
        self._paged_request_seq_lens: dict[str, int] = {}  # req_id → seq_len
        self.kv_cache_dtype: mx.Dtype | None = None

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
        # Check vLLM's multimodal detection
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
        # layer, not per-head K/V. One virtual head sized kv_lora_rank +
        # qk_rope_head_dim keeps get_cache_block_size_bytes() conservative (2x)
        # without MLA-specific logic in the sizing path.
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
        if self._gdn_free_slots:
            slot = self._gdn_free_slots.pop()
        else:
            slot = len(self._gdn_req_to_slot)
        self._gdn_req_to_slot[req_id] = slot
        return slot

    def _gdn_free_slot(self, req_id: str) -> None:
        """Release a GDN state pool slot and zero its state."""
        slot = self._gdn_req_to_slot.pop(req_id, None)
        if slot is None:
            return
        # Zero conv and recurrent state so the next request doesn't
        # inherit the previous request's linear-attention history.
        backend = self._paged_attention_backend
        if backend is not None and hasattr(backend, "_state_cache"):
            sc = backend._state_cache
            if sc is not None:
                for layer_idx in range(sc.num_layers):
                    conv = sc.conv_states[layer_idx]
                    conv[slot] = 0
                    sc.conv_states[layer_idx] = conv
                    rec = sc.recurrent_states[layer_idx]
                    rec[slot] = 0
                    sc.recurrent_states[layer_idx] = rec
        self._gdn_free_slots.append(slot)

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
        return (
            2
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
        conv_bytes = (
            (self.linear_conv_kernel_dim - 1) * self.linear_conv_dim * dtype_size
        )
        recurrent_bytes = (
            self.linear_num_v_heads
            * self.linear_value_head_dim
            * self.linear_key_head_dim
            * dtype_size
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
            output = self.model(dummy_tokens)
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
            vocab_size=self.model_args.get("vocab_size", DEFAULT_VOCAB_SIZE),
            device=self.device,
            logitsprocs=getattr(self, "_logitsprocs", None),
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
        cache_model = (
            self.model.language_model
            if self._is_vlm and hasattr(self.model, "language_model")
            else self.model
        )

        # Create cache to check if model supports prefix caching
        cache = make_prompt_cache(cache_model)
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
                _ = self.model(prefix_ids, cache=cache)
                self._prefix_cache.insert(prefix, cache)
                cached_prefix_len = len(prefix)

        # Prefill: process remaining tokens (always at least the last token)
        tokens_to_process = token_ids[cached_prefix_len:]
        input_ids = mx.array([tokens_to_process], dtype=mx.int32)
        model_output = self.model(input_ids, cache=cache)

        logits = self._extract_logits(model_output)

        # Extract last token logits
        last_logits = logits[:, -1, :]

        if SamplingBatch.can_use_native_greedy([sampling_params]):
            # Fast path: native MLX greedy sampling
            next_token_mlx = _mlx_greedy_sample(last_logits)
            # Single eval for logits, token, and cache state together
            mx.eval(next_token_mlx, *[c.state for c in cache])
            next_token = int(next_token_mlx.item())
        else:
            # Slow path: use vLLM sampler for advanced sampling
            # Single eval for logits and cache state together
            mx.eval(last_logits, *[c.state for c in cache])
            # Convert to torch for sampling
            logits_torch = mlx_to_torch(
                last_logits.astype(mx.float32), device=self.device
            )
            generators = {} if generator is None else {0: generator}
            metadata = self._make_sampling_metadata(
                [sampling_params],
                [token_ids],
                [[]],
                generators=generators,
            )
            output = self._sampler.forward(logits_torch, metadata)
            next_token = int(output.sampled_token_ids[0, 0].item())

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
        batch_size = len(decode_reqs)

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
        model_output = self.model(batched_input, cache=batch_cache)
        logits = self._extract_logits(model_output)

        # Extract next token logits
        next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
        sampling_params_list = [state.sampling_params for _, state in decode_reqs]

        if SamplingBatch.can_use_native_greedy(sampling_params_list):
            # Fast path: native MLX greedy sampling for entire batch
            next_tokens_mlx = _mlx_greedy_sample(next_token_logits)
            # Single eval - no intermediate sync needed
            mx.eval(next_tokens_mlx)
            next_tokens: list[int] = next_tokens_mlx.tolist()
        else:
            # Slow path: use vLLM sampler for advanced sampling
            mx.eval(next_token_logits)
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
            logits_torch = mlx_to_torch(
                next_token_logits.astype(mx.float32), device=self.device
            )
            metadata = self._make_sampling_metadata(
                sampling_params_list,
                prompt_token_ids_list,
                output_tokens_list,
                generators=generators,
            )
            output = self._sampler.forward(logits_torch, metadata)
            next_tokens = [
                int(output.sampled_token_ids[i, 0].item()) for i in range(batch_size)
            ]

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

            model_output = self.model(input_ids, cache=state.cache)
            logits = self._extract_logits(model_output)
            last_logits = logits[:, -1, :]

            sp = state.sampling_params
            if SamplingBatch.can_use_native_greedy([sp]):
                # Fast path: native MLX greedy sampling
                next_token_mlx = _mlx_greedy_sample(last_logits)
                mx.eval(next_token_mlx)
                next_token = int(next_token_mlx.item())
            else:
                # Slow path: use vLLM sampler
                mx.eval(last_logits)
                logits_torch = mlx_to_torch(
                    last_logits.astype(mx.float32), device=self.device
                )
                generators = {} if state.generator is None else {0: state.generator}
                metadata = self._make_sampling_metadata(
                    [state.sampling_params],
                    [state.token_ids[: state.prompt_len]],
                    [state.token_ids[state.prompt_len :]],
                    generators=generators,
                )
                output = self._sampler.forward(logits_torch, metadata)
                next_token = int(output.sampled_token_ids[0, 0].item())

            next_tokens.append(next_token)

            # Update state
            state.token_ids.append(next_token)
            state.generated_tokens += 1

        return next_tokens

    # ------------------------------------------------------------------
    # Unified prefill + decode (single forward pass)
    # ------------------------------------------------------------------

    def _unified_prefill_decode_paged(
        self,
        prefill_reqs: list[PrefillRequest],
        decode_reqs: list[tuple[str, RequestState]],
    ) -> tuple[list[int], list[int]]:
        """Single forward pass for mixed prefill + decode requests.

        Packs decode tokens (1 per request) followed by prefill tokens into
        a flat ``(1, total_tokens)`` input.  The varlen kernel uses
        ``cu_seqlens`` to handle variable-length subsequences.

        Returns:
            ``(prefill_next_tokens, decode_next_tokens)``
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

        # ---- forward ----
        offset_caches = [OffsetCache(0) for _ in range(self.num_layers)]
        input_ids = mx.array([all_token_ids], dtype=mx.int32)
        try:
            model_output = self.model(input_ids, cache=offset_caches)
            logits = self._extract_logits(model_output)
        finally:
            clear_context()

        # ---- build cu_seqlens for logit extraction ----
        cu_seqlens: list[int] = [0]
        for _ in decode_reqs:
            cu_seqlens.append(cu_seqlens[-1] + 1)
        for pr in prefill_reqs:
            cu_seqlens.append(cu_seqlens[-1] + len(pr.token_ids))

        # ---- sample decode tokens ----
        decode_next_tokens: list[int] = []
        if decode_reqs:
            # All decode logits are at positions 0..num_decode-1
            decode_logits = logits[0, :num_decode, :]  # (num_decode, vocab)

            sampling_params_list = [state.sampling_params for _, state in decode_reqs]
            if SamplingBatch.can_use_native_greedy(sampling_params_list):
                next_tokens_mlx = _mlx_greedy_sample(decode_logits)
                mx.eval(next_tokens_mlx)
                decode_next_tokens = next_tokens_mlx.tolist()
            else:
                mx.eval(decode_logits)
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
                logits_torch = mlx_to_torch(
                    decode_logits.astype(mx.float32), device=self.device
                )
                metadata = self._make_sampling_metadata(
                    sampling_params_list,
                    prompt_token_ids_list,
                    output_tokens_list,
                    generators=generators,
                )
                output = self._sampler.forward(logits_torch, metadata)
                decode_next_tokens = [
                    int(output.sampled_token_ids[i, 0].item())
                    for i in range(num_decode)
                ]

            # Update decode state
            for i, (req_id, state) in enumerate(decode_reqs):
                state.token_ids.append(decode_next_tokens[i])
                state.generated_tokens += 1
                self._paged_request_seq_lens[req_id] = (
                    self._paged_request_seq_lens.get(req_id, len(state.token_ids) - 2)
                    + 1
                )

        # ---- sample prefill tokens ----
        prefill_next_tokens: list[int] = []
        for j, pr in enumerate(prefill_reqs):
            last_idx = cu_seqlens[num_decode + j + 1] - 1
            last_logits = logits[:, last_idx : last_idx + 1, :]

            if pr.full_prompt_token_ids is not None:
                prompt_len = len(pr.full_prompt_token_ids)
            elif pr.prompt_len is not None:
                prompt_len = pr.prompt_len
            else:
                prompt_len = len(pr.token_ids)

            if SamplingBatch.can_use_native_greedy([pr.sampling_params]):
                next_token_mlx = _mlx_greedy_sample(last_logits[0])
                mx.eval(next_token_mlx)
                next_token = int(next_token_mlx.item())
            else:
                mx.eval(last_logits)
                logits_torch = mlx_to_torch(
                    last_logits[0].astype(mx.float32), device=self.device
                )
                generators = {} if pr.generator is None else {0: pr.generator}
                # Use full prompt for penalty computation when available
                # (prefix cache hit supplies suffix-only token_ids).
                prompt_for_meta = (
                    pr.full_prompt_token_ids
                    if pr.full_prompt_token_ids is not None
                    else pr.token_ids
                )
                metadata = self._make_sampling_metadata(
                    [pr.sampling_params],
                    [prompt_for_meta[:prompt_len]],
                    [prompt_for_meta[prompt_len:]],
                    generators=generators,
                )
                output = self._sampler.forward(logits_torch, metadata)
                next_token = int(output.sampled_token_ids[0, 0].item())

            self._paged_request_seq_lens[pr.req_id] = pr.start_pos + len(pr.token_ids)
            prefill_next_tokens.append(next_token)

        return prefill_next_tokens, decode_next_tokens

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

    def _run_paged_batch(
        self,
        batch: _ExecutionBatch,
        prefill_pack: list[PrefillRequest],
    ) -> None:
        """Run paged prefill/decode and write results back into ``batch``."""
        prefill_tokens, decode_tokens = self._unified_prefill_decode_paged(
            prefill_pack, batch.paged_decode_reqs
        )

        for i, entry in enumerate(batch.paged_prefill_entries):
            next_token = prefill_tokens[i]
            prefill = prefill_pack[i]

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

            state = self._request_states[prefill.req_id]
            state.token_ids.append(next_token)
            state.generated_tokens = len(state.token_ids) - state.prompt_len

        for i, (req_id, _) in enumerate(batch.paged_decode_reqs):
            batch.add_output(req_id, [decode_tokens[i]])

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
            return

        for req_id in finished_req_ids:
            state = self._request_states.pop(req_id, None)
            if state is not None:
                if state.cache:
                    del state.cache
                del state

            # Block freeing is handled by the scheduler's kv_cache_manager.
            self._paged_request_seq_lens.pop(req_id, None)
            self._gdn_free_slot(req_id)

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

    def _finalize_output(
        self,
        batch: _ExecutionBatch,
    ) -> ModelRunnerOutput | None:
        """Store execute-time output for ``sample_tokens()`` or return empty."""
        if not batch.req_ids:
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )

        self._pending_output = ModelRunnerOutput(
            req_ids=batch.req_ids,
            req_id_to_index=batch.req_id_to_index,
            sampled_token_ids=batch.sampled_tokens,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None] * len(batch.req_ids),
        )
        return None

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        """Execute model inference with true batched decode."""
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
            prefill_pack = self._build_prefill_pack(batch)
            self._run_paged_batch(batch, prefill_pack)
        elif self._paged_attention_backend is None:
            self._run_non_paged_decode_batch(batch)

        self._validate_scheduled_outputs(batch, scheduler_output)
        self._cleanup_finished_requests(scheduler_output.finished_req_ids)
        return self._finalize_output(batch)

    def sample_tokens(
        self, grammar_output: GrammarOutput | None
    ) -> ModelRunnerOutput | None:
        """Return sampled tokens produced by the last execute_model call.

        vLLM's v1 engine calls ``sample_tokens`` after a successful
        ``execute_model`` call that returned ``None``. When async scheduling is
        enabled, vLLM may still call ``sample_tokens`` even if ``execute_model``
        failed; returning ``None`` in that case allows vLLM to surface the
        original exception from ``execute_model``.
        """
        del grammar_output
        if self._pending_output is None:
            model_id = None
            model_config = getattr(self, "model_config", None)
            if model_config is not None:
                model_id = getattr(model_config, "model", None)

            if getattr(self, "use_async_scheduling", False):
                logger.error(
                    "sample_tokens called without pending output from "
                    "execute_model (model=%r). Returning None so vLLM can "
                    "surface the original execute_model error.",
                    model_id,
                )
                return None

            raise RuntimeError(
                "State error: sample_tokens called without pending output from "
                f"execute_model (model={model_id!r})."
            )
        output = self._pending_output
        self._pending_output = None
        return output

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
