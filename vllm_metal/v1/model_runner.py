# SPDX-License-Identifier: Apache-2.0
"""Metal Model Runner for vLLM v1 engine.

Optimized for performance with:
- True batched decode using BatchKVCache for O(1) forward passes per batch
- Async evaluation pipeline for pipelined computation
- Pre-allocated input buffers to reduce allocation overhead
- Rust-based token state management for efficient batch operations
- Global model cache for fast repeated loads
"""

import time
from dataclasses import dataclass
from threading import Lock
from typing import Any, TypeAlias

import mlx.core as mx
import torch
from mlx_lm import load as mlx_lm_load
from mlx_lm import stream_generate
from mlx_lm.models.cache import (
    ArraysCache,
    BatchKVCache,
    KVCache,
    MambaCache,
    make_prompt_cache,
)

# mlx_vlm for vision-language models
from mlx_vlm import load as mlx_vlm_load
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.sampling_params import SamplingParams
from vllm.utils.torch_utils import make_tensor_with_pad
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.sample.logits_processor import LogitsProcessors
from vllm.v1.sample.metadata import SamplingMetadata
from vllm.v1.sample.sampler import Sampler

from vllm_metal.config import get_config
from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
from vllm_metal.utils import get_model_download_path

logger = init_logger(__name__)

# Global model cache for fast repeated loads
_model_cache: dict[str, tuple[Any, Any]] = {}  # model_name -> (model, tokenizer)
_model_cache_lock = Lock()

# Try to import Rust extension for high-performance token state management
try:
    from vllm_metal._rs import RequestStateManager as RustRequestStateManager

    _RUST_AVAILABLE = True
except ImportError:
    _RUST_AVAILABLE = False
    logger.debug("Rust extension not available, using Python fallback")

# Configuration for batched operations
_MIN_BATCH_SIZE_FOR_BATCHING = 2  # Minimum requests to use BatchKVCache
_MAX_BATCH_SIZE = 64  # Maximum batch size for decode

# Performance tuning
_CACHE_CLEAR_INTERVAL = 50  # Clear cache every N finished requests

# Type alias for any cache type supported by the model
AnyCache: TypeAlias = KVCache | MambaCache | ArraysCache


class BatchMambaCache:
    """Batched cache for Mamba/SSM layers.

    Wraps multiple MambaCache instances into a single batched cache
    for efficient batched forward passes on hybrid models.
    """

    def __init__(self, caches: list[MambaCache | ArraysCache]):
        """Create a batched Mamba cache from individual caches.

        Args:
            caches: List of MambaCache instances to batch
        """
        self._batch_size = len(caches)
        self._cache_size = len(caches[0].cache) if caches else 0

        # Stack each state array across the batch dimension
        self.cache: list[mx.array | None] = []
        for i in range(self._cache_size):
            states = [c.cache[i] for c in caches]
            if all(s is not None for s in states):
                self.cache.append(mx.concatenate(states, axis=0))
            else:
                self.cache.append(None)

    def __getitem__(self, idx: int) -> mx.array | None:
        return self.cache[idx]

    def __setitem__(self, idx: int, value: mx.array | None) -> None:
        self.cache[idx] = value

    def extract(self, idx: int) -> MambaCache:
        """Extract a single request's cache from the batch.

        Args:
            idx: Index of the request in the batch

        Returns:
            MambaCache for the individual request
        """
        cache = MambaCache()
        for i in range(self._cache_size):
            if self.cache[i] is not None:
                cache.cache[i] = self.cache[i][idx : idx + 1]
        return cache


def _is_mamba_cache(cache: AnyCache) -> bool:
    """Check if a cache is a Mamba-style cache (ArraysCache or MambaCache)."""
    return isinstance(cache, (MambaCache, ArraysCache))


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
    if sampling_params.temperature < 1e-5:
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
    cache: list[AnyCache]  # Per-layer caches (KVCache or MambaCache for hybrid models)
    sampling_params: SamplingParams  # Sampling parameters for this request
    generator: torch.Generator | None = None
    generated_tokens: int = 0


def _merge_kv_caches(
    caches_list: list[list[AnyCache]],
) -> list[BatchKVCache | BatchMambaCache]:
    """Merge multiple per-request caches into batched caches.

    Args:
        caches_list: List of per-request caches, each is a list of per-layer caches

    Returns:
        List of batched caches, one per layer
    """
    if not caches_list:
        return []

    num_layers = len(caches_list[0])
    merged: list[BatchKVCache | BatchMambaCache] = []

    for layer_idx in range(num_layers):
        layer_caches = [caches[layer_idx] for caches in caches_list]
        if _is_mamba_cache(layer_caches[0]):
            batch_cache = BatchMambaCache(layer_caches)
        else:
            batch_cache = BatchKVCache.merge(layer_caches)
        merged.append(batch_cache)

    return merged


def _extract_kv_cache(
    batch_caches: list[BatchKVCache | BatchMambaCache], idx: int
) -> list[AnyCache]:
    """Extract a single request's cache from batched caches.

    Args:
        batch_caches: List of batched caches, one per layer
        idx: Index of the request in the batch

    Returns:
        List of caches for the request, one per layer
    """
    return [cache.extract(idx) for cache in batch_caches]


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
        self.device = device
        self.metal_config = get_config()

        self.model: Any = None
        self.tokenizer: Any = None
        self.model_args: dict[str, Any] = {}
        self._is_vlm: bool = False  # Will be set during model loading

        # KV cache state
        self.kv_cache_initialized = False
        self.num_kv_cache_blocks = 0

        # Request state cache for incremental decoding
        self._request_states: dict[str, RequestState] = {}

        # Rust-based token state manager (optional, for batch operations)
        self._rust_state_manager: Any = None
        if _RUST_AVAILABLE:
            self._rust_state_manager = RustRequestStateManager()

        # Pre-allocated buffer for decode input tokens
        self._max_batch_size = _MAX_BATCH_SIZE

        # vLLM Sampler for token sampling with temperature, top_k, top_p support
        self._sampler = Sampler()

        # Track finished requests for lazy cache clearing
        self._finished_request_count = 0

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
        is_vlm = self._is_vlm_model()

        logger.info(f"Loading model: {model_name} (VLM: {is_vlm})")
        start_time = time.time()

        # Check global cache first for fast repeated loads
        with _model_cache_lock:
            if model_name in _model_cache:
                self.model, self.tokenizer = _model_cache[model_name]
                load_time = time.time() - start_time
                logger.info(
                    f"Model loaded from cache in {load_time:.3f}s: {model_name}"
                )
                self._extract_model_args()
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
        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s: {model_name}")

    def _extract_model_args(self) -> None:
        """Extract model configuration from loaded model.

        Handles both text-only models and VLMs (which have nested text_config).
        """
        if hasattr(self.model, "args"):
            self.model_args = vars(self.model.args)
        elif hasattr(self.model, "config"):
            config = self.model.config
            # VLMs often have text config nested inside main config
            if self._is_vlm and hasattr(config, "text_config"):
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
                self.model_args = config.to_dict()
            else:
                self.model_args = vars(config)
        else:
            # Fallback: try to get from model attributes
            self.model_args = {
                "num_hidden_layers": getattr(self.model, "n_layers", 32),
                "num_attention_heads": getattr(self.model, "n_heads", 32),
                "num_key_value_heads": getattr(
                    self.model, "n_kv_heads", getattr(self.model, "n_heads", 32)
                ),
                "hidden_size": getattr(self.model, "dim", 4096),
                "head_dim": getattr(self.model, "head_dim", 128),
                "vocab_size": getattr(self.model, "vocab_size", 32000),
            }
        if self.metal_config.debug:
            logger.info(f"Model args: {self.model_args}")

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
        # Handle None values explicitly - model configs may have keys set to None
        num_layers = (
            self.model_args.get("num_hidden_layers")
            or self.model_args.get("n_layers")
            or 32
        )
        num_attention_heads = self.model_args.get("num_attention_heads") or 32
        num_kv_heads = (
            self.model_args.get("num_key_value_heads")
            or self.model_args.get("n_kv_heads")
            or num_attention_heads
        )
        hidden_size = self.model_args.get("hidden_size") or 4096
        head_size = self.model_args.get("head_dim") or (
            hidden_size // num_attention_heads
        )
        block_size = self.metal_config.block_size

        # Create a spec for each layer
        specs: dict[str, KVCacheSpec] = {}
        for layer_idx in range(num_layers):
            layer_name = f"layers.{layer_idx}.self_attn"
            specs[layer_name] = FullAttentionSpec(
                block_size=block_size,
                num_kv_heads=num_kv_heads,
                head_size=head_size,
                dtype=torch.float16,
            )

        return specs

    def initialize_kv_cache(self, kv_cache_config: KVCacheConfig) -> None:
        """Initialize KV cache from configuration.

        Args:
            kv_cache_config: KV cache configuration for this worker
        """
        self.num_kv_cache_blocks = kv_cache_config.num_blocks
        logger.info(f"KV cache initialized with {self.num_kv_cache_blocks} blocks")
        self.kv_cache_initialized = True

    def get_cache_block_size_bytes(self) -> int:
        """Get the size of a single cache block in bytes.

        Returns:
            Block size in bytes
        """
        # Handle None values explicitly - model configs may have keys set to None
        num_layers = (
            self.model_args.get("num_hidden_layers")
            or self.model_args.get("n_layers")
            or 32
        )
        num_attention_heads = self.model_args.get("num_attention_heads") or 32
        num_kv_heads = (
            self.model_args.get("num_key_value_heads")
            or self.model_args.get("n_kv_heads")
            or num_attention_heads
        )
        hidden_size = self.model_args.get("hidden_size") or 4096
        head_dim = self.model_args.get("head_dim") or (
            hidden_size // num_attention_heads
        )
        block_size = self.metal_config.block_size

        # Each block stores key and value for all layers
        # Block memory = 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size
        dtype_size = 2  # float16
        return 2 * num_layers * block_size * num_kv_heads * head_dim * dtype_size

    def warm_up(self) -> None:
        """Warm up the model with a dummy forward pass."""
        if self.model is None:
            logger.warning("Model not loaded, skipping warm-up")
            return

        logger.info("Warming up model...")

        # Run a small dummy inference
        try:
            dummy_tokens = mx.array([[1, 2, 3]], dtype=mx.int32)
            output = self.model(dummy_tokens)
            logits = self._extract_logits(output)
            mx.eval(logits)
            logger.info("Model warm-up complete")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def _make_sampling_metadata(
        self,
        sampling_params_list: list[SamplingParams],
        prompt_token_id_lists: list[list[int]],
        output_token_id_lists: list[list[int]],
        generators: dict[int, torch.Generator] | None = None,
    ) -> SamplingMetadata:
        """Create SamplingMetadata from per-request SamplingParams.

        Args:
            sampling_params_list: List of SamplingParams, one per request
            prompt_token_id_lists: Prompt token IDs per request (prefix used for
                repetition penalty).
            output_token_id_lists: Generated token IDs per request (used for
                presence/frequency penalties, and also repetition penalty).
            generators: Optional per-request torch generators keyed by batch index.
                If omitted, sampler falls back to the global RNG for those entries.

        Returns:
            SamplingMetadata for the batch
        """
        batch_size = len(sampling_params_list)
        if len(prompt_token_id_lists) != batch_size:
            raise ValueError(
                "Expected prompt token ids for each request in the batch "
                f"(len(prompt_token_id_lists)={len(prompt_token_id_lists)} "
                f"!= batch_size={batch_size})."
            )
        if len(output_token_id_lists) != batch_size:
            raise ValueError(
                "Expected output token ids for each request in the batch "
                f"(len(output_token_id_lists)={len(output_token_id_lists)} "
                f"!= batch_size={batch_size})."
            )

        # Determine sampling mode
        all_greedy = all(sp.temperature < 1e-5 for sp in sampling_params_list)
        all_random = not all_greedy and all(
            sp.temperature >= 1e-5 for sp in sampling_params_list
        )

        # Check if any penalties are applied
        no_penalties = all(
            sp.frequency_penalty == 0
            and sp.presence_penalty == 0
            and sp.repetition_penalty == 1.0
            for sp in sampling_params_list
        )

        generators = generators or {}

        # top_k: pass None if all values indicate no filtering
        # -1 = vLLM default (no filtering), 0 = OpenAI API convention (no filtering)
        # vLLM's sampler expects None to skip top-k entirely
        top_k_values = [sp.top_k for sp in sampling_params_list]
        top_k = (
            None
            if all(k <= 0 for k in top_k_values)
            else torch.tensor(top_k_values, dtype=torch.int32, device=self.device)
        )

        # top_p: pass None if all values are 1.0 (no filtering)
        # vLLM's sampler expects None to skip top-p entirely
        top_p_values = [sp.top_p for sp in sampling_params_list]
        top_p = (
            None
            if all(p == 1.0 for p in top_p_values)
            else torch.tensor(top_p_values, dtype=torch.float32, device=self.device)
        )

        vocab_size = self.model_args.get("vocab_size", 32000)
        prompt_token_ids_tensor = None
        if not no_penalties:
            prompt_token_ids_tensor = make_tensor_with_pad(
                prompt_token_id_lists,
                pad=vocab_size,
                device=self.device,
                dtype=torch.int64,
                pin_memory=False,
            )

        return SamplingMetadata(
            temperature=None
            if all_greedy
            else torch.tensor(
                [sp.temperature for sp in sampling_params_list],
                dtype=torch.float32,
                device=self.device,
            ),
            all_greedy=all_greedy,
            all_random=all_random,
            top_p=top_p,
            top_k=top_k,
            generators=generators,
            max_num_logprobs=None,
            prompt_token_ids=prompt_token_ids_tensor,
            output_token_ids=output_token_id_lists,
            frequency_penalties=torch.tensor(
                [sp.frequency_penalty for sp in sampling_params_list],
                dtype=torch.float32,
                device=self.device,
            ),
            presence_penalties=torch.tensor(
                [sp.presence_penalty for sp in sampling_params_list],
                dtype=torch.float32,
                device=self.device,
            ),
            repetition_penalties=torch.tensor(
                [sp.repetition_penalty for sp in sampling_params_list],
                dtype=torch.float32,
                device=self.device,
            ),
            no_penalties=no_penalties,
            allowed_token_ids_mask=None,
            bad_words_token_ids={},
            logitsprocs=LogitsProcessors(),
        )

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
        # Create a new prompt cache for this request
        # For VLMs, use the language_model component; for text models, use model directly
        cache_model = (
            self.model.language_model
            if self._is_vlm and hasattr(self.model, "language_model")
            else self.model
        )
        cache = make_prompt_cache(cache_model)

        # Prefill: process the entire prompt with cache
        input_ids = mx.array([token_ids], dtype=mx.int32)
        model_output = self.model(input_ids, cache=cache)
        logits = self._extract_logits(model_output)

        # Extract last token logits
        last_logits = logits[:, -1, :]

        # Use native MLX greedy sampling when possible (avoids PyTorch round-trip)
        is_greedy = sampling_params.temperature < 1e-5
        needs_advanced_sampling = (
            sampling_params.top_k > 0
            or sampling_params.top_p < 1.0
            or sampling_params.frequency_penalty != 0
            or sampling_params.presence_penalty != 0
            or sampling_params.repetition_penalty != 1.0
        )

        if is_greedy and not needs_advanced_sampling:
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

        # Use Rust extension for efficient batch token retrieval if available
        if self._rust_state_manager is not None:
            last_tokens = self._rust_state_manager.get_last_tokens_batch(
                [req_id for req_id, _ in decode_reqs]
            )
        else:
            last_tokens = [
                state.token_ids[-1] if state.token_ids else 0
                for _, state in decode_reqs
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

        # Check if all requests can use fast greedy sampling
        all_greedy = all(sp.temperature < 1e-5 for sp in sampling_params_list)
        any_advanced = any(
            sp.top_k > 0
            or sp.top_p < 1.0
            or sp.frequency_penalty != 0
            or sp.presence_penalty != 0
            or sp.repetition_penalty != 1.0
            for sp in sampling_params_list
        )

        if all_greedy and not any_advanced:
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
        for i, (req_id, state) in enumerate(decode_reqs):
            state.cache = _extract_kv_cache(batch_cache, i)
            state.token_ids.append(next_tokens[i])
            state.generated_tokens += 1

            # Update Rust state manager if available
            if self._rust_state_manager is not None:
                self._rust_state_manager.append_token(req_id, next_tokens[i])

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

        for req_id, state in decode_reqs:
            last_token = state.token_ids[-1] if state.token_ids else 0
            input_ids = mx.array([[last_token]], dtype=mx.int32)

            model_output = self.model(input_ids, cache=state.cache)
            logits = self._extract_logits(model_output)
            last_logits = logits[:, -1, :]

            # Use native MLX greedy sampling when possible
            sp = state.sampling_params
            is_greedy = sp.temperature < 1e-5
            needs_advanced = (
                sp.top_k > 0
                or sp.top_p < 1.0
                or sp.frequency_penalty != 0
                or sp.presence_penalty != 0
                or sp.repetition_penalty != 1.0
            )

            if is_greedy and not needs_advanced:
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

            # Update Rust state manager if available
            if self._rust_state_manager is not None:
                self._rust_state_manager.append_token(req_id, next_token)

        return next_tokens

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        """Execute model inference with true batched decode.

        Key optimization: Uses BatchKVCache.merge() to combine individual
        KV caches and run a SINGLE forward pass for all decode requests.

        Args:
            scheduler_output: Scheduler output with batch information

        Returns:
            Model runner output with generated tokens
        """
        if self.model is None:
            raise RuntimeError("Model not loaded")

        # Collect all requests to process
        req_ids: list[str] = []
        req_id_to_index: dict[str, int] = {}
        sampled_tokens: list[list[int]] = []

        # === PHASE 1: Process new requests (prefill phase) ===
        new_reqs = scheduler_output.scheduled_new_reqs

        for new_req in new_reqs:
            req_id = new_req.req_id
            token_ids = new_req.prompt_token_ids or []
            sampling_params = new_req.sampling_params or SamplingParams()

            req_ids.append(req_id)
            req_id_to_index[req_id] = len(req_ids) - 1

            if token_ids:
                generator = _create_request_generator(self.device, sampling_params)
                next_token, cache = self._prefill_single(
                    req_id,
                    token_ids,
                    sampling_params,
                    generator=generator,
                )
                sampled_tokens.append([next_token])

                # Store request state with cache for future decoding
                self._request_states[req_id] = RequestState(
                    token_ids=list(token_ids) + [next_token],
                    prompt_len=len(token_ids),
                    cache=cache,
                    sampling_params=sampling_params,
                    generator=generator,
                    generated_tokens=1,
                )

                # Register with Rust state manager if available
                if self._rust_state_manager is not None:
                    self._rust_state_manager.add_request(
                        req_id, list(token_ids) + [next_token]
                    )
            else:
                sampled_tokens.append([0])  # Fallback

        # === PHASE 2: Process cached requests (TRUE batched decode) ===
        cached_reqs = scheduler_output.scheduled_cached_reqs
        decode_req_ids = list(cached_reqs.req_ids)

        if decode_req_ids:
            # Collect all valid decode requests
            valid_decode_reqs: list[tuple[str, RequestState]] = []
            for req_id in decode_req_ids:
                state = self._request_states.get(req_id)
                if state is not None:
                    valid_decode_reqs.append((req_id, state))

            if valid_decode_reqs:
                # Use batched decode for multiple requests, sequential for single
                if len(valid_decode_reqs) >= _MIN_BATCH_SIZE_FOR_BATCHING:
                    decode_tokens = self._batched_decode(valid_decode_reqs)
                else:
                    decode_tokens = self._sequential_decode(valid_decode_reqs)

                # Add decode results to output
                for i, (req_id, _) in enumerate(valid_decode_reqs):
                    req_ids.append(req_id)
                    req_id_to_index[req_id] = len(req_ids) - 1
                    sampled_tokens.append([decode_tokens[i]])

            # Handle requests with no cached state (edge case)
            for req_id in decode_req_ids:
                if req_id not in req_id_to_index:
                    req_ids.append(req_id)
                    req_id_to_index[req_id] = len(req_ids) - 1
                    sampled_tokens.append([0])

        # === PHASE 3: Clean up finished requests ===
        if scheduler_output.finished_req_ids:
            for req_id in scheduler_output.finished_req_ids:
                state = self._request_states.pop(req_id, None)
                if state is not None:
                    del state.cache
                    del state

                # Remove from Rust state manager if available
                if self._rust_state_manager is not None:
                    self._rust_state_manager.remove_request(req_id)

            # Lazy cache clearing - only clear periodically to avoid sync overhead
            self._finished_request_count += len(scheduler_output.finished_req_ids)
            if self._finished_request_count >= _CACHE_CLEAR_INTERVAL:
                mx.clear_cache()
                self._finished_request_count = 0

        # Handle empty case
        if not req_ids:
            return ModelRunnerOutput(
                req_ids=[],
                req_id_to_index={},
                sampled_token_ids=[],
                logprobs=None,
                prompt_logprobs_dict={},
                pooler_output=[],
            )

        return ModelRunnerOutput(
            req_ids=req_ids,
            req_id_to_index=req_id_to_index,
            sampled_token_ids=sampled_tokens,
            logprobs=None,
            prompt_logprobs_dict={},
            pooler_output=[None] * len(req_ids),
        )

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
            if temperature < 1e-5:
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
