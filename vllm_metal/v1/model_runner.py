# SPDX-License-Identifier: Apache-2.0
"""Metal Model Runner for vLLM v1 engine.

Optimized for performance with:
- True batched decode using BatchKVCache for O(1) forward passes per batch
- Async evaluation pipeline for pipelined computation
- Pre-allocated input buffers to reduce allocation overhead
- Rust-based token state management for efficient batch operations
"""

import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import torch
from mlx_lm import load as mlx_load
from mlx_lm import stream_generate
from mlx_lm.models.cache import BatchKVCache, KVCache, make_prompt_cache
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

logger = init_logger(__name__)

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


@dataclass
class SamplerOutput:
    """Output from the sampler."""

    token_ids: list[int]
    logprobs: list[float] | None = None


@dataclass
class RequestState:
    """State for an ongoing request with KV cache."""

    token_ids: list[int]
    cache: list[KVCache]  # Per-layer KV caches
    sampling_params: SamplingParams  # Sampling parameters for this request
    generated_tokens: int = 0


def _merge_kv_caches(caches_list: list[list[KVCache]]) -> list[BatchKVCache]:
    """Merge multiple per-request caches into batched caches.

    Args:
        caches_list: List of per-request caches, each is a list of per-layer KVCache

    Returns:
        List of BatchKVCache, one per layer
    """
    if not caches_list:
        return []

    num_layers = len(caches_list[0])
    merged = []

    for layer_idx in range(num_layers):
        layer_caches = [caches[layer_idx] for caches in caches_list]
        batch_cache = BatchKVCache.merge(layer_caches)
        merged.append(batch_cache)

    return merged


def _extract_kv_cache(batch_caches: list[BatchKVCache], idx: int) -> list[KVCache]:
    """Extract a single request's cache from batched caches.

    Args:
        batch_caches: List of BatchKVCache, one per layer
        idx: Index of the request in the batch

    Returns:
        List of KVCache for the request, one per layer
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

    def load_model(self) -> None:
        """Load the model using MLX."""
        model_name = self.model_config.model

        logger.info(f"Loading model: {model_name}")
        start_time = time.time()

        # Load model and tokenizer using mlx_lm
        self.model, self.tokenizer = mlx_load(
            model_name,
            tokenizer_config={"trust_remote_code": self.model_config.trust_remote_code},
        )

        # Extract model configuration
        if hasattr(self.model, "args"):
            self.model_args = vars(self.model.args)
        elif hasattr(self.model, "config"):
            if hasattr(self.model.config, "to_dict"):
                self.model_args = self.model.config.to_dict()
            else:
                self.model_args = vars(self.model.config)
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

        load_time = time.time() - start_time
        logger.info(f"Model loaded in {load_time:.2f}s: {model_name}")
        if self.metal_config.debug:
            logger.info(f"Model args: {self.model_args}")

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
            _ = self.model(dummy_tokens)
            mx.eval(_)
            logger.info("Model warm-up complete")
        except Exception as e:
            logger.warning(f"Model warm-up failed: {e}")

    def _make_sampling_metadata(
        self,
        sampling_params_list: list[SamplingParams],
        output_token_ids: list[list[int]],
    ) -> SamplingMetadata:
        """Create SamplingMetadata from per-request SamplingParams.

        Args:
            sampling_params_list: List of SamplingParams, one per request
            output_token_ids: List of output token IDs per request (for penalties)

        Returns:
            SamplingMetadata for the batch
        """
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

        # Create generators for random sampling
        generators = {}
        for i, sp in enumerate(sampling_params_list):
            if sp.temperature >= 1e-5:
                gen = torch.Generator()
                if sp.seed is not None:
                    gen.manual_seed(sp.seed)
                generators[i] = gen

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

        # Create empty prompt_token_ids tensor to satisfy vLLM's assertion
        # Use make_tensor_with_pad to match vLLM's expected format
        # Pass empty lists and set pin_memory=False to avoid device mismatch
        batch_size = len(sampling_params_list)
        vocab_size = self.model_args.get("vocab_size", 32000)
        empty_prompt_lists = [[] for _ in range(batch_size)]
        prompt_token_ids = make_tensor_with_pad(
            empty_prompt_lists,
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
            prompt_token_ids=prompt_token_ids,
            output_token_ids=output_token_ids,
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
        cache = make_prompt_cache(self.model)

        # Prefill: process the entire prompt with cache
        input_ids = mx.array([token_ids], dtype=mx.int32)
        logits = self.model(input_ids, cache=cache)

        # Evaluate to materialize results before conversion
        mx.eval(logits)

        # Convert MLX logits to torch and sample using vLLM's Sampler
        # Cast to float32 for numpy conversion (numpy doesn't support bfloat16)
        logits_torch = mlx_to_torch(logits[:, -1, :].astype(mx.float32), device=self.device)
        metadata = self._make_sampling_metadata([sampling_params], [[]])
        output = self._sampler.forward(logits_torch, metadata)
        next_token = int(output.sampled_token_ids[0, 0].item())

        # Evaluate to materialize cache state
        mx.eval([c.state for c in cache])

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
        logits = self.model(batched_input, cache=batch_cache)

        # Evaluate to materialize results before conversion
        mx.eval(logits)

        # Extract next tokens using vLLM's Sampler with per-request params
        next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
        sampling_params_list = [state.sampling_params for _, state in decode_reqs]
        output_tokens_list = [state.token_ids for _, state in decode_reqs]

        logits_torch = mlx_to_torch(next_token_logits, device=self.device)
        metadata = self._make_sampling_metadata(
            sampling_params_list, output_tokens_list
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

            logits = self.model(input_ids, cache=state.cache)
            mx.eval(logits)

            # Sample using vLLM's Sampler with request's params
            # Cast to float32 for numpy conversion (numpy doesn't support bfloat16)
            logits_torch = mlx_to_torch(
                logits[:, -1, :].astype(mx.float32), device=self.device
            )
            metadata = self._make_sampling_metadata(
                [state.sampling_params], [state.token_ids]
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
                next_token, cache = self._prefill_single(
                    req_id, token_ids, sampling_params
                )
                sampled_tokens.append([next_token])

                # Store request state with cache for future decoding
                self._request_states[req_id] = RequestState(
                    token_ids=list(token_ids) + [next_token],
                    cache=cache,
                    sampling_params=sampling_params,
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

            # Clear MLX's memory cache after finishing requests
            mx.clear_cache()

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

        # Generate tokens using stream_generate
        generated_text = ""

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            temp=temperature,
        ):
            generated_text = response.text

        return generated_text
