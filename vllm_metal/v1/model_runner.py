#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Metal Model Runner for vLLM v1 engine (patched for hybrid caches).

Optimized for performance with:
- True batched decode using BatchKVCache for O(1) forward passes per batch
- Async evaluation pipeline for pipelined computation
- Pre-allocated input buffers to reduce allocation overhead
- Rust-based token state management for efficient batch operations

This patched version adds support for hybrid models that mix attention (KVCache)
and Mamba/SSM layers (MambaCache). It keeps the original fast path for
attention layers via BatchKVCache.merge while batching Mamba caches by
concatenating their internal state across the batch dimension.
"""

import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import torch
from mlx_lm import load as mlx_load
from mlx_lm import stream_generate
from mlx_lm.models.cache import (
    BatchKVCache,
    KVCache,
    MambaCache,
    make_prompt_cache,
)
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput

from vllm_metal.config import get_config

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
    generated_tokens: int = 0


def _merge_kv_caches(caches_list: list[list[KVCache]]) -> list[Any]:
    """Merge multiple per-request caches into batched caches.

    This patched version supports mixed per-layer cache types:
    - Attention layers (KVCache) are merged using BatchKVCache.merge.
    - Mamba/SSM layers (MambaCache/ArraysCache) are batched by concatenating
      internal state arrays along the batch dimension.

    Args:
        caches_list: List of per-request caches, each is a list of per-layer caches

    Returns:
        List of batched caches (BatchKVCache or MambaCache), one per layer
    """
    if not caches_list:
        return []

    num_layers = len(caches_list[0])
    merged: list[Any] = []

    for layer_idx in range(num_layers):
        # Gather this layer's caches across requests
        layer_caches = [caches[layer_idx] for caches in caches_list]
        non_none = [c for c in layer_caches if c is not None]

        if not non_none:
            merged.append(None)
            continue

        first = non_none[0]

        # Attention KV cache branch: has keys/values
        if hasattr(first, "keys") and hasattr(first, "values"):
            merged.append(BatchKVCache.merge(non_none))
            continue

        # Mamba/ArraysCache branch: build a batched MambaCache by stacking along batch dim
        if isinstance(first, MambaCache) or hasattr(first, "cache"):
            # Determine number of slots from first cache's state
            slots = len(first.state) if hasattr(first, "state") else len(first.cache)  # type: ignore[attr-defined]
            batched = MambaCache()
            batched_list: list[Any] = []

            for i in range(slots):
                comps: list[Any] = []
                ref = None
                for c in layer_caches:
                    if c is None:
                        comps.append(None)
                        continue
                    comp = (c.state if hasattr(c, "state") else c.cache)[i]  # type: ignore[attr-defined]
                    comps.append(comp)
                    if ref is None and comp is not None:
                        ref = comp

                if ref is None:
                    batched_list.append(None)
                else:
                    filled = [p if p is not None else mx.zeros_like(ref) for p in comps]
                    batched_list.append(mx.concatenate(filled, axis=0))

            batched.state = batched_list  # type: ignore[assignment]
            merged.append(batched)
            continue

        # Unknown cache type: leave unmerged
        merged.append(None)

    return merged


def _extract_kv_cache(batch_caches: list[Any], idx: int) -> list[Any]:
    """Extract a single request's cache from batched caches.

    Attention layers use BatchKVCache.extract().
    Mamba layers slice the batched state along the batch dimension.

    Args:
        batch_caches: List of batched caches, one per layer
        idx: Index of the request in the batch

    Returns:
        List of per-layer caches for a single request
    """
    extracted: list[Any] = []
    for bc in batch_caches:
        if bc is None:
            extracted.append(None)
            continue
        # BatchKVCache supports extract
        if hasattr(bc, "extract"):
            try:
                extracted.append(bc.extract(idx))
                continue
            except Exception:
                pass
        # Mamba path: rebuild a per-request MambaCache
        if isinstance(bc, MambaCache) or hasattr(bc, "cache"):
            mc = MambaCache()
            src_list = bc.state if hasattr(bc, "state") else bc.cache  # type: ignore[attr-defined]
            out_list: list[Any] = []
            for comp in src_list:
                if comp is None:
                    out_list.append(None)
                else:
                    out_list.append(mx.contiguous(comp[idx : idx + 1]))
            mc.state = out_list  # type: ignore[assignment]
            extracted.append(mc)
            continue
        extracted.append(None)
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

    def _prefill_single(
        self, req_id: str, token_ids: list[int]
    ) -> tuple[int, list[KVCache]]:
        """Process a single prefill request.

        Args:
            req_id: Request ID
            token_ids: Prompt token IDs

        Returns:
            Tuple of (next_token, cache)
        """
        # Create a new prompt cache for this request
        cache = make_prompt_cache(self.model)

        # Prefill: process the entire prompt with cache
        input_ids = mx.array([token_ids], dtype=mx.int32)
        logits = self.model(input_ids, cache=cache)

        # Get next token (greedy sampling)
        next_token_logits = logits[:, -1, :]
        next_token = int(mx.argmax(next_token_logits, axis=-1)[0].item())

        # Evaluate to materialize cache state
        mx.eval(logits)
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

        # Merge individual caches into batched caches (one per layer)
        batch_cache = _merge_kv_caches(caches_list)

        # Create batched input: shape (batch_size, 1) for single-token decode
        batched_input = mx.array(last_tokens, dtype=mx.int32)[:, None]

        # === SINGLE FORWARD PASS FOR ALL REQUESTS ===
        logits = self.model(batched_input, cache=batch_cache)

        # Evaluate to materialize results
        mx.eval(logits)

        # Extract next tokens (greedy sampling)
        next_token_logits = logits[:, -1, :]  # Shape: (batch_size, vocab_size)
        next_tokens_arr = mx.argmax(next_token_logits, axis=-1)
        mx.eval(next_tokens_arr)
        next_tokens = [int(next_tokens_arr[i].item()) for i in range(batch_size)]

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

            next_token_logits = logits[:, -1, :]
            next_token = int(mx.argmax(next_token_logits, axis=-1)[0].item())
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

            req_ids.append(req_id)
            req_id_to_index[req_id] = len(req_ids) - 1

            if token_ids:
                next_token, cache = self._prefill_single(req_id, token_ids)
                sampled_tokens.append([next_token])

                # Store request state with cache for future decoding
                self._request_states[req_id] = RequestState(
                    token_ids=list(token_ids) + [next_token],
                    cache=cache,
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

