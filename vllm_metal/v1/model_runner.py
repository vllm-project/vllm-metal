# SPDX-License-Identifier: Apache-2.0
"""Metal Model Runner for vLLM v1 engine."""

import time
from dataclasses import dataclass
from typing import Any

import mlx.core as mx
import torch
from mlx_lm import load as mlx_load
from mlx_lm import stream_generate
from vllm.config import VllmConfig
from vllm.logger import init_logger
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheConfig, KVCacheSpec
from vllm.v1.outputs import ModelRunnerOutput

from vllm_metal.config import get_config

logger = init_logger(__name__)


@dataclass
class SamplerOutput:
    """Output from the sampler."""

    token_ids: list[int]
    logprobs: list[float] | None = None


class MetalModelRunner:
    """Model runner for MLX-based inference on Metal.

    Implements the vLLM v1 model runner interface for Apple Silicon.
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
        head_size = self.model_args.get("head_dim") or (hidden_size // num_attention_heads)
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
        head_dim = self.model_args.get("head_dim") or (hidden_size // num_attention_heads)
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

    def execute_model(
        self, scheduler_output: SchedulerOutput
    ) -> ModelRunnerOutput | None:
        """Execute model inference.

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
        num_tokens = 0

        # Track request state for cached requests (ongoing generation)
        if not hasattr(self, "_request_cache"):
            self._request_cache: dict[str, list[int]] = {}

        # Process new requests
        for new_req in scheduler_output.scheduled_new_reqs:
            req_id = new_req.req_id
            token_ids = new_req.prompt_token_ids or []
            num_scheduled = scheduler_output.num_scheduled_tokens.get(
                req_id, len(token_ids)
            )

            # Cache the token ids for this request
            self._request_cache[req_id] = list(token_ids)

            req_ids.append(req_id)
            req_id_to_index[req_id] = len(req_ids) - 1
            num_tokens += num_scheduled

            # Run forward pass on prompt
            if token_ids:
                input_ids = mx.array([token_ids], dtype=mx.int32)
                logits = self.model(input_ids)
                mx.eval(logits)

                # Get next token (greedy sampling for now)
                next_token_logits = logits[:, -1, :]
                next_token = int(mx.argmax(next_token_logits, axis=-1)[0].item())
                sampled_tokens.append([next_token])

                # Add generated token to cache
                self._request_cache[req_id].append(next_token)
            else:
                sampled_tokens.append([0])  # Fallback

        # Process cached requests (continuation)
        cached_reqs = scheduler_output.scheduled_cached_reqs
        for idx, req_id in enumerate(cached_reqs.req_ids):
            # Get cached tokens and new tokens
            cached_tokens = self._request_cache.get(req_id, [])
            new_tokens = (
                cached_reqs.new_token_ids[idx] if cached_reqs.new_token_ids else []
            )

            # Update cache with new tokens
            if new_tokens:
                cached_tokens.extend(new_tokens)
                self._request_cache[req_id] = cached_tokens

            num_scheduled = scheduler_output.num_scheduled_tokens.get(req_id, 1)

            req_ids.append(req_id)
            req_id_to_index[req_id] = len(req_ids) - 1
            num_tokens += num_scheduled

            # Run forward pass
            if cached_tokens:
                input_ids = mx.array([cached_tokens], dtype=mx.int32)
                logits = self.model(input_ids)
                mx.eval(logits)

                # Get next token
                next_token_logits = logits[:, -1, :]
                next_token = int(mx.argmax(next_token_logits, axis=-1)[0].item())
                sampled_tokens.append([next_token])

                # Add to cache
                self._request_cache[req_id].append(next_token)
            else:
                sampled_tokens.append([0])

        # Clean up finished requests
        for req_id in scheduler_output.finished_req_ids:
            self._request_cache.pop(req_id, None)

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
