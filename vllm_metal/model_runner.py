# SPDX-License-Identifier: Apache-2.0
"""Metal Model Runner for vLLM inference."""

import logging
from typing import TYPE_CHECKING, Any

import mlx.core as mx
from mlx_lm import load as mlx_load
from mlx_lm import stream_generate
from mlx_lm.sample_utils import make_sampler

from vllm_metal.config import get_config
from vllm_metal.mlx_backend.cache import PagedKVCache

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = logging.getLogger(__name__)


class MetalModelRunner:
    """Model runner for MLX-based inference on Metal.

    Handles model loading, KV cache management, and inference execution.
    """

    def __init__(self, vllm_config: "VllmConfig"):
        """Initialize model runner.

        Args:
            vllm_config: vLLM configuration
        """
        self.vllm_config = vllm_config
        self.config = get_config()

        self.model: Any = None
        self.tokenizer: Any = None
        self.model_config: dict[str, Any] | None = None
        self.kv_cache: PagedKVCache | None = None

    def load_model(self) -> None:
        """Load the model using MLX."""
        model_config = self.vllm_config.model_config
        model_name = model_config.model

        logger.info(f"Loading model: {model_name}")

        # Load model and tokenizer using mlx_lm
        self.model, self.tokenizer = mlx_load(
            model_name,
            tokenizer_config={"trust_remote_code": True},
        )

        # Extract model configuration
        if hasattr(self.model, "config"):
            self.model_config = self.model.config
        elif hasattr(self.model, "args"):
            self.model_config = vars(self.model.args)
        else:
            # Fallback: try to get from model attributes
            self.model_config = {
                "num_hidden_layers": getattr(self.model, "n_layers", 32),
                "num_attention_heads": getattr(self.model, "n_heads", 32),
                "num_key_value_heads": getattr(
                    self.model, "n_kv_heads", getattr(self.model, "n_heads", 32)
                ),
                "hidden_size": getattr(self.model, "dim", 4096),
            }

        logger.info(f"Model loaded: {model_name}")
        if self.config.debug:
            logger.info(f"Model config: {self.model_config}")

    def initialize_cache(self, num_blocks: int) -> None:
        """Initialize the KV cache.

        Args:
            num_blocks: Number of cache blocks
        """
        if self.model_config is None:
            msg = "Model must be loaded before initializing cache"
            raise RuntimeError(msg)

        config = self.model_config
        num_layers = config.get("num_hidden_layers", 32)
        num_kv_heads = config.get(
            "num_key_value_heads", config.get("num_attention_heads", 32)
        )
        hidden_size = config.get("hidden_size", 4096)
        num_heads = config.get("num_attention_heads", 32)
        head_dim = hidden_size // num_heads

        self.kv_cache = PagedKVCache(
            num_layers=num_layers,
            num_kv_heads=num_kv_heads,
            head_dim=head_dim,
            num_blocks=num_blocks,
            block_size=self.config.block_size,
            dtype=mx.float16,
        )

        logger.info(
            f"KV cache initialized: {num_blocks} blocks, "
            f"{num_layers} layers, {num_kv_heads} kv_heads"
        )

    def _prepare_inputs(
        self, seq_group_metadata_list: list[Any]
    ) -> tuple[mx.array, mx.array]:
        """Prepare input tensors from sequence group metadata.

        Args:
            seq_group_metadata_list: List of sequence group metadata

        Returns:
            Tuple of (input_ids, positions) as MLX arrays
        """
        input_ids_list: list[int] = []
        positions_list: list[int] = []

        for seq_group in seq_group_metadata_list:
            if hasattr(seq_group, "seq_data"):
                for _seq_id, seq_data in seq_group.seq_data.items():
                    if hasattr(seq_data, "get_token_ids"):
                        tokens = seq_data.get_token_ids()
                    else:
                        tokens = seq_data.get("token_ids", [])

                    # For prefill, use all tokens; for decode, use last token
                    if seq_group.is_prompt:
                        input_ids_list.extend(tokens)
                        positions_list.extend(range(len(tokens)))
                    else:
                        input_ids_list.append(tokens[-1])
                        positions_list.append(len(tokens) - 1)

        if not input_ids_list:
            return mx.array([[]], dtype=mx.int32), mx.array([[]], dtype=mx.int32)

        input_ids = mx.array([input_ids_list], dtype=mx.int32)
        positions = mx.array([positions_list], dtype=mx.int32)

        return input_ids, positions

    def execute_model(
        self,
        seq_group_metadata_list: list[Any] | None = None,
        **kwargs: Any,
    ) -> list[Any]:
        """Execute model inference.

        Args:
            seq_group_metadata_list: Sequence group metadata
            **kwargs: Additional arguments

        Returns:
            List of sampler outputs
        """
        if self.model is None:
            msg = "Model not loaded"
            raise RuntimeError(msg)

        if seq_group_metadata_list is None or len(seq_group_metadata_list) == 0:
            return []

        # Prepare inputs
        input_ids, positions = self._prepare_inputs(seq_group_metadata_list)

        if input_ids.size == 0:
            return []

        # Run inference using MLX
        try:
            logits = self.model(input_ids)
            mx.eval(logits)

            # Get next token predictions
            # Take the last position's logits for each sequence
            next_token_logits = logits[:, -1, :]

            # Simple greedy sampling for now
            next_tokens = mx.argmax(next_token_logits, axis=-1)
            mx.eval(next_tokens)

            # Convert to output format
            outputs = []
            for i, seq_group in enumerate(seq_group_metadata_list):
                token_id = (
                    int(next_tokens[0].item())
                    if i == 0
                    else int(next_tokens[min(i, next_tokens.shape[0] - 1)].item())
                )
                outputs.append(
                    {
                        "seq_group": seq_group,
                        "token_id": token_id,
                        "logprobs": None,
                    }
                )

            return outputs

        except Exception as e:
            logger.error(f"Model execution failed: {e}")
            raise

    def generate(
        self,
        prompt: str,
        max_tokens: int = 100,
        temperature: float = 0.0,
        **kwargs: Any,
    ) -> str:
        """Generate text from a prompt.

        This is a simplified interface for direct text generation.

        Args:
            prompt: Input prompt
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature (0 = greedy)
            **kwargs: Additional generation parameters

        Returns:
            Generated text
        """
        if self.model is None or self.tokenizer is None:
            msg = "Model and tokenizer must be loaded"
            raise RuntimeError(msg)

        # Generate tokens using stream_generate
        segments: list[str] = []

        # Create sampler with temperature
        sampler = make_sampler(temp=temperature)

        for response in stream_generate(
            self.model,
            self.tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            # Accumulate incremental text from each token
            segments.append(response.text)

        return "".join(segments)

    def __del__(self) -> None:
        """Cleanup model resources."""
        self.model = None
        self.tokenizer = None
        self.kv_cache = None
