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
from vllm_metal.utils import get_model_download_path, set_wired_limit

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
        model_name = get_model_download_path(model_config.model)

        logger.info(f"Loading model: {model_name}")
        set_wired_limit()

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
    ) -> tuple[mx.array, mx.array, list[int]]:
        """Prepare batched input tensors from sequence group metadata.

        Args:
            seq_group_metadata_list: List of sequence group metadata

        Returns:
            Tuple of (input_ids, positions, seq_lens)
        """
        input_ids_list: list[list[int]] = []
        seq_lens: list[int] = []

        for seq_group in seq_group_metadata_list:
            try:
                seq_data_map = seq_group.seq_data
            except AttributeError as exc:
                raise ValueError("Sequence group missing seq_data") from exc

            if not seq_data_map:
                raise ValueError("Sequence group has no sequence data")
            if len(seq_data_map) > 1:
                logger.warning(
                    "Sequence group has multiple sequences; using the first one."
                )

            seq_data = next(iter(seq_data_map.values()))
            try:
                tokens = list(seq_data.get_token_ids())
            except AttributeError as exc:
                if isinstance(seq_data, dict):
                    tokens = list(seq_data.get("token_ids", []))
                else:
                    raise ValueError("Sequence data lacks token ids") from exc

            # For prefill, use all tokens; for decode, use last token
            if seq_group.is_prompt:
                input_ids = tokens
            else:
                if not tokens:
                    raise ValueError("Decode sequence has no tokens")
                input_ids = [tokens[-1]]

            if not input_ids:
                raise ValueError("Prompt sequence has no tokens")

            input_ids_list.append(input_ids)
            seq_lens.append(len(input_ids))

        if not input_ids_list:
            empty = mx.array([[]], dtype=mx.int32)
            return empty, empty, []

        max_len = max(seq_lens)
        pad_id = 0
        padded_rows = [row + [pad_id] * (max_len - len(row)) for row in input_ids_list]

        input_ids_array = mx.array(padded_rows, dtype=mx.int32)
        positions = mx.array([list(range(max_len))] * len(padded_rows), dtype=mx.int32)

        return input_ids_array, positions, seq_lens

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
        input_ids, positions, seq_lens = self._prepare_inputs(seq_group_metadata_list)

        if input_ids.size == 0:
            return []

        # Run inference using MLX
        try:
            logits = self.model(input_ids)
            mx.eval(logits)

            # Get next token predictions per sequence using actual lengths
            last_positions = mx.array(
                [length - 1 for length in seq_lens], dtype=mx.int32
            )
            gathered = mx.take_along_axis(logits, last_positions[:, None, None], axis=1)
            next_token_logits = gathered[:, 0, :]

            # Simple greedy sampling for now
            next_tokens = mx.argmax(next_token_logits, axis=-1)
            mx.eval(next_tokens)

            # Convert to output format
            outputs = []
            next_token_list = next_tokens.tolist()
            for seq_group, token_id in zip(
                seq_group_metadata_list, next_token_list, strict=True
            ):
                outputs.append(
                    {
                        "seq_group": seq_group,
                        "token_id": int(token_id),
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
