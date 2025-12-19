# SPDX-License-Identifier: Apache-2.0
"""Weight converter for loading HuggingFace models into MLX format."""

import logging
from pathlib import Path
from typing import Any

import mlx.core as mx
from safetensors import safe_open
from transformers import AutoConfig

from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx

logger = logging.getLogger(__name__)


class WeightConverter:
    """Converts HuggingFace model weights to MLX format."""

    # Standard weight name mappings from HF to MLX convention
    WEIGHT_MAP: dict[str, str] = {
        "model.embed_tokens.weight": "embed_tokens.weight",
        "model.norm.weight": "norm.weight",
        "lm_head.weight": "lm_head.weight",
    }

    # Layer-specific patterns
    LAYER_PATTERNS: dict[str, str] = {
        "model.layers.{}.self_attn.q_proj.weight": "layers.{}.self_attn.q_proj.weight",
        "model.layers.{}.self_attn.k_proj.weight": "layers.{}.self_attn.k_proj.weight",
        "model.layers.{}.self_attn.v_proj.weight": "layers.{}.self_attn.v_proj.weight",
        "model.layers.{}.self_attn.o_proj.weight": "layers.{}.self_attn.o_proj.weight",
        "model.layers.{}.mlp.gate_proj.weight": "layers.{}.mlp.gate_proj.weight",
        "model.layers.{}.mlp.up_proj.weight": "layers.{}.mlp.up_proj.weight",
        "model.layers.{}.mlp.down_proj.weight": "layers.{}.mlp.down_proj.weight",
        "model.layers.{}.input_layernorm.weight": "layers.{}.input_layernorm.weight",
        "model.layers.{}.post_attention_layernorm.weight": "layers.{}.post_attention_layernorm.weight",
    }

    def __init__(self, model_path: str | Path, dtype: mx.Dtype = mx.float16):
        """Initialize weight converter.

        Args:
            model_path: Path to HuggingFace model directory or model name
            dtype: Target data type for weights
        """
        self.model_path = (
            Path(model_path) if isinstance(model_path, str) else model_path
        )
        self.dtype = dtype
        self.config: dict[str, Any] = {}

    def load_config(self) -> dict[str, Any]:
        """Load model configuration.

        Returns:
            Model configuration dictionary
        """
        try:
            hf_config = AutoConfig.from_pretrained(
                str(self.model_path), trust_remote_code=True
            )
            self.config = hf_config.to_dict()
        except Exception as e:
            logger.warning(f"Could not load HF config: {e}")
            self.config = {}

        return self.config

    def convert_weight_name(self, hf_name: str) -> str:
        """Convert HuggingFace weight name to MLX convention.

        Args:
            hf_name: HuggingFace weight name

        Returns:
            MLX weight name
        """
        # Check direct mappings first
        if hf_name in self.WEIGHT_MAP:
            return self.WEIGHT_MAP[hf_name]

        # Check layer patterns
        for hf_pattern, mlx_pattern in self.LAYER_PATTERNS.items():
            # Extract layer index
            import re

            hf_regex = hf_pattern.replace("{}", r"(\d+)")
            match = re.match(hf_regex, hf_name)
            if match:
                layer_idx = match.group(1)
                return mlx_pattern.format(layer_idx)

        # Default: return original name
        return hf_name

    def convert_weights(self, weights: dict[str, Any]) -> dict[str, mx.array]:
        """Convert a dictionary of weights to MLX format.

        Args:
            weights: Dictionary mapping weight names to tensors

        Returns:
            Dictionary mapping converted names to MLX arrays
        """
        import torch

        mlx_weights: dict[str, mx.array] = {}

        for name, tensor in weights.items():
            mlx_name = self.convert_weight_name(name)

            if isinstance(tensor, torch.Tensor):
                mlx_tensor = torch_to_mlx(tensor)
            else:
                mlx_tensor = mx.array(tensor)

            # Convert to target dtype
            if mlx_tensor.dtype != self.dtype:
                mlx_tensor = mlx_tensor.astype(self.dtype)

            mlx_weights[mlx_name] = mlx_tensor
            logger.debug(f"Converted {name} -> {mlx_name}: {mlx_tensor.shape}")

        return mlx_weights

    def load_safetensors(self) -> dict[str, mx.array]:
        """Load weights from safetensors files.

        Returns:
            Dictionary mapping weight names to MLX arrays
        """
        weights: dict[str, mx.array] = {}
        safetensor_files = list(self.model_path.glob("*.safetensors"))

        if not safetensor_files:
            msg = f"No safetensors files found in {self.model_path}"
            raise FileNotFoundError(msg)

        for sf_file in safetensor_files:
            logger.info(f"Loading weights from {sf_file.name}")
            with safe_open(sf_file, framework="numpy") as f:
                for key in f.keys():
                    tensor = f.get_tensor(key)
                    mlx_name = self.convert_weight_name(key)
                    mlx_tensor = mx.array(tensor)

                    if mlx_tensor.dtype != self.dtype:
                        mlx_tensor = mlx_tensor.astype(self.dtype)

                    weights[mlx_name] = mlx_tensor

        return weights


def load_hf_weights(
    model_name_or_path: str,
    dtype: mx.Dtype = mx.float16,
) -> tuple[dict[str, mx.array], dict[str, Any]]:
    """Load HuggingFace model weights and config.

    This is a convenience function that handles both local and remote models.

    Args:
        model_name_or_path: HuggingFace model name or local path
        dtype: Target data type for weights

    Returns:
        Tuple of (weights_dict, config_dict)
    """
    from huggingface_hub import snapshot_download

    # Download model if it's a model name
    model_path = Path(model_name_or_path)
    if not model_path.exists():
        logger.info(f"Downloading model {model_name_or_path}")
        model_path = Path(
            snapshot_download(
                model_name_or_path,
                allow_patterns=["*.safetensors", "*.json", "tokenizer*"],
            )
        )

    converter = WeightConverter(model_path, dtype=dtype)
    config = converter.load_config()
    weights = converter.load_safetensors()

    return weights, config
