# SPDX-License-Identifier: Apache-2.0
"""Shared encoder checkpoint loading helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import mlx.core as mx
import torch
from huggingface_hub import snapshot_download
from vllm.model_executor.model_loader.weight_utils import download_weights_from_hf

from vllm_metal.tensor_bridge import TORCH_TO_MLX_DTYPE, torch_to_mlx


def load_encoder_checkpoint(model: Any, model_config: Any) -> None:
    """Load and materialize checkpoint weights in an encoder model."""
    target_dtype = TORCH_TO_MLX_DTYPE[model_config.dtype]
    weights = model.sanitize(load_encoder_weights(encoder_model_path(model_config)))
    weights = {
        name: value.astype(target_dtype)
        if mx.issubdtype(value.dtype, mx.floating)
        else value
        for name, value in weights.items()
    }
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())


def encoder_model_path(model_config: Any) -> Path:
    model_path = Path(model_config.model)
    if model_path.exists():
        return model_path
    snapshot = Path(
        snapshot_download(
            repo_id=str(model_config.model),
            revision=model_config.revision,
            allow_patterns=["config.json", "sparse_linear.pt"],
        )
    )
    # Hub snapshot directories are named for their resolved commit.
    return Path(
        download_weights_from_hf(
            str(model_config.model),
            cache_dir=None,
            allow_patterns=["*.safetensors", "pytorch_model*.bin"],
            revision=snapshot.name,
        )
    )


def load_encoder_weights(model_path: Path) -> dict[str, mx.array]:
    weight_files = sorted(model_path.glob("model*.safetensors"))
    if not weight_files:
        weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        weight_files = sorted(model_path.glob("pytorch_model*.bin"))
    if not weight_files:
        raise FileNotFoundError(f"No supported encoder weights found in {model_path}.")

    weights: dict[str, mx.array] = {}
    for weight_file in weight_files:
        weights.update(load_encoder_weight_file(weight_file))
    return weights


def load_encoder_weight_file(weight_file: Path) -> dict[str, mx.array]:
    if weight_file.suffix in (".bin", ".pt"):
        state_dict = torch.load(weight_file, map_location="cpu", weights_only=True)
        return {
            name: torch_to_mlx(value)
            for name, value in state_dict.items()
            if isinstance(value, torch.Tensor)
        }
    return cast(dict[str, mx.array], mx.load(str(weight_file)))
