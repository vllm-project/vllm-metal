# SPDX-License-Identifier: Apache-2.0
"""Load native MLX encoder embedding checkpoints (Apache-2.0)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn

from vllm_metal.v1.encoder.xlm_roberta import XLMRobertaArgs, XLMRobertaModel

_ENCODER_MODEL_TYPES = frozenset({"xlm-roberta", "roberta", "xlm_roberta"})


def load_encoder_model(
    model_name: str | Path,
    *,
    tokenizer_config: dict[str, Any] | None = None,
    lazy: bool = False,
) -> tuple[XLMRobertaModel, Any]:
    """Load an XLM-RoBERTa / RoBERTa encoder + HuggingFace tokenizer."""
    model_path = _resolve_model_path(model_name)
    config = _read_config(model_path)
    model_type = str(config.get("model_type", "")).replace("_", "-")
    if model_type not in {t.replace("_", "-") for t in _ENCODER_MODEL_TYPES}:
        raise ValueError(
            "Native Metal encoder loader currently supports model_type "
            f"'xlm-roberta' / 'roberta'; got {config.get('model_type')!r}."
        )

    args = XLMRobertaArgs.from_dict(config)
    model = XLMRobertaModel(args)
    _load_weights_into_model(model, model_path, config)
    if not lazy:
        mx.eval(model.parameters())

    tokenizer = _load_tokenizer(model_path, tokenizer_config or {})
    return model, tokenizer


def _resolve_model_path(model_path: str | Path) -> Path:
    path = Path(model_path)
    if path.exists():
        return path
    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:  # pragma: no cover
        raise ValueError(
            f"Could not download model {model_path}: huggingface_hub is not installed"
        ) from exc
    try:
        return Path(snapshot_download(repo_id=str(model_path)))
    except OSError as exc:
        raise ValueError(f"Could not download model: {model_path}") from exc


def _read_config(model_path: Path) -> dict[str, Any]:
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")
    with config_path.open() as handle:
        return json.load(handle)


def _load_weights(model_path: Path) -> dict[str, mx.array]:
    weight_files = sorted(model_path.glob("model*.safetensors"))
    if not weight_files:
        weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found in {model_path}")
    weights: dict[str, mx.array] = {}
    for weight_file in weight_files:
        weights.update(mx.load(str(weight_file)))
    return weights


def _load_weights_into_model(
    model: XLMRobertaModel,
    model_path: Path,
    config: dict[str, Any],
) -> None:
    weights = _load_weights(model_path)
    weights = model.sanitize(weights)

    quantization = config.get("quantization")
    if quantization is not None:

        def class_predicate(path: str, module: nn.Module) -> bool:
            return hasattr(module, "to_quantized") and f"{path}.scales" in weights

        nn.quantize(
            model,
            group_size=quantization["group_size"],
            bits=quantization["bits"],
            mode=quantization.get("mode", "affine"),
            class_predicate=class_predicate,
        )

    model.load_weights(list(weights.items()), strict=True)


def _load_tokenizer(model_path: Path, tokenizer_config: dict[str, Any]) -> Any:
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Loading encoder embedding tokenizers requires transformers."
        ) from exc
    return AutoTokenizer.from_pretrained(model_path, **tokenizer_config)
