# SPDX-License-Identifier: Apache-2.0
"""Block-FP8 quantization preflight validation for ``mlx-lm`` model loading.

``mlx-lm`` only supports block-wise ``quant_method="fp8"`` checkpoints on
architectures whose ``Model.sanitize()`` dequantizes ``weight_scale_inv``
(either natively in ``mlx-lm`` or via ``vllm_metal.compat``), or when the
checkpoint uses ``scale_fmt="ue8m0"`` with ``weight_block_size=[32, 32]``
(which ``mlx_lm.utils.load_model`` maps onto MLX native ``mxfp8``).

All other architectures carrying block-FP8 ``quantization_config`` would
otherwise fail mid-load with weight-shape or missing-parameter errors on
``weight_scale_inv`` after downloading weights. Fail fast before loading or
downloading checkpoints instead.
"""

from __future__ import annotations

import json
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

from mlx_lm.utils import MODEL_ARCHITECTURE_REMAPPING, MODEL_REMAPPING

from vllm_metal.quant.awq_config import UnsupportedQuantizationConfigError

# Model types that dequantize block-FP8 ``weight_scale_inv`` weights inside
# their ``Model.sanitize()`` in pinned ``mlx-lm`` or ``vllm_metal.compat``.
_MLX_LM_BLOCK_FP8_MODEL_TYPES: frozenset[str] = frozenset(
    {
        "deepseek_v3",
        "deepseek_v32",
        "mimo_v2_flash",
        "minimax",
        "mistral4",
        "qwen3_5",
        "qwen3_5_moe",
    }
)


def _read_field(obj: Any, key: str) -> Any:
    """Read ``key`` from a mapping or attribute object, returning ``None`` if absent."""
    if obj is None:
        return None
    if isinstance(obj, Mapping):
        return obj.get(key)
    return getattr(obj, key, None)


def _read_local_config(model_path: str | Path | None) -> Mapping[str, Any] | None:
    """Read ``config.json`` when ``model_path`` is an existing local directory."""
    if model_path is None:
        return None
    try:
        path = Path(model_path)
        if not path.is_dir():
            return None
        config_path = path / "config.json"
        if not config_path.is_file():
            return None
        parsed = json.loads(config_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, ValueError):
        return None
    return parsed if isinstance(parsed, Mapping) else None


def extract_quantization_config(
    hf_config: Any = None,
    *,
    local_config: Mapping[str, Any] | None = None,
) -> Any:
    """Return top-level or nested ``text_config`` ``quantization_config``, if set."""
    sources = (
        hf_config,
        _read_field(hf_config, "text_config"),
        local_config,
        _read_field(local_config, "text_config"),
    )
    for source in sources:
        qc = _read_field(source, "quantization_config")
        if qc is not None:
            return qc
    return None


def resolve_mlx_lm_model_type(
    model_type: str | None,
    architectures: Iterable[str] | None = None,
) -> str | None:
    """Resolve Hugging Face ``model_type`` through ``mlx_lm`` remapping tables."""
    if not isinstance(model_type, str) or not model_type:
        return None
    resolved = model_type
    if isinstance(architectures, Iterable) and not isinstance(
        architectures, (str, bytes)
    ):
        for arch in architectures:
            if not isinstance(arch, str):
                continue
            remapped = MODEL_ARCHITECTURE_REMAPPING.get((model_type, arch))
            if remapped is not None:
                resolved = remapped
                break
    return MODEL_REMAPPING.get(resolved, resolved)


def _extract_block_fp8_size(quant_config: Any) -> list[Any] | None:
    """Return non-empty ``weight_block_size`` when ``quant_config`` is block-FP8.

    Returns ``None`` for non-FP8 configs, non-block FP8 configs (per-tensor /
    per-row without ``weight_block_size``), or MXFP8 (``scale_fmt="ue8m0"``
    with ``weight_block_size=[32, 32]``), all of which ``mlx-lm`` handles without
    per-architecture ``weight_scale_inv`` dequantization.
    """
    if quant_config is None:
        return None
    quant_method = _read_field(quant_config, "quant_method")
    if not isinstance(quant_method, str) or quant_method.lower() != "fp8":
        return None

    raw_block_size = _read_field(quant_config, "weight_block_size")
    if (
        raw_block_size is None
        or isinstance(raw_block_size, (str, bytes))
        or not isinstance(raw_block_size, Iterable)
    ):
        return None

    block_size = list(raw_block_size)
    if not block_size:
        return None

    scale_fmt = _read_field(quant_config, "scale_fmt")
    if (
        isinstance(scale_fmt, str)
        and scale_fmt.lower() == "ue8m0"
        and block_size == [32, 32]
    ):
        return None

    return block_size


def validate_mlx_lm_block_fp8_support(
    hf_config: Any = None,
    *,
    model_path: str | Path | None = None,
    is_draft: bool = False,
) -> None:
    """Raise :class:`UnsupportedQuantizationConfigError` on unsupported block-FP8.

    Checks ``hf_config`` first and falls back to reading a local directory's
    ``config.json`` when ``model_path`` points to a directory on disk. Never
    triggers network downloads.
    """
    local_config = _read_local_config(model_path)
    quant_config = extract_quantization_config(hf_config, local_config=local_config)
    block_size = _extract_block_fp8_size(quant_config)
    if block_size is None:
        return

    # Custom MLX ``model_file`` implementations bypass ``mlx_lm.models`` and
    # may carry their own weight sanitization.
    if (
        _read_field(hf_config, "model_file") is not None
        or _read_field(local_config, "model_file") is not None
    ):
        return

    model_type = (
        _read_field(hf_config, "model_type")
        or _read_field(local_config, "model_type")
        or _read_field(_read_field(hf_config, "text_config"), "model_type")
        or _read_field(_read_field(local_config, "text_config"), "model_type")
    )
    architectures = _read_field(hf_config, "architectures") or _read_field(
        local_config, "architectures"
    )
    resolved_type = resolve_mlx_lm_model_type(model_type, architectures)
    if resolved_type in _MLX_LM_BLOCK_FP8_MODEL_TYPES:
        return

    arch_label = (
        f"{model_type!r} (mapped to {resolved_type!r})"
        if resolved_type and resolved_type != model_type
        else repr(model_type or "unknown")
    )
    role = "draft model " if is_draft else ""
    raise UnsupportedQuantizationConfigError(
        f"Block-FP8 checkpoint for {role}architecture {arch_label} "
        f"(quant_method='fp8', weight_block_size={block_size!r}) is not supported "
        "on the MLX backend because mlx-lm has no FP8 block dequantization path "
        "for this architecture. Use an MLX-quantized (e.g. mlx-community) or "
        "unquantized bf16 checkpoint instead."
    )
