# SPDX-License-Identifier: Apache-2.0
"""Compatibility patches for vLLM + transformers version mismatches.

Applied once at platform registration time. Optional missing dependencies are
logged; unexpected runtime errors are allowed to surface so regressions remain
diagnosable.
"""

from __future__ import annotations

import logging
from collections.abc import Callable, Mapping
from typing import Any

logger = logging.getLogger(__name__)

_APPLIED = False
_QWEN35_FP8_BLOCK_SIZE = 128


def apply_compat_patches() -> None:
    """Apply all known compatibility patches (idempotent)."""
    global _APPLIED  # noqa: PLW0603
    if _APPLIED:
        return
    _APPLIED = True
    _patch_mlx_lm_qwen35_fp8_sanitize()
    _patch_mlx_lm_gemma4_kv_shared_sanitize()


def _ceildiv(value: int, divisor: int) -> int:
    return -(-value // divisor)


def _shape_tuple(value: Any) -> tuple[int, ...]:
    return tuple(int(dim) for dim in getattr(value, "shape", ()))


def _validate_qwen35_fp8_block_scale_shape(
    weight: Any,
    scale_inv: Any,
    *,
    block_size: int = _QWEN35_FP8_BLOCK_SIZE,
) -> None:
    """Validate the FP8 scale shape before applying the fixed block layout."""
    weight_shape = _shape_tuple(weight)
    if len(weight_shape) < 2:
        return

    scale_shape = _shape_tuple(scale_inv)
    leading_shape = weight_shape[:-2]
    rows, cols = weight_shape[-2:]
    expected_scale_shape = (
        *leading_shape,
        _ceildiv(rows, block_size),
        _ceildiv(cols, block_size),
    )
    if scale_shape == expected_scale_shape:
        return

    raise ValueError(
        "Unsupported Qwen3.5/Qwen3.6 FP8 block scale shape: "
        f"weight shape={weight_shape}, weight_scale_inv shape={scale_shape}, "
        f"expected {expected_scale_shape} for {block_size}x{block_size} FP8 "
        "blocks."
    )


def _dequantize_qwen35_fp8_weight(
    weight: Any,
    scale_inv: Any,
    mx: Any,
    *,
    block_size: int = _QWEN35_FP8_BLOCK_SIZE,
) -> Any:
    _validate_qwen35_fp8_block_scale_shape(
        weight,
        scale_inv,
        block_size=block_size,
    )

    weight = mx.from_fp8(weight, dtype=mx.bfloat16)
    if weight.ndim < 2:
        return weight.astype(mx.bfloat16)

    leading_shape = weight.shape[:-2]
    rows, cols = weight.shape[-2:]
    pad_rows = (-rows) % block_size
    pad_cols = (-cols) % block_size
    pad_width = [(0, 0)] * len(leading_shape)
    pad_width.extend(((0, pad_rows), (0, pad_cols)))
    weight = mx.pad(weight, pad_width)
    block_rows = (rows + pad_rows) // block_size
    block_cols = (cols + pad_cols) // block_size
    weight = weight.reshape(
        (*leading_shape, block_rows, block_size, block_cols, block_size)
    )
    weight = (weight * scale_inv[..., :, None, :, None]).reshape(
        *leading_shape,
        rows + pad_rows,
        cols + pad_cols,
    )
    return weight[..., :rows, :cols].astype(mx.bfloat16)


def _dequantize_qwen35_fp8_weights(
    weights: Mapping[str, Any], mx: Any
) -> Mapping[str, Any]:
    if not any("weight_scale_inv" in key for key in weights):
        return weights

    new_weights: dict[str, Any] = {}
    for key, value in weights.items():
        if "weight_scale_inv" in key:
            weight_key = key.replace("_scale_inv", "")
            if weight_key not in weights:
                raise ValueError(
                    "Qwen3.5/Qwen3.6 FP8 checkpoint has "
                    f"{key!r} but is missing matching weight {weight_key!r}."
                )
            weight = weights[weight_key]
            new_weights[weight_key] = _dequantize_qwen35_fp8_weight(
                weight,
                value,
                mx,
            )
        elif "activation_scale" in key:
            continue
        elif key not in new_weights:
            new_weights[key] = value
    return new_weights


def _patch_mlx_lm_qwen35_fp8_sanitize() -> None:
    """Teach mlx_lm's Qwen3.5 loaders to consume local FP8 ``weight_scale_inv``.

    Some Qwen3.5/Qwen3.6 local checkpoints store FP8 weights plus
    ``*_weight_scale_inv`` tensors in HuggingFace-style shards. The installed
    mlx_lm ``qwen3_5`` loaders do not currently dequantize those tensors during
    ``sanitize()``, so ``model.load_weights()`` aborts with hundreds of
    unexpected ``weight_scale_inv`` parameters.

    Patch the top-level model ``sanitize()`` methods to dequantize those FP8
    tensors before the upstream remapping logic runs. This keeps the workaround
    narrow to the affected architectures and leaves upstream control flow intact.
    """
    from importlib import import_module
    from importlib.util import find_spec

    try:
        import mlx.core as mx
    except ImportError as exc:
        logger.warning(
            "Could not install mlx_lm Qwen3.5/Qwen3.6 FP8 sanitize "
            "compatibility patch because mlx.core is unavailable: %s",
            exc,
        )
        return

    model_modules = []
    for module_name in ("mlx_lm.models.qwen3_5", "mlx_lm.models.qwen3_5_moe"):
        if find_spec(module_name) is None:
            continue
        try:
            model_modules.append(import_module(module_name))
        except ImportError as exc:
            logger.warning(
                "Could not import %s while installing mlx_lm Qwen3.5/Qwen3.6 "
                "FP8 sanitize compatibility patch: %s",
                module_name,
                exc,
            )
    if not model_modules:
        logger.warning(
            "Could not install mlx_lm Qwen3.5/Qwen3.6 FP8 sanitize "
            "compatibility patch: no qwen3_5 model modules found."
        )
        return

    def _patch_model_sanitize(model_cls) -> bool:
        return _wrap_model_sanitize(
            model_cls,
            "_vllm_metal_qwen35_fp8_patch",
            lambda _self, weights: _dequantize_qwen35_fp8_weights(weights, mx),
        )

    patched_modules = []
    unpatchable_modules = []
    for module in model_modules:
        model_cls = getattr(module, "Model", None)
        if model_cls is None:
            unpatchable_modules.append(module.__name__.rsplit(".", maxsplit=1)[-1])
            continue
        if _patch_model_sanitize(model_cls):
            patched_modules.append(module.__name__.rsplit(".", maxsplit=1)[-1])
    if patched_modules:
        logger.debug(
            "Patched mlx_lm %s FP8 sanitize compatibility",
            ", ".join(sorted(patched_modules)),
        )
    elif unpatchable_modules:
        logger.warning(
            "Could not install mlx_lm Qwen3.5/Qwen3.6 FP8 sanitize "
            "compatibility patch for modules without Model classes: %s",
            ", ".join(sorted(unpatchable_modules)),
        )


def _wrap_model_sanitize(
    model_cls: Any,
    sentinel_attr: str,
    transform: Callable[[Any, Mapping[str, Any]], Mapping[str, Any]],
) -> bool:
    """Wrap an existing ``model_cls.sanitize`` with a pre-step ``transform``.

    Trusts upstream's ``Model.sanitize`` contract: if the class does not
    already define ``sanitize``, returns False instead of synthesizing a
    new method. All current targets (qwen3_5, qwen3_5_moe, gemma4_text)
    define ``sanitize`` upstream, so synthesizing one would be a
    speculative API rather than a real compatibility shim.

    Idempotent via ``sentinel_attr``. Returns True on first patch, False
    if there is no ``sanitize`` to wrap or the sentinel says we already
    patched this class.
    """
    sanitize = getattr(model_cls, "sanitize", None)
    if sanitize is None:
        return False
    if getattr(sanitize, sentinel_attr, False):
        return False

    original_sanitize = sanitize

    def _patched_sanitize(self, weights):
        return original_sanitize(self, transform(self, weights))

    setattr(_patched_sanitize, sentinel_attr, True)
    model_cls.sanitize = _patched_sanitize
    return True


def _drop_gemma4_kv_shared_phantom_weights(
    weights: Mapping[str, Any],
    num_hidden_layers: int,
    num_kv_shared_layers: int,
) -> dict[str, Any]:
    """Strip K/V/k_norm safetensors keys for KV-shared Gemma 4 layers.

    Layers with index ``>= num_hidden_layers - num_kv_shared_layers`` reuse
    K/V from earlier same-type layers (see ``Gemma4TextModel.previous_kvs``)
    and have no destination for those tensors after mlx-lm PR #1158.
    """
    if not num_kv_shared_layers:
        return dict(weights)

    first_shared = num_hidden_layers - num_kv_shared_layers
    # Generate the exact tails for every (shared_layer, suffix) pair.
    # A key is dropped iff it ends with one of these — no parsing, no
    # fallback, no ambiguity. Unrelated keys (e.g. "model.weird.self_attn
    # .k_proj.weight") cannot match because the tail mandates ".layers.<N>.".
    drop_tails = tuple(
        f".layers.{i}.self_attn.{suffix}.weight"
        for i in range(first_shared, num_hidden_layers)
        for suffix in ("k_proj", "v_proj", "k_norm")
    )
    return {k: v for k, v in weights.items() if not k.endswith(drop_tails)}


def _patch_mlx_lm_gemma4_kv_shared_sanitize() -> None:
    """Drop phantom K/V/k_norm safetensors keys for KV-shared Gemma 4 layers.

    mlx-lm PR #1158 gated ``k_proj``/``v_proj``/``k_norm`` allocation in
    ``gemma4_text.Attention.__init__`` behind ``has_kv``, but the matching
    drop step in ``Model.sanitize`` was not added. Checkpoints that still
    serialize those tensors (e.g. ``google/gemma-4-E4B-it``) crash strict
    weight load with ``Received N parameters not in model``.

    Remove this patch once upstream lands the matching ``sanitize`` change
    and the mlx-lm pin in ``pyproject.toml`` is bumped past it.
    """
    from importlib import import_module
    from importlib.util import find_spec

    if find_spec("mlx_lm.models.gemma4_text") is None:
        return
    try:
        module = import_module("mlx_lm.models.gemma4_text")
    except ImportError as exc:
        logger.warning(
            "Could not install mlx_lm Gemma 4 KV-shared sanitize "
            "compatibility patch: %s",
            exc,
        )
        return

    model_cls = getattr(module, "Model", None)
    if model_cls is None:
        logger.warning(
            "Could not install mlx_lm Gemma 4 KV-shared sanitize "
            "compatibility patch: Model class not found in gemma4_text."
        )
        return

    def _transform(self, weights):
        return _drop_gemma4_kv_shared_phantom_weights(
            weights,
            self.args.num_hidden_layers,
            self.args.num_kv_shared_layers,
        )

    if _wrap_model_sanitize(
        model_cls, "_vllm_metal_gemma4_kv_shared_patch", _transform
    ):
        logger.debug("Patched mlx_lm gemma4_text KV-shared sanitize compatibility")
