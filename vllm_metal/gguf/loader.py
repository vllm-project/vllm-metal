# SPDX-License-Identifier: Apache-2.0
"""Dense fallback GGUF loader for MLX models."""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from mlx.utils import tree_flatten
from mlx_lm.utils import _get_classes as mlx_lm_get_classes
from mlx_lm.utils import load_config as mlx_lm_load_config
from mlx_lm.utils import load_tokenizer as mlx_lm_load_tokenizer
from vllm.logger import init_logger

from vllm_metal.gguf.refs import GGUFReference

logger = init_logger(__name__)

GGUF_QTYPE_Q8_1 = 9


class GGUFLoadError(RuntimeError):
    """Raised when a GGUF checkpoint cannot be loaded through MLX."""


class GGUFMLXLoader:
    """Load a local GGUF checkpoint into the corresponding MLX-LM model.

    This is a compatibility loader: supported quantized GGUF tensors are
    dequantized to dense MLX arrays before they are attached to the model.
    Native raw-block execution is introduced in later PRs.
    """

    def __init__(
        self,
        reference: GGUFReference,
        *,
        target_dtype: mx.Dtype,
        tokenizer_config: dict[str, Any] | None = None,
    ) -> None:
        self.reference = reference
        self.target_dtype = target_dtype
        self.tokenizer_config = tokenizer_config or {}

    def load(self) -> tuple[Any, Any]:
        config = mlx_lm_load_config(self.reference.model_path)
        model_class, model_args_class = mlx_lm_get_classes(config=config)
        model_args = model_args_class.from_dict(config)
        model = model_class(model_args)

        weights = self._load_weights(config, model=model)
        if hasattr(model, "sanitize"):
            weights = model.sanitize(weights)

        model.eval()
        model.load_weights(list(weights.items()), strict=True)
        mx.eval(model.parameters())

        tokenizer = mlx_lm_load_tokenizer(
            self.reference.model_path,
            self.tokenizer_config,
            eos_token_ids=config.get("eos_token_id", None),
        )
        return model, tokenizer

    def _load_weights(
        self,
        config: dict[str, Any],
        *,
        model: nn.Module | None = None,
    ) -> dict[str, mx.array]:
        try:
            import gguf
        except ImportError as exc:
            raise GGUFLoadError(
                "GGUF support requires the `gguf` Python package."
            ) from exc

        model_type = str(config.get("model_type", ""))
        upstream_name_map = _build_upstream_gguf_to_mlx_map(model, config)
        weights: dict[str, mx.array] = {}
        skipped: list[str] = []

        for gguf_path in self.reference.all_gguf_paths:
            reader = gguf.GGUFReader(str(gguf_path))
            for tensor in reader.tensors:
                translated = translate_gguf_tensor_name(
                    tensor.name,
                    model_type=model_type,
                    upstream_name_map=upstream_name_map,
                )
                if translated is None:
                    skipped.append(tensor.name)
                    continue
                array = _decode_tensor(tensor, target_dtype=self.target_dtype)
                array = _reshape_special_tensor(translated, array)
                array = _adjust_tensor_for_mlx_sanitize(
                    translated,
                    array,
                    model_type=model_type,
                )
                _add_weight(weights, translated, array)

        if not weights:
            gguf_paths = ", ".join(str(path) for path in self.reference.all_gguf_paths)
            raise GGUFLoadError(f"No loadable tensors found in {gguf_paths}")

        if skipped:
            logger.info(
                "Skipped %d GGUF tensors without MLX mapping from %s: %s",
                len(skipped),
                ", ".join(str(path) for path in self.reference.all_gguf_paths),
                ", ".join(skipped[:8]),
            )

        return weights


_BLOCK_RE = re.compile(r"^blk\.(\d+)\.(.+)$")

_GLOBAL_NAME_MAP = {
    "token_embd.weight": "model.embed_tokens.weight",
    "output_norm.weight": "model.norm.weight",
    "output.weight": "lm_head.weight",
    "per_layer_token_embd.weight": "model.embed_tokens_per_layer.weight",
    "per_layer_model_proj.weight": "model.per_layer_model_projection.weight",
    "per_layer_proj_norm.weight": "model.per_layer_projection_norm.weight",
}

_BLOCK_NAME_MAP = {
    "attn_norm.weight": "input_layernorm.weight",
    "post_attention_norm.weight": "post_attention_layernorm.weight",
    "ffn_norm.weight": "pre_feedforward_layernorm.weight",
    "post_ffw_norm.weight": "post_feedforward_layernorm.weight",
    "post_norm.weight": "post_per_layer_input_norm.weight",
    "ffn_gate.weight": "mlp.gate_proj.weight",
    "ffn_down.weight": "mlp.down_proj.weight",
    "ffn_up.weight": "mlp.up_proj.weight",
    "attn_q.weight": "self_attn.q_proj.weight",
    "attn_k.weight": "self_attn.k_proj.weight",
    "attn_v.weight": "self_attn.v_proj.weight",
    "attn_output.weight": "self_attn.o_proj.weight",
    "attn_q_norm.weight": "self_attn.q_norm.weight",
    "attn_k_norm.weight": "self_attn.k_norm.weight",
    "inp_gate.weight": "per_layer_input_gate.weight",
    "proj.weight": "per_layer_projection.weight",
    "layer_output_scale.weight": "layer_scalar",
    "attn_qkv.weight": "linear_attn.in_proj_qkv.weight",
    "attn_gate.weight": "linear_attn.in_proj_z.weight",
    "ssm_alpha.weight": "linear_attn.in_proj_a.weight",
    "ssm_beta.weight": "linear_attn.in_proj_b.weight",
    "ssm_conv1d.weight": "linear_attn.conv1d.weight",
    "ssm_dt.bias": "linear_attn.dt_bias",
    "ssm_a": "linear_attn.A_log",
    "ssm_norm.weight": "linear_attn.norm.weight",
    "ssm_out.weight": "linear_attn.out_proj.weight",
}

_SKIPPED_GLOBAL_NAMES = {
    "rope_freqs.weight",
}


_MLX_QWEN_SHIFTED_NORM_MODEL_TYPES = {
    "qwen3_5",
    "qwen3_5_moe",
    "qwen3_next",
}

_MLX_QWEN_SHIFTED_NORM_SUFFIXES = (
    ".input_layernorm.weight",
    ".post_attention_layernorm.weight",
    "model.norm.weight",
    ".q_norm.weight",
    ".k_norm.weight",
)


def translate_gguf_tensor_name(
    name: str,
    *,
    model_type: str,
    upstream_name_map: dict[str, str] | None = None,
) -> str | None:
    if name in _SKIPPED_GLOBAL_NAMES:
        return None

    if upstream_name_map is not None:
        translated = upstream_name_map.get(name)
        if translated is not None:
            return translated

    translated = _GLOBAL_NAME_MAP.get(name)
    if translated is None:
        block_match = _BLOCK_RE.match(name)
        if block_match is None:
            return None
        layer, suffix = block_match.groups()
        block_suffix = _BLOCK_NAME_MAP.get(suffix)
        if block_suffix is None:
            return None
        translated = f"model.layers.{layer}.{block_suffix}"

    if model_type == "gemma4" and translated.startswith("model."):
        return "model.language_model." + translated.removeprefix("model.")
    return translated


_GGUF_MODEL_TYPE_ALIASES = {
    "qwen3_5": "qwen35",
    "qwen3_5_moe": "qwen35moe",
    "qwen3_6_moe": "qwen35moe",
    "qwen3_next": "qwen3next",
    "qwen3next": "qwen3next",
    "qwen2_moe": "qwen2moe",
    "qwen3_moe": "qwen3moe",
    "gemma3_text": "gemma3",
}


def _build_upstream_gguf_to_mlx_map(
    model: nn.Module | None,
    config: dict[str, Any],
) -> dict[str, str]:
    if model is None:
        return {}

    try:
        import gguf
    except ImportError:
        return {}

    arch = _gguf_arch_from_config(gguf, config)
    if arch is None:
        return {}

    num_layers = _num_hidden_layers(config)
    if num_layers <= 0:
        return {}

    try:
        tensor_name_map = gguf.get_tensor_name_map(arch, num_layers)
    except Exception as exc:  # pragma: no cover - gguf version compatibility
        logger.debug("Unable to build upstream GGUF tensor map: %s", exc)
        return {}

    leaves = tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module)
    if not isinstance(leaves, list):
        return {}

    reverse: dict[str, str] = {}
    for module_path, module in leaves:
        for param_name in module.parameters():
            mlx_name = f"{module_path}.{param_name}" if module_path else param_name
            gguf_name = _gguf_name_for_mlx_param(tensor_name_map, mlx_name)
            if gguf_name is not None:
                reverse.setdefault(gguf_name, mlx_name)
    return reverse


def _gguf_arch_from_config(gguf: Any, config: dict[str, Any]) -> Any | None:
    model_type = str(config.get("model_type", "")).lower().replace("-", "_")
    arch_name = _GGUF_MODEL_TYPE_ALIASES.get(model_type, model_type)
    for arch, name in getattr(gguf, "MODEL_ARCH_NAMES", {}).items():
        if str(name).lower().replace("-", "_") == arch_name:
            return arch
    return None


def _num_hidden_layers(config: dict[str, Any]) -> int:
    for key in ("num_hidden_layers", "n_layers", "num_layers", "n_layer"):
        value = config.get(key)
        if value is not None:
            return int(value)
    return 0


def _gguf_name_for_mlx_param(tensor_name_map: Any, mlx_name: str) -> str | None:
    translated = tensor_name_map.get_name(
        mlx_name,
        try_suffixes=(".weight", ".bias"),
    )
    if translated is not None:
        return translated

    if mlx_name.startswith("model.language_model."):
        stripped_name = "model." + mlx_name.removeprefix("model.language_model.")
        return tensor_name_map.get_name(
            stripped_name,
            try_suffixes=(".weight", ".bias"),
        )
    return None


def _reshape_special_tensor(name: str, array: mx.array) -> mx.array:
    if name.endswith(".linear_attn.conv1d.weight") and len(array.shape) == 2:
        return mx.expand_dims(array, axis=1)
    return array


def _adjust_tensor_for_mlx_sanitize(
    name: str,
    array: mx.array,
    *,
    model_type: str,
) -> mx.array:
    if (
        model_type in _MLX_QWEN_SHIFTED_NORM_MODEL_TYPES
        and len(array.shape) == 1
        and any(name.endswith(suffix) for suffix in _MLX_QWEN_SHIFTED_NORM_SUFFIXES)
    ):
        return array - 1.0
    return array


def _add_weight(weights: dict[str, mx.array], key: str, value: mx.array) -> None:
    if key in weights:
        raise GGUFLoadError(f"Duplicate GGUF tensor mapped to MLX weight: {key}")
    weights[key] = value


def _decode_tensor(tensor: Any, *, target_dtype: mx.Dtype) -> mx.array:
    try:
        import gguf
    except ImportError as exc:
        raise GGUFLoadError("GGUF support requires the `gguf` Python package.") from exc

    try:
        tensor_type = gguf.GGMLQuantizationType(tensor.tensor_type)
    except ValueError as exc:
        raise GGUFLoadError(
            f"Unsupported GGUF tensor type {tensor.tensor_type} for tensor "
            f"{tensor.name}."
        ) from exc
    final_shape = _final_shape(tensor.shape)

    if tensor_type == gguf.GGMLQuantizationType.F32:
        data = np.asarray(tensor.data, dtype=np.float32).reshape(final_shape)
        return mx.array(data)
    if tensor_type == gguf.GGMLQuantizationType.F16:
        data = np.asarray(tensor.data, dtype=np.float16).reshape(final_shape)
        return mx.array(data).astype(target_dtype)
    if tensor_type == gguf.GGMLQuantizationType.BF16:
        data = _decode_bf16(tensor.data).reshape(final_shape)
        return mx.array(data).astype(target_dtype)
    if tensor_type == gguf.GGMLQuantizationType.Q8_0:
        data = _dequantize_q8_0(tensor.data, tensor.shape)
        return mx.array(data).astype(target_dtype)
    if int(tensor_type) == GGUF_QTYPE_Q8_1:
        data = _dequantize_q8_1(tensor.data, tensor.shape)
        return mx.array(data).astype(target_dtype)
    if tensor_type in gguf.GGML_QUANT_SIZES:
        try:
            data = gguf.dequantize(
                np.asarray(tensor.data, dtype=np.uint8),
                tensor_type,
            ).reshape(final_shape)
        except NotImplementedError as exc:
            raise GGUFLoadError(
                f"Unsupported GGUF tensor type {tensor_type.name} for tensor "
                f"{tensor.name}: gguf.dequantize does not implement this type."
            ) from exc
        return mx.array(data).astype(target_dtype)

    raise GGUFLoadError(
        f"Unsupported GGUF tensor type {tensor_type.name} for tensor {tensor.name}. "
        "This experimental loader supports F32, F16, BF16, Q8_0/Q8_1, and "
        "quantized tensor types implemented by gguf.dequantize."
    )


def _final_shape(shape: Iterable[int]) -> tuple[int, ...]:
    return tuple(int(dim) for dim in reversed(tuple(shape)))


def _decode_bf16(data: Any) -> np.ndarray:
    raw = np.asarray(data, dtype=np.uint16)
    return (raw.astype(np.uint32) << 16).view(np.float32)


def _dequantize_q8_0(data: Any, shape: Iterable[int]) -> np.ndarray:
    dims = tuple(int(dim) for dim in shape)
    if not dims:
        raise GGUFLoadError("Q8_0 tensor has no shape.")

    cols = dims[0]
    if cols % 32 != 0:
        raise GGUFLoadError(f"Q8_0 tensor column count must be divisible by 32: {dims}")

    final_shape = _final_shape(dims)
    rows = int(np.prod(final_shape[:-1], dtype=np.int64)) if len(final_shape) > 1 else 1
    blocks = cols // 32
    raw = np.asarray(data, dtype=np.uint8).reshape(rows, blocks, 34)
    scales = raw[:, :, :2].copy().view(np.float16).astype(np.float32)
    quants = raw[:, :, 2:].copy().view(np.int8).astype(np.float32)
    dequantized = (quants * scales).reshape(rows, cols)
    return dequantized.reshape(final_shape)


def _dequantize_q8_1(data: Any, shape: Iterable[int]) -> np.ndarray:
    dims = tuple(int(dim) for dim in shape)
    if not dims:
        raise GGUFLoadError("Q8_1 tensor has no shape.")

    cols = dims[0]
    if cols % 32 != 0:
        raise GGUFLoadError(f"Q8_1 tensor column count must be divisible by 32: {dims}")

    final_shape = _final_shape(dims)
    rows = int(np.prod(final_shape[:-1], dtype=np.int64)) if len(final_shape) > 1 else 1
    blocks = cols // 32
    raw = np.asarray(data, dtype=np.uint8).reshape(rows, blocks, 40)
    scales = raw[:, :, :2].copy().view(np.float16).astype(np.float32)
    quants = raw[:, :, 4:36].copy().view(np.int8).astype(np.float32)
    dequantized = (quants * scales).reshape(rows, cols)
    return dequantized.reshape(final_shape)
