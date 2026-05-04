# SPDX-License-Identifier: Apache-2.0
"""Compatibility patches for vLLM + transformers version mismatches.

Applied once at platform registration time. Optional missing dependencies are
logged; unexpected runtime errors are allowed to surface so regressions remain
diagnosable.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Callable, Mapping
from pathlib import Path
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
    _patch_transformers_bytelevel_tokenizer_decoder()
    _patch_mlx_lm_qwen35_fp8_sanitize()
    _patch_mlx_lm_gemma4_kv_shared_sanitize()


def _decoder_tree_contains_type(value: Any, decoder_type: str) -> bool:
    if isinstance(value, dict):
        if value.get("type") == decoder_type:
            return True
        return any(
            _decoder_tree_contains_type(child, decoder_type) for child in value.values()
        )
    if isinstance(value, list):
        return any(_decoder_tree_contains_type(child, decoder_type) for child in value)
    return False


def _tokenizer_json_decoder_uses_bytelevel(tokenizer_json: Path) -> bool:
    try:
        with tokenizer_json.open("r", encoding="utf-8") as handle:
            tokenizer_data = json.load(handle)
    except (OSError, json.JSONDecodeError):
        return False
    return _decoder_tree_contains_type(tokenizer_data.get("decoder"), "ByteLevel")


def _loaded_tokenizer_decoder_uses_bytelevel(tokenizer: Any) -> bool:
    backend_tokenizer = getattr(tokenizer, "backend_tokenizer", None)
    decoder = getattr(backend_tokenizer, "decoder", None)
    return "ByteLevel" in repr(decoder)


def _candidate_tokenizer_json_paths(
    tokenizer: Any,
    path_or_repo_id: str | Path,
) -> list[Path]:
    paths: list[Path] = []

    init_kwargs = getattr(tokenizer, "init_kwargs", {}) or {}
    tokenizer_file = init_kwargs.get("tokenizer_file")
    if tokenizer_file:
        paths.append(Path(tokenizer_file))

    path = Path(path_or_repo_id)
    if path.is_dir():
        paths.append(path / "tokenizer.json")

    name_or_path = getattr(tokenizer, "name_or_path", None)
    if name_or_path:
        name_path = Path(name_or_path)
        if name_path.is_dir():
            paths.append(name_path / "tokenizer.json")

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in paths:
        if candidate in seen:
            continue
        seen.add(candidate)
        if candidate.is_file():
            deduped.append(candidate)
    return deduped


def _cached_tokenizer_json_path(
    path_or_repo_id: str | Path,
    from_pretrained_kwargs: Mapping[str, Any],
) -> Path | None:
    try:
        from transformers.utils import cached_file
    except ImportError:
        return None

    cached_kwargs: dict[str, Any] = {
        "_raise_exceptions_for_connection_errors": False,
        "_raise_exceptions_for_gated_repo": False,
        "_raise_exceptions_for_missing_entries": False,
    }
    for key in (
        "cache_dir",
        "force_download",
        "proxies",
        "token",
        "revision",
        "local_files_only",
        "subfolder",
        "repo_type",
        "user_agent",
        "_commit_hash",
    ):
        value = from_pretrained_kwargs.get(key)
        if value is not None:
            cached_kwargs[key] = value
    if "token" not in cached_kwargs:
        token = from_pretrained_kwargs.get("use_auth_token")
        if token is not None:
            cached_kwargs["token"] = token

    try:
        cached = cached_file(path_or_repo_id, "tokenizer.json", **cached_kwargs)
    except TypeError:
        # Older transformers releases have a narrower cached_file signature.
        fallback_kwargs = {
            key: value
            for key, value in cached_kwargs.items()
            if not key.startswith("_") and key != "token"
        }
        try:
            cached = cached_file(
                path_or_repo_id,
                "tokenizer.json",
                **fallback_kwargs,
            )
        except Exception:
            return None
    except Exception:
        return None

    if not cached:
        return None
    path = Path(cached)
    return path if path.is_file() else None


def _find_bytelevel_tokenizer_json(
    tokenizer: Any,
    path_or_repo_id: str | Path,
    from_pretrained_kwargs: Mapping[str, Any],
) -> Path | None:
    for tokenizer_json in _candidate_tokenizer_json_paths(tokenizer, path_or_repo_id):
        if _tokenizer_json_decoder_uses_bytelevel(tokenizer_json):
            return tokenizer_json

    tokenizer_json = _cached_tokenizer_json_path(
        path_or_repo_id,
        from_pretrained_kwargs,
    )
    if tokenizer_json is None:
        return None
    if not _tokenizer_json_decoder_uses_bytelevel(tokenizer_json):
        return None
    return tokenizer_json


def _rebuild_bytelevel_fast_tokenizer(tokenizer: Any, tokenizer_json: Path) -> Any:
    from transformers import PreTrainedTokenizerFast

    init_kwargs = dict(getattr(tokenizer, "init_kwargs", {}) or {})
    for key in ("backend", "tokenizer_file", "vocab_file", "name_or_path"):
        init_kwargs.pop(key, None)

    for attr in (
        "bos_token",
        "eos_token",
        "unk_token",
        "pad_token",
        "sep_token",
        "cls_token",
        "mask_token",
        "additional_special_tokens",
        "chat_template",
        "model_max_length",
        "clean_up_tokenization_spaces",
    ):
        value = getattr(tokenizer, attr, None)
        if value is not None:
            init_kwargs[attr] = value

    rebuilt = PreTrainedTokenizerFast(
        tokenizer_file=str(tokenizer_json),
        **init_kwargs,
    )
    name_or_path = getattr(tokenizer, "name_or_path", None)
    if name_or_path:
        rebuilt._name_or_path = name_or_path
    return rebuilt


def _maybe_rebuild_bytelevel_tokenizer(
    tokenizer: Any,
    path_or_repo_id: str | Path,
    from_pretrained_kwargs: Mapping[str, Any],
) -> Any:
    if _loaded_tokenizer_decoder_uses_bytelevel(tokenizer):
        return tokenizer

    tokenizer_json = _find_bytelevel_tokenizer_json(
        tokenizer,
        path_or_repo_id,
        from_pretrained_kwargs,
    )
    if tokenizer_json is None:
        return tokenizer

    rebuilt = _rebuild_bytelevel_fast_tokenizer(tokenizer, tokenizer_json)
    logger.info(
        "Rebuilt tokenizer from ByteLevel tokenizer.json after transformers "
        "loaded a non-ByteLevel decoder: %s",
        tokenizer_json,
    )
    return rebuilt


def _patch_transformers_bytelevel_tokenizer_decoder() -> None:
    """Use tokenizer.json's ByteLevel decoder when AutoTokenizer picks wrong.

    Some MLX-community Qwen/DeepSeek redistributions ship a ByteLevel
    `tokenizer.json` but a `tokenizer_config.json` that can make older
    transformers versions instantiate a Llama/SentencePiece-style decoder.
    That decoder leaves ByteLevel token pieces such as "\u0120" and "\u010a"
    in served text. If the tokenizer JSON itself is ByteLevel, rebuild a fast
    tokenizer from that file so vLLM's serving decoder matches the checkpoint.
    """
    try:
        from transformers import AutoTokenizer
    except ImportError as exc:
        logger.warning(
            "Could not install transformers ByteLevel tokenizer compatibility "
            "patch because transformers is unavailable: %s",
            exc,
        )
        return

    sentinel = "_vllm_metal_bytelevel_decoder_patch"
    if getattr(AutoTokenizer, sentinel, False):
        return

    original_from_pretrained = AutoTokenizer.from_pretrained

    def _patched_from_pretrained(cls, path_or_repo_id, *args, **kwargs):
        tokenizer = original_from_pretrained(path_or_repo_id, *args, **kwargs)
        return _maybe_rebuild_bytelevel_tokenizer(
            tokenizer,
            path_or_repo_id,
            kwargs,
        )

    AutoTokenizer.from_pretrained = classmethod(_patched_from_pretrained)
    setattr(AutoTokenizer, sentinel, True)


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


def _stack_qwen36_moe_per_expert_weights(
    weights: Mapping[str, Any], mx: Any
) -> Mapping[str, Any]:
    """Combine per-expert MoE tensors into the stacked layout mlx_lm expects.

    ``Qwen/Qwen3.6-35B-A3B-FP8`` ships expert MLPs as one tensor per expert
    per projection: ``...mlp.experts.{E}.{gate,up,down}_proj.weight``. The
    bf16 master ``Qwen/Qwen3.6-35B-A3B`` is already pre-stacked and falls
    through to the existing combined-format branch in
    ``mlx_lm.qwen3_5_moe.sanitize`` unchanged. ``mlx_lm.qwen3_5_moe``'s
    ``sanitize`` expects experts concatenated as
    ``...mlp.experts.gate_up_proj`` (gate then up along the intermediate axis)
    and ``...mlp.experts.down_proj``, both stacked along axis 0 over experts.

    Mirrors the (scan -> validate -> walk) structure of upstream
    ml-explore/mlx-lm#1224. Removable once vllm-metal's mlx-lm pin bumps
    past that merge.

    No-op when no per-expert keys are present (dense Qwen3.5/3.6 or already-
    stacked MoE checkpoints).
    """
    experts_marker = ".mlp.experts."
    proj_suffixes = (".gate_proj.weight", ".up_proj.weight", ".down_proj.weight")
    # Scan: discover per-layer experts prefixes and per-projection index sets
    # for all three projection families, so a checkpoint missing one family
    # (or with a mismatched index set across families) fails validation
    # cleanly instead of leaking a KeyError during the walk.
    layer_proj_indices: dict[str, dict[str, set[int]]] = {}
    for key in weights:
        marker_pos = key.find(experts_marker)
        if marker_pos == -1:
            continue
        suffix = next((s for s in proj_suffixes if key.endswith(s)), None)
        if suffix is None:
            continue
        index_start = marker_pos + len(experts_marker)
        index_end = len(key) - len(suffix)
        tail = key[index_start:index_end]
        if not tail.isdigit():
            continue
        prefix = key[: marker_pos + len(".mlp.experts")]
        proj = suffix[1 : -len(".weight")]  # ".gate_proj.weight" -> "gate_proj"
        layer_proj_indices.setdefault(prefix, {}).setdefault(proj, set()).add(int(tail))

    if not layer_proj_indices:
        return weights

    logger.debug(
        "Stacking per-expert MoE tensors at %d prefixes",
        len(layer_proj_indices),
    )
    required_projs = ("gate_proj", "up_proj", "down_proj")
    new_weights = dict(weights)
    for prefix, proj_to_indices in layer_proj_indices.items():
        # Validate: every prefix must have all three projection families, and
        # all three must share the same contiguous {0..N-1} index set.
        missing_projs = [p for p in required_projs if p not in proj_to_indices]
        if missing_projs:
            raise ValueError(
                f"Per-expert MoE weights at {prefix!r} are missing "
                f"projection families: {missing_projs}."
            )
        gate_indices = proj_to_indices["gate_proj"]
        expected = set(range(len(gate_indices)))
        if gate_indices != expected:
            missing = sorted(expected - gate_indices)
            extra = sorted(gate_indices - expected)
            raise ValueError(
                f"Per-expert MoE weights at {prefix!r} have "
                f"non-contiguous gate_proj indices: missing={missing}, "
                f"unexpected={extra}."
            )
        for proj in ("up_proj", "down_proj"):
            if proj_to_indices[proj] != gate_indices:
                missing = sorted(gate_indices - proj_to_indices[proj])
                extra = sorted(proj_to_indices[proj] - gate_indices)
                raise ValueError(
                    f"Per-expert MoE weights at {prefix!r} have "
                    f"mismatched {proj} indices vs gate_proj: "
                    f"missing={missing}, unexpected={extra}."
                )
        # Walk: pop per-expert tensors in order, stack, and emit the combined
        # form upstream sanitize already handles.
        gates, ups, downs = [], [], []
        for e in range(len(gate_indices)):
            gates.append(new_weights.pop(f"{prefix}.{e}.gate_proj.weight"))
            ups.append(new_weights.pop(f"{prefix}.{e}.up_proj.weight"))
            downs.append(new_weights.pop(f"{prefix}.{e}.down_proj.weight"))
        new_weights[f"{prefix}.gate_up_proj"] = mx.concatenate(
            [mx.stack(gates), mx.stack(ups)], axis=-2
        )
        new_weights[f"{prefix}.down_proj"] = mx.stack(downs)
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

    # qwen3_5 (dense) checkpoints only need FP8 dequant — they have no expert
    # tensors to stack. Keep the dense patch narrow.
    def _transform_dense(_self, weights):
        return _dequantize_qwen35_fp8_weights(weights, mx)

    # qwen3_5_moe (Qwen-org Qwen3.6-MoE FP8) needs FP8 dequant followed by
    # per-expert stacking. The stacking step is the temporary downstream
    # complement to ml-explore/mlx-lm#1224 and short-circuits when no
    # per-expert keys are present.
    def _transform_moe(_self, weights):
        weights = _dequantize_qwen35_fp8_weights(weights, mx)
        weights = _stack_qwen36_moe_per_expert_weights(weights, mx)
        return weights

    transforms_by_module: dict[str, Any] = {
        "mlx_lm.models.qwen3_5": _transform_dense,
        "mlx_lm.models.qwen3_5_moe": _transform_moe,
    }

    patched_modules = []
    unpatchable_modules = []
    for module in model_modules:
        short_name = module.__name__.rsplit(".", maxsplit=1)[-1]
        model_cls = getattr(module, "Model", None)
        if model_cls is None:
            unpatchable_modules.append(short_name)
            continue
        transform = transforms_by_module.get(module.__name__)
        if transform is None:
            unpatchable_modules.append(short_name)
            continue
        if _wrap_model_sanitize(
            model_cls,
            "_vllm_metal_qwen35_fp8_patch",
            transform,
        ):
            patched_modules.append(short_name)
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
