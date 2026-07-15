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
from functools import lru_cache
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
    _patch_vllm_gemma4_mtp_config_loading()
    _patch_vllm_bytelevel_tokenizer_loading()
    ensure_vllm_auto_fit_null_block_patch()
    _patch_mlx_lm_qwen35_fp8_sanitize()
    _patch_mlx_lm_gemma4_kv_shared_sanitize()
    _patch_transformers_exaone4_config()


def ensure_vllm_auto_fit_null_block_patch() -> None:
    """Reserve the null block in vLLM's auto-fit max_model_len estimate.

    ``_estimate_max_model_len_from_groups`` accepts the largest length whose KV
    demand fits the reported memory, but ``BlockPool`` permanently takes one of
    ``num_blocks`` as the null placeholder block, so a request at the
    auto-fitted length needs one more block than the pool can ever allocate.
    The scheduler admits it, allocation falls one block short, and the request
    preempts and requeues forever (observed live: gemma-4-31B-it at its fitted
    length sat in Waiting with 0% KV cache usage indefinitely). Shrink the
    searched budget by one pool block so every admissible length stays
    schedulable.

    Idempotent and safe to call repeatedly: plugin activation runs while vLLM
    is still partially initialized (the import below can fail there), so
    ``MetalPlatform.update_block_size_for_backend`` re-ensures it inside the
    engine process after vLLM is fully imported, before KV sizing runs.

    SCAFFOLDING: remove when the pinned vLLM includes the fix from
    vllm-project/vllm#48724.
    """
    try:
        from vllm.v1.core import kv_cache_utils
    except ImportError as exc:
        logger.debug("Skipping vLLM auto-fit null-block patch: %s", exc)
        return

    if getattr(kv_cache_utils, "_vllm_metal_auto_fit_null_block_patched", False):
        return

    original = kv_cache_utils._estimate_max_model_len_from_groups
    pool_bytes_per_block = kv_cache_utils._pool_bytes_per_block

    def _patched_estimate(
        vllm_config: Any, kv_cache_groups: Any, available_memory: int
    ) -> int:
        reserved = pool_bytes_per_block(vllm_config, kv_cache_groups)
        return original(
            vllm_config, kv_cache_groups, max(0, available_memory - reserved)
        )

    kv_cache_utils._estimate_max_model_len_from_groups = _patched_estimate
    kv_cache_utils._vllm_metal_auto_fit_null_block_patched = True
    logger.debug("Installed vLLM auto-fit null-block reservation patch")


def _patch_transformers_exaone4_config() -> None:
    """Guard ``Exaone4Config`` against a div-by-zero on no-sliding-window configs.

    transformers' ``Exaone4Config.__post_init__`` sets
    ``sliding_window_pattern = 0`` when ``sliding_window is None`` (EXAONE 4.0
    checkpoints without a sliding window, e.g. ``EXAONE-4.0-1.2B``), then derives
    ``layer_types`` via ``(i + 1) % sliding_window_pattern`` — a
    ``ZeroDivisionError``. The config, and therefore the whole model, fails to
    load through ``AutoConfig`` / ``mlx_lm`` / vLLM before any forward runs.

    Pre-populate ``layer_types`` as all ``"full_attention"`` (the real layout of
    a no-sliding-window model) so the original ``__post_init__`` skips the
    crashing derivation. The wrapper only fires on the exact crash condition
    (``layer_types is None and sliding_window is None``), so sliding-window
    variants (e.g. the 32B) keep transformers' own derivation untouched.

    TODO: remove once the supported transformers release no longer divides by a
    zero ``sliding_window_pattern`` (tracked upstream in transformers).
    """
    try:
        from transformers.models.exaone4 import configuration_exaone4
    except ImportError as exc:
        logger.debug("Skipping Exaone4 config compatibility patch: %s", exc)
        return

    config_cls = getattr(configuration_exaone4, "Exaone4Config", None)
    if config_cls is None or getattr(config_cls, "_vllm_metal_exaone4_patched", False):
        return

    original_post_init = config_cls.__post_init__

    def _patched_post_init(self: Any, **kwargs: Any) -> Any:
        if self.layer_types is None and self.sliding_window is None:
            self.layer_types = ["full_attention"] * self.num_hidden_layers
        return original_post_init(self, **kwargs)

    config_cls.__post_init__ = _patched_post_init
    config_cls._vllm_metal_exaone4_patched = True
    logger.debug("Installed Exaone4 no-sliding-window config compatibility patch")


def _metal_ray_local_gpu_ids(
    node_id: str,
    assigned: Mapping[str, float] | None,
    key: str,
) -> tuple[str, list[int]]:
    """Resolve a Metal Ray worker's ``(node_id, local_gpu_ids)`` from its resources.

    The Apple GPU is advertised as a custom Ray resource (``key``, e.g. "mlx")
    that never appears in ``get_accelerator_ids()`` — only in the worker's
    assigned resources. One Apple GPU per node means the local index list is
    ``[0]``. Fails loud if the worker was not scheduled onto a ``key`` resource:
    that is the only way the override is reachable in a correct setup, so a miss
    means a misconfigured ``ray start`` rather than a recoverable state.

    Kept module-level (rather than nested in the override) so the resolution
    logic is unit-testable without a live Ray cluster.
    """
    count = int((assigned or {}).get(key, 0))
    if count < 1:
        raise RuntimeError(
            f"Metal Ray worker on node {node_id} was not assigned a {key!r} "
            f"resource (assigned={assigned}). Start each node with "
            f"ray start --resources='{{\"{key}\": 1}}'"
        )
    return node_id, list(range(count))


def _patch_ray_distributed() -> None:
    """Override the Ray V2 worker actor's ``get_node_and_gpu_ids`` for Metal.

    Apple GPUs are not a Ray-recognized accelerator family, so a custom Ray
    resource ("mlx") never appears in
    ``ray.get_runtime_context().get_accelerator_ids()`` — only in
    ``get_assigned_resources()``.  The stock method indexes
    ``get_accelerator_ids()[ray_device_key]`` and would ``KeyError``.  We read
    the assigned custom resource instead (one Apple GPU per node -> local index
    ``[0]``), failing loud if a worker was not scheduled onto an "mlx" resource.

    vllm-metal supports only the default Ray V2 executor, whose worker actor is
    ``RayWorkerProc``.

    Installed in each Ray worker via the ``worker_process_setup_hook`` wired in
    ``MetalPlatform.check_and_update_config`` — it runs at worker startup, before
    the first actor call.  This is the only mechanism that works for real Ray: the
    driver never calls ``get_node_and_gpu_ids`` locally, actors re-import the class
    fresh in their own processes, and the lazy plugin-load path inside a worker
    runs too late (it would fire *inside* the unpatched method's first call).
    """
    from vllm.v1.executor.ray_executor_v2 import RayWorkerProc

    cls: Any = RayWorkerProc
    if getattr(cls, "_metal_patched", False):
        return

    def get_node_and_gpu_ids(self):  # noqa: ANN001, ANN202
        import ray as _ray
        from vllm.platforms import current_platform

        rc = _ray.get_runtime_context()
        return _metal_ray_local_gpu_ids(
            rc.get_node_id(),
            rc.get_assigned_resources(),
            current_platform.ray_device_key,
        )

    cls.get_node_and_gpu_ids = get_node_and_gpu_ids
    cls._metal_patched = True
    logger.info(
        "vllm_metal: patched Ray V2 worker get_node_and_gpu_ids on RayWorkerProc "
        "(Apple-GPU custom Ray resource)"
    )


def _gemma4_assistant_config_class() -> type[Any]:
    """Return a minimal Transformers config for raw Gemma4 assistant metadata."""
    from transformers.configuration_utils import PretrainedConfig
    from transformers.models.gemma4.configuration_gemma4 import Gemma4TextConfig

    class Gemma4AssistantCompatConfig(PretrainedConfig):
        model_type = "gemma4_assistant"
        sub_configs = {"text_config": Gemma4TextConfig}

        def __init__(self, text_config: Any | None = None, **kwargs: Any) -> None:
            if text_config is None:
                self.text_config = Gemma4TextConfig()
            elif isinstance(text_config, Gemma4TextConfig):
                self.text_config = text_config
            elif isinstance(text_config, Mapping):
                self.text_config = Gemma4TextConfig(**text_config)
            else:
                self.text_config = text_config
            super().__init__(**kwargs)

        def get_text_config(self, decoder: bool = False) -> Any:
            # Match Transformers' ``PretrainedConfig.get_text_config`` API.
            return self.text_config

    return Gemma4AssistantCompatConfig


def _transformers_knows_gemma4_assistant(auto_config: Any) -> bool:
    try:
        auto_config.for_model("gemma4_assistant")
    except ValueError:
        return False
    return True


def _patch_vllm_gemma4_mtp_config_loading() -> None:
    """Register raw Gemma4 assistant config parsing for the pinned stack.

    vLLM 0.21 already knows how to normalize ``gemma4_assistant`` into the
    ``gemma4_mtp`` draft wrapper, but the pinned Transformers release cannot
    parse the raw assistant checkpoint's ``model_type`` before that override
    runs. Keep the shim to config loading only; model construction remains owned
    by vLLM/vllm-metal Gemma4 MTP modules.
    """
    try:
        from transformers import AutoConfig
    except ImportError as exc:
        logger.warning(
            "Could not install Gemma4 MTP config compatibility patch: %s",
            exc,
        )
        return

    if _transformers_knows_gemma4_assistant(AutoConfig):
        return

    try:
        config_cls = _gemma4_assistant_config_class()
    except ImportError as exc:
        logger.warning(
            "Could not install Gemma4 MTP config compatibility patch because "
            "Gemma4 config classes are unavailable: %s",
            exc,
        )
        return

    # TODO: remove once the supported dependency stack parses raw
    # ``model_type=gemma4_assistant`` configs before vLLM's Gemma4 MTP
    # override runs.
    AutoConfig.register("gemma4_assistant", config_cls, exist_ok=True)
    logger.debug("Registered Gemma4 assistant config compatibility")


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
    if "cache_dir" not in cached_kwargs:
        download_dir = from_pretrained_kwargs.get("download_dir")
        if download_dir is not None:
            cached_kwargs["cache_dir"] = download_dir
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


def _tokenizers_backend_kwargs(kwargs: Mapping[str, Any]) -> dict[str, Any]:
    backend_kwargs = dict(kwargs)
    download_dir = backend_kwargs.pop("download_dir", None)
    if download_dir is not None and "cache_dir" not in backend_kwargs:
        backend_kwargs["cache_dir"] = download_dir
    return backend_kwargs


def _load_bytelevel_tokenizers_backend(
    path_or_repo_id: str | Path,
    args: tuple[Any, ...],
    kwargs: Mapping[str, Any],
) -> Any:
    from transformers.tokenization_utils_tokenizers import TokenizersBackend
    from vllm.tokenizers.hf import get_cached_tokenizer

    tokenizer = TokenizersBackend.from_pretrained(
        path_or_repo_id,
        *args,
        **_tokenizers_backend_kwargs(kwargs),
    )
    return get_cached_tokenizer(tokenizer)


def _maybe_load_bytelevel_tokenizers_backend(
    tokenizer: Any,
    path_or_repo_id: str | Path,
    args: tuple[Any, ...],
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

    tokenizer = _load_bytelevel_tokenizers_backend(
        path_or_repo_id,
        args,
        from_pretrained_kwargs,
    )
    logger.info(
        "Loaded TokenizersBackend from ByteLevel tokenizer.json after the "
        "serving tokenizer path loaded a non-ByteLevel decoder: %s",
        tokenizer_json,
    )
    return tokenizer


def _patch_vllm_bytelevel_tokenizer_loading() -> None:
    """Use TokenizersBackend at vLLM's serving tokenizer boundary when needed.

    Some MLX-community Qwen/DeepSeek redistributions ship a ByteLevel
    `tokenizer.json` but a `tokenizer_config.json` that can make newer
    transformers versions instantiate a Llama/SentencePiece-style decoder.
    That decoder leaves ByteLevel token pieces such as "\u0120" and "\u010a"
    in served text.
    """
    try:
        import vllm.tokenizers.registry as tokenizer_registry
        from vllm.tokenizers.protocol import TokenizerLike
    except ImportError as exc:
        logger.warning(
            "Could not install vLLM ByteLevel tokenizer compatibility patch "
            "because vLLM tokenizer registry is unavailable: %s",
            exc,
        )
        return

    sentinel = "_vllm_metal_bytelevel_decoder_patch"
    if getattr(tokenizer_registry, sentinel, False):
        return

    original_get_tokenizer = tokenizer_registry.get_tokenizer

    def _patched_get_tokenizer(
        tokenizer_name,
        *args,
        tokenizer_cls=TokenizerLike,
        trust_remote_code=False,
        revision=None,
        download_dir=None,
        **kwargs,
    ):
        tokenizer_mode, resolved_name, resolved_args, resolved_kwargs = (
            tokenizer_registry.cached_resolve_tokenizer_args(
                tokenizer_name,
                *args,
                trust_remote_code=trust_remote_code,
                revision=revision,
                download_dir=download_dir,
                **kwargs,
            )
        )
        tokenizer = original_get_tokenizer(
            tokenizer_name,
            *args,
            tokenizer_cls=tokenizer_cls,
            trust_remote_code=trust_remote_code,
            revision=revision,
            download_dir=download_dir,
            **kwargs,
        )

        if tokenizer_cls is not TokenizerLike or tokenizer_mode != "hf":
            return tokenizer

        # TODO(#337, transformers#45739): remove this local compatibility path
        # once vllm-metal's supported Transformers version handles ByteLevel
        # tokenizer.json + incorrect tokenizer_class mismatches upstream.
        return _maybe_load_bytelevel_tokenizers_backend(
            tokenizer,
            resolved_name,
            resolved_args,
            resolved_kwargs,
        )

    tokenizer_registry.get_tokenizer = _patched_get_tokenizer
    tokenizer_registry.cached_get_tokenizer = lru_cache(_patched_get_tokenizer)
    setattr(tokenizer_registry, sentinel, True)


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
