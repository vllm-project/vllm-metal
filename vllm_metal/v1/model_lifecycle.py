# SPDX-License-Identifier: Apache-2.0
"""Model load and metadata derivation for MetalModelRunner."""

from __future__ import annotations

import json
import time
from collections.abc import Mapping
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory
from threading import Lock
from typing import TYPE_CHECKING, Any

import torch
from mlx_lm import load as mlx_lm_load
from mlx_vlm import load as mlx_vlm_load
from vllm.logger import init_logger

from vllm_metal.compat import apply_compat_patches
from vllm_metal.paged_attention_backend.mla import MLA_DEFAULT_QK_ROPE_HEAD_DIM
from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx
from vllm_metal.stt.detection import is_stt_model
from vllm_metal.utils import get_model_download_path
from vllm_metal.v1.model_adapter import ModelAdapter

# Engine-core subprocesses don't always re-invoke `vllm_metal._register()`,
# so the compat patches applied there may be missing here. Reapply on import
# (idempotent via the `_APPLIED` guard in compat.py) to ensure mlx_lm sanitize
# patches are in place before any model load.
apply_compat_patches()

if TYPE_CHECKING:
    from vllm_metal.v1.model_runner import MetalModelRunner

logger = init_logger(__name__)

_MODEL_CACHE: dict[tuple[str, str], tuple[Any, Any]] = {}
_MODEL_CACHE_LOCK = Lock()


def reset_model_cache() -> None:
    """Clear the process-level model cache.

    Intended for tests that load multiple large models in sequence and
    need a deterministic start between variants.  Uses the same lock
    that protects every other ``_MODEL_CACHE`` access.

    This is a narrow, test-oriented API so callers do not need to reach
    into the private module global directly.
    """
    with _MODEL_CACHE_LOCK:
        _MODEL_CACHE.clear()


def _generation_cache_key(model_name: str, *, is_vlm: bool) -> tuple[str, str]:
    loader = "mlx_vlm" if is_vlm else "mlx_lm"
    return (model_name, loader)


def _stt_cache_key(model_name: str) -> tuple[str, str]:
    return (model_name, "stt")


@contextmanager
def _mlx_lm_compatible_model_path(model_name: str):
    """Yield a model path compatible with ``mlx_lm.load`` shard discovery.

    Some local checkpoints ship valid ``.safetensors`` shards and a
    ``model.safetensors.index.json`` file, but use custom shard names such as
    ``layers-*.safetensors`` or ``outside.safetensors``. ``mlx_lm.load`` only
    discovers shards whose basename matches ``model*.safetensors``.

    For those checkpoints, create a temporary directory that mirrors the
    original config/tokenizer files and exposes the indexed shard files via
    ``model-xxxxx-of-yyyyy.safetensors`` symlinks. The actual weight bytes stay
    in place; only the filenames are adapted for ``mlx_lm``.
    """
    model_path = Path(model_name)
    if not model_path.is_dir():
        yield model_name
        return

    if any(model_path.glob("model*.safetensors")):
        yield model_name
        return

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        yield model_name
        return

    with index_path.open("r") as fid:
        weight_map = json.load(fid).get("weight_map", {})

    shard_names = sorted(
        {
            shard_name
            for shard_name in weight_map.values()
            if isinstance(shard_name, str) and shard_name.endswith(".safetensors")
        }
    )
    if not shard_names:
        yield model_name
        return

    with TemporaryDirectory(prefix="vllm-metal-mlx-lm-") as tmpdir:
        compat_path = Path(tmpdir)

        for src in model_path.iterdir():
            if not src.is_file() or src.name.endswith(".safetensors"):
                continue
            (compat_path / src.name).symlink_to(src)

        total_shards = len(shard_names)
        for shard_index, shard_name in enumerate(shard_names, start=1):
            shard_path = model_path / shard_name
            compat_name = (
                "model.safetensors"
                if total_shards == 1
                else f"model-{shard_index:05d}-of-{total_shards:05d}.safetensors"
            )
            (compat_path / compat_name).symlink_to(shard_path)

        logger.info(
            "Using mlx_lm shard compatibility view for %s (%d shard files)",
            model_name,
            total_shards,
        )
        yield str(compat_path)


class ModelLifecycle:
    def __init__(
        self,
        runner: MetalModelRunner,
        model_adapter: ModelAdapter,
    ) -> None:
        self._runner = runner
        self._model_adapter = model_adapter

    def load(self) -> None:
        runner = self._runner
        model_name = get_model_download_path(runner.model_config.model)
        if is_stt_model(model_name):
            self._load_stt(model_name)
            return

        model_config = runner.model_config
        # vLLM model_config shape varies across backends.
        hf_config = getattr(model_config, "hf_config", None)
        is_vlm = bool(getattr(model_config, "is_multimodal_model", False))
        if self._model_adapter.should_force_text_backbone(hf_config):
            is_vlm = False

        model, tokenizer = self._load_generation_model(model_name, is_vlm)

        runner.model = model
        runner.tokenizer = tokenizer
        runner._is_vlm = is_vlm
        runner._is_stt = False
        runner._stt_runtime_adapter = None

        model_args = self._extract_model_args(model, is_vlm)
        runner.model_args = model_args
        runner._vocab_size = int(model_args["vocab_size"])
        if runner.metal_config.debug:
            logger.info("Model args: %s", model_args)
        self.resolve_model_dims()
        runner.kv_cache_dtype = torch_to_mlx(
            torch.empty(0, dtype=model_config.dtype)
        ).dtype

    def _load_generation_model(self, model_name: str, is_vlm: bool) -> tuple[Any, Any]:
        logger.info("Loading model: %s (VLM: %s)", model_name, is_vlm)
        start_time = time.time()
        cache_key = _generation_cache_key(model_name, is_vlm=is_vlm)

        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            logger.info(
                "Model loaded from cache in %.3fs: %s",
                time.time() - start_time,
                model_name,
            )
            return cached

        if is_vlm:
            logger.info("Using mlx-vlm for vision-language model")
            logger.warning(
                "VLM loaded in text-only mode: multimodal (image) inputs are "
                "not yet supported. Vision encoder will be bypassed."
            )
            model, tokenizer = mlx_vlm_load(model_name)
        else:
            with _mlx_lm_compatible_model_path(model_name) as compatible_model_name:
                model, tokenizer = mlx_lm_load(
                    compatible_model_name,
                    tokenizer_config={
                        "trust_remote_code": self._runner.model_config.trust_remote_code
                    },
                )

        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[cache_key] = (model, tokenizer)
        logger.info("Model loaded in %.2fs: %s", time.time() - start_time, model_name)
        return model, tokenizer

    def _load_stt(self, model_name: str) -> None:
        start_time = time.time()
        cache_key = _stt_cache_key(model_name)

        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(cache_key)
        if cached is not None:
            model, _ = cached
            load_time = time.time() - start_time
            logger.info(
                "STT model loaded from cache in %.3fs: %s",
                load_time,
                model_name,
            )
        else:
            from vllm_metal.stt.loader import load_model as stt_load_model

            logger.info("Loading STT model: %s", model_name)
            model = stt_load_model(model_name)
            with _MODEL_CACHE_LOCK:
                _MODEL_CACHE[cache_key] = (model, None)
            load_time = time.time() - start_time
            logger.info("STT model loaded in %.2fs: %s", load_time, model_name)

        self._runner.model = model
        self._runner.tokenizer = None
        self._runner.model_args = {}
        self._runner.kv_cache_dtype = None
        self._runner._is_vlm = False
        self._runner._is_stt = True
        self._runner._stt_runtime_adapter = model.create_runtime_adapter(model_name)

    def resolve_model_dims(self) -> None:
        args = self._runner.model_args
        num_layers = args.get("num_hidden_layers") or args.get("n_layers")
        num_attention_heads = args.get("num_attention_heads")
        num_kv_heads = (
            args.get("num_key_value_heads")
            or args.get("n_kv_heads")
            or num_attention_heads
        )
        hidden_size = args.get("hidden_size")
        base_head_dim = args.get("head_dim") or (
            hidden_size // num_attention_heads
            if hidden_size and num_attention_heads
            else None
        )
        head_dim = self._model_adapter.resolve_max_head_dim(args, base_head_dim)

        missing = []
        if not num_layers:
            missing.append("num_layers (num_hidden_layers / n_layers)")
        if not num_kv_heads:
            missing.append("num_kv_heads (num_key_value_heads / n_kv_heads)")
        if not head_dim:
            missing.append("head_dim")
        if missing:
            raise ValueError(
                f"Cannot resolve model dimensions: {', '.join(missing)}. "
                f"Available keys: {sorted(args.keys())}"
            )

        self._runner.num_layers = int(num_layers)
        self._runner.num_attention_heads = (
            int(num_attention_heads) if num_attention_heads is not None else None
        )
        self._runner.num_kv_heads = int(num_kv_heads)
        self._runner.hidden_size = int(hidden_size) if hidden_size is not None else None
        self._runner.head_dim = int(head_dim)

        if self._runner.is_mla:
            self._runner.num_kv_heads = 1
            self._runner.head_dim = int(args["kv_lora_rank"]) + int(
                args.get("qk_rope_head_dim", MLA_DEFAULT_QK_ROPE_HEAD_DIM)
            )

        yoco = self._model_adapter.build_yoco_cache_mapping(args)
        self._runner._yoco_cache_mapping = yoco
        self._runner.num_kv_cache_layers = (
            yoco[0] if yoco is not None else self._runner.num_layers
        )

        # Per-layer KV shapes for heterogeneous models (Gemma4 26B/31B).
        # Uses the unresolved ``base_head_dim`` so sliding-attention layers
        # get their true head_dim (256) rather than the max-with-global used
        # for cache allocation (512).  Returns None for uniform models,
        # leaving the scalar paths on the runner unchanged.
        #
        # ``base_head_dim`` is None only when neither ``head_dim`` nor
        # ``hidden_size / num_attention_heads`` could be resolved — the
        # missing-check above already raises in that case, but we guard
        # here too so ``int()`` never receives None.
        if base_head_dim is not None:
            per_layer = self._model_adapter.build_per_layer_kv_shapes(
                args,
                num_layers=self._runner.num_layers,
                num_kv_heads=self._runner.num_kv_heads,
                head_dim=int(base_head_dim),
            )
        else:
            per_layer = None
        if per_layer is not None:
            self._runner.kv_heads_per_layer, self._runner.head_dim_per_layer = per_layer
        else:
            self._runner.kv_heads_per_layer = None
            self._runner.head_dim_per_layer = None

        self._runner.sliding_window_per_layer = (
            self._model_adapter.build_sliding_window_per_layer(
                args, self._runner.num_layers
            )
        )

        if self._runner.is_hybrid:
            fai = int(args["full_attention_interval"])
            self._runner.full_attention_interval = fai
            self._runner.sdpa_layer_indices = frozenset(
                i for i in range(self._runner.num_layers) if (i + 1) % fai == 0
            )
            self._runner.num_sdpa_layers = len(self._runner.sdpa_layer_indices)
            self._runner.num_linear_layers = (
                self._runner.num_layers - self._runner.num_sdpa_layers
            )
            self._runner.linear_num_k_heads = int(args["linear_num_key_heads"])
            self._runner.linear_num_v_heads = int(args["linear_num_value_heads"])
            self._runner.linear_key_head_dim = int(args["linear_key_head_dim"])
            self._runner.linear_value_head_dim = int(args["linear_value_head_dim"])
            self._runner.linear_conv_kernel_dim = int(args["linear_conv_kernel_dim"])
            # Qwen3.5 GDN packs q/k at key_dim and v at value_dim.
            self._runner.linear_conv_dim = (
                self._runner.linear_num_k_heads * self._runner.linear_key_head_dim * 2
                + self._runner.linear_num_v_heads * self._runner.linear_value_head_dim
            )

    def _extract_model_args(self, model: Any, is_vlm: bool) -> dict[str, Any]:
        # Both the .args (mlx-lm) and .config (HF) paths may expose a nested
        # ``text_config`` (e.g. Gemma4 via mlx-lm); the merge below flattens
        # its keys onto the top level so every key sits in one flat dict.
        model_args = getattr(model, "args", None)
        if model_args is not None:
            model_values = self._config_to_mapping(model_args, label="model.args")
        else:
            config = getattr(model, "config", None)
            if config is None:
                raise ValueError(
                    "Cannot extract model config: model has neither .args nor "
                    ".config attribute."
                )

            config_values = self._config_to_mapping(config, label="config")
            if is_vlm and "text_config" in config_values:
                model_values = self._config_to_mapping(
                    config_values["text_config"],
                    label="text_config",
                )
            else:
                model_values = config_values

        text_config = model_values.get("text_config")
        if text_config is None:
            return model_values

        merged_values = dict(model_values)
        text_values = self._config_to_mapping(text_config, label="text_config")
        for key, value in text_values.items():
            merged_values.setdefault(key, value)
        return merged_values

    def _config_to_mapping(self, config: Any, *, label: str) -> dict[str, Any]:
        missing = object()

        if isinstance(config, Mapping):
            return dict(config)

        to_dict = getattr(config, "to_dict", None)
        if callable(to_dict):
            values = to_dict()
            if isinstance(values, Mapping):
                return dict(values)
            raise TypeError(f"{label}.to_dict() must return a mapping.")

        instance_dict = getattr(config, "__dict__", None)
        if instance_dict is not None:
            return dict(instance_dict)

        slot_values: dict[str, Any] = {}
        for cls in type(config).__mro__:
            slots = cls.__dict__.get("__slots__", ())
            if isinstance(slots, str):
                slots = (slots,)
            for name in slots:
                if not isinstance(name, str) or name.startswith("__"):
                    continue
                value = getattr(config, name, missing)
                if value is not missing:
                    slot_values[name] = value
        if slot_values:
            return slot_values

        raise TypeError(
            f"{label} must expose a mapping, to_dict(), __dict__, or __slots__."
        )
