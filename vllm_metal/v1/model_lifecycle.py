# SPDX-License-Identifier: Apache-2.0
"""Model lifecycle coordination for MetalModelRunner."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from threading import Lock
from typing import TYPE_CHECKING, Any

import torch
from mlx_lm import load as mlx_lm_load
from mlx_vlm import load as mlx_vlm_load
from vllm.logger import init_logger

from vllm_metal.paged_attention_backend.mla import MLA_DEFAULT_QK_ROPE_HEAD_DIM
from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx
from vllm_metal.stt.detection import is_stt_model
from vllm_metal.utils import get_model_download_path
from vllm_metal.v1.model_adapter import ModelAdapter

if TYPE_CHECKING:
    from vllm_metal.v1.model_runner import MetalModelRunner

logger = init_logger(__name__)

_MODEL_CACHE: dict[str, tuple[Any, Any]] = {}
_MODEL_CACHE_LOCK = Lock()


@dataclass(frozen=True)
class ResolvedModelDims:
    """Normalized runtime dimensions derived from model config."""

    num_layers: int
    num_attention_heads: int | None
    num_kv_heads: int
    hidden_size: int | None
    head_dim: int


class ModelLifecycle:
    """Owns model load and metadata extraction for MetalModelRunner."""

    def __init__(
        self,
        runner: MetalModelRunner,
        model_adapter: ModelAdapter,
    ) -> None:
        self._runner = runner
        self._model_adapter = model_adapter

    def is_vlm_model(self) -> bool:
        """Whether the configured model should load through mlx-vlm."""
        hf_config = getattr(self._runner.model_config, "hf_config", None)
        if hf_config is not None and self._model_adapter.should_force_text_backbone(
            hf_config
        ):
            return False
        return bool(getattr(self._runner.model_config, "is_multimodal_model", False))

    def load(self) -> None:
        """Load the configured model and derive runtime metadata."""
        model_name = get_model_download_path(self._runner.model_config.model)

        if is_stt_model(model_name):
            self.load_stt(model_name)
            return

        is_vlm = self.is_vlm_model()
        logger.info("Loading model: %s (VLM: %s)", model_name, is_vlm)
        start_time = time.time()

        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(model_name)
        if cached is not None:
            self._runner.model, self._runner.tokenizer = cached
            self._runner._is_vlm = is_vlm
            self._runner._is_stt = False
            self._runner._stt_runtime_adapter = None
            load_time = time.time() - start_time
            logger.info(
                "Model loaded from cache in %.3fs: %s", load_time, model_name
            )
            self._finalize_load()
            return

        if is_vlm:
            logger.info("Using mlx-vlm for vision-language model")
            logger.warning(
                "VLM loaded in text-only mode: multimodal (image) inputs are "
                "not yet supported. Vision encoder will be bypassed."
            )
            self._runner.model, self._runner.tokenizer = mlx_vlm_load(model_name)
            self._runner._is_vlm = True
        else:
            self._runner.model, self._runner.tokenizer = mlx_lm_load(
                model_name,
                tokenizer_config={
                    "trust_remote_code": self._runner.model_config.trust_remote_code
                },
            )
            self._runner._is_vlm = False

        self._runner._is_stt = False
        self._runner._stt_runtime_adapter = None

        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[model_name] = (self._runner.model, self._runner.tokenizer)

        self._finalize_load()
        load_time = time.time() - start_time
        logger.info("Model loaded in %.2fs: %s", load_time, model_name)

    def load_stt(self, model_name: str) -> None:
        """Load a Speech-to-Text model and create its runtime adapter."""
        start_time = time.time()

        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(model_name)
        if cached is not None:
            self._runner.model, _ = cached
            load_time = time.time() - start_time
            logger.info(
                "STT model loaded from cache in %.3fs: %s",
                load_time,
                model_name,
            )
        else:
            from vllm_metal.stt.loader import load_model as stt_load_model

            logger.info("Loading STT model: %s", model_name)
            self._runner.model = stt_load_model(model_name)
            with _MODEL_CACHE_LOCK:
                _MODEL_CACHE[model_name] = (self._runner.model, None)
            load_time = time.time() - start_time
            logger.info("STT model loaded in %.2fs: %s", load_time, model_name)

        self._runner.tokenizer = None
        self._runner.model_args = {}
        self._runner.kv_cache_dtype = None
        self._runner._is_vlm = False
        self._runner._is_stt = True
        self._runner._stt_runtime_adapter = self._runner.model.create_runtime_adapter(
            model_name
        )

    def initialize_kv_cache_dtype(self) -> None:
        """Resolve the MLX KV cache dtype from the configured torch dtype."""
        self._runner.kv_cache_dtype = torch_to_mlx(
            torch.empty(0, dtype=self._runner.model_config.dtype)
        ).dtype

    def extract_model_args(self) -> None:
        """Normalize model config into the runner's shared mapping shape."""
        if self._runner._is_vlm:
            model_args = self._vlm_model_args()
        else:
            model_args = self._text_model_args()

        merged_args = self._merge_text_config(model_args)
        self._runner.model_args = merged_args
        self._runner._vocab_size = int(merged_args["vocab_size"])

        if self._runner.metal_config.debug:
            logger.info("Model args: %s", merged_args)

    def resolve_model_dims(self) -> None:
        """Derive validated runtime dimensions from ``runner.model_args``."""
        args = self._runner.model_args
        dims = self._resolve_base_dims(args)

        self._runner.num_layers = dims.num_layers
        self._runner.num_attention_heads = dims.num_attention_heads
        self._runner.num_kv_heads = dims.num_kv_heads
        self._runner.hidden_size = dims.hidden_size
        self._runner.head_dim = dims.head_dim

        if self._runner.is_mla:
            self._apply_mla_dims(args)

        yoco = self._model_adapter.build_yoco_cache_mapping(args)
        self._runner._yoco_cache_mapping = yoco
        self._runner.num_kv_cache_layers = (
            yoco[0] if yoco is not None else self._runner.num_layers
        )

        if self._runner.is_hybrid:
            self._apply_hybrid_dims(args)

    def _finalize_load(self) -> None:
        self.extract_model_args()
        self.resolve_model_dims()
        self.initialize_kv_cache_dtype()

    def _text_model_args(self) -> dict[str, Any]:
        model_args = getattr(self._runner.model, "args", None)
        if model_args is not None:
            return self._config_to_mapping(model_args, label="model.args")

        config = getattr(self._runner.model, "config", None)
        if config is not None:
            return self._config_to_mapping(config, label="config")

        raise ValueError(
            "Cannot extract model config: model has neither .args nor .config "
            "attribute."
        )

    def _vlm_model_args(self) -> dict[str, Any]:
        config = getattr(self._runner.model, "config", None)
        if config is None:
            raise ValueError("Cannot extract VLM config: model has no .config.")

        config_dict = self._config_to_mapping(config, label="config")
        text_config = config_dict.get("text_config")
        if text_config is None:
            return config_dict
        return self._config_to_mapping(text_config, label="text_config")

    def _merge_text_config(self, model_args: dict[str, Any]) -> dict[str, Any]:
        merged = dict(model_args)
        text_config = merged.get("text_config")
        if text_config is None:
            return merged

        if isinstance(text_config, Mapping):
            text_values = dict(text_config)
        else:
            text_values = self._config_to_mapping(text_config, label="text_config")

        for key, value in text_values.items():
            merged.setdefault(key, value)
        return merged

    def _config_to_mapping(self, config: Any, *, label: str) -> dict[str, Any]:
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

        slot_values = self._slots_to_mapping(config)
        if slot_values is not None:
            return slot_values

        raise TypeError(
            f"{label} must expose a mapping, to_dict(), __dict__, or __slots__."
        )

    def _slots_to_mapping(self, config: Any) -> dict[str, Any] | None:
        slots = getattr(type(config), "__slots__", ())
        if isinstance(slots, str):
            slots = (slots,)
        if not slots:
            return None

        values: dict[str, Any] = {}
        for name in slots:
            if not isinstance(name, str) or name.startswith("__"):
                continue
            if hasattr(config, name):
                values[name] = getattr(config, name)
        return values

    def _resolve_base_dims(self, args: dict[str, Any]) -> ResolvedModelDims:
        num_layers = args.get("num_hidden_layers") or args.get("n_layers")
        num_attention_heads = args.get("num_attention_heads")
        num_kv_heads = (
            args.get("num_key_value_heads")
            or args.get("n_kv_heads")
            or num_attention_heads
        )
        hidden_size = args.get("hidden_size")
        head_dim = args.get("head_dim") or (
            hidden_size // num_attention_heads
            if hidden_size and num_attention_heads
            else None
        )
        head_dim = self._model_adapter.resolve_max_head_dim(args, head_dim)

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

        return ResolvedModelDims(
            num_layers=int(num_layers),
            num_attention_heads=(
                int(num_attention_heads) if num_attention_heads is not None else None
            ),
            num_kv_heads=int(num_kv_heads),
            hidden_size=int(hidden_size) if hidden_size is not None else None,
            head_dim=int(head_dim),
        )

    def _apply_mla_dims(self, args: dict[str, Any]) -> None:
        self._runner.num_kv_heads = 1
        self._runner.head_dim = int(args["kv_lora_rank"]) + int(
            args.get("qk_rope_head_dim", MLA_DEFAULT_QK_ROPE_HEAD_DIM)
        )

    def _apply_hybrid_dims(self, args: dict[str, Any]) -> None:
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
        self._runner.linear_conv_dim = (
            self._runner.linear_num_k_heads * self._runner.linear_key_head_dim * 2
            + self._runner.linear_num_v_heads * self._runner.linear_value_head_dim
        )
