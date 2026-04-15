# SPDX-License-Identifier: Apache-2.0
"""Model load and metadata derivation for MetalModelRunner."""

from __future__ import annotations

import time
from collections.abc import Mapping
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
        if hf_config is not None and self._model_adapter.should_force_text_backbone(
            hf_config
        ):
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

        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(model_name)
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
            model, tokenizer = mlx_lm_load(
                model_name,
                tokenizer_config={
                    "trust_remote_code": self._runner.model_config.trust_remote_code
                },
            )

        with _MODEL_CACHE_LOCK:
            _MODEL_CACHE[model_name] = (model, tokenizer)
        logger.info("Model loaded in %.2fs: %s", time.time() - start_time, model_name)
        return model, tokenizer

    def _load_stt(self, model_name: str) -> None:
        start_time = time.time()

        with _MODEL_CACHE_LOCK:
            cached = _MODEL_CACHE.get(model_name)
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
                _MODEL_CACHE[model_name] = (model, None)
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
        # mlx-lm exposes .args while HF-backed models expose .config.
        model_args = getattr(model, "args", None)
        if model_args is not None:
            return self._config_to_mapping(model_args, label="model.args")

        config = getattr(model, "config", None)
        if config is None:
            raise ValueError(
                "Cannot extract model config: model has neither .args nor .config "
                "attribute."
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
