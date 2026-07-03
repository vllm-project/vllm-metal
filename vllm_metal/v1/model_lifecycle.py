# SPDX-License-Identifier: Apache-2.0
"""Model load and metadata derivation for MetalModelRunner."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import torch
from mlx_lm import load as mlx_lm_load
from mlx_vlm import load as mlx_vlm_load
from vllm.logger import init_logger

from vllm_metal.attention.impls.mla import MLA_DEFAULT_QK_ROPE_HEAD_DIM
from vllm_metal.compat import apply_compat_patches
from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx
from vllm_metal.quant.awq_loader import AWQQuantLoader
from vllm_metal.utils import get_model_download_path
from vllm_metal.v1.gemma4_mtp import Gemma4MTPAssistantLoader
from vllm_metal.v1.mlx_lm_paths import (
    mlx_lm_compatible_model_path as _mlx_lm_compatible_model_path,
)
from vllm_metal.v1.mm import EncoderCache
from vllm_metal.v1.model_adapter import ModelAdapter

# Engine-core subprocesses don't always re-invoke `vllm_metal._register()`,
# so the compat patches applied there may be missing here. Reapply on import
# (idempotent via the `_APPLIED` guard in compat.py) to ensure mlx_lm sanitize
# patches are in place before any model load.
apply_compat_patches()

if TYPE_CHECKING:
    from vllm_metal.v1.model_runner import MetalModelRunner

logger = init_logger(__name__)


def load_stt_model(model_name: str) -> Any:
    """Load an STT model.

    Returns the loaded STT model. The caller (``STTModelRunner``) builds the
    per-model runtime adapter and wires it onto the runner.
    """
    start_time = time.time()

    from vllm_metal.stt.loader import load_model as stt_load_model

    logger.info("Loading STT model: %s", model_name)
    model = stt_load_model(model_name)
    logger.info("STT model loaded in %.2fs: %s", time.time() - start_time, model_name)
    return model


@dataclass(frozen=True, slots=True)
class GenerationLoadRequest:
    """Immutable load-time view of the runner config."""

    model_name: str
    model_config: Any
    hf_config: Any
    is_vlm: bool
    target_dtype: Any
    tokenizer_config: Mapping[str, Any]

    @classmethod
    def from_runner(
        cls,
        runner: MetalModelRunner,
        model_adapter: ModelAdapter,
    ) -> GenerationLoadRequest:
        model_config = runner.model_config
        # vLLM model_config shape varies across backends.
        hf_config = getattr(model_config, "hf_config", None)
        is_vlm = bool(getattr(model_config, "is_multimodal_model", False))
        if model_adapter.should_force_text_backbone(hf_config):
            is_vlm = False

        return cls(
            model_name=get_model_download_path(model_config.model),
            model_config=model_config,
            hf_config=hf_config,
            is_vlm=is_vlm,
            target_dtype=torch_to_mlx(torch.empty(0, dtype=model_config.dtype)).dtype,
            tokenizer_config={"trust_remote_code": model_config.trust_remote_code},
        )


@dataclass(frozen=True, slots=True)
class LoadedGenerationModel:
    """Loaded generation model plus metadata needed to wire the runner."""

    model: Any
    tokenizer: Any
    model_args: dict[str, Any]


class ModelLifecycle:
    def __init__(
        self,
        runner: MetalModelRunner,
        model_adapter: ModelAdapter,
    ) -> None:
        self._runner = runner
        self._model_adapter = model_adapter

    def load(self) -> None:
        """Load the generation model and install runner runtime state."""

        request = GenerationLoadRequest.from_runner(self._runner, self._model_adapter)
        loaded_model = self._load_generation(request)

        self._install_generation_model(loaded_model, request)

        # Runtime extensions may depend on dimensions derived from model_args.
        self.resolve_model_dims()
        self._install_runtime_extensions(
            loaded_model.model_args,
            request,
        )

    def resolve_model_dims(self) -> None:
        """Resolve loaded model args into runner attention/cache dimensions."""
        args = self._runner.model_args
        default_head_dim = self._install_runner_attention_dims(args)
        self._install_yoco_cache_mapping(args)
        self._install_per_layer_attention_metadata(args, default_head_dim)
        self._reject_pipeline_parallel_with_per_layer_metadata()
        self._install_hybrid_attention_dims(args)

    def _load_generation(
        self,
        request: GenerationLoadRequest,
    ) -> LoadedGenerationModel:
        model, tokenizer = self._load_generation_model(
            request.model_name,
            request.is_vlm,
            model_config=request.model_config,
            target_dtype=request.target_dtype,
            tokenizer_config=request.tokenizer_config,
        )
        return LoadedGenerationModel(
            model=model,
            tokenizer=tokenizer,
            model_args=self._extract_model_args(model, request.is_vlm),
        )

    def _load_generation_model(
        self,
        model_name: str,
        is_vlm: bool,
        *,
        model_config: Any | None = None,
        target_dtype: Any | None = None,
        tokenizer_config: Mapping[str, Any] | None = None,
    ) -> tuple[Any, Any]:
        """Load a text or VLM generation model."""

        model_config = (
            self._runner.model_config if model_config is None else model_config
        )
        tokenizer_config = (
            {"trust_remote_code": model_config.trust_remote_code}
            if tokenizer_config is None
            else dict(tokenizer_config)
        )
        if not is_vlm and target_dtype is None:
            target_dtype = torch_to_mlx(torch.empty(0, dtype=model_config.dtype)).dtype

        start_time = time.time()
        is_gguf = not is_vlm and model_config.quantization == "gguf"
        awq_loader = None if is_gguf or is_vlm else AWQQuantLoader.for_model(model_name)

        if is_gguf:
            load_label = "GGUF model"
            model, tokenizer = self._load_gguf_text_model(
                model_config,
                target_dtype,
                tokenizer_config,
            )

        elif is_vlm:
            load_label = "MLX-VLM model"
            model, tokenizer = mlx_vlm_load(model_name)

        elif awq_loader is not None:
            load_label = "AWQ model"
            model, tokenizer = self._load_awq_text_model(
                model_name,
                awq_loader,
                target_dtype,
                tokenizer_config,
            )

        else:
            load_label = "MLX-LM model"
            model, tokenizer = self._load_mlx_lm_text_model(
                model_name,
                tokenizer_config,
            )

        # For GGUF the engine integration rewrites model_config.model to the
        # config source; the weights the user served live in model_weights.
        loaded_from = model_config.model_weights if is_gguf else model_name
        logger.info(
            "%s loaded in %.2fs: %s",
            load_label,
            time.time() - start_time,
            loaded_from,
        )
        return model, tokenizer

    def _load_gguf_text_model(
        self,
        model_config: Any,
        target_dtype: Any,
        tokenizer_config: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        """Load a GGUF checkpoint through the optional GGUF owner."""
        from vllm_metal.gguf.loader import GGUFModelLoader

        # The GGUF engine integration rewrites model_config.model to the config
        # source and carries the .gguf path in model_config.model_weights.
        loader = GGUFModelLoader.for_model(
            model_config.model_weights,
            config_dir=model_config.model,
            target_dtype=target_dtype,
            tokenizer_config=dict(tokenizer_config),
        )
        return loader.load()

    def _load_awq_text_model(
        self,
        model_name: str,
        awq_loader: AWQQuantLoader,
        target_dtype: Any,
        tokenizer_config: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        with _mlx_lm_compatible_model_path(model_name) as compatible_model_name:
            return awq_loader.load(
                str(compatible_model_name),
                target_dtype=target_dtype,
                tokenizer_config=tokenizer_config,
            )

    def _load_mlx_lm_text_model(
        self,
        model_name: str,
        tokenizer_config: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        with _mlx_lm_compatible_model_path(model_name) as compatible_model_name:
            model, tokenizer = mlx_lm_load(
                str(compatible_model_name),
                tokenizer_config=tokenizer_config,
            )
        return model, tokenizer

    def _install_generation_model(
        self,
        loaded_model: LoadedGenerationModel,
        request: GenerationLoadRequest,
    ) -> None:
        """Install loaded generation state used by runner execution paths."""

        runner = self._runner
        runner.model = loaded_model.model
        runner.tokenizer = loaded_model.tokenizer
        runner._is_vlm = request.is_vlm

        # Adapter state follows the effective VLM mode, not the raw config flag.
        multimodal_adapter = (
            self._model_adapter.build_multimodal_adapter(
                loaded_model.model, request.hf_config
            )
            if request.is_vlm
            else None
        )
        runner._multimodal_adapter = multimodal_adapter
        runner.encoder_cache = (
            EncoderCache() if multimodal_adapter is not None else None
        )

        # Dimension resolution reads model_args immediately after this phase.
        runner.model_args = loaded_model.model_args
        runner._vocab_size = int(loaded_model.model_args["vocab_size"])
        if runner.metal_config.debug:
            logger.info("Model args: %s", loaded_model.model_args)

    def _install_runner_attention_dims(self, args: dict[str, Any]) -> int | None:
        """Install runner-wide attention dims and return the default head_dim."""
        num_layers = args.get("num_hidden_layers") or args.get("n_layers")
        num_attention_heads = args.get("num_attention_heads")
        num_kv_heads = (
            args.get("num_key_value_heads")
            or args.get("n_kv_heads")
            or num_attention_heads
        )
        hidden_size = args.get("hidden_size")
        default_head_dim = args.get("head_dim") or (
            hidden_size // num_attention_heads
            if hidden_size and num_attention_heads
            else None
        )
        head_dim = self._model_adapter.resolve_max_head_dim(args, default_head_dim)

        if not num_layers or not num_kv_heads or not head_dim:
            raise ValueError(
                "Cannot resolve model dimensions from model_args: "
                f"num_layers={num_layers!r}, num_kv_heads={num_kv_heads!r}, "
                f"head_dim={head_dim!r}. "
                f"Available keys: {sorted(args.keys())}"
            )

        runner = self._runner
        runner.num_layers = int(num_layers)
        runner.num_attention_heads = (
            int(num_attention_heads) if num_attention_heads is not None else None
        )
        runner.num_kv_heads = int(num_kv_heads)
        runner.hidden_size = int(hidden_size) if hidden_size is not None else None
        runner.head_dim = int(head_dim)

        if runner.is_mla:
            runner.num_kv_heads = 1
            runner.head_dim = int(args["kv_lora_rank"]) + int(
                args.get("qk_rope_head_dim", MLA_DEFAULT_QK_ROPE_HEAD_DIM)
            )
        return int(default_head_dim) if default_head_dim is not None else None

    def _install_yoco_cache_mapping(self, args: dict[str, Any]) -> None:
        """Install YOCO layer-to-cache mapping when the model uses shared KV."""
        yoco = self._model_adapter.build_yoco_cache_mapping(args)
        self._runner._yoco_cache_mapping = yoco
        self._runner.num_kv_cache_layers = (
            yoco[0] if yoco is not None else self._runner.num_layers
        )

    def _install_per_layer_attention_metadata(
        self,
        args: dict[str, Any],
        default_head_dim: int | None,
    ) -> None:
        """Install per-layer KV and sliding-window metadata."""
        # Use the default head_dim before any global-attention widening so
        # sliding-attention layers keep their true cache shape.
        if default_head_dim is not None:
            per_layer = self._model_adapter.build_per_layer_kv_shapes(
                args,
                num_layers=self._runner.num_layers,
                num_kv_heads=self._runner.num_kv_heads,
                head_dim=default_head_dim,
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

    def _reject_pipeline_parallel_with_per_layer_metadata(self) -> None:
        """Reject PP until per-layer metadata is stage-sliced."""
        # PP layer indices are stage-local; these metadata lists are still global.
        if (
            self._runner.pp is not None
            and self._runner.pp.size > 1
            and (
                self._runner.sliding_window_per_layer is not None
                or self._runner.kv_heads_per_layer is not None
                or self._runner.head_dim_per_layer is not None
            )
        ):
            raise NotImplementedError(
                "Pipeline parallelism is not supported for models with non-uniform "
                "per-layer KV (interleaved sliding-window / per-layer KV heads or "
                "head dims, e.g. Gemma4)."
            )

    def _install_hybrid_attention_dims(self, args: dict[str, Any]) -> None:
        """Install hybrid linear-attention dimensions for GDN-style models."""
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

    def _install_runtime_extensions(
        self,
        model_args: Mapping[str, Any],
        request: GenerationLoadRequest,
    ) -> None:
        """Install optional runtime extensions after model dimensions resolve."""

        runner = self._runner

        # Clear stale extension state before loading replacement extensions.
        runner._gemma4_mtp_assistant = None
        gemma4_mtp_assistant = Gemma4MTPAssistantLoader().load_if_needed(
            speculative_config=runner.vllm_config.speculative_config,
            target_hf_config=request.hf_config,
            target_model_args=model_args,
        )
        runner.kv_cache_dtype = request.target_dtype
        runner._gemma4_mtp_assistant = gemma4_mtp_assistant

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
