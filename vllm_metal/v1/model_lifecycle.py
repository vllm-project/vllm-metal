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
from vllm_metal.attention.runtime.families.gdn import build_gdn_hybrid_plan
from vllm_metal.compat import apply_compat_patches
from vllm_metal.compiled_mlp import CompiledMLPBlocks
from vllm_metal.gguf.source import GGUFLoadSource
from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx
from vllm_metal.quant.awq_loader import AWQQuantLoader
from vllm_metal.utils import get_model_download_path
from vllm_metal.v1.gemma4_mtp import Gemma4MTPAssistantLoader
from vllm_metal.v1.mlx_lm_paths import (
    mlx_lm_compatible_model_path as _mlx_lm_compatible_model_path,
)
from vllm_metal.v1.mm import EncoderCache
from vllm_metal.v1.model_adapter import ModelAdapter
from vllm_metal.v1.pooling.backends.decoder.factory import (
    build_decoder_pooling_backend,
)
from vllm_metal.v1.pooling.backends.encoder.factory import (
    load_encoder_pooling_backend,
    supports_encoder_pooling_backend,
)
from vllm_metal.v1.pooling.contract import LoadedEncoderBackend

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
    gguf_source: GGUFLoadSource | None
    lazy_weights: bool

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
        if is_vlm and model_config.quantization == "gguf":
            raise NotImplementedError(
                "Multimodal GGUF checkpoints are not supported by vllm-metal."
            )
        gguf_source = (
            None
            if is_vlm
            else GGUFLoadSource.from_model_config(
                model_config,
                runner.vllm_config.load_config,
            )
        )

        # A pipeline-parallel stage prunes its non-owned layers right after load, so
        # the generic MLX loaders stay lazy until the stage-owned weights are known.
        # The custom GGUF and AWQ loaders cannot honor this contract.
        pp = runner.pp
        lazy_weights = pp is not None and pp.size > 1

        return cls(
            model_name=(
                gguf_source.weights_path
                if gguf_source is not None
                else get_model_download_path(model_config.model)
            ),
            model_config=model_config,
            hf_config=hf_config,
            is_vlm=is_vlm,
            target_dtype=torch_to_mlx(torch.empty(0, dtype=model_config.dtype)).dtype,
            tokenizer_config={"trust_remote_code": model_config.trust_remote_code},
            gguf_source=gguf_source,
            lazy_weights=lazy_weights,
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
        """Load the configured model and install runner runtime state."""

        loaded_encoder_model = self._load_encoder_pooling()
        if loaded_encoder_model is not None:
            self._install_encoder_pooling_model(loaded_encoder_model)
            return

        request = GenerationLoadRequest.from_runner(self._runner, self._model_adapter)
        loaded_model = self._load_generation(request)

        self._install_generation_model(loaded_model, request)

        # Runtime extensions may depend on dimensions derived from model_args.
        self.resolve_model_dims()
        self._install_runtime_extensions(
            loaded_model.model_args,
            request,
        )

    def install_pooling_backend(self) -> None:
        """Install the pooling backend after load-time model mutation."""
        runner = self._runner
        runner._pooling_backend = build_decoder_pooling_backend(
            runner._forward_model,
            runner.model_config,
            runner.tokenizer,
        )

    def install_decode_dispatch(self) -> None:
        """Install the compiled-MLP decode dispatch."""
        if self._runner._is_pooling:
            # Pooling runners never decode; their backends may also wrap the
            # model in non-nn.Module shims the installer must not walk.
            return
        if getattr(self._runner.vllm_config, "lora_config", None) is not None:
            # LoRA setup rebinds projection modules inside the blocks after
            # install and swaps adapter state per batch; a compiled trace
            # would freeze that state at first call. LoRA serves stay eager.
            return
        CompiledMLPBlocks.install(self._runner._forward_model)

    def resolve_model_dims(self) -> None:
        """Resolve loaded model args into runner attention/cache dimensions."""
        args = self._runner.model_args
        default_head_dim = self._install_runner_attention_dims(args)
        self._install_yoco_cache_mapping(args)
        self._install_per_layer_attention_metadata(args, default_head_dim)
        self._reject_pipeline_parallel_with_per_layer_metadata()
        self._install_hybrid_state_plan(args)

    def _load_encoder_pooling(self) -> LoadedEncoderBackend | None:
        runner = self._runner
        if not runner._is_pooling or not supports_encoder_pooling_backend(
            runner.model_config
        ):
            return None

        if runner.pp is not None and runner.pp.size > 1:
            raise NotImplementedError(
                "Metal encoder pooling does not support pipeline parallelism yet."
            )
        if runner.vllm_config.lora_config is not None:
            raise NotImplementedError(
                "Metal encoder pooling does not support LoRA yet."
            )

        return load_encoder_pooling_backend(runner.model_config)

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
            gguf_source=request.gguf_source,
            lazy_weights=request.lazy_weights,
        )
        return LoadedGenerationModel(
            model=model,
            tokenizer=tokenizer,
            model_args=self._extract_model_args(model, request.is_vlm),
        )

    def _install_encoder_pooling_model(
        self,
        loaded_model: LoadedEncoderBackend,
    ) -> None:
        runner = self._runner
        runner.model = loaded_model.model
        runner.tokenizer = loaded_model.tokenizer
        runner._is_vlm = False
        runner._multimodal_adapter = None
        runner.encoder_cache = None
        runner.model_args = loaded_model.model_args
        runner._vocab_size = int(loaded_model.model_args["vocab_size"])
        runner._pooling_backend = loaded_model.pooling_backend
        logger.debug("Model args: %s", loaded_model.model_args)

    def _load_generation_model(
        self,
        model_name: str,
        is_vlm: bool,
        *,
        model_config: Any | None = None,
        target_dtype: Any | None = None,
        tokenizer_config: Mapping[str, Any] | None = None,
        gguf_source: GGUFLoadSource | None = None,
        lazy_weights: bool = False,
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
        if gguf_source is None and not is_vlm:
            gguf_source = GGUFLoadSource.from_model_config(
                model_config,
                self._runner.vllm_config.load_config,
            )
        is_gguf = gguf_source is not None
        awq_loader = None if is_gguf or is_vlm else AWQQuantLoader.for_model(model_name)

        if is_gguf:
            load_label = "GGUF model"
            model, tokenizer = self._load_gguf_text_model(
                gguf_source,
                target_dtype,
                tokenizer_config,
            )

        elif is_vlm:
            load_label = "MLX-VLM model"
            model, tokenizer = mlx_vlm_load(model_name, lazy=lazy_weights)

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
                lazy=lazy_weights,
            )

        loaded_from = (
            gguf_source.weights_path if gguf_source is not None else model_name
        )
        logger.info(
            "%s loaded in %.2fs: %s",
            load_label,
            time.time() - start_time,
            loaded_from,
        )
        return model, tokenizer

    def _load_gguf_text_model(
        self,
        source: GGUFLoadSource,
        target_dtype: Any,
        tokenizer_config: Mapping[str, Any],
    ) -> tuple[Any, Any]:
        """Load a GGUF checkpoint through the optional GGUF owner."""
        from vllm_metal.gguf.loader import GGUFModelLoader

        loader = GGUFModelLoader(
            source.weights_path,
            config_dir=source.config_dir,
            tokenizer_dir=source.tokenizer_dir,
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
        *,
        lazy: bool = False,
    ) -> tuple[Any, Any]:
        with _mlx_lm_compatible_model_path(model_name) as compatible_model_name:
            model, tokenizer = mlx_lm_load(
                str(compatible_model_name),
                tokenizer_config=tokenizer_config,
                lazy=lazy,
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
        logger.debug("Model args: %s", loaded_model.model_args)

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

    def _install_hybrid_state_plan(self, args: dict[str, Any]) -> None:
        """Resolve the state family that owns this model's hybrid layers."""
        if self._runner.is_hybrid:
            self._runner.hybrid_runtime_plan = build_gdn_hybrid_plan(
                args, self._runner.num_layers
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
            model_values = self._config_to_mapping(model_args)
        else:
            config = getattr(model, "config", None)
            if config is None:
                raise ValueError(
                    "Cannot extract model config: model has neither .args nor "
                    ".config attribute."
                )

            config_values = self._config_to_mapping(config)
            if is_vlm and "text_config" in config_values:
                model_values = self._config_to_mapping(config_values["text_config"])
            else:
                model_values = config_values

        text_config = model_values.get("text_config")
        if text_config is None:
            return model_values

        merged_values = dict(model_values)
        text_values = self._config_to_mapping(text_config)
        for key, value in text_values.items():
            merged_values.setdefault(key, value)
        return merged_values

    def _config_to_mapping(self, config: Any) -> dict[str, Any]:
        if isinstance(config, Mapping):
            return dict(config)
        return dict(vars(config))
