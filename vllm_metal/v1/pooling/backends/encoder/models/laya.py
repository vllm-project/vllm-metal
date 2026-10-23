# SPDX-License-Identifier: Apache-2.0
"""Native Laya loading, registration, and upstream MLX encoder pooling."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import mlx.core as mx
import mlx.nn as nn
import torch
from mlx_vlm.models.laya import Model as LayaModel
from mlx_vlm.models.laya import ModelConfig as LayaModelConfig
from transformers import ModernBertConfig, PreTrainedConfig
from transformers.utils.hub import cached_file
from vllm.pooling_params import PoolingParams
from vllm.tasks import PoolingTask
from vllm.tokenizers.hf import CachedHfTokenizer, HfTokenizer
from vllm.tokenizers.protocol import TokenizerLike
from vllm.tokenizers.registry import cached_tokenizer_from_config
from vllm.transformers_utils.config_parser_base import ConfigParserBase

from vllm_metal.tensor_bridge import TORCH_TO_MLX_DTYPE, mlx_to_torch
from vllm_metal.v1.pooling.backends.encoder.models.loading import (
    encoder_model_path,
    load_encoder_weights,
)
from vllm_metal.v1.pooling.backends.encoder.runtime import MetalEncoderPoolingBackend
from vllm_metal.v1.pooling.contract import (
    TOKEN_CLASSIFY_TASK,
    EncoderPoolingRequest,
    LoadedEncoderBackend,
)
from vllm_metal.v1.pooling.validation import PoolingConfigView


class LayaMetadataConfig(PreTrainedConfig):
    """Recognize the Hub tracking config probed by vLLM's tokenizer registry."""

    model_type = "laya"


class LayaConfigParser(ConfigParserBase):
    def parse(
        self,
        model: str | Path,
        trust_remote_code: bool,
        revision: str | None = None,
        code_revision: str | None = None,
        **kwargs: Any,
    ) -> tuple[dict, PreTrainedConfig]:
        configs = []
        for filename in ("encoder/config.json", "rl_agent_config.json"):
            path = cached_file(
                model,
                filename,
                revision=revision,
                token=kwargs.get("token"),
                cache_dir=kwargs.get("cache_dir"),
            )
            assert path is not None
            configs.append(json.loads(Path(path).read_text()))
        encoder, agent = configs
        if encoder["model_type"] != "modernbert":
            raise NotImplementedError("Metal Laya currently supports ModernBERT only.")
        encoder.update(
            architectures=["LayaForDecision"], dtype="float32", laya_config=agent
        )
        return encoder, ModernBertConfig.from_dict(encoder)


class LayaTokenizer(TokenizerLike):
    @classmethod
    def from_pretrained(
        cls, path_or_repo_id: str | Path, *args: Any, **kwargs: Any
    ) -> HfTokenizer:
        kwargs.setdefault("subfolder", "tokenizer")
        # The tokenizer subdirectory has no model config for AutoTokenizer.
        kwargs.setdefault("config", LayaMetadataConfig())
        return CachedHfTokenizer.from_pretrained(path_or_repo_id, *args, **kwargs)


class LayaForDecision(torch.nn.Module):
    """Architecture metadata until vLLM ships its LayaForDecision registration."""

    is_pooling_model = True
    attn_type = "encoder_only"
    default_tok_pooling_type = "ALL"

    def __init__(self, *, vllm_config, prefix=""):
        raise RuntimeError("Laya inference requires the Metal encoder pooling backend.")

    def embed_input_ids(self, input_ids):
        raise RuntimeError("Laya embeddings are owned by the Metal encoder loader.")

    def forward(self, input_ids, positions):
        raise RuntimeError("Laya inference requires the Metal encoder pooling backend.")


def register() -> None:
    from transformers import CONFIG_MAPPING, AutoConfig
    from vllm.model_executor.models import ModelRegistry
    from vllm.renderers.registry import RENDERER_REGISTRY
    from vllm.tokenizers.registry import TokenizerRegistry
    from vllm.transformers_utils.config import (
        get_config_parser,
        register_config_parser,
    )

    if "laya" not in CONFIG_MAPPING:
        AutoConfig.register("laya", LayaMetadataConfig)

    try:
        get_config_parser("laya")
    except ValueError:
        register_config_parser("laya")(LayaConfigParser)
    if "laya" not in TokenizerRegistry.tokenizers:
        TokenizerRegistry.register(
            "laya",
            "vllm_metal.v1.pooling.backends.encoder.models.laya",
            "LayaTokenizer",
        )
    if "laya" not in RENDERER_REGISTRY.renderers:
        RENDERER_REGISTRY.register("laya", "vllm.renderers.hf", "HfRenderer")

    if "LayaForDecision" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "LayaForDecision",
            "vllm_metal.v1.pooling.backends.encoder.models.laya:LayaForDecision",
        )


def _create_model(config: dict[str, Any], agent_config: dict[str, Any]) -> LayaModel:
    if config["model_type"] != "modernbert":
        raise NotImplementedError("Metal Laya supports ModernBERT only.")
    if config["hidden_activation"] != "gelu":
        raise NotImplementedError("Metal Laya supports GELU only.")
    if any(v["rope_type"] != "default" for v in config["rope_parameters"].values()):
        raise NotImplementedError("Metal Laya supports default RoPE only.")
    if len(agent_config.get("act_costs", {})) != 1:
        raise NotImplementedError("Metal Laya requires two action classes.")
    model = LayaModel(
        LayaModelConfig(
            encoder_config=config,
            decision_config=agent_config,
            head_layers=agent_config["head_layers"],
        )
    )
    # mlx-vlm 0.7.6 uses tanh GELU in ModernBERT; this checkpoint requires GELU.
    for layer in model.encoder.layers:
        layer.mlp.act = nn.GELU(approx="none")
    return model


def clamp_temperature(value):
    if isinstance(value, bool):
        return 1.0
    try:
        value = float(value)
    except (TypeError, ValueError):
        return 1.0
    return min(5.0, max(0.5, value)) if math.isfinite(value) else 1.0


def temperature_for(config, qtype, options):
    bucket = (
        "2"
        if options <= 2
        else "3-5"
        if options <= 5
        else "6-10"
        if options <= 10
        else "11+"
    )
    key = f"{['choice', 'score', 'noul'][qtype]}:{bucket}"
    return clamp_temperature(
        config.get("temperature_by_options", {}).get(
            key, config.get("temperature", [1.0] * 3)[qtype]
        )
    )


class LayaPoolingBackend(MetalEncoderPoolingBackend):
    def __init__(
        self, config: PoolingConfigView, model: LayaModel, agent_config: dict[str, Any]
    ):
        super().__init__(config, self._encode)
        self.model = model
        self.agent_config = agent_config

    def _encode(self, input_ids: mx.array, attention_mask: mx.array) -> mx.array:
        return self.model.encoder(input_ids, attention_mask).last_hidden_state

    def supported_tasks(self) -> tuple[PoolingTask, ...]:
        if (
            self.config.is_text_only
            and self.config.task in (None, TOKEN_CLASSIFY_TASK)
            and self.config.pooler_config.tok_pooling_type in (None, "ALL")
            and not self.config.chunked_processing_enabled
        ):
            return (TOKEN_CLASSIFY_TASK,)
        return ()

    def validate_params(self, pooling_params: PoolingParams) -> None:
        if not self.supported_tasks() or pooling_params.task not in (
            None,
            TOKEN_CLASSIFY_TASK,
        ):
            raise NotImplementedError(
                "Metal Laya supports only task='token_classify' with ALL token pooling."
            )
        if pooling_params.dimensions is not None:
            raise NotImplementedError(
                "Metal Laya does not support truncated output dimensions."
            )

    def _pool_request(self, request: EncoderPoolingRequest) -> torch.Tensor:
        self.validate_params(request.pooling_params)
        markers = [
            i
            for i, token in enumerate(request.token_ids)
            if token == self.agent_config["mask_token_id"]
        ]
        if not markers:
            return torch.empty((0, 3), dtype=torch.float32)
        type_token = request.token_ids[1] if len(request.token_ids) > 1 else -1
        types = self.agent_config["qtype_token_ids"]
        qtype = types.index(type_token) if type_token in types else 0
        count = len(markers)
        # Upstream computes top-2 features even for a single option.
        padding = max(0, 2 - count)
        input_ids = mx.array([request.token_ids], dtype=mx.int32)
        logits, action_logits = self.model(
            input_ids,
            mx.ones_like(input_ids),
            mx.array([markers + [0] * padding], dtype=mx.int32),
            mx.array([[True] * count + [False] * padding], dtype=mx.bool_),
            mx.array([qtype], dtype=mx.int32),
        )
        logits = logits[:, :count]
        if request.pooling_params.use_activation is not False:
            logits = mx.softmax(
                logits / temperature_for(self.agent_config, qtype, count), axis=-1
            )
            action_logits = mx.softmax(action_logits.astype(mx.float32), axis=-1)
        actions = mx.broadcast_to(action_logits[:, None, :], (1, count, 2))
        output = mx.concatenate([logits[:, :, None], actions], axis=-1)[0]
        return mlx_to_torch(output, device="cpu").detach().clone()

    def profile_forward(self, input_ids: mx.array) -> mx.array:
        batch, length = input_ids.shape
        width = max(2, length)
        positions = mx.broadcast_to(
            mx.minimum(mx.arange(width), length - 1)[None, :], (batch, width)
        )
        valid = mx.broadcast_to((mx.arange(width) < length)[None, :], (batch, width))
        logits, actions = self.model(
            input_ids,
            mx.ones_like(input_ids),
            positions,
            valid,
            mx.zeros((batch,), dtype=mx.int32),
        )
        # Keep both branches in the lazy graph used for memory profiling.
        return mx.concatenate([logits, actions], axis=-1)


def supports_laya_encoder(model_config: Any) -> bool:
    return "LayaForDecision" in (model_config.hf_config.architectures or ())


def load_laya_backend(model_config: Any) -> LoadedEncoderBackend:
    config = model_config.hf_config.to_dict()
    if model_config.quantization is not None or config.get("quantization_config"):
        raise NotImplementedError("Metal Laya does not support quantization yet.")
    target_dtype = TORCH_TO_MLX_DTYPE[model_config.dtype]
    agent_config = dict(config["laya_config"])
    model = _create_model(config, agent_config)
    tokenizer = cached_tokenizer_from_config(model_config)
    if tokenizer is None:
        raise ValueError("Laya requires tokenizer initialization.")
    qtype_ids = [
        tokenizer(f"{name} question: ", add_special_tokens=False)["input_ids"][0]
        for name in ("choice", "score", "noul")
    ]
    if tokenizer.mask_token_id is None or len(set(qtype_ids)) != 3:
        raise ValueError("Laya requires a MASK token and three distinct type tokens.")
    agent_config.update(
        mask_token_id=tokenizer.mask_token_id, qtype_token_ids=qtype_ids
    )
    config["laya_config"] = agent_config
    weights = model.sanitize(load_encoder_weights(encoder_model_path(model_config)))
    weights = {
        name: value.astype(target_dtype)
        if mx.issubdtype(value.dtype, mx.floating)
        else value
        for name, value in weights.items()
    }
    model.load_weights(list(weights.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    return LoadedEncoderBackend(
        model,
        tokenizer,
        config,
        LayaPoolingBackend(PoolingConfigView(model_config), model, agent_config),
    )
