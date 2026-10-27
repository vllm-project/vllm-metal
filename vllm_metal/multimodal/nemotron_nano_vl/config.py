# SPDX-License-Identifier: Apache-2.0
"""Local configuration reader for Nano VL V2 checkpoint exports.

Some MLX exports retain NVIDIA's AutoConfig entry but omit its Python files.
Only configuration parsing lives here; execution belongs to MLX-VLM.
"""

from __future__ import annotations

from typing import Any

from transformers import PretrainedConfig


def image_tile_limit(mm_config: Any = None) -> int:
    """The server's tile ceiling, shared by admission and encoder profiling."""
    from vllm.transformers_utils.processors.nano_nemotron_vl import DEFAULT_NUM_TILES

    kwargs = mm_config.merge_mm_processor_kwargs({}) if mm_config is not None else {}
    limit = kwargs.get("max_num_tiles")
    if limit is None:
        limit = DEFAULT_NUM_TILES
    if isinstance(limit, bool) or not isinstance(limit, int) or limit < 1:
        raise ValueError("Nano VL V2 max_num_tiles must be a positive integer.")
    return limit


class NemotronNanoVLConfig(PretrainedConfig):
    model_type = "NemotronH_Nano_VL_V2"
    is_composition = True
    sub_configs = {"text_config": PretrainedConfig, "vision_config": PretrainedConfig}

    def __init__(
        self,
        llm_config: dict[str, Any] | None = None,
        text_config: dict[str, Any] | None = None,
        vision_config: dict[str, Any] | None = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(**kwargs)
        text = dict(text_config if text_config is not None else llm_config or {})
        pattern = text.get("hybrid_override_pattern")
        if pattern is not None:
            names = {"M": "mamba", "*": "attention", "-": "mlp"}
            if not pattern or set(pattern) - names.keys():
                raise ValueError("Nano VL V2 requires a nonempty M/*/- layer pattern.")
            if len(pattern) != text.get("num_hidden_layers", len(pattern)):
                raise ValueError("Nano VL V2 layer pattern disagrees with layer count.")
            text["layers_block_type"] = [names[block] for block in pattern]
        self.text_config = PretrainedConfig(**text)
        self.vision_config = PretrainedConfig(**(vision_config or {}))


def register_config() -> None:
    """Register only the exact VL V2 architecture, leaving Omni independent."""
    from transformers import AutoConfig
    from transformers.models.auto.configuration_auto import CONFIG_MAPPING
    from vllm.transformers_utils.config import _CONFIG_REGISTRY

    _CONFIG_REGISTRY.setdefault(NemotronNanoVLConfig.model_type, NemotronNanoVLConfig)
    if NemotronNanoVLConfig.model_type not in CONFIG_MAPPING:
        AutoConfig.register(NemotronNanoVLConfig.model_type, NemotronNanoVLConfig)


def register_processing() -> None:
    from vllm.model_executor.models import ModelRegistry

    ModelRegistry.register_model(
        "NemotronH_Nano_VL_V2",
        "vllm_metal.multimodal.nemotron_nano_vl.processing:NemotronNanoVLForProcessing",
    )
