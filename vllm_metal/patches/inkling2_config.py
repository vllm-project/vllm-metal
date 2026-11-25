# SPDX-License-Identifier: Apache-2.0
"""Inkling Mini's Hugging Face config for vLLM versions before 0.35.

The model-compat hook imports this in every vLLM process, so it must not
import vLLM model code; the model class lives in ``inkling2_registry``.
"""

from transformers import PretrainedConfig


class Inkling2LoopConfig(PretrainedConfig):
    model_type = "inkling2_loop"
    sub_configs = {"text_config": PretrainedConfig, "vision_config": PretrainedConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        self.text_config = PretrainedConfig(**(text_config or {}))
        self.vision_config = PretrainedConfig(**(vision_config or {}))
