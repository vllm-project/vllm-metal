# SPDX-License-Identifier: Apache-2.0
"""Text-serving metadata for Inkling Mini on vLLM versions before 0.35."""

from transformers import PretrainedConfig
from vllm.model_executor.models.llama import LlamaForCausalLM


class Inkling2LoopConfig(PretrainedConfig):
    model_type = "inkling2_loop"
    sub_configs = {"text_config": PretrainedConfig, "vision_config": PretrainedConfig}

    def __init__(self, text_config=None, vision_config=None, **kwargs):
        super().__init__(**kwargs)
        self.text_config = PretrainedConfig(**(text_config or {}))
        self.vision_config = PretrainedConfig(**(vision_config or {}))


class Inkling2LoopForConditionalGeneration(LlamaForCausalLM):
    """Expose the text generation interface; MLX-VLM executes the model."""

    def __init__(self, *, vllm_config, prefix=""):
        raise NotImplementedError("Inkling Mini requires the Metal MLX model runner")
