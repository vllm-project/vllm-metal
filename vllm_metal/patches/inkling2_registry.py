# SPDX-License-Identifier: Apache-2.0
"""Text-serving metadata for Inkling Mini on vLLM versions before 0.35."""

from vllm.model_executor.models.llama import LlamaForCausalLM


class Inkling2LoopForConditionalGeneration(LlamaForCausalLM):
    """Expose the text generation interface; MLX-VLM executes the model."""

    def __init__(self, *, vllm_config, prefix=""):
        raise NotImplementedError("Inkling Mini requires the Metal MLX model runner")
