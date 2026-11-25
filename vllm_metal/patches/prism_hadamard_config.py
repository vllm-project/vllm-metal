# SPDX-License-Identifier: Apache-2.0
"""PrismML Hadamard-rotated Qwen3.5 config for Transformers versions without it.

The model-compat hook imports this in every vLLM process, so it imports only
Transformers. mlx-vlm and mlx-lm run the model.
"""

from typing import Any

from transformers.models.qwen3_5.configuration_qwen3_5 import Qwen3_5Config


class PrismHadamardQwen35Config(Qwen3_5Config):
    model_type = "prism_hadamard_qwen35"

    def __post_init__(self, **kwargs: Any) -> None:
        # Qwen3_5Config.__post_init__ populates a default vision_config when None.
        has_vision = self.vision_config is not None
        super().__post_init__(**kwargs)
        if not self.architectures:
            self.architectures = [
                "Qwen3_5ForConditionalGeneration"
                if has_vision
                else "Qwen3_5ForCausalLM"
            ]
