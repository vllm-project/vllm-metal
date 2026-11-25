# SPDX-License-Identifier: Apache-2.0
"""Nanbeige's Hugging Face config for Transformers versions without it.

The model-compat hook imports this in every vLLM process, so it imports only
Transformers. mlx-lm runs the model.
"""

from transformers import PretrainedConfig


class NanbeigeConfig(PretrainedConfig):
    model_type = "nanbeige"
