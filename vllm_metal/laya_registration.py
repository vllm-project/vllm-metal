# SPDX-License-Identifier: Apache-2.0
"""vLLM architecture metadata for the Metal-owned Laya encoder loader.

Remove this fallback when vLLM ships its LayaForDecision registration.
"""

import torch.nn as nn


class LayaForDecision(nn.Module):
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
    from vllm.model_executor.models import ModelRegistry

    if "LayaForDecision" not in ModelRegistry.get_supported_archs():
        ModelRegistry.register_model(
            "LayaForDecision", "vllm_metal.laya_registration:LayaForDecision"
        )
