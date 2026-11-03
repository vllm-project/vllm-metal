# SPDX-License-Identifier: Apache-2.0
"""Adapt mlx-vlm hidden states for encoder pooling."""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn


class MLXVLMEncoderAdapter(nn.Module):
    """Expose mlx-vlm token hidden states to encoder poolers."""

    def __init__(self, model: Any) -> None:
        super().__init__()
        self.model = model
        self.config = model.config

    def __call__(
        self,
        input_ids: mx.array,
        attention_mask: mx.array | None = None,
    ) -> mx.array:
        return cast(
            mx.array,
            self.model(input_ids, attention_mask=attention_mask).last_hidden_state,
        )
