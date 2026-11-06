# SPDX-License-Identifier: Apache-2.0
"""GLM-5-Next multimodal adapter exports."""

from __future__ import annotations

from vllm_metal.multimodal.glm5_next.adapter import (
    Glm5NextMultimodalAdapter,
    Glm5NextVisionEncodeResult,
)

__all__ = ["Glm5NextMultimodalAdapter", "Glm5NextVisionEncodeResult"]
