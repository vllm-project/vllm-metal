# SPDX-License-Identifier: Apache-2.0
"""LFM2-VL multimodal adapter exports."""

from __future__ import annotations

from vllm_metal.multimodal.lfm2_vl.adapter import (
    Lfm2VlMultimodalAdapter,
    Lfm2VlVisionEncodeResult,
)

__all__ = ["Lfm2VlMultimodalAdapter", "Lfm2VlVisionEncodeResult"]
