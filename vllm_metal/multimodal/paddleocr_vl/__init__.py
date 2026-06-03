# SPDX-License-Identifier: Apache-2.0
"""PaddleOCR-VL multimodal adapter exports."""

from __future__ import annotations

from vllm_metal.multimodal.paddleocr_vl.adapter import (
    PaddleOCRVLMultimodalAdapter,
    PaddleOCRVLVisionEncodeResult,
)

__all__ = ["PaddleOCRVLMultimodalAdapter", "PaddleOCRVLVisionEncodeResult"]
