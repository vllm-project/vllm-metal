# SPDX-License-Identifier: Apache-2.0
"""VLM utility helpers."""

from typing import Any


def _vlm_text_model(model: Any) -> Any:
    """Return the text sub-model for a VLM, or the model itself.

    mlx-vlm wraps the language model in a top-level ``Model`` that requires
    ``pixel_values`` and ``mask`` for multimodal forward passes.  For
    text-only inference the ``language_model`` sub-module must be used
    directly.  Non-VLM models are returned unchanged.
    """
    if hasattr(model, "language_model"):
        return model.language_model
    return model
