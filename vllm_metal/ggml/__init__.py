# SPDX-License-Identifier: Apache-2.0
"""ggml model-execution backend (Rust + ggml Metal), an alternative to MLX.

Enabled with ``VLLM_METAL_BACKEND=ggml``. The heavy lifting lives in the
``_ggml_engine`` extension built from ``rust/ggml-engine``; this package holds
the Python glue (loader, vLLM worker and model runner). Nothing here imports
MLX.
"""

from __future__ import annotations

import importlib
import os
from types import ModuleType

SUPPORTED_MODEL_TYPES = ("qwen3_5", "qwen3_5_text", "gemma4", "gemma4_text")


def load_extension() -> ModuleType:
    """Import ``_ggml_engine``, building it first if requested/needed."""
    try:
        return importlib.import_module("vllm_metal.ggml._ggml_engine")
    except ImportError:
        if os.environ.get("VLLM_METAL_BUILD_FROM_SOURCE", "0") != "1":
            raise ImportError(
                "vllm_metal ggml engine is not built. Run "
                "`python -m vllm_metal.ggml.build` (requires `brew install ggml` "
                "and a Rust toolchain) or set VLLM_METAL_BUILD_FROM_SOURCE=1."
            ) from None
    from vllm_metal.ggml.build import build

    build()
    return importlib.import_module("vllm_metal.ggml._ggml_engine")
