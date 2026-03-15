# SPDX-License-Identifier: Apache-2.0
"""Qwen3-ASR configuration and entrypoints.

Keep this module MLX-free so ``vllm_metal.stt.hf_config`` can import
``vllm_metal.stt.qwen3_asr.config`` without pulling in the model stack.
"""

from .config import Qwen3ASRAudioConfig, Qwen3ASRConfig, Qwen3ASRTextConfig

__all__ = [
    "Qwen3ASRAudioConfig",
    "Qwen3ASRConfig",
    "Qwen3ASRTextConfig",
]
