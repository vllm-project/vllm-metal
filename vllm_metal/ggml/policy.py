# SPDX-License-Identifier: Apache-2.0
"""Config-time policy for ``VLLM_METAL_BACKEND=ggml``.

Called from ``MetalPlatform.check_and_update_config`` so unsupported
combinations fail before any worker spawns.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from vllm.logger import init_logger

from vllm_metal.ggml import SUPPORTED_MODEL_TYPES

if TYPE_CHECKING:
    from vllm.config import VllmConfig

logger = init_logger(__name__)

WORKER_CLS = "vllm_metal.ggml.worker.GGMLWorker"
DEFAULT_BLOCK_SIZE = 16


def apply_ggml_config_policy(vllm_config: VllmConfig) -> None:
    model_config = vllm_config.model_config
    parallel_config = vllm_config.parallel_config
    cache_config = vllm_config.cache_config

    if parallel_config.worker_cls == "auto":
        parallel_config.worker_cls = WORKER_CLS

    if model_config is None:
        return

    model_type = getattr(model_config.hf_config, "model_type", None)
    if model_type not in SUPPORTED_MODEL_TYPES:
        raise NotImplementedError(
            f"VLLM_METAL_BACKEND=ggml does not support model_type={model_type!r} yet "
            f"(supported: {', '.join(SUPPORTED_MODEL_TYPES)})"
        )
    if model_config.quantization is not None:
        raise NotImplementedError(
            "VLLM_METAL_BACKEND=ggml loads unquantized HF safetensors only"
        )
    for name, value in (
        ("speculative decoding", vllm_config.speculative_config),
        ("LoRA", vllm_config.lora_config),
    ):
        if value is not None:
            raise NotImplementedError(
                f"VLLM_METAL_BACKEND=ggml does not support {name} yet"
            )
    if (
        parallel_config.pipeline_parallel_size > 1
        or parallel_config.data_parallel_size > 1
    ):
        raise NotImplementedError(
            "VLLM_METAL_BACKEND=ggml supports a single worker only"
        )

    # The engine is text-only: skip multimodal processor setup.
    if model_config.multimodal_config is not None:
        model_config.multimodal_config = None
        logger.info("ggml backend: serving %s on its text backbone", model_type)

    # Recurrent (linear-attention) state is kept per sequence by the runner,
    # not in vLLM blocks, so cached prefixes cannot restore it.
    if model_config.is_hybrid and cache_config.enable_prefix_caching:
        logger.warning(
            "ggml backend: disabling prefix caching for hybrid model %s "
            "(recurrent state is not block-cached)",
            model_type,
        )
        cache_config.enable_prefix_caching = False
        cache_config.mamba_cache_mode = "none"
        cache_config.mamba_block_size = model_config.max_model_len


def update_block_size(vllm_config: VllmConfig) -> None:
    """The engine takes any block size; skip Metal's hybrid alignment."""
    cache_config = vllm_config.cache_config
    if not cache_config.user_specified_block_size:
        cache_config.block_size = DEFAULT_BLOCK_SIZE
