# SPDX-License-Identifier: Apache-2.0
"""Lifecycle glue for GGUF generation model loading."""

from __future__ import annotations

import time
from threading import Lock
from typing import Any

import torch
from vllm.logger import init_logger

from vllm_metal.gguf.loader import GGUFMLXLoader
from vllm_metal.gguf.refs import resolve_gguf_reference
from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx

logger = init_logger(__name__)


def load_gguf_generation_model(
    model_name: str,
    *,
    model_config: Any,
    model_cache: dict[tuple[str, str], tuple[Any, Any]],
    model_cache_lock: Lock,
    start_time: float,
) -> tuple[Any, Any]:
    target_dtype = torch_to_mlx(torch.empty(0, dtype=model_config.dtype)).dtype
    reference = resolve_gguf_reference(model_name, model_config=model_config)
    cache_key = reference.dense_cache_key(target_dtype=target_dtype)

    with model_cache_lock:
        cached = model_cache.get(cache_key)
    if cached is not None:
        logger.info(
            "GGUF model loaded from cache in %.3fs: %s",
            time.time() - start_time,
            reference.gguf_path,
        )
        return cached

    tokenizer_config = {"trust_remote_code": model_config.trust_remote_code}
    logger.info(
        "Using experimental dense GGUF loader: weights=%s config=%s",
        reference.gguf_path,
        reference.model_path,
    )
    model, tokenizer = GGUFMLXLoader(
        reference,
        target_dtype=target_dtype,
        tokenizer_config=tokenizer_config,
    ).load()

    with model_cache_lock:
        model_cache[cache_key] = (model, tokenizer)
    logger.info(
        "GGUF model loaded in %.2fs: %s",
        time.time() - start_time,
        reference.gguf_path,
    )
    return model, tokenizer
