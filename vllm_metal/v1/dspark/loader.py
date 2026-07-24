# SPDX-License-Identifier: Apache-2.0
"""DSpark drafter checkpoint loader.

Adapted from ARahim3/mlx-dspark ``load.py`` (MIT) — vendored here so vllm-metal
has no runtime dependency on the standalone library. Only the drafter loader is
ported: the target model is loaded by vllm-metal's own model lifecycle.
"""

from __future__ import annotations

import glob
import os
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download

from .config import DSparkConfig
from .model import DSparkDrafter


def _resolve(repo_or_path: str) -> str:
    if os.path.isdir(repo_or_path):
        return repo_or_path
    return snapshot_download(repo_or_path)


def _flatten_params(module) -> list[tuple[str, Any]]:
    from mlx.utils import tree_flatten

    params = tree_flatten(module.parameters())
    assert isinstance(params, list)  # Module.parameters() is a dict -> list of pairs
    return params


def load_drafter(
    repo_or_path: str,
    *,
    quantize: bool = True,
    bits: int = 4,
    group_size: int = 64,
    strict: bool = True,
) -> tuple[DSparkDrafter, DSparkConfig]:
    """Load a DeepSpec-native standalone DSpark drafter. Returns ``(drafter, config)``.

    Weights load 1:1 by tensor name (``strict=True``); a name mismatch raises
    instead of silently loading a partial drafter (which would draft with
    near-zero acceptance). 4-bit quantized by default — the drafter runs every
    round, so quantizing it is what makes speculation a net win on Apple Silicon.
    Output correctness is unaffected (the target verifies every token).
    """
    path = _resolve(repo_or_path)
    config = DSparkConfig.from_json(os.path.join(path, "config.json"))
    drafter = DSparkDrafter(config)

    weights: dict[str, mx.array] = {}
    for st in glob.glob(os.path.join(path, "*.safetensors")):
        weights.update(mx.load(st))

    model_keys = {k for k, _ in _flatten_params(drafter)}
    ckpt_keys = set(weights.keys())
    missing = sorted(model_keys - ckpt_keys)
    unexpected = sorted(ckpt_keys - model_keys)
    if missing or unexpected:
        detail = ""
        if missing:
            detail += f"\n  missing in checkpoint ({len(missing)}): {missing[:8]}"
        if unexpected:
            detail += f"\n  unexpected in checkpoint ({len(unexpected)}): {unexpected[:8]}"
        if strict:
            raise ValueError(
                f"{repo_or_path}: drafter tensor names don't match a DeepSpec-format "
                f"DSpark drafter — the checkpoint may be a different packaging or "
                f"variant (e.g. vLLM 'speculators' format).{detail}"
            )
        print(f"[load_drafter] WARNING key mismatch:{detail}")

    drafter.load_weights(list(weights.items()), strict=not (missing or unexpected))

    if quantize:
        nn.quantize(drafter, group_size=group_size, bits=bits)

    mx.eval(drafter.parameters())
    return drafter, config


def is_dspark_drafter(draft_model_config: object) -> bool:
    """True if a draft-model config points at a DeepSpec DSpark drafter checkpoint.

    Distinguishing fields vs an ordinary Qwen3/Gemma-4 target: the drafter's
    ``config.json`` carries ``block_size`` and ``target_layer_ids``.
    """
    hf_config = getattr(draft_model_config, "hf_config", None)
    if hf_config is None:
        return False
    return (
        getattr(hf_config, "block_size", None) is not None
        and getattr(hf_config, "target_layer_ids", None) is not None
    )
