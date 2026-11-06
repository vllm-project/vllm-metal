# SPDX-License-Identifier: Apache-2.0
"""Local path helpers for mlx-lm checkpoint loading."""

from __future__ import annotations

import json
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from tempfile import TemporaryDirectory

from vllm.logger import init_logger

logger = init_logger(__name__)


@contextmanager
def mlx_lm_compatible_model_path(model_name: str | Path) -> Iterator[Path]:
    """Yield a model path compatible with ``mlx_lm`` shard discovery.

    Some local checkpoints ship valid ``.safetensors`` shards and a
    ``model.safetensors.index.json`` file, but use custom shard names such as
    ``layers-*.safetensors`` or ``outside.safetensors``. ``mlx_lm`` only
    discovers shards whose basename matches ``model*.safetensors``.

    For those checkpoints, create a temporary directory that mirrors the
    original config/tokenizer files and exposes the indexed shard files via
    ``model-xxxxx-of-yyyyy.safetensors`` symlinks. The actual weight bytes stay
    in place; only the filenames are adapted for ``mlx_lm``.
    """
    model_path = Path(model_name)
    if not model_path.is_dir():
        yield model_path
        return

    if any(model_path.glob("model*.safetensors")):
        yield model_path
        return

    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        yield model_path
        return

    with index_path.open("r") as fid:
        weight_map = json.load(fid).get("weight_map", {})

    shard_names = sorted(
        {
            shard_name
            for shard_name in weight_map.values()
            if isinstance(shard_name, str) and shard_name.endswith(".safetensors")
        }
    )
    if not shard_names:
        yield model_path
        return

    with TemporaryDirectory(prefix="vllm-metal-mlx-lm-") as tmpdir:
        compat_path = Path(tmpdir)

        for src in model_path.iterdir():
            if not src.is_file() or src.name.endswith(".safetensors"):
                continue
            (compat_path / src.name).symlink_to(src)

        total_shards = len(shard_names)
        for shard_index, shard_name in enumerate(shard_names, start=1):
            shard_path = model_path / shard_name
            compat_name = (
                "model.safetensors"
                if total_shards == 1
                else f"model-{shard_index:05d}-of-{total_shards:05d}.safetensors"
            )
            (compat_path / compat_name).symlink_to(shard_path)

        logger.info(
            "Using mlx_lm shard compatibility view for %s (%d shard files)",
            model_path,
            total_shards,
        )
        yield compat_path


_MLX_LM_MODEL_TYPE_REMAP = {
    # Text-only Qwen3.5 checkpoints declare model_type "qwen3_5_text"; mlx-lm
    # ships the shared "qwen3_5" module instead (the HF-side *_text split is
    # newer than its MODEL_REMAPPING, e.g. "deepseek_v41_text").
    "qwen3_5_text": "qwen3_5",
}


def mlx_lm_model_type_override(hf_config) -> dict | None:
    """``model_config`` overrides routing ``hf_config``'s model_type at an
    mlx-lm module that exists; None when the checkpoint's type resolves
    directly."""
    model_type = getattr(hf_config, "model_type", None)
    remapped = _MLX_LM_MODEL_TYPE_REMAP.get(model_type)
    return {"model_type": remapped} if remapped is not None else None
