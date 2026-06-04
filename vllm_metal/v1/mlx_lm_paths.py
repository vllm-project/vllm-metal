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

    model_path = model_path.resolve()
    index_path = model_path / "model.safetensors.index.json"
    if not index_path.is_file():
        yield model_path
        return

    with index_path.open("r", encoding="utf-8") as fid:
        index_data = json.load(fid)
    weight_map = index_data.get("weight_map", {})

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
            if (
                not src.is_file()
                or src.name.endswith(".safetensors")
                or src.name == "model.safetensors.index.json"
            ):
                continue
            (compat_path / src.name).symlink_to(src)

        total_shards = len(shard_names)
        shard_name_map: dict[str, str] = {}
        for shard_index, shard_name in enumerate(shard_names, start=1):
            shard_path = model_path / shard_name
            compat_name = (
                "model.safetensors"
                if total_shards == 1
                else f"model-{shard_index:05d}-of-{total_shards:05d}.safetensors"
            )
            (compat_path / compat_name).symlink_to(shard_path)
            shard_name_map[shard_name] = compat_name

        rewritten_index = dict(index_data)
        rewritten_index["weight_map"] = {
            weight_name: (
                shard_name_map.get(shard_name, shard_name)
                if isinstance(shard_name, str)
                else shard_name
            )
            for weight_name, shard_name in weight_map.items()
        }
        with (compat_path / "model.safetensors.index.json").open(
            "w", encoding="utf-8"
        ) as fid:
            json.dump(rewritten_index, fid, ensure_ascii=False)

        logger.info(
            "Using mlx_lm shard compatibility view for %s (%d shard files)",
            model_path,
            total_shards,
        )
        yield compat_path
