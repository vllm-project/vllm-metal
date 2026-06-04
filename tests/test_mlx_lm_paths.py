# SPDX-License-Identifier: Apache-2.0
"""Tests for local mlx-lm checkpoint compatibility views."""

from __future__ import annotations

import json
import logging
import sys
from importlib.util import find_spec
from pathlib import Path
from types import ModuleType

import pytest

if find_spec("vllm") is None:
    vllm_module = ModuleType("vllm")
    vllm_logger_module = ModuleType("vllm.logger")
    vllm_logger_module.init_logger = logging.getLogger
    vllm_module.logger = vllm_logger_module
    sys.modules.setdefault("vllm", vllm_module)
    sys.modules.setdefault("vllm.logger", vllm_logger_module)
    from vllm_metal.v1.mlx_lm_paths import mlx_lm_compatible_model_path

    sys.modules.pop("vllm.logger", None)
    sys.modules.pop("vllm", None)
else:
    from vllm_metal.v1.mlx_lm_paths import mlx_lm_compatible_model_path


def _write_indexed_checkpoint(
    model_dir: Path,
    *,
    shard_names: tuple[str, ...],
    weight_map: dict[str, object] | None = None,
    metadata: dict[str, object] | None = None,
) -> None:
    model_dir.mkdir(parents=True)
    for name in ("config.json", "tokenizer.json", "tokenizer_config.json"):
        (model_dir / name).write_text("{}", encoding="utf-8")
    for name in shard_names:
        (model_dir / name).write_text("", encoding="utf-8")

    effective_weight_map = weight_map or {
        f"weight_{index}": name for index, name in enumerate(shard_names)
    }
    (model_dir / "model.safetensors.index.json").write_text(
        json.dumps(
            {
                "metadata": metadata or {"total_size": len(shard_names)},
                "weight_map": effective_weight_map,
            }
        ),
        encoding="utf-8",
    )


def test_relative_custom_shards_resolve_inside_compat_view(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    parent = tmp_path / "parent"
    model_dir = parent / "model"
    _write_indexed_checkpoint(
        model_dir,
        shard_names=("layers-0.safetensors", "outside.safetensors"),
    )
    monkeypatch.chdir(parent)

    with mlx_lm_compatible_model_path("model") as compat:
        compat_path = Path(compat)
        compat_shards = sorted(compat_path.glob("model*.safetensors"))

        assert [path.name for path in compat_shards] == [
            "model-00001-of-00002.safetensors",
            "model-00002-of-00002.safetensors",
        ]
        assert all(path.resolve().is_file() for path in compat_shards)


def test_custom_shard_index_is_rewritten_to_generated_names(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    _write_indexed_checkpoint(
        model_dir,
        shard_names=("layers-0.safetensors", "outside.safetensors"),
        weight_map={
            "a": "outside.safetensors",
            "b": "layers-0.safetensors",
        },
        metadata={"total_size": 2, "format": "pt"},
    )

    with mlx_lm_compatible_model_path(str(model_dir)) as compat:
        rewritten_index = json.loads(
            (Path(compat) / "model.safetensors.index.json").read_text(encoding="utf-8")
        )

        assert rewritten_index["metadata"] == {"total_size": 2, "format": "pt"}
        assert rewritten_index["weight_map"] == {
            "a": "model-00002-of-00002.safetensors",
            "b": "model-00001-of-00002.safetensors",
        }


def test_single_custom_shard_index_is_rewritten_to_model_safetensors(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_indexed_checkpoint(
        model_dir,
        shard_names=("weights.safetensors",),
        weight_map={"embed.weight": "weights.safetensors"},
    )

    with mlx_lm_compatible_model_path(str(model_dir)) as compat:
        compat_path = Path(compat)
        rewritten_index = json.loads(
            (compat_path / "model.safetensors.index.json").read_text(encoding="utf-8")
        )

        assert (compat_path / "model.safetensors").resolve().is_file()
        assert rewritten_index["weight_map"] == {"embed.weight": "model.safetensors"}


def test_index_rewrite_preserves_non_string_weight_map_values(
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_indexed_checkpoint(
        model_dir,
        shard_names=("weights.safetensors",),
        weight_map={
            "embed.weight": "weights.safetensors",
            "quant_state": ["weights.safetensors"],
        },
    )

    with mlx_lm_compatible_model_path(str(model_dir)) as compat:
        rewritten_index = json.loads(
            (Path(compat) / "model.safetensors.index.json").read_text(encoding="utf-8")
        )

        assert rewritten_index["weight_map"] == {
            "embed.weight": "model.safetensors",
            "quant_state": ["weights.safetensors"],
        }


def test_standard_model_shards_use_original_directory(tmp_path: Path) -> None:
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "model-00001-of-00001.safetensors").write_text("", encoding="utf-8")

    with mlx_lm_compatible_model_path(str(model_dir)) as compat:
        assert Path(compat) == model_dir
