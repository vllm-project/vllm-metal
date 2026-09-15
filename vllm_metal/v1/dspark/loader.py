# SPDX-License-Identifier: MIT
# Copyright (c) 2026 erahim3
"""DSpark drafter checkpoint loader.

Adapted from ARahim3/mlx-dspark ``load.py`` (MIT) and vendored here so
vllm-metal has no runtime dependency on the standalone library. Only the
drafter loader is ported: the target model is loaded by vllm-metal's own
model lifecycle.
"""

from __future__ import annotations

import json
import math
import os
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from huggingface_hub import snapshot_download
from mlx.utils import tree_flatten, tree_unflatten
from safetensors import safe_open

from vllm_metal.utils import get_model_download_path

from .config import DSparkConfig
from .memory import KERNEL_RESERVE_BYTES
from .model import DSparkDrafter


def _resolve(repo_or_path: str, *, revision: str | None = None) -> str:
    repo_or_path = get_model_download_path(repo_or_path, revision=revision)
    if os.path.isdir(repo_or_path):
        return repo_or_path
    return snapshot_download(
        repo_or_path, revision=revision, allow_patterns=["*.json", "*.safetensors"]
    )


def _unique_object(pairs: list[tuple[str, object]]) -> dict:
    result: dict = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"DSpark checkpoint JSON repeats key {key!r}")
        result[key] = value
    return result


def _checkpoint_files(path: Path) -> dict[Path, set[str] | None]:
    """Follow an authoritative HF index; never combine unrelated weight files."""
    index = path / "model.safetensors.index.json"
    if not index.exists():
        files = sorted(path.glob("*.safetensors"))
        if len(files) != 1:
            raise ValueError(
                f"{path}: DSpark requires one safetensors file or an explicit "
                "model.safetensors.index.json"
            )
        return {files[0]: None}
    with index.open() as stream:
        mapping = json.load(stream, object_pairs_hook=_unique_object).get("weight_map")
    if not isinstance(mapping, dict) or not mapping:
        raise ValueError(f"{index}: weight_map must be a nonempty object")
    shards: dict[Path, set[str] | None] = {}
    for name, filename in mapping.items():
        if (
            not isinstance(filename, str)
            or Path(filename).name != filename
            or not filename.endswith(".safetensors")
        ):
            raise ValueError(f"{index}: invalid shard filename {filename!r}")
        shard = path / filename
        if not shard.is_file():
            raise ValueError(f"{index}: missing shard {filename!r}")
        names = shards.setdefault(shard, set())
        assert names is not None
        names.add(name)
    return dict(sorted(shards.items()))


def _validate_weights(
    files: dict[Path, set[str] | None], drafter: DSparkDrafter
) -> dict[str, int]:
    # Header-only checks precede any weight materialization or conversion.
    parameters = tree_flatten(drafter.parameters())
    assert isinstance(parameters, list)
    expected = {name: value.shape for name, value in parameters}
    actual: dict[str, tuple[int, ...]] = {}
    dtypes = set()
    for path, indexed_names in files.items():
        with safe_open(path, framework="numpy") as stream:
            names = set(stream.keys())
            if indexed_names is not None and names != indexed_names:
                raise ValueError(f"{path}: shard tensors disagree with weight_map")
            for name in sorted(names):
                if name in actual:
                    raise ValueError(f"{path}: duplicate tensor {name!r}")
                tensor = stream.get_slice(name)
                actual[name] = tuple(tensor.get_shape())
                dtypes.add(tensor.get_dtype())
    if actual.keys() != expected.keys():
        raise ValueError(
            "DSpark drafter tensor names do not match the DeepSpec checkpoint: "
            f"missing={sorted(expected.keys() - actual.keys())[:8]}, "
            f"unexpected={sorted(actual.keys() - expected.keys())[:8]}"
        )
    for name, shape in actual.items():
        if shape != expected[name]:
            raise ValueError(
                f"DSpark tensor {name!r}: expected shape {expected[name]}, got {shape}"
            )
    if len(dtypes) != 1 or not dtypes <= {"F16", "BF16", "F32"}:
        raise ValueError(
            "DSpark requires a uniform float16, bfloat16 or float32 checkpoint; "
            f"got tensor dtypes {sorted(dtypes)}"
        )
    itemsize = {"F16": 2, "BF16": 2, "F32": 4}[next(iter(dtypes))]
    return {name: math.prod(shape) * itemsize for name, shape in actual.items()}


def load_drafter(
    repo_or_path: str,
    *,
    revision: str | None = None,
    quantize: bool = True,
    bits: int = 4,
    group_size: int = 64,
    memory_budget_bytes: int | None = None,
    expected_config: DSparkConfig | None = None,
) -> tuple[DSparkDrafter, DSparkConfig]:
    """Load a DeepSpec-native standalone DSpark drafter. Returns ``(drafter, config)``.

    Tensor names, shapes and source precision must match before loading starts.
    The serving recipe is MLX affine 4-bit/group-64, including embeddings and
    heads. ``quantize=False`` preserves source precision for numerical reference
    checks. Prepacked and mixed-precision checkpoints are not supported.
    ``memory_budget_bytes`` is the remaining startup allowance after target
    weights; admission includes final draft weights and conversion overlap.
    """
    if bits != 4 or group_size != 64:
        raise ValueError("DSpark only implements affine 4-bit/group-64 conversion")
    path = Path(_resolve(repo_or_path, revision=revision))
    with (path / "config.json").open() as stream:
        raw = json.load(stream, object_pairs_hook=_unique_object)
    if (
        raw.get("quantization") is not None
        or raw.get("quantization_config") is not None
    ):
        raise ValueError("DSpark requires an unpacked floating-point checkpoint")
    config = DSparkConfig.from_dict(raw, source=str(path / "config.json"))
    if expected_config is not None and config != expected_config:
        raise ValueError(
            "DSpark snapshot config disagrees with the resolved draft ModelConfig"
        )
    drafter = DSparkDrafter(config)
    files = _checkpoint_files(path)
    source_bytes = _validate_weights(files, drafter)
    modules = dict(drafter.named_modules())
    resident_bytes = sum(source_bytes.values())
    if quantize:
        for name, module in modules.items():
            if isinstance(module, (nn.Linear, nn.Embedding)):
                if module.weight.shape[-1] % group_size:
                    raise ValueError(
                        f"DSpark {name}: input dimension must be divisible by group_size={group_size}"
                    )
                name_bytes = source_bytes[f"{name}.weight"]
                itemsize = name_bytes // module.weight.size
                packed_bytes = module.weight.size * bits // 8
                scale_bytes = 2 * (module.weight.size // group_size) * itemsize
                resident_bytes += packed_bytes + scale_bytes - name_bytes

    # MLX's memory limit is a guideline, not a hard allocation cap. Check an
    # explicit startup estimate before reading any data. Two source-tensor
    # buffers cover conversion overlap; the allowance is separate from serving
    # context/workspace because target KV does not exist during this phase.
    startup_bytes = (
        resident_bytes + 2 * max(source_bytes.values()) + KERNEL_RESERVE_BYTES
    )
    if memory_budget_bytes is not None and startup_bytes > memory_budget_bytes:
        raise ValueError(
            "DSpark startup memory budget is too small: "
            f"draft_weights_and_conversion={startup_bytes} bytes, "
            f"available_after_target={memory_budget_bytes} bytes. Increase "
            "--gpu-memory-utilization within device capacity or use a smaller pair."
        )

    # Materialize/convert one tensor at a time. Keeping an evaluated dictionary
    # of all original weights alongside a quantized model doubles startup RAM.
    # Disable allocator retention only for this scoped load, restoring it even
    # when a read or conversion fails.
    cache_limit = mx.set_cache_limit(0)
    try:
        for shard in files:
            weights = mx.load(shard)
            assert isinstance(weights, dict)
            for name in sorted(weights):
                value = weights.pop(name)
                # Check payloads as they stream in; shape/dtype headers cannot
                # detect NaNs or infinities. Never quantize invalid source data.
                if not mx.all(mx.isfinite(value)).item():
                    raise ValueError(
                        f"DSpark tensor {name!r} contains non-finite weights"
                    )
                # The full header set was checked strictly above. Partial
                # updates here are intentional, not a permissive load mode.
                drafter.load_weights([(name, value)], strict=False)
                module_name, parameter = name.rsplit(".", 1)
                module = modules.get(module_name)
                if (
                    quantize
                    and parameter == "weight"
                    and isinstance(module, (nn.Linear, nn.Embedding))
                ):
                    converted = module.to_quantized(group_size=group_size, bits=bits)
                    mx.eval(converted.parameters())
                    drafter.update_modules(tree_unflatten([(module_name, converted)]))
                    modules.pop(module_name)
                    del converted
                else:
                    mx.eval(value)
                del value, module
        mx.eval(drafter.parameters())
    finally:
        mx.set_cache_limit(cache_limit)
    return drafter, config
