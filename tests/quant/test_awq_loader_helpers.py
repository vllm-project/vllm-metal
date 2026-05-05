# SPDX-License-Identifier: Apache-2.0
"""Tests for the loader-side helpers in `model_lifecycle`:

- `_generation_cache_key` isolates entries by target dtype, so the
  AWQ post-load mutation cannot leak across engines with different dtypes.
- `_read_raw_quantization_config` mirrors mlx-lm's fallback to
  `text_config.quantization_config` for nested / wrapper configs.

Both are pure helpers; no model load, no downloads.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest
import torch

from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx
from vllm_metal.quant.awq_config import (
    UnsupportedQuantizationConfigError,
    normalize_quant_config,
)
from vllm_metal.v1.model_lifecycle import (
    _generation_cache_key,
    _maybe_normalize_awq_model_config,
    _read_raw_quantization_config,
    _stt_cache_key,
)


def _mlx_dtype(torch_dtype):
    """Mirror what `_load_generation_model` does to derive the cache-key
    dtype: convert a torch dtype through `torch_to_mlx`. Tests use this
    instead of literal strings so we pin the *actual* MLX dtype the
    production path passes in (e.g. ``mlx.core.bfloat16``), not just
    "different strings produce different keys".
    """
    return torch_to_mlx(torch.empty(0, dtype=torch_dtype)).dtype


# ---- cache-key dtype isolation ---------------------------------------------


def test_generation_cache_key_isolates_by_dtype():
    """Two engines requesting the same model with different dtypes must NOT
    share a cache entry, since AWQ post-load alignment mutates the model."""
    bf16_key = _generation_cache_key(
        "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        is_vlm=False,
        target_dtype=_mlx_dtype(torch.bfloat16),
    )
    fp16_key = _generation_cache_key(
        "Qwen/Qwen2.5-1.5B-Instruct-AWQ",
        is_vlm=False,
        target_dtype=_mlx_dtype(torch.float16),
    )
    assert bf16_key != fp16_key


def test_generation_cache_key_uses_mlx_dtype_string():
    """Pin the wire format of the third key segment to the actual MLX
    dtype string (e.g. ``"mlx.core.bfloat16"``), so a future refactor that
    accidentally swaps to torch dtype repr would fail loudly instead of
    silently producing different keys for the same logical dtype.
    """
    key = _generation_cache_key(
        "foo", is_vlm=False, target_dtype=_mlx_dtype(torch.bfloat16)
    )
    assert key == ("foo", "mlx_lm", str(mx.bfloat16))
    assert key[2] == "mlx.core.bfloat16"


def test_generation_cache_key_isolates_by_loader():
    """VLM and mlx_lm paths must not collide even at the same dtype."""
    bf16 = _mlx_dtype(torch.bfloat16)
    vlm_key = _generation_cache_key(
        "Qwen/Qwen2.5-VL-7B-Instruct", is_vlm=True, target_dtype=bf16
    )
    lm_key = _generation_cache_key(
        "Qwen/Qwen2.5-VL-7B-Instruct", is_vlm=False, target_dtype=bf16
    )
    assert vlm_key != lm_key


def test_generation_cache_key_stable_for_same_inputs():
    bf16 = _mlx_dtype(torch.bfloat16)
    a = _generation_cache_key("foo", is_vlm=False, target_dtype=bf16)
    b = _generation_cache_key("foo", is_vlm=False, target_dtype=bf16)
    assert a == b


def test_stt_cache_key_shape_matches_generation():
    """STT key must be a 3-tuple too so the shared `_MODEL_CACHE` dict is
    homogeneous. The third element is empty since STT does not depend on
    runtime dtype."""
    key = _stt_cache_key("openai/whisper-tiny")
    assert isinstance(key, tuple)
    assert len(key) == 3
    assert key[2] == ""


# ---- text_config.quantization_config fallback ------------------------------


def _write_config(tmp_path: Path, config: dict) -> Path:
    """Drop a config.json into `tmp_path` and return the directory."""
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


_AWQ_INNER = {
    "quant_method": "awq",
    "bits": 4,
    "group_size": 128,
    "zero_point": True,
    "version": "gemm",
}


def test_read_quantization_config_top_level(tmp_path):
    model_dir = _write_config(
        tmp_path,
        {"model_type": "qwen2", "quantization_config": _AWQ_INNER},
    )
    qc = _read_raw_quantization_config(str(model_dir))
    assert qc == _AWQ_INNER


def test_read_quantization_config_nested_text_config(tmp_path):
    """Multimodal wrapper configs nest the quant config under `text_config`.
    `mlx_lm.utils.load_model` falls back to it; the vllm-metal preflight
    must do the same so the alias normalization / reject logic still runs.
    """
    model_dir = _write_config(
        tmp_path,
        {
            "model_type": "wrapper_vlm",
            "text_config": {
                "model_type": "qwen2",
                "quantization_config": _AWQ_INNER,
            },
        },
    )
    qc = _read_raw_quantization_config(str(model_dir))
    assert qc == _AWQ_INNER


def test_read_quantization_config_top_level_wins_over_text_config(tmp_path):
    """If both are present, top-level takes precedence (matches mlx-lm)."""
    nested = {**_AWQ_INNER, "bits": 8}  # would normally reject
    model_dir = _write_config(
        tmp_path,
        {
            "model_type": "wrapper",
            "quantization_config": _AWQ_INNER,
            "text_config": {
                "quantization_config": nested,
            },
        },
    )
    qc = _read_raw_quantization_config(str(model_dir))
    assert qc == _AWQ_INNER  # top-level, not the bits=8 nested one


def test_read_quantization_config_absent(tmp_path):
    model_dir = _write_config(tmp_path, {"model_type": "qwen2"})
    assert _read_raw_quantization_config(str(model_dir)) is None


def test_read_quantization_config_missing_dir():
    """Non-existent path / non-HF repo: silently inactive, returns None."""
    assert _read_raw_quantization_config("/nonexistent/path/zzz") is None


def test_maybe_normalize_passes_through_for_non_awq(tmp_path):
    """Non-AWQ/GPTQ checkpoints get a None back, no exception."""
    model_dir = _write_config(
        tmp_path,
        {
            "model_type": "qwen2",
            "quantization_config": {"quant_method": "fp8", "bits": 8},
        },
    )
    assert _maybe_normalize_awq_model_config(str(model_dir)) is None


def test_maybe_normalize_returns_kwarg_dict_for_awq(tmp_path):
    model_dir = _write_config(
        tmp_path,
        {"model_type": "qwen2", "quantization_config": _AWQ_INNER},
    )
    out = _maybe_normalize_awq_model_config(str(model_dir))
    assert out is not None
    assert "quantization_config" in out
    assert out["quantization_config"]["quant_method"] == "awq"


def test_maybe_normalize_propagates_reject_via_text_config(tmp_path):
    """The text_config fallback must surface reject errors too — nesting must
    not be a hole that lets unsupported configs through."""
    model_dir = _write_config(
        tmp_path,
        {
            "model_type": "wrapper",
            "text_config": {
                "quantization_config": {
                    "quant_method": "awq",
                    "bits": 8,  # reject
                    "group_size": 128,
                },
            },
        },
    )
    with pytest.raises(UnsupportedQuantizationConfigError):
        _maybe_normalize_awq_model_config(str(model_dir))


def test_normalize_quant_config_idempotent_via_helper(tmp_path):
    """End-to-end shape: read -> normalize is the same as the direct call."""
    model_dir = _write_config(
        tmp_path,
        {"model_type": "qwen2", "quantization_config": _AWQ_INNER},
    )
    out = _maybe_normalize_awq_model_config(str(model_dir))
    direct = normalize_quant_config(_AWQ_INNER)
    assert out == {"quantization_config": direct}
