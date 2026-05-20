# SPDX-License-Identifier: Apache-2.0
"""Tests for experimental dense GGUF loading support."""

from __future__ import annotations

from pathlib import Path
from threading import Lock
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest
import torch

from vllm_metal.gguf import lifecycle as gguf_lifecycle
from vllm_metal.gguf.loader import (
    GGUFLoadError,
    _adjust_tensor_for_mlx_sanitize,
    _decode_tensor,
    _dequantize_q8_0,
    _dequantize_q8_1,
    translate_gguf_tensor_name,
)
from vllm_metal.gguf.refs import (
    GGUFReference,
    is_local_gguf,
    resolve_gguf_reference,
)


def _write_gguf_header(path: Path) -> None:
    path.write_bytes(b"GGUF" + b"\x00" * 32)


def _write_config(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "config.json").write_text("{}", encoding="utf-8")


def _pack_q8_0_raw(qweight: np.ndarray, scales: np.ndarray) -> np.ndarray:
    rows, blocks, _ = qweight.shape
    raw = np.zeros((rows, blocks, 34), dtype=np.uint8)
    raw[:, :, :2] = scales.astype(np.float16).reshape(rows, blocks, 1).view(np.uint8)
    raw[:, :, 2:] = qweight.astype(np.int8).view(np.uint8)
    return raw.reshape(rows, blocks * 34)


def _pack_q8_1_raw(qweight: np.ndarray, scales: np.ndarray) -> np.ndarray:
    rows, blocks, _ = qweight.shape
    raw = np.zeros((rows, blocks, 40), dtype=np.uint8)
    raw[:, :, :2] = scales.astype(np.float16).reshape(rows, blocks, 1).view(np.uint8)
    raw[:, :, 4:36] = qweight.astype(np.int8).view(np.uint8)
    return raw.reshape(rows, blocks * 40)


def test_resolve_single_local_gguf_file(tmp_path: Path) -> None:
    model_dir = tmp_path / "Qwen3.5-0.8B"
    _write_config(model_dir)
    gguf_path = tmp_path / "Qwen3.5-0.8B-Q8_0.gguf"
    _write_gguf_header(gguf_path)

    reference = resolve_gguf_reference(gguf_path)

    assert reference.gguf_path == gguf_path
    assert reference.all_gguf_paths == (gguf_path,)
    assert reference.model_path == model_dir
    assert is_local_gguf(gguf_path)


def test_resolve_local_shard_group_from_any_shard(tmp_path: Path) -> None:
    _write_config(tmp_path / "model")
    shard_1 = tmp_path / "model-Q4_K-00001-of-00002.gguf"
    shard_2 = tmp_path / "model-Q4_K-00002-of-00002.gguf"
    _write_gguf_header(shard_2)
    _write_gguf_header(shard_1)

    reference = resolve_gguf_reference(shard_2)

    assert reference.gguf_path == shard_1
    assert reference.all_gguf_paths == (shard_1, shard_2)
    assert reference.model_path == tmp_path / "model"
    assert is_local_gguf(tmp_path)


def test_incomplete_local_shard_group_is_not_local_gguf(tmp_path: Path) -> None:
    _write_gguf_header(tmp_path / "model-00001-of-00002.gguf")

    assert not is_local_gguf(tmp_path)
    with pytest.raises(ValueError, match="Incomplete GGUF shard group"):
        resolve_gguf_reference(tmp_path / "model-00001-of-00002.gguf")


def test_directory_with_multiple_groups_requires_explicit_input(tmp_path: Path) -> None:
    _write_gguf_header(tmp_path / "a.gguf")
    _write_gguf_header(tmp_path / "b.gguf")

    assert not is_local_gguf(tmp_path)
    with pytest.raises(ValueError, match="multiple GGUF files"):
        resolve_gguf_reference(tmp_path)


def test_model_config_candidates_take_precedence(tmp_path: Path) -> None:
    model_dir = tmp_path / "from-config"
    _write_config(model_dir)
    gguf_path = tmp_path / "other-name.gguf"
    _write_gguf_header(gguf_path)

    reference = resolve_gguf_reference(
        gguf_path,
        model_config=SimpleNamespace(hf_config_path=model_dir),
    )

    assert reference.model_path == model_dir


def test_cache_key_includes_all_local_shards(tmp_path: Path) -> None:
    shards = (
        tmp_path / "model-00001-of-00002.gguf",
        tmp_path / "model-00002-of-00002.gguf",
    )
    reference = GGUFReference(shards[0], tmp_path / "model", shards)

    cache_key = reference.cache_key()

    assert str(shards[0]) in cache_key[0]
    assert str(shards[1]) in cache_key[0]
    assert cache_key[1] == f"gguf-dense:{tmp_path / 'model'}"


def test_load_gguf_generation_model_uses_cache_and_loader(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    model_dir = tmp_path / "model"
    _write_config(model_dir)
    gguf_path = tmp_path / "model-Q8_0.gguf"
    _write_gguf_header(gguf_path)

    loaded_model = object()
    loaded_tokenizer = object()
    captured: dict[str, object] = {}

    class FakeGGUFLoader:
        def __init__(self, reference, *, target_dtype, tokenizer_config):
            captured["reference"] = reference
            captured["target_dtype"] = target_dtype
            captured["tokenizer_config"] = tokenizer_config

        def load(self):
            captured["load_count"] = int(captured.get("load_count", 0)) + 1
            return loaded_model, loaded_tokenizer

    monkeypatch.setattr(gguf_lifecycle, "GGUFMLXLoader", FakeGGUFLoader)
    cache: dict[tuple[str, str], tuple[object, object]] = {}
    cache_lock = Lock()
    model_config = SimpleNamespace(dtype=torch.float16, trust_remote_code=True)

    model, tokenizer = gguf_lifecycle.load_gguf_generation_model(
        str(gguf_path),
        model_config=model_config,
        model_cache=cache,
        model_cache_lock=cache_lock,
        start_time=0.0,
    )

    assert model is loaded_model
    assert tokenizer is loaded_tokenizer
    assert captured["reference"].gguf_path == gguf_path
    assert captured["reference"].model_path == model_dir
    assert captured["target_dtype"] == mx.float16
    assert captured["tokenizer_config"] == {"trust_remote_code": True}
    assert cache[captured["reference"].cache_key()] == (loaded_model, loaded_tokenizer)

    cached_model, cached_tokenizer = gguf_lifecycle.load_gguf_generation_model(
        str(gguf_path),
        model_config=model_config,
        model_cache=cache,
        model_cache_lock=cache_lock,
        start_time=0.0,
    )

    assert cached_model is loaded_model
    assert cached_tokenizer is loaded_tokenizer
    assert captured["load_count"] == 1


def test_translate_prefers_upstream_name_map() -> None:
    translated = translate_gguf_tensor_name(
        "blk.0.attn_q.weight",
        model_type="qwen3_5",
        upstream_name_map={"blk.0.attn_q.weight": "model.layers.0.attn.q_proj.weight"},
    )

    assert translated == "model.layers.0.attn.q_proj.weight"


def test_translate_uses_local_fallback_for_gemma4_language_model() -> None:
    translated = translate_gguf_tensor_name(
        "blk.3.ffn_gate.weight",
        model_type="gemma4",
    )

    assert translated == "model.language_model.layers.3.mlp.gate_proj.weight"


def test_qwen35_shifted_norm_is_adjusted_before_mlx_sanitize() -> None:
    array = mx.array([2.0, 3.0], dtype=mx.float32)

    adjusted = _adjust_tensor_for_mlx_sanitize(
        "model.layers.0.input_layernorm.weight",
        array,
        model_type="qwen3_5",
    )

    np.testing.assert_allclose(np.asarray(adjusted), np.array([1.0, 2.0]))


def test_dequantize_q8_0_raw_blocks() -> None:
    qweight = np.arange(-16, 48, dtype=np.int8).reshape(2, 1, 32)
    scales = np.array([[0.5], [0.25]], dtype=np.float16)
    raw = _pack_q8_0_raw(qweight, scales)

    decoded = _dequantize_q8_0(raw, (32, 2))

    expected = qweight.astype(np.float32) * scales.astype(np.float32)[..., None]
    np.testing.assert_allclose(decoded, expected.reshape(2, 32), rtol=0, atol=0)


def test_dequantize_q8_1_raw_blocks() -> None:
    qweight = np.arange(-16, 48, dtype=np.int8).reshape(2, 1, 32)
    scales = np.array([[0.5], [0.25]], dtype=np.float16)
    raw = _pack_q8_1_raw(qweight, scales)

    decoded = _dequantize_q8_1(raw, (32, 2))

    expected = qweight.astype(np.float32) * scales.astype(np.float32)[..., None]
    np.testing.assert_allclose(decoded, expected.reshape(2, 32), rtol=0, atol=0)


def test_decode_tensor_uses_gguf_dequantize_for_supported_quant_types() -> None:
    import gguf

    if not hasattr(gguf, "quantize"):
        pytest.skip("gguf.quantize is unavailable")

    dense = np.linspace(-1.0, 1.0, 64, dtype=np.float32).reshape(2, 32)
    quant_type = gguf.GGMLQuantizationType.Q4_0
    block_size, type_size = gguf.GGML_QUANT_SIZES[quant_type]
    raw = gguf.quantize(dense.reshape(-1, block_size), quant_type)
    tensor = SimpleNamespace(
        name="blk.0.ffn_gate.weight",
        data=raw.reshape(-1, type_size),
        shape=(32, 2),
        tensor_type=quant_type,
    )

    decoded = _decode_tensor(tensor, target_dtype=mx.float32)
    expected = gguf.dequantize(raw.reshape(-1, type_size), quant_type).reshape(2, 32)

    np.testing.assert_allclose(np.asarray(decoded), expected, rtol=0, atol=0)


def test_decode_tensor_raises_for_unknown_quant_type() -> None:
    tensor = SimpleNamespace(
        name="unknown.weight",
        data=np.zeros((1,), dtype=np.uint8),
        shape=(1,),
        tensor_type=9999,
    )

    with pytest.raises(GGUFLoadError, match="Unsupported GGUF tensor type 9999"):
        _decode_tensor(tensor, target_dtype=mx.float32)
