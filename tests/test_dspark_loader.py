# SPDX-License-Identifier: Apache-2.0
"""Checkpoint selection and precision tests with actual safetensors and MLX."""

from __future__ import annotations

import json

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from tests.test_dspark_contracts import draft_hf_config
from vllm_metal.v1.dspark.config import DSparkConfig
from vllm_metal.v1.dspark.loader import load_drafter
from vllm_metal.v1.dspark.model import DSparkDrafter


def checkpoint(tmp_path, dtype=mx.float32):
    raw = draft_hf_config().to_dict()
    raw.update(hidden_size=64, intermediate_size=128, head_dim=16, markov_rank=64)
    (tmp_path / "config.json").write_text(json.dumps(raw))
    mx.random.seed(17)
    model = DSparkDrafter(DSparkConfig.from_dict(raw))
    weights = {
        name: value.astype(dtype) for name, value in tree_flatten(model.parameters())
    }
    mx.save_safetensors(tmp_path / "model.safetensors", weights)
    model.load_weights(list(weights.items()))
    return model, weights


def forward(model):
    context = model.make_ctx_cache()
    features = mx.sin(mx.arange(3 * 128).reshape(1, 3, 128) * 0.01)
    features = features.astype(model.hidden_norm.weight.dtype)
    model.update_context(features, 0, context)
    hidden = model.backbone(
        model.embed(mx.array([[1, 63, 63, 63, 63, 63, 63]])), 3, context
    )
    logits = model.compute_logits(hidden) + model.markov_head.step_bias(mx.array([1]))
    mx.eval(logits)
    return logits


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("quantize", [False, True])
def test_streamed_load_preserves_recipe_and_execution(tmp_path, dtype, quantize):
    reference, _ = checkpoint(tmp_path, dtype)
    if quantize:
        nn.quantize(reference, bits=4, group_size=64)
    mx.eval(reference.parameters())
    loaded, _ = load_drafter(str(tmp_path), quantize=quantize)
    expected = dict(tree_flatten(reference.parameters()))
    actual = dict(tree_flatten(loaded.parameters()))
    assert actual.keys() == expected.keys()
    for name in actual:
        assert actual[name].dtype == expected[name].dtype
        assert mx.array_equal(actual[name], expected[name]).item(), name
    assert mx.array_equal(forward(loaded), forward(reference)).item()


@pytest.mark.parametrize("quantize", [False, True])
def test_index_selects_only_declared_shards(tmp_path, quantize):
    reference, weights = checkpoint(tmp_path)
    names = sorted(weights)
    mapping = {}
    for i, subset in enumerate((names[::2], names[1::2])):
        filename = f"shard-{i}.safetensors"
        mx.save_safetensors(
            tmp_path / filename, {name: weights[name] for name in subset}
        )
        mapping.update(dict.fromkeys(subset, filename))
    (tmp_path / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": mapping})
    )
    # The unrelated full file intentionally remains; the index is authoritative.
    if quantize:
        nn.quantize(reference, bits=4, group_size=64)
    loaded, _ = load_drafter(str(tmp_path), quantize=quantize)
    assert mx.array_equal(forward(loaded), forward(reference)).item()


@pytest.mark.parametrize(
    "corruption,match",
    [
        ("missing", "tensor names"),
        ("extra", "tensor names"),
        ("shape", "expected shape"),
        ("integer", "tensor dtypes"),
        ("mixed", "tensor dtypes"),
        ("ambiguous", "one safetensors file"),
        ("missing_shard", "missing shard"),
        ("misindexed", "disagree with weight_map"),
        ("traversal", "invalid shard"),
        ("duplicate_json", "repeats key"),
        ("prepacked", "unpacked floating-point"),
    ],
)
def test_invalid_checkpoint_fails_before_materialization(
    tmp_path, monkeypatch, corruption, match
):
    _, weights = checkpoint(tmp_path)
    name = "embed_tokens.weight"
    if corruption == "missing":
        del weights[name]
    elif corruption == "extra":
        weights["unrelated.weight"] = mx.zeros((2, 2))
    elif corruption == "shape":
        weights[name] = weights[name][:1]
    elif corruption == "integer":
        weights = {key: value.astype(mx.int32) for key, value in weights.items()}
    elif corruption == "mixed":
        weights[name] = weights[name].astype(mx.float16)
    elif corruption == "ambiguous":
        mx.save_safetensors(tmp_path / "extra.safetensors", weights)
    elif corruption in {"missing_shard", "misindexed", "traversal"}:
        filename = {
            "missing_shard": "absent.safetensors",
            "misindexed": "model.safetensors",
            "traversal": "../model.safetensors",
        }[corruption]
        (tmp_path / "model.safetensors.index.json").write_text(
            json.dumps({"weight_map": {name: filename}})
        )
    elif corruption == "duplicate_json":
        (tmp_path / "model.safetensors.index.json").write_text(
            '{"weight_map":{"a":"model.safetensors","a":"model.safetensors"}}'
        )
    elif corruption == "prepacked":
        raw = json.loads((tmp_path / "config.json").read_text())
        raw["quantization"] = {"bits": 4, "group_size": 64}
        (tmp_path / "config.json").write_text(json.dumps(raw))
    mx.save_safetensors(tmp_path / "model.safetensors", weights)

    def unexpected_load(*args, **kwargs):
        pytest.fail("weight data was read before header validation")

    monkeypatch.setattr(mx, "load", unexpected_load)
    with pytest.raises(ValueError, match=match):
        load_drafter(str(tmp_path))


@pytest.mark.parametrize("dtype", [mx.float32, mx.float16, mx.bfloat16])
@pytest.mark.parametrize("invalid", [float("nan"), float("inf"), -float("inf")])
@pytest.mark.parametrize("quantize", [False, True])
def test_nonfinite_payload_is_rejected_and_allocator_limit_restored(
    tmp_path, dtype, invalid, quantize
):
    _, weights = checkpoint(tmp_path, dtype)
    name = "embed_tokens.weight"
    weights[name][0, 0] = invalid
    mx.save_safetensors(tmp_path / "model.safetensors", weights)
    previous = mx.set_cache_limit(123456)
    try:
        with pytest.raises(ValueError, match="embed_tokens.weight.*non-finite"):
            load_drafter(str(tmp_path), quantize=quantize)
        assert mx.set_cache_limit(123456) == 123456
    finally:
        mx.set_cache_limit(previous)


def test_conversion_failure_restores_allocator_limit(tmp_path, monkeypatch):
    checkpoint(tmp_path)
    previous = mx.set_cache_limit(123456)

    def fail(*args, **kwargs):
        raise RuntimeError("conversion failed")

    monkeypatch.setattr(nn.Linear, "to_quantized", fail)
    try:
        with pytest.raises(RuntimeError, match="conversion failed"):
            load_drafter(str(tmp_path))
        assert mx.set_cache_limit(123456) == 123456
    finally:
        mx.set_cache_limit(previous)
