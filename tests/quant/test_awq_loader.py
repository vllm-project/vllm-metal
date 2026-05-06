# SPDX-License-Identifier: Apache-2.0
"""Unit tests for ``AWQQuantLoader`` under ``vllm_metal/quant/``.

Covers detection (``for_model``), cache-key dtype isolation, and the
``text_config.quantization_config`` fallback that mirrors mlx-lm. Pure
helpers; no model load, no downloads.
"""

from __future__ import annotations

import json
from pathlib import Path

import mlx.core as mx
import pytest
import torch

from vllm_metal.pytorch_backend.tensor_bridge import torch_to_mlx
from vllm_metal.quant.awq_config import UnsupportedQuantizationConfigError
from vllm_metal.quant.awq_loader import (
    AWQQuantLoader,
    _align_non_quantized_dtypes,
    _read_raw_quantization_config,
)


def _mlx_dtype(torch_dtype):
    """Mirror what production does to derive the cache-key dtype: convert a
    torch dtype through ``torch_to_mlx``. Tests use this instead of literal
    strings so we pin the *actual* MLX dtype the production path passes
    (e.g. ``mlx.core.bfloat16``), not just "different strings produce
    different keys".
    """
    return torch_to_mlx(torch.empty(0, dtype=torch_dtype)).dtype


_AWQ_INNER = {
    "quant_method": "awq",
    "bits": 4,
    "group_size": 128,
    "zero_point": True,
    "version": "gemm",
}


def _write_config(tmp_path: Path, config: dict) -> Path:
    """Drop a config.json into ``tmp_path`` and return the directory."""
    (tmp_path / "config.json").write_text(json.dumps(config))
    return tmp_path


# ---- cache-key dtype isolation ---------------------------------------------


def test_cache_key_isolates_by_dtype():
    """Two engines requesting the same model with different dtypes must NOT
    share a cache entry, since AWQ post-load alignment mutates the model
    in place."""
    loader = AWQQuantLoader(_AWQ_INNER)
    bf16_key = loader.cache_key(
        "Qwen/Qwen2.5-1.5B-Instruct-AWQ", target_dtype=_mlx_dtype(torch.bfloat16)
    )
    fp16_key = loader.cache_key(
        "Qwen/Qwen2.5-1.5B-Instruct-AWQ", target_dtype=_mlx_dtype(torch.float16)
    )
    assert bf16_key != fp16_key


def test_cache_key_pins_loader_segment_with_dtype():
    """Pin the wire format: the second key segment is ``mlx_lm-awq:<dtype>``
    (encoding the dtype inside the loader segment, not as a third tuple
    element). A future refactor that strips the dtype or swaps to torch
    repr would fail loudly here.
    """
    loader = AWQQuantLoader(_AWQ_INNER)
    key = loader.cache_key("foo", target_dtype=_mlx_dtype(torch.bfloat16))
    assert key == ("foo", f"mlx_lm-awq:{mx.bfloat16}")


def test_cache_key_distinct_from_generic_lifecycle_key():
    """AWQ cache key must not collide with the generic mlx_lm key for the
    same model: an AWQ-mutated cached model must not be served to a
    non-AWQ caller, nor vice versa."""
    from vllm_metal.v1.model_lifecycle import _generation_cache_key

    awq_key = AWQQuantLoader(_AWQ_INNER).cache_key(
        "x", target_dtype=_mlx_dtype(torch.bfloat16)
    )
    generic_key = _generation_cache_key("x", is_vlm=False)
    assert awq_key != generic_key


def test_cache_key_stable_for_same_inputs():
    loader = AWQQuantLoader(_AWQ_INNER)
    bf16 = _mlx_dtype(torch.bfloat16)
    assert loader.cache_key("foo", target_dtype=bf16) == loader.cache_key(
        "foo", target_dtype=bf16
    )


# ---- text_config.quantization_config fallback ------------------------------


def test_read_quantization_config_top_level(tmp_path):
    model_dir = _write_config(
        tmp_path,
        {"model_type": "qwen2", "quantization_config": _AWQ_INNER},
    )
    assert _read_raw_quantization_config(str(model_dir)) == _AWQ_INNER


def test_read_quantization_config_nested_text_config(tmp_path):
    """Multimodal wrapper configs nest the quant config under
    ``text_config``. ``mlx_lm.utils.load_model`` falls back to it; the
    AWQ owner must do the same so the alias normalization / reject logic
    still runs.
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
    assert _read_raw_quantization_config(str(model_dir)) == _AWQ_INNER


def test_read_quantization_config_top_level_wins_over_text_config(tmp_path):
    """If both are present, top-level takes precedence (matches mlx-lm)."""
    nested = {**_AWQ_INNER, "bits": 8}  # would normally reject
    model_dir = _write_config(
        tmp_path,
        {
            "model_type": "wrapper",
            "quantization_config": _AWQ_INNER,
            "text_config": {"quantization_config": nested},
        },
    )
    assert _read_raw_quantization_config(str(model_dir)) == _AWQ_INNER


def test_read_quantization_config_absent(tmp_path):
    model_dir = _write_config(tmp_path, {"model_type": "qwen2"})
    assert _read_raw_quantization_config(str(model_dir)) is None


def test_read_quantization_config_missing_dir():
    """Non-existent path / non-HF repo: silently inactive, returns None."""
    assert _read_raw_quantization_config("/nonexistent/path/zzz") is None


# ---- AWQQuantLoader.for_model ---------------------------------------------


def test_for_model_returns_none_for_non_awq(tmp_path):
    """Non-AWQ checkpoints get None back, no exception."""
    model_dir = _write_config(
        tmp_path,
        {
            "model_type": "qwen2",
            "quantization_config": {"quant_method": "fp8", "bits": 8},
        },
    )
    assert AWQQuantLoader.for_model(str(model_dir)) is None


def test_for_model_returns_none_for_no_quant_config(tmp_path):
    model_dir = _write_config(tmp_path, {"model_type": "qwen2"})
    assert AWQQuantLoader.for_model(str(model_dir)) is None


def test_for_model_raises_for_gptq(tmp_path):
    """GPTQ is rejected loudly at the loader entry point. Falling through
    to the generic loader would bypass the dtype-alignment contract this
    owner enforces, so the loader must not silently take ownership of a
    quant_method it has not yet validated.
    """
    model_dir = _write_config(
        tmp_path,
        {
            "model_type": "qwen2",
            "quantization_config": {
                "quant_method": "gptq",
                "bits": 4,
                "group_size": 128,
                "zero_point": True,
                "version": "gemm",
            },
        },
    )
    with pytest.raises(UnsupportedQuantizationConfigError) as excinfo:
        AWQQuantLoader.for_model(str(model_dir))
    assert "GPTQ" in str(excinfo.value)


def test_for_model_returns_loader_for_awq(tmp_path):
    """AWQ checkpoint: ``for_model`` returns a configured loader whose
    cache key reflects the requested dtype."""
    model_dir = _write_config(
        tmp_path,
        {"model_type": "qwen2", "quantization_config": _AWQ_INNER},
    )
    loader = AWQQuantLoader.for_model(str(model_dir))
    assert loader is not None
    key = loader.cache_key(str(model_dir), target_dtype=_mlx_dtype(torch.bfloat16))
    assert key[1].startswith("mlx_lm-awq:")


def test_for_model_propagates_reject_via_text_config(tmp_path):
    """The text_config fallback must surface reject errors too — nesting
    must not be a hole that lets unsupported configs through."""
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
        AWQQuantLoader.for_model(str(model_dir))


def test_for_model_accepts_aliased_config(tmp_path):
    """``for_model`` runs ``normalize_quant_config`` on detection, so an
    AutoAWQ-style aliased config (``w_bit``, ``q_group_size``, uppercase
    ``GEMM``) yields a valid loader rather than raising.
    """
    aliased = {
        "quant_method": "awq",
        "w_bit": 4,
        "q_group_size": 128,
        "zero_point": True,
        "version": "GEMM",
    }
    model_dir = _write_config(
        tmp_path,
        {"model_type": "qwen2", "quantization_config": aliased},
    )
    assert AWQQuantLoader.for_model(str(model_dir)) is not None


# ---- _align_non_quantized_dtypes -------------------------------------------
#
# Every MLX-quantize-protocol leaf — ``QuantizedLinear``,
# ``QuantizedEmbedding``, mlx_lm's MoE / MLA peers — has ``scales`` /
# ``biases`` parameters owned by the AWQ transform that must NOT be cast
# to the runtime dtype. The duck-typed signal for "this is quantized"
# is the instance-level ``bits`` / ``group_size`` attributes set by
# every class adhering to the protocol.
#
# A quantized layer's ordinary ``bias`` (Qwen2 q/k/v projections, MoE
# per-expert biases) is a normal floating param and IS cast, otherwise
# the projection emits mixed-dtype activations into a bf16 KV cache.


import mlx.nn as nn  # noqa: E402  (kept near its sole user)


class _SingleLeaf(nn.Module):
    """Tiny container so ``leaf_modules()`` yields one leaf to align."""

    def __init__(self, leaf: nn.Module) -> None:
        super().__init__()
        self.leaf = leaf


def _make_quantized_linear(*, bias: bool) -> nn.QuantizedLinear:
    """Build a QuantizedLinear and force its float buffers to fp16 to
    mirror the dtype distribution that ``mlx_lm._transform_awq_weights``
    produces from a real AWQ checkpoint."""
    qlinear = nn.QuantizedLinear(
        input_dims=128, output_dims=64, bias=bias, group_size=64, bits=4
    )
    fp16_updates = {
        "scales": qlinear.scales.astype(mx.float16),
        "biases": qlinear.biases.astype(mx.float16),
    }
    if bias:
        fp16_updates["bias"] = qlinear.bias.astype(mx.float16)
    qlinear.update(fp16_updates)
    return qlinear


def test_align_dtypes_casts_quantized_linear_regular_bias():
    """A ``QuantizedLinear`` with ``bias=True`` (Qwen2 q/k/v shape) must
    have its ``bias`` parameter cast to the runtime target dtype, while
    ``scales``/``biases`` stay at the AWQ transform's dtype.
    """
    qlinear = _make_quantized_linear(bias=True)
    wrapper = _SingleLeaf(qlinear)

    n_cast = _align_non_quantized_dtypes(wrapper, mx.bfloat16)

    assert qlinear.scales.dtype == mx.float16
    assert qlinear.biases.dtype == mx.float16
    assert qlinear.bias.dtype == mx.bfloat16
    assert n_cast == 1


def test_align_dtypes_leaves_quantized_linear_without_regular_bias():
    """A ``QuantizedLinear`` with ``bias=False`` has nothing for align to
    touch; ``scales``/``biases`` stay at the transform's dtype.
    """
    qlinear = _make_quantized_linear(bias=False)
    wrapper = _SingleLeaf(qlinear)

    n_cast = _align_non_quantized_dtypes(wrapper, mx.bfloat16)

    assert qlinear.scales.dtype == mx.float16
    assert qlinear.biases.dtype == mx.float16
    assert n_cast == 0


def test_align_dtypes_casts_non_quantized_floating_params():
    """A plain ``nn.Linear`` next to a quant layer should still have its
    own ``weight`` and ``bias`` aligned (the prior behavior, asserted
    here so the QuantizedLinear-specific handling does not regress it).
    """
    plain = nn.Linear(input_dims=8, output_dims=4, bias=True)
    plain.update(
        {
            "weight": plain.weight.astype(mx.float16),
            "bias": plain.bias.astype(mx.float16),
        }
    )
    wrapper = _SingleLeaf(plain)

    n_cast = _align_non_quantized_dtypes(wrapper, mx.bfloat16)

    assert plain.weight.dtype == mx.bfloat16
    assert plain.bias.dtype == mx.bfloat16
    assert n_cast == 2


def test_align_dtypes_treats_quantized_embedding_as_quantized():
    """``nn.QuantizedEmbedding`` is NOT a subclass of ``nn.QuantizedLinear``;
    an ``isinstance`` check would silently let its ``scales`` / ``biases``
    be cast away from the AWQ transform's dtype. The duck-typed
    protocol detector (``bits`` / ``group_size`` attributes) must still
    classify it as quantized so the AWQ buffers are exempted.
    """
    qemb = nn.QuantizedEmbedding(num_embeddings=64, dims=128, group_size=64, bits=4)
    qemb.update(
        {
            "scales": qemb.scales.astype(mx.float16),
            "biases": qemb.biases.astype(mx.float16),
        }
    )
    wrapper = _SingleLeaf(qemb)

    n_cast = _align_non_quantized_dtypes(wrapper, mx.bfloat16)

    assert qemb.scales.dtype == mx.float16
    assert qemb.biases.dtype == mx.float16
    # No regular ``bias`` on QuantizedEmbedding; nothing to cast.
    assert n_cast == 0


class _DuckTypedQuantized(nn.Module):
    """Synthetic stand-in for mlx_lm peers (``QuantizedSwitchLinear`` for
    MoE, ``QuantizedMultiLinear`` for MLA) that follow the MLX quantize
    protocol but are not subclasses of ``nn.QuantizedLinear``. The unit
    test relies on attribute-based detection rather than importing
    those classes directly so a future mlx_lm version bump does not
    invalidate the regression coverage.
    """

    def __init__(self) -> None:
        super().__init__()
        self.bits = 4
        self.group_size = 128
        self.weight = mx.zeros((4, 128 // 8), dtype=mx.uint32)
        self.scales = mx.zeros((4, 1), dtype=mx.float16)
        self.biases = mx.zeros((4, 1), dtype=mx.float16)
        self.bias = mx.zeros((4,), dtype=mx.float16)


def test_align_dtypes_treats_duck_typed_protocol_module_as_quantized():
    """Any leaf adhering to MLX's quantize protocol — ``bits`` /
    ``group_size`` set on the instance — must have its ``scales`` /
    ``biases`` exempted from alignment, regardless of class identity.
    Covers ``QuantizedSwitchLinear`` (MoE AWQ) and ``QuantizedMultiLinear``
    (MLA AWQ) without taking an mlx_lm import dependency.
    """
    fake = _DuckTypedQuantized()
    wrapper = _SingleLeaf(fake)

    n_cast = _align_non_quantized_dtypes(wrapper, mx.bfloat16)

    # AWQ-transform buffers stay at the transform's dtype.
    assert fake.scales.dtype == mx.float16
    assert fake.biases.dtype == mx.float16
    # The regular per-expert / per-head bias still aligns.
    assert fake.bias.dtype == mx.bfloat16
    assert n_cast == 1


@pytest.mark.parametrize(
    "make_module",
    [
        lambda: _make_quantized_linear(bias=True),
        _DuckTypedQuantized,
    ],
    ids=["nn.QuantizedLinear", "duck-typed-protocol-module"],
)
def test_align_dtypes_distinguishes_biases_buffer_from_regular_bias(make_module):
    """``biases`` (plural — the AWQ-transform per-group quant buffer) and
    ``bias`` (singular — the layer's regular linear bias) coexist on
    every quantized layer with ``bias=True`` and must follow OPPOSITE
    dtype policies: ``biases`` stays at the transform's fp16 while
    ``bias`` aligns to the runtime bf16.

    The two parameter names differ by a single trailing character — a
    refactor that special-cases by substring (``"bias" in name`` etc.)
    or that drops one of the two from the exempt set would silently
    swap the policies. This test pins the distinction on a real
    ``nn.QuantizedLinear`` and on a duck-typed protocol module standing
    in for mlx_lm's MoE / MLA peers.
    """
    module = make_module()
    wrapper = _SingleLeaf(module)

    n_cast = _align_non_quantized_dtypes(wrapper, mx.bfloat16)

    assert module.biases.dtype == mx.float16, (
        "AWQ-transform `biases` (plural) must NOT be cast away from fp16"
    )
    assert module.bias.dtype == mx.bfloat16, (
        "Regular linear `bias` (singular) MUST be cast to runtime bfloat16"
    )
    assert n_cast == 1
