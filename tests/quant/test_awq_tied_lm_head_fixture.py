# SPDX-License-Identifier: Apache-2.0
"""Tied-`lm_head` AWQ quant-triple drop, exercised at the model layer.

mlx-lm's load order is `model.sanitize(weights)` -> `_transform_awq_weights`
-> `model.load_weights(strict=True)`. `Qwen2.sanitize` only drops
`lm_head.weight` when `tie_word_embeddings=True`; it has no awareness of
`lm_head.qweight/qzeros/scales`. AWQ checkpoints that quantize `lm_head`
on a tied model would survive sanitize, get transformed into
`lm_head.weight/scales/biases`, and fail strict-load.

The compat-layer wrap in `vllm_metal/compat.py` patches `Qwen2.sanitize`
to drop the AWQ quant triple when `self.args.tie_word_embeddings` is
true, before delegating to upstream sanitize.

This test exercises the patch directly via `Model.sanitize(...)` —
faster and more isolated than the full `mlx_lm.load` path, and
sufficient to pin the contract that the dropping happens iff the model
is tied. End-to-end load is covered by the `slow`-marked e2e suite.
"""

from __future__ import annotations

import mlx.core as mx
import pytest
from mlx_lm.models.qwen2 import Model, ModelArgs

from vllm_metal.compat import apply_compat_patches

_AWQ_LM_HEAD_KEYS = (
    "lm_head.qweight",
    "lm_head.qzeros",
    "lm_head.scales",
)


def _make_model_args(*, tie_word_embeddings: bool) -> ModelArgs:
    """Minimal Qwen2 ModelArgs sufficient to instantiate `Model` and call
    sanitize. Attention/MLP weight shapes do not matter — sanitize only
    walks the weights dict by key."""
    return ModelArgs(
        model_type="qwen2",
        hidden_size=128,
        num_hidden_layers=1,
        intermediate_size=256,
        num_attention_heads=2,
        num_key_value_heads=1,
        rms_norm_eps=1e-6,
        vocab_size=1000,
        tie_word_embeddings=tie_word_embeddings,
    )


def _stub_weights_with_lm_head_quant_triple() -> dict[str, mx.array]:
    """Minimal weights dict containing the AWQ `lm_head` quant triple plus
    a few unrelated keys, so the test asserts that sanitize touches *only*
    the lm_head triple."""
    return {
        "lm_head.qweight": mx.zeros((128, 16), dtype=mx.int32),
        "lm_head.qzeros": mx.zeros((1, 16), dtype=mx.int32),
        "lm_head.scales": mx.zeros((1, 128), dtype=mx.float16),
        "model.embed_tokens.weight": mx.zeros((1000, 128), dtype=mx.float16),
        "model.layers.0.input_layernorm.weight": mx.zeros((128,), dtype=mx.float16),
    }


@pytest.fixture(autouse=True)
def _ensure_compat_patches_applied():
    """The patch is registered in `apply_compat_patches`. The vllm-metal
    plugin entry point invokes it at platform registration; some test
    runners may import the model_lifecycle module out of order, so call
    explicitly here. `apply_compat_patches` is idempotent.
    """
    apply_compat_patches()


def test_tied_lm_head_quant_triple_dropped():
    """Tied embedding + AWQ `lm_head` quant triple: sanitize drops the
    triple before mlx-lm's transform sees it, so strict load does not
    fail with `Received N parameters not in model`.
    """
    model = Model(_make_model_args(tie_word_embeddings=True))
    weights = _stub_weights_with_lm_head_quant_triple()
    out = model.sanitize(weights)

    for key in _AWQ_LM_HEAD_KEYS:
        assert key not in out, (
            f"tied lm_head AWQ key {key!r} should be dropped by patched sanitize"
        )
    # Other keys preserved.
    assert "model.embed_tokens.weight" in out
    assert "model.layers.0.input_layernorm.weight" in out


def test_untied_lm_head_quant_triple_preserved():
    """Untied embedding: `lm_head` is a real `QuantizedLinear` parameter
    target, so the quant triple MUST survive sanitize. The patch only
    drops when tied; this regression test catches the foot-gun of a
    too-eager unconditional drop.
    """
    model = Model(_make_model_args(tie_word_embeddings=False))
    weights = _stub_weights_with_lm_head_quant_triple()
    out = model.sanitize(weights)

    for key in _AWQ_LM_HEAD_KEYS:
        assert key in out, (
            f"untied lm_head AWQ key {key!r} must NOT be dropped — it is a "
            f"real parameter target"
        )


def test_tied_sanitize_idempotent_under_repeated_apply():
    """`apply_compat_patches` is idempotent (sentinel-guarded); calling
    it twice must not re-wrap the wrapper (which would still produce
    correct behavior but with extra delegation overhead per call). Pin
    both identity (no double-wrap) and behavior.
    """
    apply_compat_patches()
    sanitize_after_first = Model.sanitize
    apply_compat_patches()
    sanitize_after_second = Model.sanitize

    # Identity check: a future regression that drops the sentinel guard
    # would re-wrap the already-wrapped sanitize, swapping the function
    # object on every apply. The behavior assertions below would still
    # pass under double-wrap, so this identity assertion is the load-
    # bearing one.
    assert sanitize_after_first is sanitize_after_second

    model = Model(_make_model_args(tie_word_embeddings=True))
    weights = _stub_weights_with_lm_head_quant_triple()
    out = model.sanitize(weights)

    for key in _AWQ_LM_HEAD_KEYS:
        assert key not in out


def test_sanitize_still_drops_lm_head_weight_for_tied():
    """Sanity: the patch must NOT regress upstream sanitize's existing
    behavior of dropping `lm_head.weight` for tied embeddings. The
    wrap delegates to the original after the new pre-step.
    """
    model = Model(_make_model_args(tie_word_embeddings=True))
    weights = {
        "lm_head.weight": mx.zeros((1000, 128), dtype=mx.float16),
        "model.embed_tokens.weight": mx.zeros((1000, 128), dtype=mx.float16),
    }
    out = model.sanitize(weights)
    assert "lm_head.weight" not in out
    assert "model.embed_tokens.weight" in out
