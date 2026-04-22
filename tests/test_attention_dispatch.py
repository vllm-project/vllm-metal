# SPDX-License-Identifier: Apache-2.0
"""Tests for attention backend dispatch.

Unit tests verify detection heuristics against real mlx_lm modules
(no model weights, just module instantiation).  The slow integration
test covers the full paged attention dispatch on Qwen3.5.
"""

from __future__ import annotations

import pytest

from vllm_metal.metal_kernel_backend.attention_linear import is_linear_attention
from vllm_metal.metal_kernel_backend.attention_sdpa import is_sdpa
from vllm_metal.paged_attention_common import find_attn_attr, find_layers

# ---------------------------------------------------------------------------
# Minimal ModelArgs for real mlx_lm module instantiation (no weights needed)
# ---------------------------------------------------------------------------

_QWEN3_ARGS_KWARGS = {
    "model_type": "qwen3",
    "hidden_size": 64,
    "num_hidden_layers": 2,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "vocab_size": 100,
    "max_position_embeddings": 512,
    "rope_theta": 10000.0,
    "head_dim": 16,
    "tie_word_embeddings": False,
}

_QWEN35_ARGS_KWARGS = {
    "hidden_size": 64,
    "num_hidden_layers": 4,
    "intermediate_size": 128,
    "num_attention_heads": 4,
    "num_key_value_heads": 2,
    "rms_norm_eps": 1e-6,
    "vocab_size": 100,
    "max_position_embeddings": 512,
    "rope_theta": 10000.0,
    "head_dim": 16,
    "tie_word_embeddings": False,
    "full_attention_interval": 4,
}


# ---------------------------------------------------------------------------
# Detection against real mlx_lm modules
# ---------------------------------------------------------------------------


def test_qwen3_attention_detected_as_sdpa():
    """Real Qwen3 Attention module should be detected as SDPA."""
    from mlx_lm.models.qwen3 import Attention, ModelArgs

    args = ModelArgs(**_QWEN3_ARGS_KWARGS)
    attn = Attention(args)

    assert is_sdpa(attn)
    assert not is_linear_attention(attn)


def test_qwen35_sdpa_layer_detected():
    """Qwen3.5 SDPA layer (every full_attention_interval-th) should have
    self_attn detected as SDPA."""
    from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

    args = TextModelArgs(**_QWEN35_ARGS_KWARGS)
    # layer_idx=3 with full_attention_interval=4 → SDPA layer
    layer = DecoderLayer(args, layer_idx=3)

    assert find_attn_attr(layer) == "self_attn"
    assert is_sdpa(layer.self_attn)
    assert not is_linear_attention(layer.self_attn)


def test_qwen35_linear_layer_detected():
    """Qwen3.5 linear attention layer (GatedDeltaNet) should have
    linear_attn detected as linear attention."""
    from mlx_lm.models.qwen3_5 import DecoderLayer, TextModelArgs

    args = TextModelArgs(**_QWEN35_ARGS_KWARGS)
    # layer_idx=0 with full_attention_interval=4 → linear attention layer
    layer = DecoderLayer(args, layer_idx=0)

    assert find_attn_attr(layer) == "linear_attn"
    assert is_linear_attention(layer.linear_attn)
    assert not is_sdpa(layer.linear_attn)


def test_gemma4_attention_contract_detected_as_sdpa():
    """Gemma4-like SDPA modules should match the dispatch contract.

    This test intentionally avoids importing the real ``mlx-lm`` Gemma4
    ``Attention`` class. Its internal attribute layout has drifted across
    minor releases, which makes unit tests flaky without changing the
    actual Metal dispatch contract we care about.

    The contract is:
    - sliding/full SDPA modules expose ``q_proj`` / ``k_proj`` / ``o_proj``
    - values arrive either through ``v_proj`` OR the explicit
      ``use_k_eq_v=True`` opt-in
    """

    class _Gemma4SlidingLike:
        q_proj = object()
        k_proj = object()
        v_proj = object()
        o_proj = object()

    class _Gemma4FullKEqVLike:
        q_proj = object()
        k_proj = object()
        o_proj = object()
        use_k_eq_v = True

    sliding_attn = _Gemma4SlidingLike()
    full_attn = _Gemma4FullKEqVLike()

    assert is_sdpa(sliding_attn)
    assert is_sdpa(full_attn)
    assert not is_linear_attention(sliding_attn)
    assert not is_linear_attention(full_attn)


def test_is_sdpa_rejects_modules_without_v_proj_or_use_k_eq_v():
    """A module exposing only q_proj / k_proj / o_proj must NOT classify as
    SDPA.  This is the case the hybrid dispatcher must not silently send
    down the SDPA path — it would miss the values projection entirely.
    """

    class _QkoOnly:
        q_proj = object()
        k_proj = object()
        o_proj = object()

    assert not is_sdpa(_QkoOnly())

    class _QkoWithUseKEqVFalse:
        q_proj = object()
        k_proj = object()
        o_proj = object()
        use_k_eq_v = False

    assert not is_sdpa(_QkoWithUseKEqVFalse())

    class _QkoWithUseKEqVTrue:
        q_proj = object()
        k_proj = object()
        o_proj = object()
        use_k_eq_v = True

    assert is_sdpa(_QkoWithUseKEqVTrue())


def test_find_layers_on_qwen3_model():
    """find_layers should return the layer list from a real Qwen3 Model."""
    from mlx_lm.models.qwen3 import Model, ModelArgs

    args = ModelArgs(**_QWEN3_ARGS_KWARGS)
    model = Model(args)
    layers = find_layers(model)

    assert len(layers) == args.num_hidden_layers
    assert find_attn_attr(layers[0]) == "self_attn"


# ---------------------------------------------------------------------------
# Slow integration test
# ---------------------------------------------------------------------------


@pytest.mark.slow
def test_qwen35_paged_attention_hybrid():
    """Qwen3.5 hybrid model loads and generates with paged attention."""
    from vllm import LLM, SamplingParams

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        mp.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        mp.setenv("VLLM_METAL_MEMORY_FRACTION", "0.3")

        llm = LLM(model="Qwen/Qwen3.5-0.8B", max_model_len=512, max_num_seqs=1)
        sp = SamplingParams(temperature=0, max_tokens=5)
        outputs = llm.generate(["The capital of France is"], sp)
        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].token_ids) > 0
