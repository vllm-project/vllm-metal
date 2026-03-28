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
def test_qwen35_paged_attention_raises_on_hybrid():
    """Loading Qwen/Qwen3.5-0.8B with paged attention raises RuntimeError
    at setup — hybrid models are not yet supported on the paged path."""
    from vllm import LLM

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        mp.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        mp.setenv("VLLM_METAL_MEMORY_FRACTION", "0.2")

        with pytest.raises(RuntimeError, match="not yet supported for hybrid"):
            LLM(model="Qwen/Qwen3.5-0.8B", max_model_len=512, max_num_seqs=1)
