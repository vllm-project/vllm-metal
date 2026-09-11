# SPDX-License-Identifier: Apache-2.0
"""Tests for attention backend dispatch.

Unit tests verify detection heuristics against real mlx_lm modules
(no model weights, just module instantiation).  The slow integration
test covers the full paged attention dispatch on Qwen3.5.
"""

from __future__ import annotations

import pytest

from tests.stub_runner import NEMOTRON_H_TINY_ARGS
from vllm_metal.attention.attention_contracts import attention_contract_for
from vllm_metal.attention.impls.linear import is_linear_attention
from vllm_metal.attention.impls.mamba2 import is_mamba2_mixer
from vllm_metal.attention.impls.sdpa import is_sdpa
from vllm_metal.attention.patching import (
    DEFAULT_ATTN_ATTR_NAMES,
    find_attn_attr,
    find_layers,
)

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
    assert not is_mamba2_mixer(layer.linear_attn)
    assert not is_sdpa(layer.linear_attn)


def test_nemotron_h_mamba2_mixer_is_not_gdn():
    """Mamba-2 mixers carry conv1d too; the GDN predicate keys on its projections."""
    from mlx_lm.models.nemotron_h import ModelArgs, NemotronHMamba2Mixer

    mixer = NemotronHMamba2Mixer(ModelArgs(**NEMOTRON_H_TINY_ARGS))

    assert is_mamba2_mixer(mixer)
    assert not is_linear_attention(mixer)
    assert not is_sdpa(mixer)


def test_nemotron_h_attention_detected_as_sdpa():
    from mlx_lm.models.nemotron_h import ModelArgs, NemotronHAttention

    attn = NemotronHAttention(ModelArgs(**NEMOTRON_H_TINY_ARGS))

    assert is_sdpa(attn)
    assert not is_linear_attention(attn)
    assert attention_contract_for(attn).use_rope is False


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


def test_is_sdpa_rejects_packed_qkv_modules_missing_runtime_contract():
    """Packed-qkv modules must expose the runtime contract SDPA needs."""

    class _PackedQKVOnly:
        qkv_proj = object()
        o_proj = object()

    assert not is_sdpa(_PackedQKVOnly())

    class _PackedQKVWithoutRope:
        qkv_proj = object()
        o_proj = object()
        n_heads = 4
        n_kv_heads = 2
        head_dim = 16
        scale = 0.25

    assert not is_sdpa(_PackedQKVWithoutRope())

    class _PackedQKVWithRotaryEmb:
        qkv_proj = object()
        o_proj = object()
        n_heads = 4
        n_kv_heads = 2
        head_dim = 16
        scale = 0.25
        rotary_emb = object()

    assert is_sdpa(_PackedQKVWithRotaryEmb())


def test_is_sdpa_falls_back_to_split_contract_when_packed_fields_incomplete():
    """Mixed modules should still classify as SDPA via the split contract."""

    class _MixedAttention:
        qkv_proj = object()
        q_proj = object()
        k_proj = object()
        v_proj = object()
        o_proj = object()

    assert is_sdpa(_MixedAttention())


def test_phi3_attention_detected_as_sdpa():
    """Real Phi3/Phi4-style packed-projection attention should be SDPA."""
    from mlx_lm.models.phi3 import Attention, ModelArgs

    args = ModelArgs(
        model_type="phi3",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        num_key_value_heads=2,
        rms_norm_eps=1e-6,
        vocab_size=100,
        max_position_embeddings=512,
        original_max_position_embeddings=512,
        tie_word_embeddings=False,
    )
    attn = Attention(args)

    assert is_sdpa(attn)
    assert not is_linear_attention(attn)


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
        mp.setenv("VLLM_METAL_MEMORY_FRACTION", "0.3")

        llm = LLM(model="Qwen/Qwen3.5-0.8B", max_model_len=512, max_num_seqs=1)
        sp = SamplingParams(temperature=0, max_tokens=5)
        outputs = llm.generate(["The capital of France is"], sp)
        assert len(outputs) == 1
        assert len(outputs[0].outputs[0].token_ids) > 0


def test_mixer_is_probed_only_when_a_family_opts_in() -> None:
    from mlx_lm.models.nemotron_h import ModelArgs, NemotronHBlock

    block = NemotronHBlock(ModelArgs(**NEMOTRON_H_TINY_ARGS), "M")

    assert find_attn_attr(block) is None
    assert find_attn_attr(block, (*DEFAULT_ATTN_ATTR_NAMES, "mixer")) == "mixer"


# ---------------------------------------------------------------------------
# validate_paged_attention_support refuses unmanaged native cache topologies
# ---------------------------------------------------------------------------


def _make_policy_runner(model, *, num_layers: int, hybrid_plan=None, **attrs):
    """Stub runner around a real mlx_lm model for cache-policy validation."""
    from tests.stub_runner import make_stub_runner

    return make_stub_runner(
        model=model,
        num_layers=num_layers,
        num_kv_cache_layers=num_layers,
        num_kv_heads=2,
        is_hybrid=hybrid_plan is not None,
        hybrid_runtime_plan=hybrid_plan,
        **attrs,
    )


def _tiny_gdn_plan(num_layers, attention_indices):
    """GDN plan with explicit topology; geometry is irrelevant to validation."""
    from tests.stub_runner import make_gdn_hybrid_plan

    return make_gdn_hybrid_plan(
        num_layers,
        attention_indices,
        conv_kernel_dim=2,
        conv_dim=4,
        num_v_heads=1,
        value_head_dim=4,
        key_head_dim=32,
    )


def test_validate_rejects_falcon_h1_parallel_mamba():
    """Falcon-H1 runs a Mamba-2 mixer next to self_attn in every layer (#655).

    mlx-lm declares that topology as a ``CacheList`` per layer; no state
    family owns it, and the dense paged runtimes manage one KV-style cache
    per slot, so setup must refuse at validation time rather than crash on
    the first request.
    """
    from mlx_lm.models.falcon_h1 import Model, ModelArgs

    args = ModelArgs(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        mamba_n_heads=4,
        mamba_d_head=16,
        mamba_d_ssm=64,
        mamba_d_state=16,
        vocab_size=100,
    )
    runner = _make_policy_runner(Model(args), num_layers=2)

    with pytest.raises(NotImplementedError) as excinfo:
        runner.validate_paged_attention_support()

    message = str(excinfo.value)
    assert "CacheList" in message
    assert "native cache slot 0" in message
    assert "VLLM_METAL_USE_PAGED_ATTENTION=0" in message


def test_validate_rejects_native_cache_count_mismatch():
    """A model declaring fewer native slots than KV layers must refuse."""
    from types import SimpleNamespace

    from mlx_lm.models.cache import KVCache

    model = SimpleNamespace(make_cache=lambda: [KVCache()])
    runner = _make_policy_runner(model, num_layers=2)

    with pytest.raises(
        NotImplementedError, match=r"expected 2 native cache slots.*declares 1"
    ):
        runner.validate_paged_attention_support()


def test_validate_accepts_dense_and_gdn_hybrid_models():
    """Qwen3 (all KV slots) and Qwen3.5 (matching GDN plan) both pass."""
    from mlx_lm.models.qwen3 import Model as Qwen3Model
    from mlx_lm.models.qwen3 import ModelArgs as Qwen3Args
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    dense = _make_policy_runner(
        Qwen3Model(Qwen3Args(**_QWEN3_ARGS_KWARGS)), num_layers=2
    )
    dense.validate_paged_attention_support()

    hybrid = _make_policy_runner(
        TextModel(TextModelArgs(**_QWEN35_ARGS_KWARGS)),
        num_layers=4,
        hybrid_plan=_tiny_gdn_plan(4, [3]),
    )
    hybrid.validate_paged_attention_support()


def test_validate_accepts_shortconv_and_mamba2_state_families():
    """LFM2 conv slots and Nemotron-H mixer slots match their family plans.

    Nemotron-H also carries a stateless MLP block ('-'), which mlx-lm skips
    when declaring caches; validation must line native slots up against the
    plan's attention and state layers only.
    """
    import torch
    from mlx_lm.models.lfm2 import Model as LFM2Model
    from mlx_lm.models.lfm2 import ModelArgs as LFM2Args
    from mlx_lm.models.nemotron_h import Model as NemotronHModel
    from mlx_lm.models.nemotron_h import ModelArgs as NemotronHArgs

    from tests.stub_runner import NEMOTRON_H_TINY_ARGS, make_nemotron_hybrid_plan
    from vllm_metal.attention.runtime.factory import build_hybrid_runtime_plan

    lfm2_args = {
        "model_type": "lfm2",
        "vocab_size": 100,
        "hidden_size": 64,
        "num_hidden_layers": 2,
        "num_attention_heads": 4,
        "num_key_value_heads": 2,
        "max_position_embeddings": 512,
        "norm_eps": 1e-5,
        "conv_bias": False,
        "conv_L_cache": 3,
        "block_dim": 64,
        "block_ff_dim": 128,
        "block_multiple_of": 64,
        "block_ffn_dim_multiplier": 1.0,
        "block_auto_adjust_ff_dim": False,
        "layer_types": ["conv", "full_attention"],
    }
    lfm2 = _make_policy_runner(
        LFM2Model(LFM2Args(**lfm2_args)),
        num_layers=2,
        hybrid_plan=build_hybrid_runtime_plan(lfm2_args, 2, (torch.float16,)),
    )
    lfm2.validate_paged_attention_support()

    nemotron_args = {
        **NEMOTRON_H_TINY_ARGS,
        "num_hidden_layers": 3,
        "hybrid_override_pattern": "M*-",
    }
    nemotron = _make_policy_runner(
        NemotronHModel(NemotronHArgs(**nemotron_args)),
        num_layers=3,
        hybrid_plan=make_nemotron_hybrid_plan("M*-"),
    )
    nemotron.validate_paged_attention_support()


def test_validate_rejects_hybrid_plan_disagreement():
    """A hybrid whose declared split disagrees with the layer plan refuses.

    The plan expects attention at slots 1 and 3, but the Qwen3.5 model
    built with ``full_attention_interval=4`` declares ArraysCache at
    slot 1 -- state the runtime would map to a KV-style cache.
    """
    from mlx_lm.models.qwen3_5 import TextModel, TextModelArgs

    runner = _make_policy_runner(
        TextModel(TextModelArgs(**_QWEN35_ARGS_KWARGS)),
        num_layers=4,
        hybrid_plan=_tiny_gdn_plan(4, [1, 3]),
    )

    with pytest.raises(
        NotImplementedError, match=r"ArraysCache at native cache slot 1"
    ):
        runner.validate_paged_attention_support()
