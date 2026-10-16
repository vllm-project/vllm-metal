# SPDX-License-Identifier: Apache-2.0
"""Paged-attention regressions for Muse Glimmer."""

import mlx.core as mx
import pytest
from mlx_lm.models import muse_glimmer

from vllm_metal.attention.attention_contracts import attention_contract_for
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.sdpa import sdpa_forward
from vllm_metal.attention.impls.sdpa_wrapper import patch_sdpa_attention

# Four layers in Glimmer's [local, local, local, global] pattern.
_CONFIG = {
    "hidden_size": 128,
    "num_hidden_layers": 4,
    "intermediate_size": 128,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 64,
    "vocab_size": 32,
}


def _mlx_lm_attention(layer_idx: int):
    return muse_glimmer.Attention(muse_glimmer.ModelArgs(**_CONFIG), layer_idx)


def _mlx_vlm_attention(layer_idx: int):
    # The mlx-vlm floor (0.6.8) predates Muse Glimmer; mlx-lm is pinned.
    language = pytest.importorskip(
        "mlx_vlm.models.muse_glimmer.language",
        reason="installed mlx-vlm predates Muse Glimmer",
    )
    return language.Attention(language.TextConfig(**_CONFIG), layer_idx)


@pytest.mark.parametrize("layer_idx", [0, 3], ids=["local_rope", "global_nope"])
@pytest.mark.parametrize(
    "build", [_mlx_lm_attention, _mlx_vlm_attention], ids=["mlx_lm", "mlx_vlm"]
)
def test_paged_attention_matches_module_forward(build, layer_idx: int) -> None:
    """The output gate, weightless QK-norm and query scale must reach the kernel."""
    mx.random.seed(0)
    attention = build(layer_idx)
    x = mx.random.normal((1, 6, 128))
    cache = MetalPagedKVCache(
        num_layers=1,
        num_kv_heads=1,
        head_dim=64,
        num_blocks=1,
        block_size=16,
        dtype=mx.float32,
    )
    ctx = PagedAttentionContext(
        slot_mapping=list(range(6)),
        block_tables=[[0]],
        context_lens=[6],
        offsets=[],
        cu_seqlens=[0, 6],
    )

    actual, _ = sdpa_forward(
        attention,
        x,
        ctx,
        cache,
        layer_idx=0,
        attention_contract=attention_contract_for(attention),
    )
    expected = attention(x, mask="causal")
    mx.eval(actual, expected)

    assert mx.allclose(actual, expected, atol=2e-5, rtol=2e-5).item()


def test_patched_mlx_lm_model_keeps_its_mask_selection() -> None:
    """mlx-lm's Glimmer forward reads ``self_attn.layer_type`` through the wrapper."""
    model = muse_glimmer.Model(muse_glimmer.ModelArgs(**_CONFIG))
    tokens = mx.array([[1, 2, 3]])
    expected = model(tokens)

    patch_sdpa_attention(model, object(), 16)

    assert mx.array_equal(model(tokens), expected).item()
