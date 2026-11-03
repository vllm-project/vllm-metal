# SPDX-License-Identifier: Apache-2.0
"""Paged-attention regressions for Flint."""

import mlx.core as mx
import pytest
from mlx_lm.models import flint
from mlx_lm.models.base import create_causal_mask

from vllm_metal.attention.attention_contracts import attention_contract_for
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.sdpa import sdpa_forward

_WINDOW = 4
_SEQ_LEN = 9
_ARGS = flint.ModelArgs(
    model_type="flint",
    hidden_size=128,
    num_hidden_layers=2,
    intermediate_size=128,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=64,
    num_experts=4,
    num_experts_per_tok=2,
    moe_intermediate_size=64,
    shared_expert_intermediate_size=64,
    mlp_only_layers=[0, 1],
    layer_types=["full_attention", "sliding_attention"],
    sliding_window=_WINDOW,
    rms_norm_eps=1e-6,
    vocab_size=32,
)


@pytest.mark.parametrize(
    ("layer_idx", "window"),
    [(0, -1), (1, _WINDOW)],
    ids=["full_nope", "sliding_rope"],
)
def test_paged_attention_matches_mlx_lm(layer_idx: int, window: int) -> None:
    """Position-free full layers, windowed RoPE layers and the learned
    ``softmax_offset`` sinks must all reach the kernel."""
    mx.random.seed(0)
    attention = flint.Attention(_ARGS, layer_idx)
    # Checkpoint sinks are nonzero; the zero init would hide dropped sinks.
    attention.softmax_offset = mx.random.normal((_ARGS.num_attention_heads,))
    x = mx.random.normal((1, _SEQ_LEN, _ARGS.hidden_size))
    cache = MetalPagedKVCache(
        num_layers=1,
        num_kv_heads=_ARGS.num_key_value_heads,
        head_dim=_ARGS.head_dim,
        num_blocks=1,
        block_size=16,
        dtype=mx.float32,
        sliding_window_per_layer=[window],
    )
    ctx = PagedAttentionContext(
        slot_mapping=list(range(_SEQ_LEN)),
        block_tables=[[0]],
        context_lens=[_SEQ_LEN],
        offsets=[],
        cu_seqlens=[0, _SEQ_LEN],
    )

    actual, _ = sdpa_forward(
        attention,
        x,
        ctx,
        cache,
        layer_idx=0,
        attention_contract=attention_contract_for(attention),
    )
    mask = create_causal_mask(_SEQ_LEN, window_size=window if window > 0 else None)
    expected = attention(x, mask=mask)
    mx.eval(actual, expected)

    assert mx.allclose(actual, expected, atol=2e-5, rtol=2e-5).item()
