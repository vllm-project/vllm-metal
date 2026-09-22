# SPDX-License-Identifier: Apache-2.0
"""Paged-attention regression for Cohere2 (Command R7B)."""

import mlx.core as mx
import pytest
from mlx_lm.models.cohere2 import Attention, ModelArgs

from vllm_metal.attention.attention_contracts import attention_contract_for
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.sdpa import prepare_sdpa_qkv


@pytest.mark.parametrize(
    ("layer_idx", "rotated"),
    [(0, True), (2, True), (3, False), (7, False)],
)
def test_rope_follows_the_layer_kind(layer_idx: int, rotated: bool) -> None:
    """Sliding layers rotate Q/K; Cohere2's global layers are NoPE."""
    args = ModelArgs(
        model_type="cohere2",
        hidden_size=8,
        head_dim=4,
        num_hidden_layers=8,
        intermediate_size=16,
        num_attention_heads=2,
        num_key_value_heads=1,
        vocab_size=32,
        sliding_window=4_096,
        sliding_window_pattern=4,
    )
    attention = Attention(args, layer_idx)
    assert attention.use_sliding_window is rotated

    attention.q_proj.weight = mx.linspace(-1.0, 1.0, 64).reshape(8, 8)
    attention.k_proj.weight = mx.linspace(0.75, -0.75, 32).reshape(4, 8)
    attention.v_proj.weight = mx.ones((4, 8))
    x = mx.linspace(-1.0, 1.0, 24).reshape(1, 3, 8)

    expected_q = attention.q_proj(x).reshape(1, 3, 2, 4).transpose(0, 2, 1, 3)
    expected_k = attention.k_proj(x).reshape(1, 3, 1, 4).transpose(0, 2, 1, 3)
    if rotated:
        expected_q = attention.rope(expected_q, offset=17)
        expected_k = attention.rope(expected_k, offset=17)
    context = PagedAttentionContext(
        slot_mapping=[17, 18, 19],
        block_tables=[[0, 1]],
        context_lens=[20],
        offsets=[17],
        cu_seqlens=[0, 3],
    )

    queries, keys, _, _, _ = prepare_sdpa_qkv(
        attention,
        x,
        context,
        attention.n_heads,
        attention.n_kv_heads,
        attention_contract=attention_contract_for(attention),
    )
    mx.eval(queries, keys, expected_q, expected_k)

    assert mx.array_equal(queries, expected_q).item()
    assert mx.array_equal(keys, expected_k).item()
