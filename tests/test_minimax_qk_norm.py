# SPDX-License-Identifier: Apache-2.0
"""Numeric regression coverage for MiniMax full-projection Q/K norms."""

import mlx.core as mx
from mlx_lm.models.minimax import MiniMaxAttention, ModelArgs

from vllm_metal.attention.attention_contracts import attention_contract_for
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.sdpa import prepare_sdpa_qkv


def _context(seq_len: int) -> PagedAttentionContext:
    return PagedAttentionContext(
        slot_mapping=list(range(seq_len)),
        block_tables=[[0]],
        context_lens=[seq_len],
        offsets=[],
        cu_seqlens=[0, seq_len],
    )


def test_minimax_qk_norm_is_applied_before_splitting_heads() -> None:
    args = ModelArgs(
        model_type="minimax_m2",
        hidden_size=16,
        intermediate_size=32,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        num_experts_per_tok=1,
        num_local_experts=2,
        shared_intermediate_size=0,
        num_hidden_layers=1,
        rms_norm_eps=1e-5,
        rope_theta=10_000.0,
        rotary_dim=4,
        vocab_size=32,
        head_dim=4,
        use_qk_norm=True,
    )
    attention = MiniMaxAttention(args)
    contract = attention_contract_for(attention)

    attention.q_norm.weight = mx.linspace(0.5, 1.5, 16)
    attention.k_norm.weight = mx.linspace(1.5, 0.5, 8)
    attention.q_proj.weight = mx.linspace(-1.0, 1.0, 256).reshape(16, 16)
    attention.k_proj.weight = mx.linspace(0.75, -0.75, 128).reshape(8, 16)
    attention.v_proj.weight = mx.ones((8, 16))

    x = mx.linspace(-1.0, 1.0, 48).reshape(1, 3, 16)
    expected_q = attention.q_norm(attention.q_proj(x)).reshape(1, 3, 4, 4)
    expected_k = attention.k_norm(attention.k_proj(x)).reshape(1, 3, 2, 4)
    expected_q = attention.rope(expected_q.transpose(0, 2, 1, 3))
    expected_k = attention.rope(expected_k.transpose(0, 2, 1, 3))

    queries, keys, _, _, _ = prepare_sdpa_qkv(
        attention,
        x,
        _context(seq_len=3),
        args.num_attention_heads,
        args.num_key_value_heads,
        attention_contract=contract,
    )
    mx.eval(queries, keys, expected_q, expected_k)

    assert mx.array_equal(queries, expected_q).item()
    assert mx.array_equal(keys, expected_k).item()
