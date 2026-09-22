# SPDX-License-Identifier: Apache-2.0
"""Paged-attention regressions for EXAONE 4.0."""

import mlx.core as mx
from mlx_lm.models.exaone4 import Attention, Model, ModelArgs

from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.sdpa import prepare_sdpa_qkv
from vllm_metal.v1.model_adapter import DefaultModelAdapter


def _model_args(*, num_hidden_layers: int = 4) -> ModelArgs:
    return ModelArgs(
        model_type="exaone4",
        hidden_size=8,
        num_hidden_layers=num_hidden_layers,
        intermediate_size=16,
        num_attention_heads=2,
        rms_norm_eps=1e-5,
        vocab_size=32,
        num_key_value_heads=1,
        max_position_embeddings=32_768,
        rope_theta=1_000_000.0,
        head_dim=4,
        tie_word_embeddings=False,
        rope_scaling={},
        sliding_window=4_096,
        sliding_window_pattern="LLLG",
    )


def test_cache_factory_drives_per_layer_sliding_window_policy() -> None:
    """EXAONE's own cache layout must drive Metal cache geometry."""
    args = _model_args(num_hidden_layers=8)

    sliding_windows = DefaultModelAdapter().build_sliding_window_per_layer(
        vars(args), args.num_hidden_layers, model=Model(args)
    )

    assert sliding_windows == [4_096, 4_096, 4_096, -1] * 2


def test_global_attention_preserves_unrotated_qk() -> None:
    """EXAONE 4.0 32B's global layers explicitly omit RoPE."""
    args = _model_args()
    attention = Attention(args, is_local=False)
    assert attention.use_rope is False

    attention.q_proj.weight = (
        mx.arange(64).reshape(8, 8).astype(mx.float32) / 31.0 - 1.0
    )
    attention.k_proj.weight = (
        mx.arange(32).reshape(4, 8).astype(mx.float32) / 19.0 - 0.75
    )
    attention.v_proj.weight = mx.ones((4, 8), dtype=mx.float32)
    x = mx.arange(24).reshape(1, 3, 8).astype(mx.float32) / 11.0 - 1.0

    expected_q = attention.q_norm(attention.q_proj(x).reshape(1, 3, 2, 4)).transpose(
        0, 2, 1, 3
    )
    expected_k = attention.k_norm(attention.k_proj(x).reshape(1, 3, 1, 4)).transpose(
        0, 2, 1, 3
    )
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
    )
    mx.eval(queries, keys, expected_q, expected_k)

    assert mx.array_equal(queries, expected_q).item()
    assert mx.array_equal(keys, expected_k).item()
