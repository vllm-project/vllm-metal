# SPDX-License-Identifier: Apache-2.0
"""Paged-attention regressions for Muse Glimmer."""

import mlx.core as mx
import pytest
from mlx_lm.models import muse_glimmer
from mlx_vlm.models.muse_glimmer import language

from vllm_metal.attention.attention_contracts import attention_contract_for
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.context import PagedAttentionContext
from vllm_metal.attention.impls.sdpa import prepare_sdpa_qkv, sdpa_forward
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
_LAYERS = pytest.mark.parametrize(
    "layer_idx", [0, 3], ids=["local_rope", "global_nope"]
)


def _mlx_lm_attention(layer_idx: int):
    return muse_glimmer.Attention(muse_glimmer.ModelArgs(**_CONFIG), layer_idx)


def _mlx_vlm_attention(layer_idx: int):
    return language.Attention(language.TextConfig(**_CONFIG), layer_idx)


def _context() -> PagedAttentionContext:
    return PagedAttentionContext(
        slot_mapping=list(range(6)),
        block_tables=[[0]],
        context_lens=[6],
        offsets=[],
        cu_seqlens=[0, 6],
    )


@_LAYERS
@pytest.mark.parametrize(
    "build", [_mlx_lm_attention, _mlx_vlm_attention], ids=["mlx_lm", "mlx_vlm"]
)
# bf16 allows about one ulp near 1; mlx-vlm's RoPE also rounds in bf16.
@pytest.mark.parametrize(
    ("dtype", "tol"), [(mx.float32, 2e-5), (mx.bfloat16, 1e-2)], ids=["fp32", "bf16"]
)
def test_paged_attention_matches_upstream(
    build, layer_idx: int, dtype: mx.Dtype, tol: float
) -> None:
    """The weightless QK-norm, query scale and output gate must reach the kernel.

    Both ports are served; the reference is mlx-vlm's port with the same weights.
    """
    mx.random.seed(0)
    attention = build(layer_idx)
    attention.set_dtype(dtype)
    upstream = _mlx_vlm_attention(layer_idx)
    upstream.update(attention.parameters())
    x = mx.random.normal((1, 6, 128)).astype(dtype)
    cache = MetalPagedKVCache(
        num_layers=1,
        num_kv_heads=1,
        head_dim=64,
        num_blocks=1,
        block_size=16,
        dtype=dtype,
    )

    actual, _ = sdpa_forward(
        attention,
        x,
        _context(),
        cache,
        layer_idx=0,
        attention_contract=attention_contract_for(attention),
    )
    expected = upstream(x, mask="causal")
    mx.eval(actual, expected)

    assert mx.allclose(actual, expected, atol=tol, rtol=tol).item()


@_LAYERS
def test_bf16_queries_are_scaled_in_fp32_before_rope(layer_idx: int) -> None:
    """Normed queries are scaled in fp32 and cast back before RoPE, as vLLM and
    mlx-vlm do; mlx-lm's bf16 multiply rounds differently."""
    mx.random.seed(0)
    attention = _mlx_lm_attention(layer_idx)
    attention.set_dtype(mx.bfloat16)
    x = mx.random.normal((1, 6, 128)).astype(mx.bfloat16)
    heads, head_dim = attention.n_heads, attention.head_dim

    queries, *_ = prepare_sdpa_qkv(
        attention,
        x,
        _context(),
        n_heads=heads,
        n_kv_heads=attention.n_kv_heads,
        attention_contract=attention_contract_for(attention),
    )

    def rope(q: mx.array) -> mx.array:
        q = q.transpose(0, 2, 1, 3)
        return attention.rope(q) if attention.use_rope else q

    normed = mx.fast.rms_norm(
        attention.q_proj(x).reshape(1, 6, heads, head_dim), None, attention.eps
    )
    scale = attention.qk_scale_factor
    expected = rope((normed.astype(mx.float32) * scale).astype(mx.bfloat16))
    assert mx.array_equal(queries, expected).item()
    # The comparison is sensitive enough to tell the two roundings apart.
    assert not mx.array_equal(rope(normed * scale), expected).item()


def test_patched_mlx_lm_model_keeps_its_mask_selection() -> None:
    """mlx-lm's Glimmer forward reads ``self_attn.layer_type`` through the wrapper."""
    model = muse_glimmer.Model(muse_glimmer.ModelArgs(**_CONFIG))
    tokens = mx.array([[1, 2, 3]])
    expected = model(tokens)

    patch_sdpa_attention(model, object(), 16)

    assert mx.array_equal(model(tokens), expected).item()
