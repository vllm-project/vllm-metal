# SPDX-License-Identifier: Apache-2.0
"""Correctness gates for the contiguous native-SDPA decode fast path.

``_native_sdpa_decode_fast_path`` routes *contiguous* single-sequence decode
to MLX native SDPA over zero-copy strided views. Non-contiguous runs return
``None`` so ``sdpa_forward`` falls through to ``paged_attention_primitive``
(GQA-shared flash-decode at long context, split-KV otherwise) — those
shapes are covered by ``tests/test_gqa_paged_decode.py``.
"""

from __future__ import annotations

import inspect
import math
from types import SimpleNamespace

import mlx.core as mx
import pytest

import vllm_metal.attention.impls.sdpa as sdpa_mod
from vllm_metal.attention.impls.sdpa import _native_sdpa_decode_fast_path


def _ref_sdpa(q: mx.array, k: mx.array, v: mx.array, scale: float) -> mx.array:
    """float32 reference: q (Hq, D), k/v (seq, Hkv, D) -> (Hq, D)."""
    n_heads, dim = q.shape
    n_kv_heads = k.shape[1]
    group = n_heads // n_kv_heads
    qf = q.astype(mx.float32).reshape(n_kv_heads, group, dim)
    kf = k.astype(mx.float32).transpose(1, 0, 2)
    vf = v.astype(mx.float32).transpose(1, 0, 2)
    s = mx.einsum("kgd,ksd->kgs", qf, kf) * scale
    return mx.einsum("kgs,ksd->kgd", mx.softmax(s, axis=-1), vf).reshape(n_heads, dim)


def _make_ctx(seq: int):
    return SimpleNamespace(context_lens=[seq], kernel_metadata_cache={})


def _run_fast_path(
    n_kv_heads: int,
    group: int,
    dim: int,
    block: int,
    seq: int,
    dtype,
    contig: bool,
    seed: int = 0,
):
    mx.random.seed(seed)
    n_blocks = (seq + block - 1) // block
    n_cache_blocks = n_blocks * 2 + 16
    kc = mx.random.normal((n_cache_blocks, block, n_kv_heads, dim)).astype(dtype)
    vc = mx.random.normal((n_cache_blocks, block, n_kv_heads, dim)).astype(dtype)
    if contig:
        table = list(range(7, 7 + n_blocks))
    else:
        table = (
            [3, 4, 6, 7][:n_blocks]
            if n_blocks <= 4
            else ([3, 4] + list(range(6, 6 + n_blocks - 2)))
        )
    q = mx.random.normal((1, n_kv_heads * group, dim)).astype(dtype)
    scale = 1.0 / math.sqrt(dim)
    ctx = _make_ctx(seq)
    out = _native_sdpa_decode_fast_path(
        q,
        kc,
        vc,
        ctx,
        None,
        [table],
        block,
        n_kv_heads,
        scale,
        0.0,
        None,
        None,
        None,
    )
    return out, q, kc, vc, table, scale, ctx


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("seq", [1, 100, 4096, 4097, 16384])
def test_contiguous_path_matches_reference(seq, dtype):
    out, q, kc, vc, table, scale, _ = _run_fast_path(
        8, 4, 128, 16, seq, dtype, contig=True
    )
    assert out is not None, "contiguous native-SDPA path did not fire"
    block = 16
    rows = [table[t // block] * block + t % block for t in range(seq)]
    kg = kc.reshape(-1, 8, 128)[mx.array(rows)]
    vg = vc.reshape(-1, 8, 128)[mx.array(rows)]
    ref = _ref_sdpa(q[0], kg, vg, scale)
    diff = mx.max(mx.abs(out.reshape(-1, 128).astype(mx.float32) - ref)).item()
    assert diff < 0.01, f"max abs diff {diff}"


def test_non_contiguous_defers_to_paged_kernel():
    out, *_ = _run_fast_path(8, 4, 128, 16, 4096, mx.bfloat16, contig=False)
    assert out is None


def test_plan_memoized_per_forward_step():
    """Second layer on the same ctx must reuse the plan, not rescan."""
    _, _, _, _, _, _, ctx = _run_fast_path(
        8, 4, 128, 16, 4096, mx.bfloat16, contig=True
    )
    assert len(ctx.kernel_metadata_cache) == 1
    mx.random.seed(1)
    n_kv_heads, group, dim, block, seq = 8, 4, 128, 16, 4096
    n_blocks = (seq + block - 1) // block
    kc = mx.random.normal((n_blocks * 2 + 16, block, n_kv_heads, dim)).astype(
        mx.bfloat16
    )
    q = mx.random.normal((1, n_kv_heads * group, dim)).astype(mx.bfloat16)
    out = _native_sdpa_decode_fast_path(
        q,
        kc,
        kc,
        ctx,
        None,
        [list(range(20, 20 + n_blocks))],  # different table; memoized plan wins
        block,
        n_kv_heads,
        1.0 / math.sqrt(dim),
        0.0,
        None,
        None,
        None,
    )
    assert out is not None
    assert len(ctx.kernel_metadata_cache) == 1


def _gating_kwargs(*, contig: bool = False):
    table = [3, 4, 5, 6] if contig else [3, 4, 6, 7]
    return {
        "q_3d": mx.zeros((1, 32, 128), dtype=mx.bfloat16),
        "k_cache": mx.zeros((64, 16, 8, 128), dtype=mx.bfloat16),
        "v_cache": mx.zeros((64, 16, 8, 128), dtype=mx.bfloat16),
        "ctx": _make_ctx(60),
        "group_index": None,
        "raw_block_tables": [table],
        "cache_block_size": 16,
        "cache_kv_heads": 8,
        "attn_scale": 0.088,
        "attn_softcap": 0.0,
        "layer_sliding_window": None,
        "sinks": None,
        "verify_window_q": None,
    }


def _fast_path_named(**kw):
    return _native_sdpa_decode_fast_path(
        kw["q_3d"],
        kw["k_cache"],
        kw["v_cache"],
        kw["ctx"],
        kw["group_index"],
        kw["raw_block_tables"],
        kw["cache_block_size"],
        kw["cache_kv_heads"],
        kw["attn_scale"],
        kw["attn_softcap"],
        kw["layer_sliding_window"],
        kw["sinks"],
        kw["verify_window_q"],
    )


def test_gating_fallbacks():
    kw = _gating_kwargs(contig=True)
    assert _fast_path_named(**kw) is not None  # contiguous fires native SDPA

    assert _fast_path_named(**_gating_kwargs(contig=False)) is None

    bad = dict(kw, attn_softcap=50.0)
    assert _fast_path_named(**bad) is None, "softcap must fall back"

    bad = dict(kw, sinks=mx.zeros((32,), dtype=mx.float32))
    assert _fast_path_named(**bad) is None, "sinks must fall back"

    bad = dict(kw, layer_sliding_window=1024)
    assert _fast_path_named(**bad) is None, "sliding window must fall back"

    bad = dict(kw, verify_window_q=4)
    assert _fast_path_named(**bad) is None, "verify window must fall back"

    bad = dict(kw, q_3d=mx.zeros((2, 32, 128), dtype=mx.bfloat16))
    assert _fast_path_named(**bad) is None, "multi-token must fall back"

    bad = dict(kw, ctx=SimpleNamespace(context_lens=[30, 30], kernel_metadata_cache={}))
    assert _fast_path_named(**bad) is None, "multi-seq must fall back"

    bad = dict(kw, q_3d=mx.zeros((1, 32, 128), dtype=mx.float32))
    assert _fast_path_named(**bad) is None, "fp32 must fall back"

    bad = dict(kw, q_3d=mx.zeros((1, 32, 80), dtype=mx.bfloat16))
    assert _fast_path_named(**bad) is None, "unsupported head_dim must fall back"

    bad = dict(kw, raw_block_tables=[[3, 4]], ctx=_make_ctx(60))  # needs 4 blocks
    assert _fast_path_named(**bad) is None, "short table must fall back"


def test_env_disable(monkeypatch):
    monkeypatch.setenv("VLLM_METAL_NATIVE_SDPA_DECODE", "0")
    assert _fast_path_named(**_gating_kwargs(contig=True)) is None


def test_no_python_jit_gqa_kernels():
    """Shipped decode must not construct mx.fast.metal_kernel GQA pass1/merge."""
    assert not hasattr(sdpa_mod, "_gqa_decode_kernels")
    assert not hasattr(sdpa_mod, "_GQA_DECODE_PASS1_TEMPLATE")
    src = inspect.getsource(sdpa_mod._native_sdpa_decode_fast_path)
    assert "metal_kernel" not in src
    assert "_gqa_decode_kernels" not in src
