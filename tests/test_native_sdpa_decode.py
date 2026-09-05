# SPDX-License-Identifier: Apache-2.0
"""Correctness gates for the single-sequence decode fast path.

``_native_sdpa_decode_fast_path`` routes decode attention either to MLX's
native SDPA over zero-copy strided views (contiguous block runs) or to a
block-table-driven flash-decode kernel (non-contiguous runs, e.g. hybrid
GDN interleave). Both paths are compared against a float32 einsum reference
here — the flash-decode launch config is in threads, not threadgroups, a
mistake these tests catch immediately (it computes only tile 0 / kv head 0).
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import mlx.core as mx
import pytest

from vllm_metal.attention.impls.sdpa import (
    _GQA_TILE,
    _native_sdpa_decode_fast_path,
)


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


def _interleaved_table(n_blocks: int) -> list[int]:
    """Run-of-2, skip-1 pattern produced by hybrid GDN block interleave."""
    table: list[int] = []
    b = 3
    while len(table) < n_blocks:
        table += [b, b + 1]
        b += 3
    return table[:n_blocks]


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
        table = _interleaved_table(n_blocks)
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
    assert out is not None, "fast path did not fire"
    rows = [table[t // block] * block + t % block for t in range(seq)]
    kg = kc.reshape(-1, n_kv_heads, dim)[mx.array(rows)]
    vg = vc.reshape(-1, n_kv_heads, dim)[mx.array(rows)]
    ref = _ref_sdpa(q[0], kg, vg, scale)
    diff = mx.max(mx.abs(out.reshape(-1, dim).astype(mx.float32) - ref)).item()
    return diff, ctx


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize(
    "n_kv_heads,group,dim,block,seq",
    [
        (8, 4, 128, 16, 1),  # single token
        (8, 4, 128, 16, 4096),
        (8, 4, 128, 16, 4097),  # seq % block != 0
        (4, 8, 128, 32, 5000),
        (1, 8, 64, 16, 3333),  # MQA, head_dim 64
        (2, 2, 256, 8, 2000),  # head_dim 256
        (8, 4, 96, 16, 1000),  # head_dim 96
        (8, 4, 128, 16, _GQA_TILE - 1),
        (8, 4, 128, 16, _GQA_TILE),  # exactly one tile
        (8, 4, 128, 16, _GQA_TILE + 1),
        (8, 1, 128, 16, 2000),  # no GQA grouping
        (8, 4, 128, 544, 30000),  # hybrid align block size
    ],
)
def test_kernel_path_matches_reference(n_kv_heads, group, dim, block, seq, dtype):
    diff, _ = _run_fast_path(n_kv_heads, group, dim, block, seq, dtype, contig=False)
    assert diff < 0.01, f"max abs diff {diff}"


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16])
@pytest.mark.parametrize("seq", [1, 100, 4096, 30000])
def test_contiguous_path_matches_reference(seq, dtype):
    diff, _ = _run_fast_path(8, 4, 128, 16, seq, dtype, contig=True)
    assert diff < 0.01, f"max abs diff {diff}"


def test_plan_memoized_per_forward_step():
    """Second layer on the same ctx must reuse the plan, not rescan."""
    _, ctx = _run_fast_path(8, 4, 128, 16, 4096, mx.bfloat16, contig=False)
    assert len(ctx.kernel_metadata_cache) == 1
    # Repeat with a different table object under the same ctx: the memoized
    # plan (not the new list) drives the dispatch.
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
        [list(range(5, 5 + n_blocks))],  # contiguous, but memoized plan wins
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


def _gating_kwargs():
    return {
        "q_3d": mx.zeros((1, 32, 128), dtype=mx.bfloat16),
        "k_cache": mx.zeros((64, 16, 8, 128), dtype=mx.bfloat16),
        "v_cache": mx.zeros((64, 16, 8, 128), dtype=mx.bfloat16),
        "ctx": _make_ctx(60),
        "group_index": None,
        "raw_block_tables": [[3, 4, 6, 7]],
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
    kw = _gating_kwargs()
    assert _fast_path_named(**kw) is not None  # baseline fires

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
    assert _fast_path_named(**_gating_kwargs()) is None
