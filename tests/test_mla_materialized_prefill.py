# SPDX-License-Identifier: Apache-2.0
"""End-to-end test for materialized-MLA prefill. Absorbed-MLA prefill is routed
through materialized full K/V + standard MHA (MLX SDPA), which must match the
absorbed kv_lora-space path (the absorption identity). On by default for
absorbed models; no custom kernel; works on any GPU."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from vllm_metal.attention import context as pac
from vllm_metal.attention.caches.mla_cache import MLAPagedLatentCache
from vllm_metal.attention.impls.mla import MLAPagedAttentionWrapper

MultiLinear = pytest.importorskip("mlx_lm.models.mla").MultiLinear

# GLM-4.7-Flash dims (small num_heads / hidden for a fast test).
_H, _NOPE, _ROPE, _KVL, _VD, _HID, _BLK = 4, 128, 64, 512, 128, 256, 16


class _AbsorbedInner(nn.Module):
    """Absorbed-MLA stub shaped like glm4_moe_lite (MultiLinear embed_q/unembed_out)."""

    def __init__(self) -> None:
        super().__init__()
        self.q_lora_rank = None
        self.num_heads = _H
        self.q_head_dim = _NOPE + _ROPE
        self.qk_nope_head_dim = _NOPE
        self.qk_rope_head_dim = _ROPE
        self.kv_lora_rank = _KVL
        self.v_head_dim = _VD
        self.scale = (_NOPE + _ROPE) ** -0.5
        self.q_proj = nn.Linear(_HID, _H * (_NOPE + _ROPE), bias=False)
        self.kv_a_proj_with_mqa = nn.Linear(_HID, _KVL + _ROPE, bias=False)
        self.kv_a_layernorm = nn.LayerNorm(_KVL)
        self.embed_q = MultiLinear(_NOPE, _KVL, _H)
        self.unembed_out = MultiLinear(_KVL, _VD, _H)
        self.o_proj = nn.Linear(_H * _VD, _HID, bias=False)

    def rope(self, x: mx.array, offset: int = 0) -> mx.array:
        return x


@pytest.fixture(autouse=True)
def _clear_ctx():
    pac.clear_context()
    yield
    pac.clear_context()


def _make(quantize: bool = False, num_blocks: int = 8):
    mx.random.seed(0)
    inner = _AbsorbedInner()
    inner.apply(lambda p: p.astype(mx.float16))
    if quantize:
        # GLM-4.7-Flash-4bit ships embed_q/unembed_out as QuantizedMultiLinear,
        # whose quantized_matmul broadcasts the per-head weights differently from
        # the dense `x @ weight` — guards the 4bit materialization shape path.
        inner.embed_q = inner.embed_q.to_quantized(64, 4)
        inner.unembed_out = inner.unembed_out.to_quantized(64, 4)
    cache = MLAPagedLatentCache(
        num_layers=1,
        latent_dim=_KVL + _ROPE,
        num_blocks=num_blocks,
        block_size=_BLK,
        dtype=mx.float16,
    )
    return (
        inner,
        cache,
        MLAPagedAttentionWrapper(inner, layer_idx=0, latent_cache=cache),
    )


@pytest.mark.parametrize(
    ("quantize", "atol"),
    [(False, 2e-2), (True, 6e-2)],
    ids=["dense", "quantized-4bit"],
)
def test_materialized_prefill_matches_absorbed_loop(
    quantize: bool, atol: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    inner, cache, wrapper = _make(quantize=quantize)
    lens = [16, 48]  # 2 prefill requests, past=0, block-aligned
    total = sum(lens)
    cu = [0] + [int(c) for c in np.cumsum(lens)]
    ctx = pac.PagedAttentionContext(
        slot_mapping=list(range(total)),
        block_tables=[[0], [1, 2, 3]],
        context_lens=list(lens),
        cu_seqlens=cu,
        offsets=[0, 0],
    )
    x = mx.random.normal((1, total, _HID)).astype(mx.float16)

    def run() -> mx.array:
        cache.latent_caches[0] = mx.zeros_like(cache.latent_caches[0])
        pac.set_context(ctx)
        out = wrapper(x, mask=None, cache=None)
        mx.eval(out)
        pac.clear_context()
        return out

    # Reference: force the gate off → absorbed kv_lora-space (512-wide MQA) loop.
    monkeypatch.setattr(
        MLAPagedAttentionWrapper, "_materialized_prefill_ok", lambda *a, **k: False
    )
    ref = run()
    monkeypatch.undo()  # restore the real gate → materialized path (on by default)
    mat = run()

    assert mat.shape == (1, total, _HID)
    np.testing.assert_allclose(np.array(mat), np.array(ref), atol=atol, rtol=1e-2)


def test_materialized_prefill_gate() -> None:
    """The gate engages for an absorbed model on pure prefill and on
    continuation chunks with >= ``_MATERIALIZED_MIN_NEW_TOKENS_WITH_PAST``
    new tokens; small chunks with cached context, decode-shaped rows, and
    pure decode batches fall back to the absorbed loop."""
    inner, _, wrapper = _make()
    # pure prefill (past=0): ctx_len == num_new → engage even when small
    assert wrapper._materialized_prefill_ok(
        inner,
        pac.PagedAttentionContext(
            slot_mapping=[0, 1],
            block_tables=[[0]],
            context_lens=[2],
            cu_seqlens=[0, 2],
            offsets=[0],
        ),
    )
    # chunked prefill below the new-token threshold → fall back
    for num_new in (2, 511):
        assert not wrapper._materialized_prefill_ok(
            inner,
            pac.PagedAttentionContext(
                slot_mapping=list(range(num_new)),
                block_tables=[[0]],
                context_lens=[4 + num_new],
                cu_seqlens=[0, num_new],
                offsets=[4],
            ),
        )
    # chunked prefill at the threshold: past>0, num_new==512 → engage
    assert wrapper._materialized_prefill_ok(
        inner,
        pac.PagedAttentionContext(
            slot_mapping=list(range(512)),
            block_tables=[[0]],
            context_lens=[4 + 512],
            cu_seqlens=[0, 512],
            offsets=[4],
        ),
    )
    # decode-shaped row (num_new=1, ctx_len>1) mixed with prefill → fall back
    assert not wrapper._materialized_prefill_ok(
        inner,
        pac.PagedAttentionContext(
            slot_mapping=[0, 1, 2],
            block_tables=[[0], [1]],
            context_lens=[4, 2],
            cu_seqlens=[0, 1, 3],
            offsets=[3, 0],
        ),
    )
    # pure decode → fall back
    assert not wrapper._materialized_prefill_ok(
        inner,
        pac.PagedAttentionContext(
            slot_mapping=[0],
            block_tables=[[0]],
            context_lens=[4],
            cu_seqlens=[0, 1],
            offsets=[3],
        ),
    )


@pytest.mark.parametrize(
    ("quantize", "atol"),
    [(False, 2e-2), (True, 6e-2)],
    ids=["dense", "quantized-4bit"],
)
def test_chunked_prefill_matches_absorbed_loop(
    quantize: bool, atol: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Mixed batch: one fresh prefill plus two continuation chunks (past
    non-block-aligned and block-aligned). Materialized output must match the
    absorbed kv_lora-space loop."""
    # The continuation chunks below are smaller than the real 512-token
    # threshold; drop it so they exercise the materialized path.
    monkeypatch.setattr(
        MLAPagedAttentionWrapper, "_MATERIALIZED_MIN_NEW_TOKENS_WITH_PAST", 1
    )
    inner, cache, wrapper = _make(quantize=quantize, num_blocks=12)

    def slots(block_ids: list[int], start: int, num: int) -> list[int]:
        return [
            block_ids[pos // _BLK] * _BLK + pos % _BLK
            for pos in range(start, start + num)
        ]

    # Phase 1: prefill request B (20 tokens → blocks 4,5) and request C
    # (32 tokens → blocks 6,7) to seed past context in the cache.
    ctx1 = pac.PagedAttentionContext(
        slot_mapping=slots([4, 5], 0, 20) + slots([6, 7], 0, 32),
        block_tables=[[4, 5], [6, 7]],
        context_lens=[20, 32],
        cu_seqlens=[0, 20, 52],
        offsets=[0, 0],
    )
    x1 = mx.random.normal((1, 52, _HID)).astype(mx.float16)

    # Phase 2: A fresh 16-token prefill (block 0); B continues past=20
    # (non-block-aligned: next slots land mid-block in block 5, then block 8);
    # C continues past=32 (block-aligned, one new token-block: block 9).
    ctx2 = pac.PagedAttentionContext(
        slot_mapping=(
            slots([0], 0, 16) + slots([4, 5, 8], 20, 24) + slots([6, 7, 9], 32, 8)
        ),
        block_tables=[[0], [4, 5, 8], [6, 7, 9]],
        context_lens=[16, 44, 40],
        cu_seqlens=[0, 16, 40, 48],
        offsets=[0, 20, 32],
    )
    x2 = mx.random.normal((1, 48, _HID)).astype(mx.float16)

    # Under the patched threshold the phase-2 batch engages the materialized
    # path, so the `mat` arm below really exercises it.
    assert wrapper._materialized_prefill_ok(inner, ctx2)

    def run() -> mx.array:
        # Identical cache for both arms: reset, then re-run phase 1.
        cache.latent_caches[0] = mx.zeros_like(cache.latent_caches[0])
        pac.set_context(ctx1)
        mx.eval(wrapper(x1, mask=None, cache=None))
        pac.clear_context()
        pac.set_context(ctx2)
        out = wrapper(x2, mask=None, cache=None)
        mx.eval(out)
        pac.clear_context()
        return out

    # Reference: force the gate off → absorbed kv_lora-space (512-wide MQA) loop.
    monkeypatch.setattr(
        MLAPagedAttentionWrapper, "_materialized_prefill_ok", lambda *a, **k: False
    )
    ref = run()
    monkeypatch.undo()  # restores gate AND threshold — re-patch the threshold
    monkeypatch.setattr(
        MLAPagedAttentionWrapper, "_MATERIALIZED_MIN_NEW_TOKENS_WITH_PAST", 1
    )
    mat = run()

    assert mat.shape == (1, 48, _HID)
    np.testing.assert_allclose(np.array(mat), np.array(ref), atol=atol, rtol=1e-2)
