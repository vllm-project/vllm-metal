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
from vllm_metal.attention.impls.mla import (
    MLAPagedAttentionWrapper,
    materialized_min_new_tokens_with_past,
)

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
        MLAPagedAttentionWrapper, "_materialized_segments", lambda *a, **k: None
    )
    ref = run()
    monkeypatch.undo()  # restore the real gate → materialized path (on by default)
    mat = run()

    assert mat.shape == (1, total, _HID)
    np.testing.assert_allclose(np.array(mat), np.array(ref), atol=atol, rtol=1e-2)


def _ctx(context_lens: list[int], cu_seqlens: list[int]) -> pac.PagedAttentionContext:
    return pac.PagedAttentionContext(
        slot_mapping=list(range(cu_seqlens[-1])),
        block_tables=[[i] for i in range(len(context_lens))],
        context_lens=context_lens,
        cu_seqlens=cu_seqlens,
        offsets=[
            c - (e - s)
            for c, s, e in zip(
                context_lens, cu_seqlens[:-1], cu_seqlens[1:], strict=True
            )
        ],
    )


def test_materialized_segments_routing(monkeypatch: pytest.MonkeyPatch) -> None:
    """Segments route independently: pure prefill and continuation chunks with
    >= ``_materialized_min_new_with_past`` new tokens materialize; small
    chunks with cached context and decode-shaped rows stay absorbed. With no
    routed multi-token segment the batch keeps the absorbed/kernel paths."""
    inner, _, wrapper = _make()
    route = wrapper._materialized_segments
    # Fixture attention dims (nope/rope/v = 128/64/128, kv_lora 512) are the
    # DeepSeek-V2/V3 family's → threshold 256.
    assert wrapper._materialized_min_new_with_past == 256
    # pure prefill (past=0): ctx_len == num_new → routed even when small
    assert route(inner, _ctx([2], [0, 2])) == [True]
    # chunked prefill below the new-token threshold → nothing routed
    for num_new in (2, 255):
        assert route(inner, _ctx([4 + num_new], [0, num_new])) is None
    # chunked prefill at the threshold: past>0, num_new==256 → routed
    assert route(inner, _ctx([4 + 256], [0, 256])) == [True]
    # decode row packed ahead of a prefill: only the prefill is routed
    assert route(inner, _ctx([4, 2], [0, 1, 3])) == [False, True]
    # decode rows + small continuation + large continuation + fresh prefill
    assert route(inner, _ctx([9, 7, 4 + 8, 4 + 512, 16], [0, 1, 2, 10, 522, 538])) == [
        False,
        False,
        False,
        True,
        True,
    ]
    # pure decode → nothing routed
    assert route(inner, _ctx([4, 5], [0, 1, 2])) is None
    # decode rows never route, even with the threshold at 1
    monkeypatch.setattr(wrapper, "_materialized_min_new_with_past", 1)
    assert route(inner, _ctx([4, 3], [0, 1, 3])) == [False, True]


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
    inner, cache, wrapper = _make(quantize=quantize, num_blocks=12)
    # The continuation chunks below are smaller than the dims-derived
    # threshold; drop it so they exercise the materialized path.
    monkeypatch.setattr(wrapper, "_materialized_min_new_with_past", 1)

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

    # Under the patched threshold every phase-2 segment is routed to the
    # materialized path, so the `mat` arm below really exercises it.
    assert wrapper._materialized_segments(inner, ctx2) == [True, True, True]

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
        MLAPagedAttentionWrapper, "_materialized_segments", lambda *a, **k: None
    )
    ref = run()
    monkeypatch.undo()  # restores gate AND threshold — re-patch the threshold
    monkeypatch.setattr(wrapper, "_materialized_min_new_with_past", 1)
    mat = run()

    assert mat.shape == (1, 48, _HID)
    np.testing.assert_allclose(np.array(mat), np.array(ref), atol=atol, rtol=1e-2)


@pytest.mark.parametrize(
    ("quantize", "atol"),
    [(False, 2e-2), (True, 6e-2)],
    ids=["dense", "quantized-4bit"],
)
def test_mixed_decode_prefill_batch_routes_per_segment(
    quantize: bool, atol: float, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Continuous-batching shape: decode rows packed ahead of a fresh prefill,
    a large continuation and a small continuation. Only the prefill-shaped
    segments above the threshold materialize; decode rows and the small chunk
    take the absorbed attention, and the combined output matches the
    all-absorbed loop."""
    inner, cache, wrapper = _make(quantize=quantize, num_blocks=16)
    # Scaled-down threshold: the 24-token continuation clears it, the
    # 8-token one does not.
    monkeypatch.setattr(wrapper, "_materialized_min_new_with_past", 16)

    def slots(block_ids: list[int], start: int, num: int) -> list[int]:
        return [
            block_ids[pos // _BLK] * _BLK + pos % _BLK
            for pos in range(start, start + num)
        ]

    # Phase 1 seeds past context: D1 (5 tokens, block 10), D2 (17, blocks
    # 11,12), B (20, blocks 4,5), C (32, blocks 6,7).
    ctx1 = pac.PagedAttentionContext(
        slot_mapping=(
            slots([10], 0, 5)
            + slots([11, 12], 0, 17)
            + slots([4, 5], 0, 20)
            + slots([6, 7], 0, 32)
        ),
        block_tables=[[10], [11, 12], [4, 5], [6, 7]],
        context_lens=[5, 17, 20, 32],
        cu_seqlens=[0, 5, 22, 42, 74],
        offsets=[0, 0, 0, 0],
    )
    x1 = mx.random.normal((1, 74, _HID)).astype(mx.float16)

    # Phase 2 (decode first, then prefill — the runner's packed order):
    # D1, D2 decode one token each; A fresh 16-token prefill; B continues
    # past=20 with 24 new (routed); C continues past=32 with 8 new (absorbed).
    ctx2 = pac.PagedAttentionContext(
        slot_mapping=(
            slots([10], 5, 1)
            + slots([11, 12], 17, 1)
            + slots([0], 0, 16)
            + slots([4, 5, 8], 20, 24)
            + slots([6, 7, 9], 32, 8)
        ),
        block_tables=[[10], [11, 12], [0], [4, 5, 8], [6, 7, 9]],
        context_lens=[6, 18, 16, 44, 40],
        cu_seqlens=[0, 1, 2, 18, 42, 50],
        offsets=[5, 17, 0, 20, 32],
        num_decode_requests=2,
    )
    x2 = mx.random.normal((1, 50, _HID)).astype(mx.float16)
    assert wrapper._materialized_segments(inner, ctx2) == [
        False,
        False,
        True,
        True,
        False,
    ]

    absorbed_calls: list[int] = []
    real_absorbed = MLAPagedAttentionWrapper._absorbed_segment

    def spy(self, *args, **kwargs):
        absorbed_calls.append(args[-1])
        return real_absorbed(self, *args, **kwargs)

    def run() -> mx.array:
        cache.latent_caches[0] = mx.zeros_like(cache.latent_caches[0])
        pac.set_context(ctx1)
        mx.eval(wrapper(x1, mask=None, cache=None))
        pac.clear_context()
        absorbed_calls.clear()
        pac.set_context(ctx2)
        out = wrapper(x2, mask=None, cache=None)
        mx.eval(out)
        pac.clear_context()
        return out

    monkeypatch.setattr(MLAPagedAttentionWrapper, "_absorbed_segment", spy)
    mat = run()
    assert absorbed_calls == [0, 1, 4]  # decode rows + the small continuation

    # Reference: routing off → every segment on the absorbed loop.
    monkeypatch.setattr(
        MLAPagedAttentionWrapper, "_materialized_segments", lambda *a, **k: None
    )
    ref = run()
    assert absorbed_calls == [0, 1, 2, 3, 4]

    assert mat.shape == (1, 50, _HID)
    np.testing.assert_allclose(np.array(mat), np.array(ref), atol=atol, rtol=1e-2)


@pytest.mark.parametrize(
    ("nope", "rope", "v", "expected"),
    [
        (128, 64, 128, 256),  # DeepSeek-V2 / V2-Lite / V3, Kimi-K2
        (192, 64, 256, 512),  # GLM-4.7-Flash
        (512, 64, 512, None),  # materialized attention wider than absorbed
    ],
    ids=["deepseek", "glm-4.7-flash", "never"],
)
def test_materialized_threshold_from_attention_dims(
    nope: int, rope: int, v: int, expected: int | None
) -> None:
    """The continuation threshold follows the attention dims: the FLOP
    break-even times the measured margin, rounded up to 64 tokens."""
    assert (
        materialized_min_new_tokens_with_past(
            kv_lora_rank=512, qk_nope_head_dim=nope, qk_rope_head_dim=rope, v_head_dim=v
        )
        == expected
    )


def test_threshold_none_blocks_only_cached_context_segments(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """With no profitable threshold, continuation chunks stay absorbed while
    pure-prefill segments still materialize."""
    inner, _, wrapper = _make()
    monkeypatch.setattr(wrapper, "_materialized_min_new_with_past", None)
    assert wrapper._materialized_segments(inner, _ctx([4 + 1024], [0, 1024])) is None
    assert wrapper._materialized_segments(
        inner, _ctx([4 + 1024, 32], [0, 1024, 1056])
    ) == [False, True]


def test_decode_batch_rows_gating(monkeypatch: pytest.MonkeyPatch) -> None:
    """Batched decode picks single-token rows under the per-row context cap,
    and only when enough rows batch to amortize the dispatch; rows over the
    cap or past the padded-volume cap keep the per-segment loop."""
    import vllm_metal.attention.impls.mla as mla_mod

    _, _, wrapper = _make()
    route = wrapper._decode_batch_rows

    # Below the row-count floor: nothing batches.
    assert route(_ctx([16] * 8, list(range(9)))) is None

    monkeypatch.setattr(mla_mod, "_DECODE_BATCH_MIN_ROWS", 4)
    ctx = _ctx([16] * 4, list(range(5)))
    assert route(ctx) == [0, 1, 2, 3]

    # Rows over the context cap are excluded; the rest still batch.
    ctx = _ctx([16, 4096, 16, 16, 16], list(range(6)))
    assert route(ctx) == [0, 2, 3, 4]

    # Eligible rows come out sorted by context length so chunks group
    # similar-sized rows and stay tight under the token cap.
    ctx = _ctx([32, 8, 16, 8, 24], list(range(6)))
    assert route(ctx) == [1, 3, 2, 4, 0]

    # Multi-token segments (fresh prefill, continuation chunk) never batch.
    ctx = _ctx([16, 16, 32, 16, 16], [0, 1, 2, 4, 5, 6])
    assert route(ctx) == [0, 1, 3, 4]

    # Padded volume past the cap no longer rejects: the batch chunks
    # inside _absorbed_decode_batch instead.
    monkeypatch.setattr(mla_mod, "_DECODE_BATCH_MAX_TOKENS", 512)
    ctx = _ctx([64] * 16, list(range(17)))
    assert route(ctx) == list(range(16))

    # The row list is memoized on the per-forward metadata.
    assert mla_mod._mla_metadata(ctx).decode_batch_rows == list(range(16))


@pytest.mark.parametrize(
    ("quantize", "atol"),
    [(False, 2e-2), (True, 6e-2)],
    ids=["dense", "quantized-4bit"],
)
@pytest.mark.parametrize(
    "token_cap",
    [65536, 64],
    ids=["single-chunk", "chunked"],
)
def test_batched_decode_matches_absorbed_loop(
    quantize: bool, atol: float, token_cap: int, monkeypatch: pytest.MonkeyPatch
) -> None:
    """18 decode rows with varied contexts, one over the cap: the short rows
    take the batched absorbed pass (unequal contexts exercise the padding
    mask), the long row stays on the per-segment loop, and the packed output
    matches the all-looped reference.  The chunked variant caps the padded
    volume so the batch splits into multiple passes instead of falling back
    to the loop."""
    import math

    import vllm_metal.attention.impls.mla as mla_mod

    n_rows = 18
    attn_passes: list[int] = []
    real_attn = MLAPagedAttentionWrapper._apply_absorbed_mla_attention

    def attn_spy(self, *args, **kwargs):
        # rq_nope is [n_rows, nheads, 1, dim]: the pass's batch size.
        attn_passes.append(kwargs["rq_nope"].shape[0])
        return real_attn(self, *args, **kwargs)

    # _apply_mla_attention is bound to _apply_absorbed_mla_attention at
    # __init__, so the spy must be installed before the wrapper is built.
    monkeypatch.setattr(
        MLAPagedAttentionWrapper, "_apply_absorbed_mla_attention", attn_spy
    )
    inner, cache, wrapper = _make(quantize=quantize, num_blocks=64)
    # Scaled-down gates: the row with 48 cached tokens stays on the
    # per-segment loop while the other 17 rows batch.  max_ctx is 29, so a
    # 64-token cap chunks the batch into groups of 2.
    monkeypatch.setattr(mla_mod, "_DECODE_BATCH_MAX_CTX", 40)
    monkeypatch.setattr(mla_mod, "_DECODE_BATCH_MIN_ROWS", 4)
    monkeypatch.setattr(mla_mod, "_DECODE_BATCH_MAX_TOKENS", token_cap)

    # Varied past lengths (16..30) so batched rows pad to the max context.
    pasts = [16 + 3 * (i % 5) for i in range(n_rows)]
    pasts[9] = 47  # context 48 > cap -> loops
    tables: list[list[int]] = []
    nb = 0
    for p in pasts:
        n_blocks = math.ceil((p + 1) / _BLK)
        tables.append(list(range(nb, nb + n_blocks)))
        nb += n_blocks

    def slots(block_ids: list[int], start: int, num: int) -> list[int]:
        return [
            block_ids[pos // _BLK] * _BLK + pos % _BLK
            for pos in range(start, start + num)
        ]

    # Phase 1 seeds each row's past context.
    ctx1 = pac.PagedAttentionContext(
        slot_mapping=[
            s for t, p in zip(tables, pasts, strict=True) for s in slots(t, 0, p)
        ],
        block_tables=tables,
        context_lens=pasts,
        cu_seqlens=[0] + [int(c) for c in np.cumsum(pasts)],
        offsets=[0] * n_rows,
    )
    x1 = mx.random.normal((1, int(np.cumsum(pasts)[-1]), _HID)).astype(mx.float16)

    # Phase 2: one decode token per row, appended at each row's last slot.
    ctx2 = pac.PagedAttentionContext(
        slot_mapping=[slots(t, p, 1)[0] for t, p in zip(tables, pasts, strict=True)],
        block_tables=tables,
        context_lens=[p + 1 for p in pasts],
        cu_seqlens=list(range(n_rows + 1)),
        offsets=pasts,
        num_decode_requests=n_rows,
    )
    x2 = mx.random.normal((1, n_rows, _HID)).astype(mx.float16)

    absorbed_calls: list[int] = []
    batched_groups: list[list[int]] = []
    batch_calls: list[tuple[tuple, mx.array]] = []
    real_segment = MLAPagedAttentionWrapper._absorbed_segment
    real_batch = MLAPagedAttentionWrapper._absorbed_decode_batch

    def segment_spy(self, *args, **kwargs):
        absorbed_calls.append(args[-1])
        return real_segment(self, *args, **kwargs)

    def batch_spy(self, *args, **kwargs):
        batched_groups.append(list(args[-1]))
        result = real_batch(self, *args, **kwargs)
        batch_calls.append((args, result))
        return result

    def run() -> mx.array:
        cache.latent_caches[0] = mx.zeros_like(cache.latent_caches[0])
        pac.set_context(ctx1)
        mx.eval(wrapper(x1, mask=None, cache=None))
        pac.clear_context()
        absorbed_calls.clear()
        batched_groups.clear()
        batch_calls.clear()
        attn_passes.clear()
        pac.set_context(ctx2)
        out = wrapper(x2, mask=None, cache=None)
        mx.eval(out)
        pac.clear_context()
        return out

    monkeypatch.setattr(MLAPagedAttentionWrapper, "_absorbed_segment", segment_spy)
    monkeypatch.setattr(MLAPagedAttentionWrapper, "_absorbed_decode_batch", batch_spy)
    out = run()
    assert absorbed_calls == [9]  # only the over-cap row loops
    batched = [i for i in range(n_rows) if i != 9]
    # The batch rows are sorted by context length so chunk bounds group
    # similar-sized rows.
    batched.sort(key=lambda i: pasts[i])
    assert batched_groups == [batched]
    # Greedy chunks: take sorted rows until count * ctx of the next row would
    # pass the cap — one pass when the cap fits everything, chunks otherwise.
    expected_chunks: list[int] = []
    run_rows = 0
    for i in batched:
        if run_rows and (run_rows + 1) * (pasts[i] + 1) > token_cap:
            expected_chunks.append(run_rows)
            run_rows = 0
        run_rows += 1
    expected_chunks.append(run_rows)
    assert attn_passes == expected_chunks + [1]

    # The memoized device index is keyed on the row list itself.
    idx = mla_mod._mla_metadata(ctx2).decode_batch_idx
    assert idx is not None and idx.tolist() == batched

    # A different same-length row list must get its own index, not the memoized
    # one, and the token cap must hold for rows that are not sorted.
    ((call_args, first),) = batch_calls
    rev_rows = batched[::-1]
    attn_passes.clear()
    rev = real_batch(wrapper, *call_args[:-1], rev_rows)
    mx.eval(rev)
    expected_rev: list[int] = []
    run_rows = run_max = 0
    for i in rev_rows:
        if run_rows and (run_rows + 1) * max(run_max, pasts[i] + 1) > token_cap:
            expected_rev.append(run_rows)
            run_rows = run_max = 0
        run_rows += 1
        run_max = max(run_max, pasts[i] + 1)
    expected_rev.append(run_rows)
    assert attn_passes == expected_rev
    np.testing.assert_allclose(
        np.array(rev), np.array(first)[::-1], atol=atol, rtol=1e-2
    )
    assert idx.tolist() == batched  # the memoized index is left untouched

    # Reference: batching gate off -> every row on the per-segment loop.
    monkeypatch.setattr(
        MLAPagedAttentionWrapper, "_decode_batch_rows", lambda *a, **k: None
    )
    ref = run()
    assert absorbed_calls == list(range(n_rows))
    assert batched_groups == []

    assert out.shape == (1, n_rows, _HID)
    np.testing.assert_allclose(np.array(out), np.array(ref), atol=atol, rtol=1e-2)


@pytest.mark.parametrize("quantize", [False, True], ids=["dense", "quantized-4bit"])
def test_absorbed_attention_folds_query_heads(
    quantize: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The query heads share one latent K/V head, so SDPA gets one head with
    nheads * q_len query rows (a per-head layout rereads K and V per head) and
    the result matches the per-head layout, with a request axis and a causal
    mask."""
    import vllm_metal.attention.impls.mla as mla_mod

    inner, _, wrapper = _make(quantize=quantize)
    b, q_len, ctx = 3, 2, 24
    rq_nope = mx.random.normal((b, _H, q_len, _NOPE)).astype(mx.float16)
    rq_pe = mx.random.normal((b, _H, q_len, _ROPE)).astype(mx.float16)
    all_kv_norm = mx.random.normal((b, ctx, _KVL)).astype(mx.float16)
    k_pe = mx.random.normal((b, 1, ctx, _ROPE)).astype(mx.float16)
    rows = mx.arange(q_len).reshape(-1, 1)
    causal_mask = (mx.arange(ctx) <= ctx - q_len + rows).reshape(1, 1, q_len, ctx)

    queries: list[tuple[int, ...]] = []
    real_sdpa = mla_mod.scaled_dot_product_attention

    def sdpa_spy(q, k, v, **kwargs):
        queries.append(tuple(q.shape))
        return real_sdpa(q, k, v, **kwargs)

    monkeypatch.setattr(mla_mod, "scaled_dot_product_attention", sdpa_spy)
    out = wrapper._apply_absorbed_mla_attention(
        rq_nope=rq_nope,
        rq_pe=rq_pe,
        all_kv_norm=all_kv_norm,
        k_pe=k_pe,
        causal_mask=causal_mask,
    )
    assert queries == [(b, 1, _H * q_len, _KVL)]

    pe = mx.where(
        causal_mask,
        (rq_pe * inner.scale) @ k_pe.swapaxes(-1, -2),
        mx.finfo(mx.float16).min,
    )
    kv = all_kv_norm[:, None]
    ref = inner.unembed_out(
        mx.fast.scaled_dot_product_attention(
            inner.embed_q(rq_nope), kv, kv, scale=inner.scale, mask=pe
        )
    )
    assert out.shape == ref.shape == (b, _H, q_len, _VD)
    np.testing.assert_allclose(np.array(out), np.array(ref), atol=1e-2, rtol=1e-2)
