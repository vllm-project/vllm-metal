# SPDX-License-Identifier: Apache-2.0
"""Host-side NAX split/unsplit pipeline choice (vllm_metal.attention.nax_split)."""

import mlx.core as mx
import numpy as np
import pytest

from tests.gqa_test_utils import _primitive
from tests.test_gqa_decode_routing import _mixed_call, _mixed_inputs
from vllm_metal.attention.nax_split import (
    nax_split_hint,
    nax_split_kwargs,
    nax_tile_work,
)


@pytest.fixture(params=[(144, 3, 0), (1328, 6, 1)], ids=["h3", "h6"])
def grouped_prefill_inputs(nax_prefill, request):
    """Exercise both native widths with shuffled, interleaved 784-token pages."""
    import os

    if os.environ.get("VLLM_METAL_NAX_GQA_PREFILL", "1") != "1":
        pytest.skip("Grouped layout coverage requires GQA prefill enabled")
    ops = nax_prefill
    query_tokens, width, hint = request.param
    prefix = 31360
    # Fix host planning to cover both widths on every NAX-capable test host.
    # This override is not a measurement of a 40-core GPU.
    ops._override_detected_gpu_core_count_for_test(40)
    try:
        inputs = _mixed_inputs(
            [1],
            prefill_kv=prefix + query_tokens,
            geometry=(24, 4, 256, 16),
            cache_block=784,
        )
        inputs.update(
            query=mx.random.normal((query_tokens, 24, 256)).astype(mx.bfloat16),
            block_tables=inputs["block_tables"][-1:],
            seq_lens=mx.array([prefix + query_tokens], mx.int32),
            cu_seqlens_q=mx.array([0, query_tokens], mx.int32),
            num_decode_requests=0,
            num_decode_tokens=0,
            max_decode_context_len=0,
        )
        yield ops, inputs, width, hint
    finally:
        mx.synchronize()
        ops._override_detected_gpu_core_count_for_test(-1)


def test_nax_grouped_translated_pages_match_compact_cache(grouped_prefill_inputs):
    """Each selected width must execute and preserve logical-page attention."""
    ops, inputs, width, hint = grouped_prefill_inputs
    assert inputs["key_cache"].shape[1] == 784 and inputs["block_size"] == 16
    translated = _mixed_call(inputs, nax_split_hint=hint)
    assert ops.last_paged_dispatch() == "nax_gqa6_prefill"
    assert ops.last_nax_heads_per_threadgroup() == width

    # Independent contiguous kernel-page storage removes physical-page and
    # interleaved-head strides while preserving logical tokens/block IDs.
    compact = inputs | {
        "key_cache": mx.contiguous(inputs["key_cache"]).reshape(-1, 16, 4, 256),
        "value_cache": mx.contiguous(inputs["value_cache"]).reshape(-1, 16, 4, 256),
    }
    direct = _mixed_call(compact, nax_split_hint=hint)
    assert ops.last_paged_dispatch() == "nax_gqa6_prefill"
    assert ops.last_nax_heads_per_threadgroup() == width
    assert mx.array_equal(translated, direct)

    ops._override_detected_gpu_core_count_for_test(0)
    # A fresh input forces a new encode after changing the host override.
    compact["query"] = mx.array(
        np.array(inputs["query"].astype(mx.float32)), mx.bfloat16
    )
    upstream = _mixed_call(compact, nax_split_hint=hint)
    assert ops.last_paged_dispatch() == "nax_prefill"
    assert ops.last_nax_heads_per_threadgroup() == 1
    assert mx.all(mx.isfinite(translated)) and mx.array_equal(translated, upstream)
    ops._set_paged_dispatch_diagnostics(False)
    assert ops.last_nax_heads_per_threadgroup() == 0
    ops._set_paged_dispatch_diagnostics(True)


def _work(seqs):
    """seqs: (query tokens, cached context tokens) per sequence."""
    cu, kv = [0], []
    for q, context in seqs:
        cu.append(cu[-1] + q)
        kv.append(q + context)
    return nax_tile_work(cu, kv)


# Query-block and work-ratio boundaries: (batch, D128 hint, D256 hint).
_CASES = [
    ([(63, 8129)], 0, 0),  # short prefill
    ([(64, 8128)], 0, 0),  # one full block
    ([(193, 7999)], 0, 0),  # three full blocks + residual
    ([(257, 7935)], 0, 1),  # D256 admits four full blocks + residual
    ([(513, 7679)], 0, 1),  # D128 full-tile work is still below 8x partial
    ([(1025, 7167)], 1, 1),  # D128 crosses the work ratio
    ([(2048, 6144), (1, 8191), (1, 8191), (1, 8191)], 1, 1),  # 2048 + decode
    ([(1024, 7168), (63, 32705)], 0, 1),  # weight the long partial history
]


@pytest.mark.parametrize(("seqs", "d128", "d256"), _CASES)
def test_pipeline_choice_respects_full_and_partial_work(seqs, d128, d256):
    work = _work(seqs)
    assert nax_split_hint(work, 128) == d128
    assert nax_split_hint(work, 64) == d128
    assert nax_split_hint(work, 256) == d256
    assert nax_split_hint(work, 512) == d128


def test_tile_counts_follow_the_causal_mask():
    work = _work([(129, 64)])
    # Two full blocks end at KV 128 and 192 (4 and 6 tiles); the residual
    # row sees all 193 tokens (7 tiles).
    assert (work.full_blocks, work.full_tiles, work.partial_tiles) == (2, 10, 7)


def test_split_kwargs_defaults_to_the_original_loop():
    caps = {"nax_split_hint": True}
    work = _work([(2048, 6144)])
    assert nax_split_kwargs(caps, work, 128) == {"nax_split_hint": 1}
    assert nax_split_kwargs(caps, None, 128) == {"nax_split_hint": 0}


def test_separated_decode_work_changes_the_prefill_hint():
    cu, kv = [0, 1, 257], [131072, 8192]
    whole = nax_tile_work(cu, kv)
    prefill = nax_tile_work(cu[1:], kv[1:])
    assert (whole.full_blocks, whole.full_tiles, whole.partial_tiles) == (4, 1012, 4096)
    assert (prefill.full_blocks, prefill.full_tiles, prefill.partial_tiles) == (
        4,
        1012,
        0,
    )
    assert nax_split_hint(whole, 256) == 0
    assert nax_split_hint(prefill, 256) == 1


@pytest.mark.parametrize(
    "caps,expected",
    [
        ({}, {}),
        ({"nax_split_hint": True}, {"nax_split_hint": 0}),
        (
            {"nax_split_hint": True, "nax_prefill_split_hint": True},
            {"nax_split_hint": 0, "nax_prefill_split_hint": 1},
        ),
    ],
    ids=["legacy", "whole-batch-only", "prefill-subset"],
)
def test_prefill_hint_keeps_whole_batch_and_old_build_compatibility(caps, expected):
    whole = nax_tile_work([0, 1, 257], [131072, 8192])
    prefill = nax_tile_work([1, 257], [8192])
    assert nax_split_kwargs(caps, whole, 256, prefill_work=prefill) == expected
    if caps.get("nax_split_hint"):
        assert nax_split_kwargs(caps, whole, 256) == {"nax_split_hint": 0}


def test_sliding_window_counts_only_tiles_inside_the_window():
    # The kernel skips KV tiles left of each block's window, so a 128-token
    # window leaves the 256-row prefill only ~6 tiles per full block: the
    # causal-prefix count would split, the windowed count must not.
    seqs = [(256, 131072)] + [(63, 0)] * 9
    cu, kv = [0], []
    for q, context in seqs:
        cu.append(cu[-1] + q)
        kv.append(q + context)
    causal = nax_tile_work(cu, kv)
    windowed = nax_tile_work(cu, kv, sliding_window=128)
    assert causal.full_blocks == windowed.full_blocks == 4
    assert windowed.full_tiles == 4 * 6 and windowed.partial_tiles == 9 * 2
    assert nax_split_hint(causal, 128) == 1
    assert nax_split_hint(windowed, 128) == 0
    # A window wider than the context changes nothing.
    assert nax_tile_work(cu, kv, sliding_window=1 << 20) == causal


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        nax_tile_work([0, 4], [4, 5])


@pytest.mark.parametrize("head_size", [128, 512])
def test_native_hint_does_not_change_outputs(head_size):
    from vllm_metal.metal import get_ops

    ops = get_ops()
    inputs = _mixed_inputs(
        [1], prefill_kv=257, geometry=(8, 2, head_size, 16), dtype=mx.float16
    )
    inputs.update(
        query=mx.random.normal((257, 8, head_size)).astype(mx.float16),
        cu_seqlens_q=mx.array([0, 1, 257], mx.int32),
    )
    nax = ops.nax_supported() and ops.nax_ready()
    previous = ops._set_paged_dispatch_diagnostics(True)
    outs, pipelines = [], []
    try:
        for hint in (-1, 0, 1):
            outs.append(_mixed_call(inputs, nax_split_hint=hint))
            pipelines.append(ops.last_nax_split_pipeline())
    finally:
        ops._set_paged_dispatch_diagnostics(previous)
    assert mx.array_equal(outs[0], outs[1]) and mx.array_equal(outs[0], outs[2])
    # Legacy callers retain the D512 fallback without per-sequence metadata.
    expected = [int(head_size < 512), 0, 1] if nax else [-1, -1, -1]
    assert pipelines == expected


@pytest.mark.parametrize("name", ["nax_split_hint", "nax_prefill_split_hint"])
def test_native_rejects_invalid_hints_before_evaluation(name):
    """Public argument validation also runs on machines without NAX."""
    inputs = _mixed_inputs([2], prefill_kv=2, geometry=(8, 2, 128, 16))
    for invalid in (-2, 2):
        with pytest.raises(ValueError, match=name):
            _mixed_call(inputs, **{name: invalid})


@pytest.mark.parametrize(
    "decode_kv,decode_rows,metadata,separated,head_size",
    [
        (4096, 1, True, True, 256),
        (128, 1, True, False, 256),
        (4096, 1, False, False, 256),
        (4096, 3, True, False, 256),
        (4096, 1, True, True, 512),
    ],
    ids=[
        "ordinary-split",
        "short-decode",
        "missing-metadata",
        "expanded-verify",
        "d512-output-slices",
    ],
)
def test_native_prefill_hint_only_applies_after_decode_separation(
    nax_prefill, decode_kv, decode_rows, metadata, separated, head_size
):
    """Conflicting hints expose the native subset choice and preserve row ownership."""
    ops = nax_prefill
    inputs = _mixed_inputs(
        [decode_kv], prefill_kv=256, geometry=(8, 2, head_size, 16), dtype=mx.float16
    )
    inputs.update(
        query=mx.random.normal((decode_rows + 256, 8, head_size)).astype(mx.float16),
        cu_seqlens_q=mx.array([0, decode_rows, decode_rows + 256], mx.int32),
        num_decode_tokens=decode_rows,
        nax_split_hint=0,
    )
    if decode_rows > 1:
        inputs.update(
            cu_seqlens_q=mx.array(
                [*range(decode_rows + 1), decode_rows + 256], mx.int32
            ),
            block_tables=inputs["block_tables"][mx.array([0] * decode_rows + [1])],
            seq_lens=mx.array(
                [*range(decode_kv - decode_rows + 1, decode_kv + 1), 256], mx.int32
            ),
        )
    if not metadata:
        for name in (
            "num_decode_requests",
            "num_decode_tokens",
            "max_decode_context_len",
        ):
            inputs.pop(name)
    whole = _mixed_call(
        inputs, num_decode_requests=0, num_decode_tokens=0, max_decode_context_len=0
    )
    assert ops.last_paged_dispatch() == "nax_prefill"
    assert ops.last_nax_split_pipeline() == 0
    decode = None
    if separated:
        decode = _primitive(
            inputs["query"][:1],
            inputs["key_cache"],
            inputs["value_cache"],
            inputs["block_tables"][:1],
            inputs["seq_lens"][:1],
            inputs["block_size"],
            decode_kv,
            [decode_kv],
            gqa_disabled=True,
        )
        mx.eval(decode)
    actual = _mixed_call(inputs, nax_prefill_split_hint=1)
    assert ops.last_paged_dispatch() == (
        "mixed_nax_prefill_decode" if separated else "nax_prefill"
    )
    assert ops.last_nax_split_pipeline() == int(separated)
    assert ops.last_nax_heads_per_threadgroup() == 1
    if separated:
        assert mx.array_equal(actual[:1], decode).item()
        assert mx.array_equal(actual[1:], whole[1:]).item()
        # Omitted and explicit -1 hints retain the whole-batch choice.
        for defaults in ({}, {"nax_prefill_split_hint": -1}):
            fallback = _mixed_call(inputs, **defaults)
            assert ops.last_paged_dispatch() == "mixed_nax_prefill_decode"
            assert ops.last_nax_split_pipeline() == 0
            assert ops.last_nax_heads_per_threadgroup() == 1
            assert mx.array_equal(fallback, actual).item()
    else:
        assert mx.array_equal(actual, whole).item()


def test_nax_trace_reports_executed_layout(grouped_prefill_inputs):
    """Trace fields describe the encoded layout without changing its output."""
    ops, inputs, width, hint = grouped_prefill_inputs
    query_tokens = inputs["query"].shape[0]
    ops._take_nax_prefill_dispatch_trace()
    plain = _mixed_call(inputs, nax_split_hint=hint)
    inputs["query"] = mx.array(
        np.array(inputs["query"].astype(mx.float32)), dtype=mx.bfloat16
    )
    try:
        ops._start_nax_prefill_dispatch_trace()
        observed = _mixed_call(inputs, nax_split_hint=hint)
        trace = ops._take_nax_prefill_dispatch_trace()
        assert mx.array_equal(plain, observed)
        assert not trace["overflow"] and trace["dropped_events"] == 0
        assert trace["dispatch_count"] == len(trace["events"]) == 1
        event = trace["events"][0]
        suffix = "_gqa6_sub_full" if width == 3 else "_gqa6_full"
        assert event["function"].endswith(suffix)
        assert event["HEADS_PER_TG"] == width and event["threads"] == width * 32
        assert (event["gridx"], event["gridy"], event["BQ"]) == (
            24 // width,
            query_tokens // 16 + 1,
            16,
        )
        assert event["max_seq_len"] == 31360 + query_tokens
        assert event["q_block_offset"] == 0 and event["key_page_tokens"] == 784
        assert event["key_block_stride"] == 16 * event["key_stride1"]
        assert event["key_head_stride"] == event["key_stride2"] == 512
    finally:
        ops._take_nax_prefill_dispatch_trace()


def test_nax_trace_overflow_restart_and_drain(nax_prefill):
    """Bound the real API, reset active traces, and clear every drained counter."""
    ops = nax_prefill
    empty = ops._take_nax_prefill_dispatch_trace()
    capacity = empty["capacity"]
    assert capacity == 8192
    inputs = _mixed_inputs([1], prefill_kv=2, geometry=(8, 2, 128, 16))

    def encode(index):
        # Each fresh lazy node must encode; re-evaluating one output would not.
        inputs["query"] = mx.array(
            np.full((3, 8, 128), (index % 7) / 8, dtype=np.float32), mx.bfloat16
        )
        _mixed_call(inputs, nax_split_hint=0)

    def assert_empty(trace):
        assert trace["capacity"] == capacity
        assert (
            trace["events"],
            trace["overflow"],
            trace["dropped_events"],
            trace["dispatch_count"],
            trace["append_time_ns"],
        ) == ([], False, 0, 0, 0)

    try:
        ops._start_nax_prefill_dispatch_trace()
        encode(0)
        ops._start_nax_prefill_dispatch_trace()
        ops._start_nax_prefill_dispatch_trace()
        assert_empty(ops._take_nax_prefill_dispatch_trace())

        ops._start_nax_prefill_dispatch_trace()
        for index in range(capacity + 3):
            encode(index)
        trace = ops._take_nax_prefill_dispatch_trace()
        assert len(trace["events"]) == capacity
        assert trace["dispatch_count"] == capacity + 3
        assert trace["overflow"] and trace["dropped_events"] == 3
        assert [event["event_index"] for event in trace["events"]] == list(
            range(capacity)
        )
        assert_empty(ops._take_nax_prefill_dispatch_trace())
        encode(1)
        assert_empty(ops._take_nax_prefill_dispatch_trace())

        ops._start_nax_prefill_dispatch_trace()
        encode(2)
        trace = ops._take_nax_prefill_dispatch_trace()
        assert trace["dispatch_count"] == len(trace["events"]) == 1
        assert not trace["overflow"] and trace["dropped_events"] == 0
        assert_empty(ops._take_nax_prefill_dispatch_trace())
    finally:
        ops._take_nax_prefill_dispatch_trace()
