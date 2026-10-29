# SPDX-License-Identifier: Apache-2.0
"""Host-side NAX split/unsplit pipeline choice (vllm_metal.attention.nax_split)."""

import pytest

from vllm_metal.attention.nax_split import (
    nax_gpu_core_count,
    nax_split_hint,
    nax_split_kwargs,
    nax_tile_work,
)


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
    wide = nax_tile_work(cu, kv, sliding_window=1 << 20)
    assert wide.full_tiles == causal.full_tiles
    assert wide.partial_tiles == causal.partial_tiles


def test_length_mismatch_is_rejected():
    with pytest.raises(ValueError):
        nax_tile_work([0, 4], [4, 5])


def test_sequence_boundaries_retain_kv_loop_depth():
    # Equal aggregate work must not turn four independent one-block streams
    # into the better-amortized stream of one four-block prefill.
    single = _work([(256, 0)])
    multiple = _work([(64, 96)] * 4)
    for work in (single, multiple):
        assert (work.full_blocks, work.full_tiles, work.partial_tiles) == (4, 20, 0)
    assert single.full_stream_tiles == 5
    assert multiple.full_stream_tiles == 20
    kwargs = {"num_heads": 32, "num_kv_heads": 32, "gpu_core_count": 20}
    assert nax_split_hint(single, 96, **kwargs) == 1
    assert nax_split_hint(multiple, 96, **kwargs) == 0


def test_shared_cached_pages_do_not_count_as_independent_streams():
    cu, kv = [64 * i for i in range(6)], [576] * 5
    independent = nax_tile_work(cu, kv)
    shared = nax_tile_work(
        cu,
        kv,
        block_tables=[[*range(8), 8 + i] for i in range(5)],
        block_size=64,
    )
    assert shared.full_stream_tiles == independent.full_stream_tiles == 90
    assert shared.repeated_prefix_tiles == 64
    kwargs = {"num_heads": 4, "num_kv_heads": 4, "gpu_core_count": 4}
    assert nax_split_hint(independent, 128, **kwargs) == 0
    assert nax_split_hint(shared, 128, **kwargs) == 1


def test_prefix_reuse_counts_complete_scheduler_pages_only():
    kwargs = {"block_tables": [[5, 6], [5, 7]], "block_size": 784}
    work = nax_tile_work([0, 64, 128], [896, 896], **kwargs)
    # 832 cached tokens: only the complete 784-token page is shared.
    assert work.full_stream_tiles == 56
    assert work.repeated_prefix_tiles == 784 / 32
    windowed = nax_tile_work([0, 64, 128], [896, 896], 128, **kwargs)
    assert windowed.repeated_prefix_tiles == 0


@pytest.mark.parametrize(
    "seqs,dimension,heads,kv_heads,cores,expected",
    [
        ([(128, 8192)], 256, 24, 4, 40, 1),
        ([(129, 8192)], 256, 24, 4, 40, 0),
        ([(257, 0)], 256, 36, 9, 40, 1),
        ([(63, 8192)] * 4, 512, 8, 2, 10, 1),
        ([(64, 8192)] * 4, 64, 14, 2, 40, 1),
        ([(1, 255)] * 4 + [(128, 8192)], 96, 32, 32, 40, 0),
        ([(128, 32768)], 128, 18, 3, 40, 0),
        ([(128, 32768)], 128, 27, 1, 40, 1),
        ([(64, 8192)] * 2, 96, 32, 32, 20, 0),
        (
            [(1, 127), (1, 255), (1, 383), (128, 6144), (255, 128), (31, 768)],
            256,
            13,
            13,
            40,
            1,
        ),
    ],
    ids=[
        "single-full-blocks",
        "expensive-tail",
        "established-complete-blocks",
        "multiple-output-slices",
        "multiple-full-blocks",
        "decode-passengers",
        "underfilled-grid",
        "shared-kv-stream",
        "no-margin-independent-streams",
        "wide-groups-with-decode-passengers",
    ],
)
def test_batch_score_accounts_for_work_and_capacity(
    seqs, dimension, heads, kv_heads, cores, expected
):
    assert (
        nax_split_hint(
            _work(seqs),
            dimension,
            num_heads=heads,
            num_kv_heads=kv_heads,
            gpu_core_count=cores,
        )
        == expected
    )


def test_short_queries_do_not_form_complete_blocks_across_requests():
    work = _work([(63, 128)] * 5)
    assert work.query_tokens > 256 and work.full_blocks == 0
    assert work.full_stream_tiles == 0
    kwargs = {"num_heads": 8, "num_kv_heads": 2, "gpu_core_count": 20}
    assert nax_split_hint(work, 128, **kwargs) == 0
    assert nax_split_hint(work, 512, **kwargs) == 1
    # One-row YOCO/decode views retain their native decode route.
    assert nax_split_hint(_work([(1, 8192)] * 4), 512, **kwargs) == 0


def test_parallel_blocks_do_not_erase_guarded_kv_traffic():
    # The full grid is large, but each MHA stream has only two complete
    # blocks before a one-row tail. Admitting this batch regressed on M5.
    assert (
        nax_split_hint(
            _work([(129, 8192)] * 4),
            96,
            num_heads=32,
            num_kv_heads=32,
            gpu_core_count=10,
        )
        == 0
    )


@pytest.mark.parametrize("missing", ["num_heads", "num_kv_heads", "gpu_core_count"])
def test_missing_layout_metadata_keeps_legacy_admission(missing):
    kwargs = {"num_heads": 8, "num_kv_heads": 2, "gpu_core_count": 20}
    work = _work([(16, 0)])
    assert nax_split_hint(work, 512, **kwargs) == 1
    kwargs.pop(missing)
    assert nax_split_hint(work, 512, **kwargs) == 0
    assert nax_gpu_core_count(object()) == 0


@pytest.mark.parametrize("dimension", [64, 96, 128, 256, 512])
@pytest.mark.parametrize("query", [2047, 2048])
def test_long_prefill_keeps_the_optimized_pipeline(dimension, query):
    assert (
        nax_split_hint(
            _work([(query, 0)]),
            dimension,
            num_heads=16,
            num_kv_heads=4,
            gpu_core_count=40,
        )
        == 1
    )


def test_cached_append_hint_describes_the_potential_prefill_batch():
    # Native may execute this whole batch or split its decode prefix under
    # either native decode-work gate. Python must provide both correct scores.
    whole = _work([(1, 131071), (64, 8192), (64, 8192)])
    prefill = _work([(64, 8192), (64, 8192)])
    caps = {"nax_split_hint": True, "nax_prefill_split_hint": True}
    kwargs = {"gpu_core_count": 20, "num_heads": 24, "num_kv_heads": 4}
    assert nax_split_kwargs(caps, whole, 256, prefill_work=prefill, **kwargs) == {
        "nax_split_hint": 0,
        "nax_prefill_split_hint": 1,
    }
    assert nax_split_kwargs(
        {"nax_split_hint": True}, whole, 256, prefill_work=prefill, **kwargs
    ) == {"nax_split_hint": 0}


@pytest.mark.parametrize("head_size", [128, 512])
def test_native_hint_does_not_change_outputs(head_size):
    import mlx.core as mx

    from tests.test_gqa_decode_routing import _mixed_call, _mixed_inputs
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
    from tests.test_gqa_decode_routing import _mixed_call, _mixed_inputs

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
    import mlx.core as mx

    from tests.gqa_test_utils import _primitive
    from tests.test_gqa_decode_routing import _mixed_call, _mixed_inputs

    ops = nax_prefill
    # D512's cached-append extension must preserve both output slices on a
    # partial Q block after the native dispatcher removes the decode prefix.
    prefill_rows = 63 if head_size == 512 else 256
    inputs = _mixed_inputs(
        [decode_kv], prefill_kv=256, geometry=(8, 2, head_size, 16), dtype=mx.float16
    )
    inputs.update(
        query=mx.random.normal((decode_rows + prefill_rows, 8, head_size)).astype(
            mx.float16
        ),
        cu_seqlens_q=mx.array([0, decode_rows, decode_rows + prefill_rows], mx.int32),
        num_decode_tokens=decode_rows,
        nax_split_hint=0,
    )
    if decode_rows > 1:
        inputs.update(
            cu_seqlens_q=mx.array(
                [*range(decode_rows + 1), decode_rows + prefill_rows], mx.int32
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
    if separated:
        assert mx.array_equal(actual[:1], decode).item()
        assert mx.array_equal(actual[1:], whole[1:]).item()
        # Omitted and explicit -1 hints retain the whole-batch choice.
        for defaults in ({}, {"nax_prefill_split_hint": -1}):
            fallback = _mixed_call(inputs, **defaults)
            assert ops.last_paged_dispatch() == "mixed_nax_prefill_decode"
            assert ops.last_nax_split_pipeline() == 0
            assert mx.array_equal(fallback, actual).item()
    else:
        assert mx.array_equal(actual, whole).item()
