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


@pytest.mark.parametrize(
    "query,expected",
    [
        (65, (0, 0, 0)),
        (128, (1, 1, 0)),
        (129, (0, 0, 0)),
        (192, (1, 1, 1)),
        (193, (1, 1, 0)),
    ],
)
def test_cached_d256_full_and_partial_work_scales_with_capacity(query, expected):
    work = _work([(query, 8192)])
    for cores, hint in zip((10, 20, 40), expected, strict=True):
        assert nax_split_hint(work, 256, gpu_core_count=cores, num_heads=24) == hint


def test_d256_parallelism_accepts_head_and_core_counts_without_profiles():
    # The complete-block grid crosses capacity between 32 and 33 cores.
    full = _work([(128, 8192)])
    assert nax_split_hint(full, 256, gpu_core_count=32, num_heads=24) == 1
    assert nax_split_hint(full, 256, gpu_core_count=33, num_heads=24) == 0
    # More query heads can also amortize the same partial-block work.
    tail = _work([(193, 8192)])
    assert nax_split_hint(tail, 256, gpu_core_count=20, num_heads=18) == 0
    assert nax_split_hint(tail, 256, gpu_core_count=20, num_heads=19) == 1


def test_d512_short_append_uses_tile_work_without_head_or_device_profiles():
    # The same short prefix boundary applies across arbitrary equal-capacity
    # grids. It follows tile work, without an 8K prefix gate.
    below, enough = _work([(63, 417)]), _work([(63, 418)])
    for heads in (1, 7, 37):
        kwargs = {"num_heads": heads, "gpu_core_count": heads}
        assert nax_split_hint(below, 512, **kwargs) == 0
        assert nax_split_hint(enough, 512, **kwargs) == 1
        assert (
            nax_split_hint(enough, 512, num_heads=heads, gpu_core_count=heads - 1) == 0
        )
    assert nax_split_hint(enough, 512, num_heads=7) == 0
    # Missing capacity retains the established rule, including for full blocks.
    assert nax_split_hint(_work([(64, 8192)]), 512, num_heads=37) == 0
    assert nax_split_hint(_work([(1, 32768)]), 512, num_heads=7, gpu_core_count=20) == 0


def test_append_metadata_gates_preserve_legacy_admission():
    kwargs = {"gpu_core_count": 20, "num_heads": 24}
    assert nax_split_hint(_work([(128, 2048)]), 256, **kwargs) == 1
    assert nax_split_hint(_work([(128, 8192)]), 128, **kwargs) == 0
    assert nax_split_hint(_work([(128, 8192)]), 256, num_heads=24) == 0
    assert nax_split_hint(_work([(63, 32768)]), 512) == 0
    assert nax_split_hint(_work([(2048, 8192)]), 256) == 1
    windowed = nax_tile_work([0, 63], [32768 + 63], sliding_window=32768)
    assert nax_split_hint(windowed, 512, **kwargs) == 0
    assert nax_gpu_core_count(object()) == 0


@pytest.mark.parametrize(
    "head_size,query,prefix,expected",
    [
        (64, 256, 0, 1),
        (96, 256, 0, 1),
        (128, 513, 8192, 0),
        (256, 128, 512, 1),
        (512, 257, 0, 1),
        (512, 63, 512, 1),
    ],
)
def test_full_and_cached_prefill_use_the_same_work_score(
    head_size, query, prefix, expected
):
    # Preserve full-tile admission, reject the measured D128 guarded-tail
    # regression, and admit profitable short workloads without a prefix floor.
    assert (
        nax_split_hint(
            _work([(query, prefix)]), head_size, num_heads=16, gpu_core_count=20
        )
        == expected
    )


@pytest.mark.parametrize(
    "head_size,heads,query,prefix",
    [(64, 14, 256, 2048), (256, 36, 257, 0), (256, 10, 257, 1536)],
)
def test_complete_blocks_keep_established_admission(head_size, heads, query, prefix):
    assert (
        nax_split_hint(
            _work([(query, prefix)]), head_size, num_heads=heads, gpu_core_count=40
        )
        == 1
    )


def test_cached_append_hint_describes_only_the_native_prefill_subset():
    whole = nax_tile_work([0, 1, 64], [32769, 32831])
    prefill = nax_tile_work([1, 64], [32831])
    caps = {"nax_split_hint": True, "nax_prefill_split_hint": True}
    kwargs = {"gpu_core_count": 20, "num_heads": 8}
    assert nax_split_kwargs(caps, whole, 512, prefill_work=prefill, **kwargs) == {
        "nax_split_hint": 0,
        "nax_prefill_split_hint": 1,
    }
    assert nax_split_kwargs(
        {"nax_split_hint": True}, whole, 512, prefill_work=prefill, **kwargs
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
