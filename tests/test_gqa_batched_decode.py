# SPDX-License-Identifier: Apache-2.0
"""Batched GQA planning, executed dispatch and independent numerical checks."""

import mlx.core as mx
import numpy as np
import pytest

from tests.gqa_test_utils import (
    _assert_close,
    _assert_fallback,
    _dispatch_family,
    _run_primitive,
)
from tests.test_gqa_paged_decode import GQA_GEOMETRIES, GQA_PARTITIONS
from vllm_metal.metal import get_ops


@pytest.fixture(autouse=True)
def _diagnostics_and_cores():
    ops = get_ops()
    previous = ops._set_paged_dispatch_diagnostics(True)
    ops._override_detected_gpu_core_count_for_test(40)
    try:
        yield
    finally:
        mx.synchronize()
        ops._override_detected_gpu_core_count_for_test(-1)
        ops._set_paged_dispatch_diagnostics(previous)


@pytest.mark.parametrize(
    "lengths,expected",
    [
        ([5375, 5375], 0),
        ([5376, 5376], 256),
        ([10751, 10751], 256),
        ([10752, 10752], 512),
        ([5375, 5631], 0),  # Summing tokens before flooring overcounts tails.
        ([5375, 5632], 256),
        ([1, 10751], 0),
        ([1, 10752], 256),  # max(length) * batch would wrongly promote to 512.
        ([1, 21504], 512),
        ([65536] * 9, 512),
        ([65536] * 10, 512),  # Batch size alone must not veto long GQA decode.
        ([1279] * 10, 0),
        ([1280] * 10, 256),
        ([2559] * 10, 256),
        ([2560] * 10, 512),
        ([255] * 128, 0),  # Partial tails do not become complete partitions.
        ([256] * 64, 256),
        ([512] * 64, 512),
        ([], 0),
        ([0, 21504], 0),
        ([-1, 21504], 0),
    ],
)
def test_batch_planner_golden_decisions(lengths, expected):
    assert (
        get_ops().gqa_decode_batch_partition_size(32, 8, 128, lengths, 40) == expected
    )


@pytest.mark.parametrize("cores", [0, -1])
def test_batch_planner_requires_known_cores(cores):
    assert (
        get_ops().gqa_decode_batch_partition_size(32, 8, 128, [65536] * 2, cores) == 0
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "partition,length,maximum,expected",
    [
        (256, 1536, 1046527, 256),
        (256, 1536, 1046528, 256),
        (256, 1536, 1046529, 0),
        (256, 1536, 1048576, 0),
        (256, 1536, 1048577, 0),
        (512, 4096, 2093055, 512),
        (512, 4096, 2093056, 512),
        (512, 4096, 2093057, 0),
        (512, 4096, 2097152, 0),
        (512, 4096, 2097153, 0),
    ],
)
def test_generic_admission_counts_static_reducer_memory(
    dtype, partition, length, maximum, expected
):
    ops = get_ops()
    batch, q, kv, head, block = 10, 32, 8, 128, 16
    # Apple GPUs allow 32 KiB per threadgroup. 4,088 partitions consume
    # 32,704 aligned dynamic bytes plus the ordinary reducer's 64 static bytes.
    # Also cover the former dynamic-only limit and its successor.
    # These are allocation bounds, not the actual KV lengths used for planning.
    lengths = [length] * batch
    assert ops.gqa_decode_batch_partition_size(q, kv, head, lengths, 40) == partition
    query = mx.zeros((batch, q, head), dtype)
    keys = mx.zeros((batch, block, kv, head), dtype)
    values = mx.stack(
        [mx.full((block, kv, head), (i + 1) / 16, dtype) for i in range(batch)]
    )
    tables = mx.array(
        [[i] * ((maximum + block - 1) // block) for i in range(batch)], mx.int32
    )
    reference = np.broadcast_to(
        np.arange(1, batch + 1)[:, None, None] / 16, query.shape
    )
    outputs = []
    for disabled in (True, False):
        out = mx.array(0)
        ops.paged_attention_primitive(
            query,
            keys,
            values,
            kv,
            head**-0.5,
            0.0,
            tables,
            mx.array(lengths, mx.int32),
            mx.arange(batch + 1, dtype=mx.int32),
            block,
            maximum,
            -1,
            out,
            num_decode_requests=batch,
            num_decode_tokens=batch,
            max_decode_context_len=length,
            gqa_context_lens=lengths,
            gqa_disabled=disabled,
        )
        mx.eval(out)
        selected = 0 if disabled else expected
        assert ops.last_gqa_partition_size() == selected
        assert ops.last_gqa_num_requests() == (batch if selected else 0)
        if selected:
            assert ops.last_paged_dispatch() == "gqa_decode"
        else:
            assert ops.last_paged_dispatch() == "per_token_ps0"
        actual = np.array(out.astype(mx.float32))
        np.testing.assert_allclose(actual, reference, atol=2e-3, rtol=0)
        outputs.append(actual)
    np.testing.assert_allclose(outputs[0], outputs[1], atol=2e-3, rtol=0)


@pytest.mark.parametrize(
    "cores,q,kv,head,batch,entry,promotion",
    [
        (10, 32, 8, 128, 3, 1024, 2048),
        (10, 24, 4, 256, 4, 1024, 2048),
        (10, 16, 2, 128, 5, 1280, 2560),
        (20, 32, 8, 128, 5, 1280, 2560),
        (20, 24, 4, 256, 7, 1024, 2048),
        (20, 16, 2, 256, 10, 1280, 2560),
    ],
)
def test_cross_device_batch_boundaries(cores, q, kv, head, batch, entry, promotion):
    # Fixed expectations independent of the planner, including integer rounding
    # at the first batch admitted above the former unsplit-grid veto.
    for length, expected in (
        (entry - 1, 0),
        (entry, 256),
        (promotion - 1, 256),
        (promotion, 512),
    ):
        assert (
            get_ops().gqa_decode_batch_partition_size(
                q, kv, head, [length] * batch, cores
            )
            == expected
        )


def test_reduce_offsets_cross_int32_boundary_without_large_buffers():
    import re
    from pathlib import Path

    import vllm_metal.metal

    # Compile the production reducer's offset helper, not a copied expression.
    # This exercises Metal integer promotion using only a few output elements.
    source = (
        Path(vllm_metal.metal.__file__).parent / "kernels_v2/pagedattention.metal"
    ).read_text()
    helper = re.search(r"inline int64_t paged_attention_reduce_offset\([^}]+\}", source)
    assert helper is not None
    rows = [
        (1023, 31, 32, 512, 128),
        (1024, 0, 32, 512, 128),
        (2048, 0, 32, 512, 128),
        (1024, 0, 32, 512, 1),  # FP32 statistics offset.
        (1024, 0, 32, 1, 128),  # Final output offset.
    ]
    probe = mx.fast.metal_kernel(
        name="test_paged_reduce_offsets",
        input_names=["args"],
        output_names=["offsets"],
        header=helper.group(),
        source="""
            uint i = thread_position_in_grid.x;
            offsets[i] = paged_attention_reduce_offset(
                args[5*i], args[5*i+1], args[5*i+2], args[5*i+3], args[5*i+4]);
        """,
    )
    actual = probe(
        inputs=[mx.array(rows, mx.int32)],
        grid=(len(rows), 1, 1),
        threadgroup=(len(rows), 1, 1),
        output_shapes=[(len(rows),)],
        output_dtypes=[mx.int64],
    )[0]
    assert actual.tolist() == [2147418112, 2147483648, 4294967296, 16777216, 4194304]


def test_forward_context_reorder_and_membership_survive_deferred_evaluation(
    monkeypatch,
):
    from types import SimpleNamespace

    import mlx.nn as nn

    from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
    from vllm_metal.attention.context import clear_context, get_context, prepare_grouped
    from vllm_metal.attention.impls.sdpa import sdpa_forward

    monkeypatch.setenv("VLLM_METAL_DISABLE_GQA_DECODE", "0")
    get_ops()._override_detected_gpu_core_count_for_test(10)
    q, kv, head, block = 32, 8, 128, 16
    past = [1023, 1535, 2047, 3071]
    counts = [(n + 2 + block - 1) // block for n in past]
    pages, cursor = [], 0
    for count in counts:
        pages.append(list(range(cursor, cursor + count)))
        cursor += count
    cache = MetalPagedKVCache(
        num_layers=1,
        num_kv_heads=kv,
        head_dim=head,
        num_blocks=cursor,
        block_size=block,
        dtype=mx.float16,
    )
    # Zero Q/K makes the oracle a plain mean. Distinct per-request histories
    # and large new values expose row mixups, stale lengths and missing writes.
    cache.key_caches[0] = mx.zeros_like(cache.key_caches[0])
    values = np.zeros((cursor, block, kv, head), np.float16)
    histories = []
    for rid, length in enumerate(past):
        history = [float(rid + 1 + (i % 5)) for i in range(length)]
        histories.append(history)
        for pos, value in enumerate(history):
            values[pages[rid][pos // block], pos % block] = value
    cache.value_caches[0] = mx.array(values)
    mx.eval(cache.key_caches[0], cache.value_caches[0])
    inner = SimpleNamespace(
        n_heads=q,
        n_kv_heads=kv,
        head_dim=head,
        scale=head**-0.5,
        q_proj=nn.Linear(1, q * head, bias=False),
        k_proj=nn.Linear(1, kv * head, bias=False),
        v_proj=nn.Linear(1, kv * head, bias=False),
        rope=lambda x, offset=0: x,
        o_proj=lambda x: x,
    )
    inner.q_proj.weight = mx.zeros((q * head, 1), mx.float16)
    inner.k_proj.weight = mx.zeros((kv * head, 1), mx.float16)
    inner.v_proj.weight = mx.ones((kv * head, 1), mx.float16)
    pending = []
    try:
        for step, ids in enumerate(([0, 1, 2], [2, 3, 0])):
            prepare_grouped([([pages[rid]], past[rid]) for rid in ids], [], (block,))
            ctx = get_context()
            assert ctx is not None
            new_values = [float(1000 + 100 * rid + step * 10) for rid in ids]
            output, _ = sdpa_forward(
                inner,
                mx.array(new_values, mx.float16).reshape(1, len(ids), 1),
                ctx,
                cache,
                layer_idx=0,
            )
            expected = []
            for rid, value in zip(ids, new_values, strict=True):
                histories[rid].append(value)
                expected.append(np.mean(histories[rid], dtype=np.float64))
                past[rid] += 1
            pending.append((output, expected, 256 if step == 0 else 512))
        clear_context()
        # Building the second graph/context must not change the first. Resolve
        # in reverse order, as allowed by the scatter dependency graph.
        for output, expected, partition in reversed(pending):
            mx.eval(output)
            assert get_ops().last_gqa_partition_size() == partition
            reference = np.broadcast_to(np.array(expected)[:, None], (3, q * head))
            np.testing.assert_allclose(
                np.array(output).reshape(3, -1), reference, atol=0.004, rtol=0.002
            )
    finally:
        clear_context()


@pytest.mark.parametrize("q,kv,head,block", GQA_GEOMETRIES)
@pytest.mark.parametrize("length", [1, 1536, 8192, 10752, 21248, 42496, 262145])
def test_batch_planner_preserves_single_request(q, kv, head, block, length):
    ops = get_ops()
    assert ops.gqa_decode_batch_partition_size(q, kv, head, [length], 40, block) == (
        ops.gqa_decode_partition_size(q, kv, head, length, 40, block)
    )


@pytest.mark.parametrize(
    "geometry", [(32, 4, 128, 16), (32, 8, 128, 32), (24, 4, 256, 8)]
)
def test_batch_planner_rejects_unshipped_geometry(geometry):
    q, kv, head, block = geometry
    assert (
        get_ops().gqa_decode_batch_partition_size(q, kv, head, [65536] * 2, 40, block)
        == 0
    )


@pytest.mark.parametrize("q,kv,head,block", GQA_GEOMETRIES)
@pytest.mark.parametrize("partition", GQA_PARTITIONS)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_forced_batched_kernels_with_ragged_tails(q, kv, head, block, partition, dtype):
    # Disjoint shuffled pages and different tail widths expose row/stride errors.
    out, ref = _run_primitive(
        [1, partition - 1, partition + 1, 3 * partition + 3],
        dtype,
        interleaved=True,
        seed=1004,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
        block_size=block,
        test_partition=partition,
    )
    assert _dispatch_family() == "gqa_decode"
    assert get_ops().last_gqa_partition_size() == partition
    assert get_ops().last_gqa_num_requests() == 4
    _assert_close(out, ref, dtype)


@pytest.mark.parametrize("q,kv,head,block", GQA_GEOMETRIES)
@pytest.mark.parametrize("partition", GQA_PARTITIONS)
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_automatic_batch_route_matches_reference(q, kv, head, block, partition, dtype):
    # 40 cores, two rows: 21 / 28 / 42 full partitions per row for Q32/24/16.
    per_row = {32: 21, 24: 28, 16: 42}[q]
    lengths = [partition * per_row + 1, partition * per_row + 3]
    out, ref = _run_primitive(
        lengths,
        dtype,
        interleaved=True,
        seed=1005,
        num_query_heads=q,
        num_kv_heads=kv,
        head_size=head,
        block_size=block,
        num_decode_requests=2,
        num_decode_tokens=2,
        gqa_context_lens=lengths,
    )
    assert _dispatch_family() == "gqa_decode"
    assert get_ops().last_gqa_partition_size() == partition
    assert get_ops().last_gqa_num_requests() == 2
    _assert_close(out, ref, dtype)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"gqa_disabled": True},
        {"num_decode_requests": -1},
        {"num_decode_requests": 1},  # Expanded two-row verification.
        {"num_decode_requests": 0},  # One-token prefill segments.
        {"num_decode_tokens": 0},
        {"num_decode_tokens": 3},
        {"softcap": 2.0},
        {"sliding_window": 1024},
    ],
)
def test_batch_gate_keeps_excluded_routes(kwargs):
    lengths = [10752, 11009]
    routing = {
        "num_decode_requests": 2,
        "num_decode_tokens": 2,
        "gqa_context_lens": lengths,
    }
    routing.update(kwargs)
    out, ref = _run_primitive(
        lengths, mx.float16, interleaved=True, seed=1006, **routing
    )
    _assert_fallback()
    _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize("cores,expected_partition", [(0, 0), (4, 512)])
def test_batch_uses_known_core_budget_without_unsplit_grid_veto(
    cores, expected_partition
):
    get_ops()._override_detected_gpu_core_count_for_test(cores)
    lengths = [1536, 1793]
    out, ref = _run_primitive(
        lengths,
        mx.float16,
        interleaved=True,
        seed=1007,
        num_decode_requests=2,
        num_decode_tokens=2,
        gqa_context_lens=lengths,
    )
    if expected_partition:
        assert _dispatch_family() == "gqa_decode"
        assert get_ops().last_gqa_partition_size() == expected_partition
        assert get_ops().last_gqa_num_requests() == 2
    else:
        _assert_fallback()
    _assert_close(out, ref, mx.float16)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_production_dispatch_above_legacy_split_budget(dtype):
    lengths = [3072 + 17 * i for i in range(10)]
    out, ref = _run_primitive(
        lengths,
        dtype,
        interleaved=True,
        seed=1041,
        num_query_heads=32,
        num_kv_heads=8,
        head_size=128,
        num_decode_requests=len(lengths),
        num_decode_tokens=len(lengths),
        gqa_context_lens=lengths,
    )
    assert _dispatch_family() == "gqa_decode"
    assert get_ops().last_gqa_partition_size() == 512
    assert get_ops().last_gqa_num_requests() == len(lengths)
    _assert_close(out, ref, dtype)


@pytest.mark.parametrize("lengths", [[1], [0, 1024], [-1, 1024], [1024, 1025]])
def test_invalid_host_lengths_fail_before_dispatch(lengths):
    with pytest.raises(ValueError, match="gqa_context_lens"):
        _run_primitive(
            [1024, 1024],
            mx.float16,
            interleaved=False,
            seed=1008,
            num_decode_requests=2,
            num_decode_tokens=2,
            gqa_context_lens=lengths,
        )


@pytest.mark.parametrize("preplanned", [False, True])
def test_host_lengths_are_captured_by_each_lazy_primitive(preplanned):
    ops = get_ops()
    q = mx.ones((2, 32, 128), mx.float16)
    k = mx.ones((1344, 16, 8, 128), mx.float16)
    v = mx.ones(k.shape, mx.float16)
    tables = mx.arange(1344, dtype=mx.int32).reshape(2, 672)
    lengths = mx.array([10752, 10752], mx.int32)
    cu = mx.array([0, 1, 2], mx.int32)
    host_lengths = [10752, 10752]
    routing = (
        {"gqa_length_plan": ops.gqa_decode_length_plan(host_lengths)}
        if preplanned
        else {"gqa_context_lens": host_lengths}
    )
    selected, fallback = mx.array(0), mx.array(0)
    ops.paged_attention_primitive(
        q,
        k,
        v,
        8,
        128**-0.5,
        0.0,
        tables,
        lengths,
        cu,
        16,
        10752,
        -1,
        selected,
        num_decode_requests=2,
        num_decode_tokens=2,
        **routing,
    )
    # Neither caller-side mutation nor destruction may change the saved plan.
    del routing
    host_lengths.clear()
    ops.paged_attention_primitive(
        q,
        k,
        v,
        8,
        128**-0.5,
        0.0,
        tables,
        lengths,
        cu,
        16,
        10752,
        -1,
        fallback,
        num_decode_requests=2,
        num_decode_tokens=2,
        gqa_context_lens=host_lengths,
    )
    mx.eval(fallback)
    _assert_fallback()
    mx.eval(selected)
    assert _dispatch_family() == "gqa_decode"
    assert ops.last_gqa_partition_size() == 512
    np.testing.assert_allclose(np.array(selected), np.array(fallback), atol=2e-3)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("partition", GQA_PARTITIONS)
@pytest.mark.parametrize(
    "q,kv,head,cache_block", [(24, 4, 256, 784), (16, 2, 256, 1056)]
)
def test_batched_shared_prefix_and_strided_scheduler_pages(
    dtype, partition, q, kv, head, cache_block
):
    from tests.test_gqa_paged_decode import _grouped_paged_reference
    from vllm_metal.attention.block_tables import build_block_tables

    ops = get_ops()
    length = 2 * cache_block + 3
    mx.random.seed(1009)
    logical_k = mx.random.normal((2, length, kv, head)).astype(dtype)
    logical_v = mx.random.normal(logical_k.shape).astype(dtype)
    logical_k[1, :cache_block] = logical_k[0, :cache_block]
    logical_v[1, :cache_block] = logical_v[0, :cache_block]
    # Different dominant suffix rows make a row mixup or lost lazy write visible.
    logical_k[0, -2], logical_k[1, -1] = 8, 8
    logical_v[0, -2], logical_v[1, -1] = 1, 3
    pages = [[3, 1, 4], [3, 0, 2]]
    raw = mx.zeros((5, cache_block, kv, 2 * head), dtype)
    strides = (cache_block * kv * 2 * head, kv * 2 * head, 2 * head, 1)
    shape = (5, cache_block, kv, head)
    key = ops.as_strided(raw, shape, strides, 0)
    value = ops.as_strided(raw, shape, strides, head)
    slots = [
        pages[0][i // cache_block] * cache_block + i % cache_block
        for i in range(length)
    ]
    slots += [
        pages[1][i // cache_block] * cache_block + i % cache_block
        for i in range(cache_block, length)
    ]
    key, value = ops.reshape_and_cache(
        mx.concatenate([logical_k[0], logical_k[1, cache_block:]]),
        mx.concatenate([logical_v[0], logical_v[1, cache_block:]]),
        key,
        value,
        mx.array(slots, mx.int64),
    )
    tables, block = build_block_tables(pages, cache_block)
    query = mx.ones((2, q, head), dtype)
    out = mx.array(0)
    ops._gqa_paged_attention_for_test(
        query,
        key,
        value,
        head**-0.5,
        tables,
        mx.array([length - 1, length], mx.int32),
        block,
        length,
        partition,
        out,
    )
    mx.eval(out)  # No evaluation boundary between scatter and attention.
    ref = _grouped_paged_reference(
        query=query,
        key_cache=logical_k,
        value_cache=logical_v,
        query_lens=[1, 1],
        kv_lens=[length - 1, length],
        block_tables=np.array([[0], [1]]),
        scale=head**-0.5,
    )
    _assert_close(out, ref, dtype)
    assert ops.last_gqa_num_requests() == 2


def test_recorded_batch_size_clears_with_diagnostics():
    _run_primitive(
        [513, 1025], mx.float16, interleaved=True, seed=1010, test_partition=256
    )
    ops = get_ops()
    assert ops.last_gqa_num_requests() == 2
    ops._set_paged_dispatch_diagnostics(False)
    assert ops.last_gqa_num_requests() == 0
    ops._set_paged_dispatch_diagnostics(True)
    assert ops.last_gqa_num_requests() == 0


@pytest.mark.parametrize("dtype,turboquant", [(mx.float32, False), (mx.float16, True)])
def test_unsupported_batch_cache_retains_fallback(dtype, turboquant):
    lengths = [10752, 11009]
    out, ref = _run_primitive(
        lengths,
        dtype,
        interleaved=True,
        seed=1011,
        num_decode_requests=2,
        num_decode_tokens=2,
        gqa_context_lens=lengths,
        turboquant=turboquant,
    )
    _assert_fallback()
    _assert_close(out, ref, dtype)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_ragged_scratch_budget_falls_back_before_allocation(dtype):
    ops = get_ops()
    batch, q, kv, head, block = 256, 24, 4, 256, 16
    lengths = [131072] + [1] * (batch - 1)
    padded_bytes = batch * q * 256 * (2 * head + 8)
    assert padded_bytes == 780 * 1024**2
    assert ops._gqa_decode_config_for_test()["max_scratch_bytes"] == 512 * 1024**2
    assert ops.gqa_decode_batch_partition_size(q, kv, head, lengths, 40) == 0
    query = mx.zeros((batch, q, head), dtype)
    keys = mx.zeros((1, block, kv, head), dtype)
    values = mx.full(keys.shape, 2, dtype)
    tables = mx.zeros((batch, max(lengths) // block), mx.int32)
    gpu_lengths = mx.array(lengths, mx.int32)
    cu = mx.arange(batch + 1, dtype=mx.int32)
    mx.eval(query, keys, values, tables, gpu_lengths, cu)
    mx.synchronize()
    before = mx.get_active_memory()
    mx.reset_peak_memory()
    out = mx.array(0)
    ops.paged_attention_primitive(
        query,
        keys,
        values,
        kv,
        head**-0.5,
        0.0,
        tables,
        gpu_lengths,
        cu,
        block,
        max(lengths),
        -1,
        out,
        num_decode_requests=batch,
        num_decode_tokens=batch,
        max_decode_context_len=max(lengths),
        gqa_length_plan=ops.gqa_decode_length_plan(lengths),
    )
    mx.eval(out)
    assert ops.last_paged_dispatch() == "per_token_ps0"
    assert ops.last_gqa_partition_size() == 0
    assert mx.get_peak_memory() - before < 64 * 1024**2
    np.testing.assert_allclose(np.array(out.astype(mx.float32)), 2, atol=2e-3)


def test_precomputed_length_plan_is_immutable_and_validated():
    ops = get_ops()
    plan = ops.gqa_decode_length_plan([10752, 10752])
    assert plan.num_requests == 2 and plan.max_length == 10752
    with pytest.raises(AttributeError):
        plan.max_length = 1
    query = mx.zeros((1, 32, 128), mx.float16)
    keys = mx.zeros((1, 16, 8, 128), mx.float16)
    with pytest.raises(ValueError, match="one positive length per decode row"):
        ops.paged_attention_primitive(
            query,
            keys,
            keys,
            8,
            128**-0.5,
            0.0,
            mx.zeros((1, 672), mx.int32),
            mx.array([10752], mx.int32),
            mx.array([0, 1], mx.int32),
            16,
            10752,
            -1,
            mx.array(0),
            gqa_length_plan=plan,
        )
