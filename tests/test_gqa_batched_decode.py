# SPDX-License-Identifier: Apache-2.0
"""Batched GQA planning, executed dispatch and independent numerical checks."""

import mlx.core as mx
import numpy as np
import pytest

from tests.test_gqa_decode_routing import _run_primitive
from tests.test_gqa_paged_decode import (
    GQA_GEOMETRIES,
    GQA_PARTITIONS,
    _assert_close,
    _assert_fallback,
    _dispatch_family,
)
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
        ([65536] * 10, 0),  # The unsplit grid already fills the split budget.
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


@pytest.mark.parametrize("cores", [0, 4])
def test_unknown_or_saturated_batch_uses_established_path(cores):
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
    _assert_fallback()
    _assert_close(out, ref, mx.float16)


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


def test_host_lengths_are_captured_by_each_lazy_primitive():
    ops = get_ops()
    q = mx.ones((2, 32, 128), mx.float16)
    k = mx.ones((1344, 16, 8, 128), mx.float16)
    v = mx.ones(k.shape, mx.float16)
    tables = mx.arange(1344, dtype=mx.int32).reshape(2, 672)
    lengths = mx.array([10752, 10752], mx.int32)
    cu = mx.array([0, 1, 2], mx.int32)
    host_lengths = [10752, 10752]
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
        gqa_context_lens=host_lengths,
    )
    # Later caller-side mutation must not alter the pending primitive's plan.
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
    from vllm_metal.attention.impls.sdpa import _build_block_tables

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
    tables, block = _build_block_tables(pages, cache_block)
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
