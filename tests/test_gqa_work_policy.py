# SPDX-License-Identifier: Apache-2.0
"""Work-budget boundaries, resource checks and executed lazy batches."""

import mlx.core as mx
import numpy as np
import pytest

from tests.gqa_test_utils import _primitive
from vllm_metal.metal import get_ops


@pytest.fixture(autouse=True)
def _native_diagnostics():
    ops = get_ops()
    previous = ops._set_paged_dispatch_diagnostics(True)
    ops._override_detected_gpu_core_count_for_test(10)
    try:
        yield
    finally:
        mx.synchronize()
        ops._override_detected_gpu_core_count_for_test(-1)
        ops._set_paged_dispatch_diagnostics(previous)


@pytest.mark.parametrize(
    "geometry,lengths,expected",
    [
        ((24, 4, 256, 16), [1023] * 4, 0),
        ((24, 4, 256, 16), [1024] * 4, 256),
        ((24, 4, 256, 16), [2048] * 4, 512),
        ((24, 4, 256, 16), [4095] * 4, 512),
        # Former M3 calibration boundaries now use the generic work rule.
        ((24, 4, 256, 16), [4096] * 4, 512),
        ((24, 4, 256, 16), [8192] * 8, 512),
        ((24, 4, 256, 16), [1] * 63 + [131072], 512),
        ((24, 4, 256, 16), [1] * 127 + [131072], 512),
        (
            (24, 4, 256, 16),
            [131072] * 128,
            512,
        ),  # The original P512-first selection fits the scratch budget.
        ((24, 4, 256, 16), [131072] * 256, 0),  # P512 also exceeds scratch.
        ((24, 4, 256, 16), [1, 255, 256, 8192], 512),
        ((16, 2, 256, 16), [4095] * 5, 512),
        ((16, 2, 256, 16), [4096] * 5, 512),
        ((16, 2, 256, 32), [1280] * 5, 256),
        ((16, 2, 256, 32), [2559] * 5, 256),
        ((16, 2, 256, 32), [2560] * 5, 512),
        ((16, 2, 256, 32), [1535] * 10, 256),
        ((16, 2, 256, 32), [1536] * 10, 512),
        ((16, 2, 256, 32), [8191] * 5, 512),
        ((16, 2, 256, 32), [8192] * 5, 512),
        ((16, 2, 256, 32), [8192] * 10, 512),
        # Complete partition work determines short-context admission.
        ((32, 8, 128, 16), [512] * 5, 0),
        ((32, 8, 128, 16), [512] * 6, 256),
        ((32, 8, 128, 16), [639] * 6, 256),
        ((32, 8, 128, 16), [640] * 6, 256),
        ((32, 8, 128, 16), [703] * 6, 256),
        ((32, 8, 128, 16), [704] * 6, 256),
        ((32, 8, 128, 16), [767] * 6, 256),
        ((32, 8, 128, 16), [512] * 7, 256),
        ((32, 8, 128, 16), [512] * 8, 256),
        ((32, 8, 128, 16), [512] * 5 + [256], 256),
        ((32, 8, 128, 16), [1024] * 3, 256),
        ((32, 8, 128, 16), [2048] * 3, 512),
        ((16, 2, 128, 16), [1280] * 5, 256),
        ((16, 2, 128, 16), [8192] * 5, 512),
        # Functional admission and malformed-length handling are unchanged.
        ((16, 4, 256, 16), [8192] * 4, 0),
        # The page size does not introduce a separate performance preference.
        ((24, 4, 256, 32), [8192] * 4, 512),
        ((24, 4, 256, 16), [], 0),
        ((24, 4, 256, 16), [0, 8192], 0),
        ((24, 4, 256, 16), [-1, 8192], 0),
    ],
)
def test_batch_planner_boundaries(geometry, lengths, expected):
    q, kv, head, block = geometry
    assert (
        get_ops().gqa_decode_batch_partition_size(q, kv, head, lengths, 10, block)
        == expected
    )


@pytest.mark.parametrize("cores", [7, 9, 10, 11, 13, 20, 40])
def test_work_budget_scales_with_core_count(cores):
    # A single row has ceil(33 * cores / 32) full partitions at the
    # transition. Head128 retains P512 promotion when it gains enough work.
    required = (33 * cores + 31) // 32
    ops = get_ops()
    for partition, expected_before in ((256, 0), (512, 256)):
        boundary = required * partition
        assert (
            ops.gqa_decode_batch_partition_size(32, 8, 128, [boundary - 1], cores, 16)
            == expected_before
        )
        assert (
            ops.gqa_decode_batch_partition_size(32, 8, 128, [boundary], cores, 16)
            == partition
        )


@pytest.mark.parametrize("cores", [0, -1])
def test_unknown_core_count_falls_back(cores):
    assert (
        get_ops().gqa_decode_batch_partition_size(24, 4, 256, [8192] * 4, cores, 16)
        == 0
    )


@pytest.mark.parametrize(
    "geometry", [(24, 4, 256, 16), (16, 2, 256, 16), (16, 2, 256, 32)]
)
def test_single_and_batch_planners_agree(geometry):
    q, kv, head, block = geometry
    ops = get_ops()
    assert (
        ops.gqa_decode_batch_partition_size(q, kv, head, [16384], 10, block)
        == ops.gqa_decode_partition_size(q, kv, head, 16384, 10, block)
        == 512
    )


@pytest.mark.parametrize(
    "maximum,expected",
    [(1046528, 512), (1046529, 512), (1048576, 512), (1048577, 512)],
)
def test_compact_plan_keeps_p512_across_former_preference_boundary(maximum, expected):
    ops = get_ops()
    # Actual lengths determine compact scratch; a larger dispatch bound
    # neither changes the work budget nor creates a device-specific preference.
    assert (
        ops.gqa_decode_batch_partition_size(
            24, 4, 256, [8192] * 4, 10, 16, max_seq_len=maximum
        )
        == expected
    )


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "geometry,length,batch,expected_partition",
    [
        ((24, 4, 256, 16), 8192, 4, 512),
        ((16, 2, 256, 16), 8192, 5, 512),
        ((16, 2, 256, 32), 8192, 4, 512),
        ((16, 2, 256, 32), 1280, 5, 256),
        ((32, 8, 128, 16), 639, 6, 256),
        ((32, 8, 128, 16), 640, 6, 256),
        ((32, 8, 128, 16), 704, 6, 256),
        ((32, 8, 128, 16), 512, 7, 256),
    ],
)
def test_default_planner_matches_executed_batch_and_constant_oracle(
    dtype, geometry, length, batch, expected_partition
):
    q, kv, head, block = geometry
    query = mx.zeros((batch, q, head), dtype)
    keys = mx.zeros((batch, block, kv, head), dtype)
    values = mx.stack([mx.full((block, kv, head), i + 1, dtype) for i in range(batch)])
    tables = mx.array(
        [[i] * ((length + block - 1) // block) for i in range(batch)], mx.int32
    )
    lengths = [length] * batch
    ops = get_ops()
    assert (
        ops.gqa_decode_batch_partition_size(q, kv, head, lengths, 10, block)
        == expected_partition
    )
    out = _primitive(
        query, keys, values, tables, mx.array(lengths, mx.int32), block, length, lengths
    )
    mx.eval(out)
    if expected_partition:
        assert ops.last_paged_dispatch() == "gqa_decode"
    else:
        assert ops.last_paged_dispatch().startswith("per_token_")
    assert ops.last_gqa_partition_size() == expected_partition
    assert ops.last_gqa_num_requests() == (batch if expected_partition else 0)
    expected = np.broadcast_to(np.arange(1, batch + 1)[:, None, None], out.shape)
    np.testing.assert_allclose(
        np.array(out.astype(mx.float32)), expected, atol=2e-3, rtol=0
    )


def test_length_transition_and_reordered_rows_survive_lazy_evaluation():
    ops = get_ops()
    q, kv, head, block = 24, 4, 256, 16
    keys = mx.zeros((5, block, kv, head), mx.float16)
    value_np = np.ones((5, block, kv, head), np.float16)
    for i in range(4):
        value_np[i + 1, -1] += 64 * (i + 1)
    values = mx.array(value_np)
    table_np = np.zeros((4, 256), np.int32)
    table_np[:, -1] = np.arange(1, 5)
    host_lengths = [4095] * 4
    first = _primitive(
        mx.zeros((4, q, head), mx.float16),
        keys,
        values,
        mx.array(table_np),
        mx.array(host_lengths, mx.int32),
        block,
        4095,
        host_lengths,
    )
    host_lengths[:] = [4096] * 4
    second = _primitive(
        mx.zeros((4, q, head), mx.float16),
        keys,
        values,
        mx.array(table_np),
        mx.array(host_lengths, mx.int32),
        block,
        4096,
        host_lengths,
    )
    order = [2, 0, 3]
    host_lengths[:] = [4096, 4095, 4096]
    third = _primitive(
        mx.zeros((3, q, head), mx.float16),
        keys,
        values,
        mx.array(table_np[order]),
        mx.array(host_lengths, mx.int32),
        block,
        4096,
        host_lengths,
    )
    host_lengths[:] = [1]  # Neither the saved plan nor its GPU lengths may drift.
    for out, partition, expected in [
        (first, 512, [1.0] * 4),
        (second, 512, [1 + (i + 1) / 64 for i in range(4)]),
        (third, 512, [1 + 3 / 64, 1.0, 1 + 4 / 64]),
    ]:
        mx.eval(out)
        assert ops.last_gqa_partition_size() == partition
        assert ops.last_gqa_num_requests() == len(expected)
        reference = np.broadcast_to(np.array(expected)[:, None, None], out.shape)
        np.testing.assert_allclose(np.array(out), reference, atol=2e-3, rtol=0)


def test_short_lengths_survive_deferred_decode():
    ops = get_ops()
    batch, q, kv, head, block = 6, 32, 8, 128, 16
    keys = mx.zeros((batch, block, kv, head), mx.float16)
    value_np = np.ones((batch, block, kv, head), np.float16)
    for row in range(batch):
        value_np[row] *= row + 1
        value_np[row, -1] += 64
    values = mx.array(value_np)
    tables = mx.array([[row] * 44 for row in range(batch)], mx.int32)
    pending = []
    for length in (703, 704):
        out = _primitive(
            mx.zeros((batch, q, head), mx.float16),
            keys,
            values,
            tables,
            mx.array([length] * batch, mx.int32),
            block,
            length,
            [length] * batch,
        )
        pending.append((length, out))
    for length, out in reversed(pending):
        mx.eval(out)
        assert ops.last_gqa_partition_size() == 256
        expected = np.arange(1, batch + 1) + 64 * (length // block) / length
        reference = np.broadcast_to(expected[:, None, None], out.shape)
        np.testing.assert_allclose(np.array(out), reference, atol=0.004, rtol=0.001)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize(
    "maximum,forced_partition",
    [(1046528, 256), (1046529, 512), (1048576, 512), (1048577, 512)],
)
def test_native_selection_keeps_resource_feasible_partition(
    dtype, maximum, forced_partition
):
    ops = get_ops()
    batch, q, kv, head, block = 4, 24, 4, 256, 16
    query = mx.ones((batch, q, head), dtype)
    keys = mx.ones((1, block, kv, head), dtype)
    values = mx.full(keys.shape, 2, dtype)
    tables = mx.zeros((batch, (maximum + block - 1) // block), mx.int32)
    lengths = [8192] * batch
    out = _primitive(
        query,
        keys,
        values,
        tables,
        mx.array(lengths, mx.int32),
        block,
        maximum,
        lengths,
    )
    mx.eval(out)
    assert ops.last_paged_dispatch() == "gqa_decode"
    assert ops.last_gqa_partition_size() == 512
    np.testing.assert_allclose(np.array(out.astype(mx.float32)), 2, atol=2e-3, rtol=0)
    # Also exercise P256 at the conservative 32-KiB boundary. Only
    # occupancy is forced; actual pipeline resource checks remain.
    selected = ops._gqa_paged_attention_for_test(
        query,
        keys,
        values,
        head**-0.5,
        tables,
        mx.array(lengths, mx.int32),
        block,
        maximum,
        forced_partition,
    )
    mx.eval(selected)
    assert ops.last_gqa_partition_size() == forced_partition
    np.testing.assert_allclose(
        np.array(selected.astype(mx.float32)), 2, atol=2e-3, rtol=0
    )
