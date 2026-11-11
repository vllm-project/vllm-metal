# SPDX-License-Identifier: Apache-2.0
"""Check the device-independent default policy against an independent oracle.

This is a planning-only differential check: it allocates no KV arrays and does
not claim numerical parity or test the compiled reducer's final safety check.
The measured geometry table and work budget are deliberately frozen here
rather than read from the implementation being checked.
"""

import random

import pytest

from vllm_metal.metal import get_ops

_GEOMETRIES = (
    (32, 8, 128, 16),
    (24, 8, 128, 16),
    (24, 4, 256, 16),
    (24, 4, 256, 32),
    (16, 2, 128, 16),
    (16, 2, 256, 16),
    (16, 2, 256, 32),
)


def _generic_partition(geometry, lengths, cores, bound):
    """Stage-one generic work rule and scratch check, without legacy M3 policy."""
    if geometry not in _GEOMETRIES or cores <= 0 or not lengths:
        return 0
    if min(lengths) <= 0:
        return 0
    heads, _, width, _ = geometry
    enough_work = {
        p for p in (256, 512) if heads * sum(n // p for n in lengths) >= 33 * cores
    }
    if not enough_work:
        return 0
    selected = max(enough_work)
    allocation = max(bound, max(lengths))
    rectangle = len(lengths) * ((allocation + selected - 1) // selected)
    actual = sum((n + selected - 1) // selected for n in lengths)
    compact = len(lengths) > 1 and actual < rectangle
    slots = actual if compact else rectangle
    offsets_bytes = 8 * (len(lengths) + 1) if compact else 0
    scratch_bytes = slots * heads * (2 * width + 8) + offsets_bytes
    return selected if scratch_bytes <= 512 * 1024**2 else 0


def _length_cases():
    # Full-partition work, former calibration boundaries, allocation
    # capacity and scratch transitions; also rotate which row owns the maximum.
    boundaries = (255, 256, 511, 512, 703, 704, 1023, 1024, 4095, 4096, 8191, 8192)
    for n in boundaries:
        for batch in (1, 4, 6, 8):
            yield [n] * batch, 0
        yield [1, n, 255, 512], n + 512
    for bound in (65536, 65537, 131072, 131073, 1046528, 1046529, 1048576, 1048577):
        yield [8192] * 4, bound
    for batch in (63, 127, 128, 256):
        yield [131072] * batch, 0
        yield [1] * (batch - 1) + [131072], 0
    yield [], 0
    yield [0, 8192], 0
    yield [8192, -1], 0
    rng = random.Random(991)
    for _ in range(40):
        lengths = [
            rng.randint(1, 131072) for _ in range(rng.choice((1, 2, 6, 17, 128)))
        ]
        yield lengths, rng.choice((0, max(lengths), max(lengths) + 65536))


@pytest.mark.parametrize("geometry", (*_GEOMETRIES, (16, 4, 256, 16), (32, 8, 128, 32)))
@pytest.mark.parametrize("cores", (-1, 0, 7, 9, 10, 11, 20, 40))
def test_default_planner_matches_generic_policy(geometry, cores):
    ops = get_ops()
    heads, kv_heads, width, page = geometry
    for lengths, bound in _length_cases():
        expected = _generic_partition(geometry, lengths, cores, bound)
        actual = ops.gqa_decode_batch_partition_size(
            heads,
            kv_heads,
            width,
            lengths,
            cores,
            page,
            max_seq_len=bound,
        )
        assert actual == expected, (geometry, lengths, cores, bound)


@pytest.mark.parametrize("geometry", [(32, 8, 128, 16), (24, 4, 256, 16)])
@pytest.mark.parametrize("bound", [65536, 65537, 131072, 131073])
def test_allocation_bound_does_not_change_work_selected_partition(geometry, bound):
    heads, kv, width, page = geometry
    assert (
        get_ops().gqa_decode_batch_partition_size(
            heads, kv, width, [8192] * 4, 10, page, max_seq_len=bound
        )
        == 512
    )


def test_allocation_bound_does_not_supply_missing_work():
    # P256 has 384 complete groups; P512 has 192, below the 330-group budget.
    assert (
        get_ops().gqa_decode_batch_partition_size(
            32, 8, 128, [512] * 6, 10, 16, max_seq_len=65537
        )
        == 256
    )


@pytest.mark.parametrize("bound,expected", [(16268800, 256), (16268801, 0)])
def test_work_selected_p256_scratch_limit_does_not_promote_to_p512(bound, expected):
    # Actual 10,752-token work admits only P256 at 40 cores. A single row
    # uses rectangular scratch: 63,550 P256 slots fit, one more does not.
    # P512 would fit the allocation budget but still lacks complete work.
    assert (
        get_ops().gqa_decode_batch_partition_size(
            32, 8, 128, [10752], 40, 16, max_seq_len=bound
        )
        == expected
    )


def test_work_selected_p512_scratch_failure_falls_back():
    # Work chooses P512 first; exceeding its scratch budget returns fallback,
    # rather than running a separate resource-driven partition search.
    assert (
        get_ops().gqa_decode_batch_partition_size(24, 4, 256, [131072] * 256, 10, 16)
        == 0
    )
