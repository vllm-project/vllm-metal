# SPDX-License-Identifier: Apache-2.0
"""Freeze the pre-refactor default policy, independently of native helpers.

This is a planning-only differential check: it allocates no KV arrays and does
not claim numerical parity or test the compiled reducer's final safety check.
The measured geometry table and calibration values are deliberately frozen here
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


def _original_partition(geometry, lengths, cores, architecture, bound):
    """Reference contract of main@3ccf7b2, before policy extraction."""
    if geometry not in _GEOMETRIES or cores <= 0 or not lengths:
        return 0
    if min(lengths) <= 0:
        return 0
    heads, _, width, page = geometry
    enough_work = {
        p for p in (256, 512) if heads * sum(n // p for n in lengths) >= 33 * cores
    }
    if not enough_work:
        return 0
    selected = max(enough_work)
    allocation = max(bound, max(lengths))
    if architecture == "applegpu_g15g" and cores == 10 and len(lengths) > 1:
        if geometry[:3] == (32, 8, 128) and len(lengths) == 6 and max(lengths) < 704:
            return 0
        # 32 KiB minus 64 static bytes leaves room for 4,088 P256
        # statistics pairs. This is the legacy preference's conservative
        # check, not a replacement for the actual compiled-pipeline check.
        threshold = 4096 if page == 16 else 8192
        if width == 256 and max(lengths) >= threshold and allocation <= 4088 * 256:
            selected = 256

    rectangle = len(lengths) * ((allocation + selected - 1) // selected)
    actual = sum((n + selected - 1) // selected for n in lengths)
    compact = len(lengths) > 1 and actual < rectangle
    slots = actual if compact else rectangle
    offsets_bytes = 8 * (len(lengths) + 1) if compact else 0
    scratch_bytes = slots * heads * (2 * width + 8) + offsets_bytes
    return selected if scratch_bytes <= 512 * 1024**2 else 0


def _length_cases():
    # Full-partition work, short M3 veto, page-specific preference, reducer
    # capacity and scratch transitions; also rotate which row owns the maximum.
    boundaries = (255, 256, 511, 512, 703, 704, 1023, 1024, 4095, 4096, 8191, 8192)
    for n in boundaries:
        for batch in (1, 4, 6, 8):
            yield [n] * batch, 0
        yield [1, n, 255, 512], n + 512
    for bound in (1046528, 1046529, 1048576, 1048577):
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
@pytest.mark.parametrize("cores", (0, 10, 20, 40))
@pytest.mark.parametrize("architecture", ("applegpu_g15g", "applegpu_g18p", "unknown"))
def test_default_planner_matches_pre_refactor_policy(geometry, cores, architecture):
    ops = get_ops()
    heads, kv_heads, width, page = geometry
    for lengths, bound in _length_cases():
        expected = _original_partition(geometry, lengths, cores, architecture, bound)
        actual = ops.gqa_decode_batch_partition_size(
            heads,
            kv_heads,
            width,
            lengths,
            cores,
            page,
            gpu_arch=architecture,
            max_seq_len=bound,
        )
        assert actual == expected, (geometry, lengths, cores, architecture, bound)
