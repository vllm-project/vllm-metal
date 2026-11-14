# SPDX-License-Identifier: Apache-2.0
"""Kernel-layer dispatch for in-place GDN state rows."""

import functools

from vllm_metal.metal import get_ops


@functools.cache
def _native_kernel():
    return get_ops().gdn_state_scatter


def _native_scatter(pool, rows, ids, *, strides=None, zero=False):
    # The native primitive reads strides directly from the MLX array.
    return _native_kernel()(pool, rows, ids, zero=zero)


# Benchmark policy, not a serving option. Small-row submission overhead and
# variable serving results do not yet justify promoting TileLang. Tests and
# controlled benchmarks can exercise the candidate at this kernel boundary.
_implementation = _native_scatter


def scatter(pool, rows, ids, *, strides=None, zero=False):
    """Return a write-dependent alias; callers supply physical layout metadata."""
    return _implementation(pool, rows, ids, strides=strides, zero=zero)
