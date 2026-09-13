"""Pytest configuration and shared fixtures."""

from __future__ import annotations

import os
import random

import numpy as np
import pytest
import torch


def pytest_configure() -> None:
    # M5 TF32 matmul can exceed FP32 parity tolerances. Set the default before
    # collection because MLX caches this flag on its first matmul.
    os.environ.setdefault("MLX_ENABLE_TF32", "0")


def _get_test_seed() -> int:
    """Return the deterministic seed used across tests.

    Override via `VLLM_METAL_TEST_SEED` for debugging.
    """

    raw_seed = os.environ.get("VLLM_METAL_TEST_SEED", "0")
    try:
        return int(raw_seed)
    except ValueError as exc:  # pragma: no cover
        raise ValueError("VLLM_METAL_TEST_SEED must be an integer") from exc


@pytest.fixture
def force_tiled_prefill():
    """Keep prefill batches on the tiled kernel for the duration of a test.

    On an M5 the NAX kernel intercepts eligible prefill batches before
    select_tile_config, so a test that means to exercise the tiled kernel
    silently exercises NAX instead -- and because CI runners are not M5,
    nothing would catch the swap. Tests that assert tiled dispatch request this
    fixture; NAX itself is covered by tools/nax_prefill_parity.py (manual, M5).
    """

    from vllm_metal.metal import get_ops

    ops = get_ops()
    ops.set_nax_enabled(False)
    try:
        yield
    finally:
        ops.set_nax_enabled(True)


@pytest.fixture(autouse=True)
def _seed_random_generators() -> None:
    """Seed common RNGs to keep tests deterministic."""

    seed = _get_test_seed()
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    try:
        import mlx.core as mx
    except ImportError:
        return

    mlx_seed = getattr(mx.random, "seed", None)
    if mlx_seed is None:
        return
    mlx_seed(seed)
