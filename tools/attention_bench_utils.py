# SPDX-License-Identifier: Apache-2.0
"""Shared helpers for attention correctness tests and benchmarks."""

from __future__ import annotations

import hashlib
import importlib.metadata
from collections.abc import Iterable
from pathlib import Path
from types import FunctionType

import mlx.core as mx
import numpy as np


def attention_tolerances(
    dtype: mx.Dtype, *, float32_tolerance: float = 1e-3
) -> tuple[float, float]:
    """Shared paged-attention atol/rtol; FP32 oracles may request tighter bounds."""
    if dtype == mx.float32:
        return float32_tolerance, float32_tolerance
    try:
        return {mx.bfloat16: (3e-2, 2e-2), mx.float16: (1.5e-2, 2e-2)}[dtype]
    except KeyError:
        raise ValueError(
            f"Unsupported attention dtype {dtype}; expected float16, bfloat16 or float32"
        ) from None


def package_versions(*names: str) -> dict[str, str | None]:
    """Record versions without requiring every distribution to be installed."""
    versions: dict[str, str | None] = {}
    for name in names:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return versions


def native_source_hashes(*functions: FunctionType) -> dict[str, str]:
    """Hash the source file behind each function, keyed by file name."""
    hashes: dict[str, str] = {}
    for function in functions:
        path = Path(function.__code__.co_filename)
        hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def source_file_hashes(root: Path, files: Iterable[Path]) -> dict[str, str]:
    """Hash source files using paths relative to the recorded source root."""
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in files
    }


def compare(
    actual: mx.array,
    expected,
    *,
    atol: float = 1e-3,
    rtol: float = 1e-3,
) -> dict[str, object]:
    """Assert a native output matches its reference within tolerance.

    ``expected`` may be an MLX array, a NumPy array, or a torch tensor
    (converted through ``detach().float().cpu().numpy()`` so this module
    does not import torch).  Shape mismatches, empty outputs and
    non-finite values are rejected before the tolerance check.  Returns
    the max absolute error and whether the outputs match bit-for-bit.
    """
    actual_np = np.array(actual.astype(mx.float32))
    if isinstance(expected, mx.array):
        expected_np = np.array(expected.astype(mx.float32))
    elif isinstance(expected, np.ndarray):
        expected_np = expected.astype(np.float32)
    else:
        expected_np = expected.detach().float().cpu().numpy()
    if actual_np.shape != expected_np.shape or not actual_np.size:
        raise ValueError(
            f"Incomplete comparison: {actual_np.shape} != {expected_np.shape}"
        )
    if not np.isfinite(actual_np).all() or not np.isfinite(expected_np).all():
        raise ValueError("Non-finite output in comparison")
    np.testing.assert_allclose(actual_np, expected_np, atol=atol, rtol=rtol)
    return {
        "max_abs_error": float(np.max(np.abs(actual_np - expected_np))),
        "exact": bool(np.array_equal(actual_np, expected_np)),
    }


def ref_paged_attn(
    query: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    query_lens: list[int],
    kv_lens: list[int],
    block_tables: np.ndarray,
    scale: float,
    sliding_window: int | None = None,
    soft_cap: float | None = None,
) -> mx.array:
    """Pure-MLX reference: gather K/V from paged cache, compute attention."""
    _, block_size, num_kv_heads, head_size = key_cache.shape

    outputs: list[mx.array] = []
    start_idx = 0
    for i, query_len in enumerate(query_lens):
        kv_len = kv_lens[i]
        q = query[start_idx : start_idx + query_len] * scale

        num_kv_blocks = (kv_len + block_size - 1) // block_size
        block_indices = mx.array(block_tables[i, :num_kv_blocks])

        k = key_cache[block_indices].reshape(-1, num_kv_heads, head_size)[:kv_len]
        v = value_cache[block_indices].reshape(-1, num_kv_heads, head_size)[:kv_len]

        if q.shape[1] != k.shape[1]:
            n_rep = q.shape[1] // k.shape[1]
            k = mx.repeat(k, n_rep, axis=1)
            v = mx.repeat(v, n_rep, axis=1)

        attn = mx.einsum("qhd,khd->hqk", q, k).astype(mx.float32)

        empty_mask = mx.ones((query_len, kv_len))
        mask = mx.triu(empty_mask, k=kv_len - query_len + 1).astype(mx.bool_)

        if sliding_window is not None:
            sliding_window_mask = mx.logical_not(
                mx.triu(empty_mask, k=kv_len - (query_len + sliding_window) + 1).astype(
                    mx.bool_
                )
            )
            mask = mx.logical_or(mask, sliding_window_mask)

        if soft_cap is not None and soft_cap > 0:
            attn = soft_cap * mx.tanh(attn / soft_cap)

        attn = mx.where(mask, float("-inf"), attn)
        attn = mx.softmax(attn, axis=-1).astype(v.dtype)
        outputs.append(mx.einsum("hqk,khd->qhd", attn, v))
        start_idx += query_len

    return mx.concatenate(outputs, axis=0)
