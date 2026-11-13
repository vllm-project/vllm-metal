# SPDX-License-Identifier: Apache-2.0
"""Experimental static FP16/FP32 state scatter through TileLang and MLX."""

import functools
import importlib.util
import math

import mlx.core as mx


def _compact_strides(shape: tuple[int, ...]) -> tuple[int, ...]:
    return tuple(math.prod(shape[axis + 1 :]) for axis in range(len(shape)))


@functools.lru_cache(maxsize=128)
def _compile(shape, strides, count, dtype, zero):
    message = (
        "TileLang state scatter requires the optional TileLang MLX adapter and "
        "native runtime. Follow docs/tilelang-state-cache.md, or set "
        "VLLM_METAL_GDN_STATE_SCATTER=native."
    )
    if importlib.util.find_spec("_tilelang_mlx") is None:
        raise ImportError(message)
    try:
        from tilelang.contrib.mlx import compile as compile_mlx
        from tilelang.metal import language as T  # noqa: N812
    except ModuleNotFoundError as exc:
        if exc.name in {"tilelang", "tilelang.contrib", "tilelang.contrib.mlx"}:
            raise ImportError(message) from exc
        raise

    row_shape = shape[1:]
    inner_strides = _compact_strides(row_shape)
    width = math.prod(row_shape)
    slots = shape[0]

    def coords(col):
        return tuple(
            (col // stride) % size
            for stride, size in zip(inner_strides, row_shape, strict=True)
        )

    if zero:

        @T.prim_func
        def clear(
            pool: T.StridedTensor(shape, strides, dtype),
            ids: T.Tensor((count,), T.int32),
        ):
            with T.Kernel(T.ceildiv(width, 256), count, threads=128) as (bx, by):
                for j in T.Parallel(256):
                    col = bx * 256 + j
                    if col < width and ids[by] >= 0 and ids[by] < slots:
                        pool[(ids[by],) + coords(col)] = 0

        return compile_mlx(clear, inplace_idx=0)

    @T.prim_func
    def update(
        pool: T.StridedTensor(shape, strides, dtype),
        rows: T.Tensor((count,) + row_shape, dtype),
        ids: T.Tensor((count,), T.int32),
    ):
        with T.Kernel(T.ceildiv(width, 256), count, threads=128) as (bx, by):
            for j in T.Parallel(256):
                col = bx * 256 + j
                if col < width and ids[by] >= 0 and ids[by] < slots:
                    pool[(ids[by],) + coords(col)] = rows[(by,) + coords(col)]

    return compile_mlx(update, inplace_idx=0)


def scatter(pool, rows, ids, *, strides=None, zero=False):
    """Return a write-dependent alias without reshaping or copying the pool.

    CacheViews supplies physical strides. Without metadata, dense storage is
    required and checked by the native adapter at evaluation. Destination IDs
    must be distinct and valid, as in the upstream scatter contract.
    """
    if pool.dtype not in (mx.float16, mx.float32):
        raise ValueError("TileLang state scatter supports only float16 and float32")
    if pool.ndim < 2:
        raise ValueError("State scatter requires a pool with at least two dimensions")
    row_shape = pool.shape[1:]
    if rows.shape[1:] != row_shape or rows.dtype != pool.dtype:
        raise ValueError("State scatter source shape and dtype must match the pool")
    if ids.ndim != 1 or ids.dtype != mx.int32:
        raise ValueError("State scatter IDs must be a one-dimensional int32 array")
    if not zero and rows.shape[0] != ids.size:
        raise ValueError("State scatter requires one ID per source row")
    if ids.size == 0:
        return pool
    shape = tuple(pool.shape)
    strides = tuple(strides) if strides is not None else _compact_strides(shape)
    if len(strides) != len(shape):
        raise ValueError("State scatter requires one stride per pool dimension")
    dtype = "float16" if pool.dtype == mx.float16 else "float32"
    kernel = _compile(shape, strides, ids.size, dtype, zero)
    return kernel(pool, ids) if zero else kernel(pool, rows, ids)
