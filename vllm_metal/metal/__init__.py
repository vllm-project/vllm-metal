# SPDX-License-Identifier: Apache-2.0
"""Native paged-attention Metal kernels dispatched through MLX.

Usage::

    from vllm_metal.metal import get_ops
    ops = get_ops()
    ops.reshape_and_cache(key, value, key_cache, value_cache, slot_mapping)
    ops.paged_attention_v1(out, query, key_cache, value_cache, ...)
"""

from __future__ import annotations

import importlib
import importlib.util
import logging
import re
from pathlib import Path
from types import ModuleType

from vllm_metal.metal.constants import PARTITION_SIZE, PARTITION_THRESHOLD

logger = logging.getLogger(__name__)

_THIS_DIR = Path(__file__).resolve().parent
_KERNELS_DIR = _THIS_DIR / "kernels_v1"
_KERNELS_V2_DIR = _THIS_DIR / "kernels_v2"

# Cached after first get_ops() call.  The Metal shaders are JIT-compiled once
# and held in MLX's library cache for the lifetime of the process.  Editing
# .metal source files requires restarting the Python interpreter to pick up
# changes (the .cpp extension itself is rebuilt automatically by build.py when
# paged_ops.cpp is newer than the .so).
_ops_module: ModuleType | None = None


def _read_metal_source(path: Path) -> str:
    """Read a .metal file and strip local #include directives."""
    text = path.read_text()
    # Remove #include "..." for our vendored files (keep <metal_stdlib> etc.)
    text = re.sub(r'#include\s+"[^"]*"', "", text)
    return text


def _build_reshape_cache_source() -> str:
    """Concatenate float8 + utils + reshape_and_cache into a single source."""
    parts = [
        _read_metal_source(_KERNELS_DIR / "float8.metal"),
        _read_metal_source(_KERNELS_DIR / "utils.metal"),
        _read_metal_source(_KERNELS_DIR / "reshape_and_cache.metal"),
    ]
    return "\n".join(parts)


def _build_paged_attention_source() -> str:
    """Concatenate float8 + utils + paged_attention into a single source."""
    parts = [
        f"#define VLLM_METAL_PARTITION_SIZE {PARTITION_SIZE}",
        _read_metal_source(_KERNELS_DIR / "float8.metal"),
        _read_metal_source(_KERNELS_DIR / "utils.metal"),
        _read_metal_source(_KERNELS_DIR / "pagedattention.metal"),
    ]
    return "\n".join(parts)


def _build_v2_paged_attention_source() -> str:
    """Concatenate float8 + utils + v2 paged_attention (online softmax)."""
    parts = [
        f"#define VLLM_METAL_PARTITION_SIZE {PARTITION_SIZE}",
        _read_metal_source(_KERNELS_V2_DIR / "float8.metal"),
        _read_metal_source(_KERNELS_V2_DIR / "utils.metal"),
        _read_metal_source(_KERNELS_V2_DIR / "pagedattention.metal"),
    ]
    return "\n".join(parts)


def metal_unified_attention(
    q,  # [total_q_tokens, num_q_heads, head_size]
    k,  # [num_blocks, block_size, num_kv_heads, head_size]
    v,  # [num_blocks, block_size, num_kv_heads, head_size]
    out,  # [total_q_tokens, num_q_heads, head_size]
    cu_seqlens_q,  # [num_seqs + 1], int32
    seqused_k,  # [num_seqs], int32
    max_seqlen_q: int,
    max_seqlen_k: int,
    softmax_scale: float,
    causal: bool,
    window_size: tuple[int, int],
    block_table,  # [num_seqs, max_blocks_per_seq], int32
    softcap: float,
) -> None:
    """Unified varlen paged attention for Metal.

    Supports variable-length queries (prefill + decode) with online softmax,
    paged KV cache, causal masking, sliding window, and soft capping.

    Grid: one threadgroup per (head, query_token). Each threadgroup uses
    binary search on cu_seqlens_q to find its sequence and computes causal
    attention against the paged KV cache.
    """
    assert causal, "Only causal attention is supported"
    import mlx.core as mx

    # Extract dimensions from cache shape
    # k shape: [num_blocks, block_size, num_kv_heads, head_size]
    num_kv_heads = k.shape[2]
    block_size = k.shape[1]

    # Convert window_size tuple to a single sliding_window int.
    # window_size = (left, right) where left = sw-1, right = 0 for causal.
    # sliding_window = left + 1 = total window size. -1 = disabled.
    if window_size == (-1, -1):
        sliding_window = -1
    else:
        sliding_window = window_size[0] + 1

    ops = get_ops()

    # Ensure all inputs are evaluated before raw Metal dispatch
    mx.eval(out, q, k, v, block_table, seqused_k, cu_seqlens_q)
    max_num_partitions = max(1, (max_seqlen_k + PARTITION_SIZE - 1) // PARTITION_SIZE)
    use_partitioning = (
        PARTITION_SIZE % block_size == 0
        and max_seqlen_q == 1
        and max_seqlen_k >= PARTITION_THRESHOLD
        and max_num_partitions > 1
    )

    if use_partitioning:
        exp_sums = mx.zeros(
            (q.shape[0], q.shape[1], max_num_partitions), dtype=mx.float32
        )
        max_logits = mx.zeros(
            (q.shape[0], q.shape[1], max_num_partitions), dtype=mx.float32
        )
        tmp_out = mx.zeros(
            (q.shape[0], q.shape[1], max_num_partitions, q.shape[2]),
            dtype=q.dtype,
        )
        mx.eval(exp_sums, max_logits, tmp_out)
        ops.paged_attention_v2_online_partitioned(
            out,
            q,
            k,
            v,
            num_kv_heads,
            softmax_scale,
            softcap,
            block_table,
            seqused_k,
            cu_seqlens_q,
            block_size,
            max_seqlen_k,
            sliding_window,
            exp_sums,
            max_logits,
            tmp_out,
        )
        mx.synchronize()
    else:
        ops.paged_attention_v2_online(
            out,
            q,
            k,
            v,
            num_kv_heads,
            softmax_scale,
            softcap,
            block_table,
            seqused_k,
            cu_seqlens_q,
            block_size,
            max_seqlen_k,
            sliding_window,
        )
        mx.synchronize()


def get_ops() -> ModuleType:
    """JIT-build and import the native paged_ops extension.

    The Metal shader sources are read, pre-processed (includes inlined),
    and passed to the C++ extension which JIT-compiles them via
    ``mlx::core::metal::Device::get_library()``.

    Returns:
        The ``_paged_ops`` module with ``reshape_and_cache()`` and
        ``paged_attention_v1()``.
    """
    global _ops_module
    if _ops_module is not None:
        return _ops_module

    # 1. JIT-build the C++ extension if needed
    from vllm_metal.metal.build import build

    so_path = build()

    # 2. Import the built extension
    spec = importlib.util.spec_from_file_location("_paged_ops", str(so_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load extension from {so_path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)

    # 3. Initialise Metal libraries (JIT-compile shaders)
    reshape_src = _build_reshape_cache_source()
    paged_attn_src = _build_paged_attention_source()
    mod.init_libraries(reshape_src, paged_attn_src)

    # 4. Initialise v2 library (online softmax kernel)
    v2_src = _build_v2_paged_attention_source()
    mod.init_v2_library(v2_src)

    _ops_module = mod
    logger.info("Native paged-attention Metal kernels loaded")
    return mod
