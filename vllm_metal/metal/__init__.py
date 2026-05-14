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


def _read_v2_metal_source(filename: str) -> str:
    """Read a kernels_v2 .metal source file."""
    return _read_metal_source(_KERNELS_V2_DIR / filename)


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
    """Concatenate float8 + utils + turboquant + v2 paged_attention (online softmax)."""
    parts = [
        f"#define VLLM_METAL_PARTITION_SIZE {PARTITION_SIZE}",
        _read_metal_source(_KERNELS_V2_DIR / "float8.metal"),
        _read_metal_source(_KERNELS_V2_DIR / "utils.metal"),
        _read_metal_source(_KERNELS_V2_DIR / "turboquant.metal"),
        _read_metal_source(_KERNELS_V2_DIR / "pagedattention.metal"),
    ]
    return "\n".join(parts)


def _build_gdn_source() -> str:
    """GDN linear attention kernel source."""
    parts = [
        _read_metal_source(_KERNELS_V2_DIR / "utils.metal"),
        _read_metal_source(_KERNELS_V2_DIR / "gdn_linear_attention.metal"),
    ]
    return "\n".join(parts)


def _build_mla_paged_attention_source() -> str:
    """Concatenate utils + mla into a single source for the MLA library."""
    parts = [
        _read_metal_source(_KERNELS_V2_DIR / "utils.metal"),
        _read_metal_source(_KERNELS_V2_DIR / "mla.metal"),
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


def metal_mla_paged_attention(
    q_nope,  # [total_q_tokens, num_heads, kv_lora_rank]
    q_pe,  # [total_q_tokens, num_heads, qk_rope_head_dim]
    latent_cache,  # [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
    out,  # [total_q_tokens, num_heads, kv_lora_rank]
    block_tables,  # [num_seqs, max_blocks_per_seq], int32
    context_lens,  # [num_seqs], uint32
    cu_seqlens_q,  # [num_seqs + 1], int32
    scale: float,
    heads_per_tg: int = 1,
) -> None:
    """Paged Multi-head Latent Attention (RFC #360).

    Phase 1 step 8 (multi-block decode): the kernel iterates the per-sequence
    block table with NUM_WARPS-strided online softmax and a cross-warp merge,
    so any ``ctx_len`` that fits into the allocated block_tables row is fine.
    Decode-only is still required (one query token per sequence) — varlen
    prefill lands in P2.

    Q is expected to be already projected through ``embed_q`` (so q_nope is
    in kv_lora_rank space) and ``q_pe`` is RoPE-applied. The caller is
    responsible for ``unembed_out`` on the result to recover v_head_dim.

    ``heads_per_tg`` (G) controls cross-head KV amortization: each
    threadgroup processes ``G`` consecutive query heads sharing the same
    latent KV. Total dispatched threadgroups drop from ``B×H`` to
    ``B×ceil(H/G)``, so total KV bandwidth is amortized G×. ``num_heads``
    must be divisible by G. Currently instantiated for G ∈ {1, 4}; G=1 is
    the existing single-head-per-TG kernel.
    """
    import mlx.core as mx

    # Shape contract — fail fast at the Python boundary so the C++ error
    # path isn't the only line of defence.
    if q_nope.shape[2] != latent_cache.shape[2] - q_pe.shape[2]:
        raise ValueError(
            f"MLA shape mismatch: q_nope.shape[2]={q_nope.shape[2]} must equal "
            f"latent_cache.shape[2] ({latent_cache.shape[2]}) - "
            f"q_pe.shape[2] ({q_pe.shape[2]})"
        )

    block_size = latent_cache.shape[1]

    # Materialise inputs before the C++ in-place dispatch. The eager
    # binding reads input buffer pointers directly (no MLX primitive
    # graph), so we need real Metal buffers backing each array here.
    # The lazy primitive variant ``metal_mla_paged_attention_primitive``
    # does not need this — MLX's primitive system materialises inputs
    # as part of graph evaluation.
    mx.eval(out, q_nope, q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q)

    ops = get_ops()
    ops.mla_paged_attention(
        out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_size,
        scale,
        heads_per_tg,
    )
    mx.synchronize()


def metal_mla_paged_attention_primitive(
    q_nope,  # [total_q_tokens, num_heads, kv_lora_rank]
    q_pe,  # [total_q_tokens, num_heads, qk_rope_head_dim]
    latent_cache,  # [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
    block_tables,  # [num_seqs, max_blocks_per_seq], int32
    context_lens,  # [num_seqs], uint32
    cu_seqlens_q,  # [num_seqs + 1], int32
    scale: float,
    heads_per_tg: int = 1,
):
    """Primitive variant of :func:`metal_mla_paged_attention` — returns
    a lazy ``mx.array`` whose evaluation triggers the kernel dispatch.

    The call participates in MLX's lazy graph instead of forcing an
    ``mx.eval`` boundary inside this entry. That saves ~200 μs at small
    workloads (B=1, H≤64) where MLX dispatch overhead is a meaningful
    fraction of total wrapper time.
    """
    import mlx.core as mx

    if q_nope.shape[2] != latent_cache.shape[2] - q_pe.shape[2]:
        raise ValueError(
            f"MLA shape mismatch: q_nope.shape[2]={q_nope.shape[2]} must equal "
            f"latent_cache.shape[2] ({latent_cache.shape[2]}) - "
            f"q_pe.shape[2] ({q_pe.shape[2]})"
        )

    block_size = latent_cache.shape[1]

    total_q_tokens = int(q_nope.shape[0])
    num_heads = int(q_nope.shape[1])
    kv_lora_rank = int(q_nope.shape[2])
    # ``mx.zeros`` here is lazy — the C++ side replaces ``out``'s
    # descriptor with the Primitive output before the zeros ever
    # evaluate, so the memset is never scheduled.
    out = mx.zeros((total_q_tokens, num_heads, kv_lora_rank), dtype=q_nope.dtype)

    ops = get_ops()
    ops.mla_paged_attention_primitive(
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_size,
        scale,
        heads_per_tg,
        out,
    )
    return out


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

    # 5. Initialise GDN linear attention library
    gdn_src = _build_gdn_source()
    mod.init_gdn_library(gdn_src)

    # 6. Initialise MLA paged-attention library (RFC #360)
    mla_src = _build_mla_paged_attention_source()
    mod.init_mla_library(mla_src)

    _ops_module = mod
    logger.info("Native paged-attention Metal kernels loaded")
    return mod
