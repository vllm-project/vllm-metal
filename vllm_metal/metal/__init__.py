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

# The MLA paged-attention shader set is loaded lazily on the first MLA
# entrypoint call (see _ensure_mla_library). Initialising it inside
# get_ops() would compile experimental MLA Metal source for every
# paged-attention user, so non-MLA models pay the compile cost and a
# compile error in MLA would block unrelated paged-attention paths.
_mla_library_initialized: bool = False


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
    import numpy as np

    # Shape contract — fail fast at the Python boundary so the C++ error
    # path isn't the only line of defence.
    if q_nope.shape[2] != latent_cache.shape[2] - q_pe.shape[2]:
        raise ValueError(
            f"MLA shape mismatch: q_nope.shape[2]={q_nope.shape[2]} must equal "
            f"latent_cache.shape[2] ({latent_cache.shape[2]}) - "
            f"q_pe.shape[2] ({q_pe.shape[2]})"
        )

    block_size = latent_cache.shape[1]

    mx.eval(out, q_nope, q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q)

    # P1 decode-only guard: q_token_idx == seq_idx is hard-wired in the
    # kernel. Reading cu_seqlens_q back to host is cheap (tens of int32 per
    # layer per step) and the guard goes away when P2 makes the kernel
    # cu_seqlens_q-aware via find_seq_idx.
    cu_q = np.asarray(cu_seqlens_q)
    deltas = np.diff(cu_q)
    if np.any(deltas != 1):
        bad = int(np.argmax(deltas != 1))
        raise NotImplementedError(
            "MLA kernel (P1) supports decode only — one query token per "
            f"sequence. Got request {bad} with {int(deltas[bad])} query "
            "tokens. Multi-token prefill / varlen support lands in P2."
        )

    # block_tables row-width guard. The kernel walks block_table_row[0..
    # num_context_blocks-1] for each sequence; if the caller-allocated row
    # is too narrow we'd silently read into the next sequence's row or off
    # the end of the buffer. Caller-side capacity bug (ValueError, not
    # NotImplementedError — this isn't a feature gap).
    ctx = np.asarray(context_lens)
    max_blocks_per_seq = int(block_tables.shape[1])
    required_blocks = (ctx + block_size - 1) // block_size
    if np.any(required_blocks > max_blocks_per_seq):
        bad = int(np.argmax(required_blocks > max_blocks_per_seq))
        raise ValueError(
            f"MLA: block_tables row width ({max_blocks_per_seq}) too small "
            f"for request {bad}: ctx_len={int(ctx[bad])} requires "
            f"{int(required_blocks[bad])} blocks at block_size={block_size}."
        )

    ops = get_ops()
    _ensure_mla_library(ops)
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


# Hard-coded for now: matches the only instantiated reduce kernel
# (see kernels_v2/mla.metal). If we add more PARTITION_SIZE specializations
# later, this becomes a parameter.
MLA_PARTITION_SIZE = 512


def metal_mla_paged_attention_partitioned(
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
    """Paged MLA with split-K + reduce (RFC #360 Phase 3 — for long contexts /
    low-batch decode where the single-pass kernel under-fills the GPU).

    Same shape contract and decode-only constraints as
    ``metal_mla_paged_attention``. Internally allocates per-partition
    ``exp_sums`` / ``max_logits`` / ``tmp_out`` scratch and dispatches the
    main kernel with ``PARTITION_SIZE=512`` followed by the reduce kernel.

    Caller is responsible for the partitioning routing decision (e.g., based
    on max ctx_len and total threadgroup count); this function unconditionally
    runs the partitioned path.
    """
    import mlx.core as mx
    import numpy as np

    # Shape contract — identical to the non-partitioned entry.
    if q_nope.shape[2] != latent_cache.shape[2] - q_pe.shape[2]:
        raise ValueError(
            f"MLA shape mismatch: q_nope.shape[2]={q_nope.shape[2]} must equal "
            f"latent_cache.shape[2] ({latent_cache.shape[2]}) - "
            f"q_pe.shape[2] ({q_pe.shape[2]})"
        )

    block_size = latent_cache.shape[1]
    if MLA_PARTITION_SIZE % block_size != 0:
        raise ValueError(
            f"MLA partitioned: PARTITION_SIZE ({MLA_PARTITION_SIZE}) must be "
            f"divisible by block_size ({block_size})."
        )

    mx.eval(out, q_nope, q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q)

    cu_q = np.asarray(cu_seqlens_q)
    deltas = np.diff(cu_q)
    if np.any(deltas != 1):
        bad = int(np.argmax(deltas != 1))
        raise NotImplementedError(
            "MLA partitioned kernel (P1) supports decode only — one query "
            f"token per sequence. Got request {bad} with {int(deltas[bad])} "
            "query tokens."
        )

    ctx = np.asarray(context_lens)
    max_blocks_per_seq = int(block_tables.shape[1])
    required_blocks = (ctx + block_size - 1) // block_size
    if np.any(required_blocks > max_blocks_per_seq):
        bad = int(np.argmax(required_blocks > max_blocks_per_seq))
        raise ValueError(
            f"MLA partitioned: block_tables row width ({max_blocks_per_seq}) "
            f"too small for request {bad}: ctx_len={int(ctx[bad])} requires "
            f"{int(required_blocks[bad])} blocks at block_size={block_size}."
        )

    total_q_tokens = int(q_nope.shape[0])
    num_heads = int(q_nope.shape[1])
    kv_lora_rank = int(q_nope.shape[2])
    max_ctx = int(ctx.max())
    max_num_partitions = max(
        1, (max_ctx + MLA_PARTITION_SIZE - 1) // MLA_PARTITION_SIZE
    )

    # Scratch buffers. Zero-initialized so partitions that return early (no
    # blocks to process for their seq) leave a no-op contribution.
    exp_sums = mx.zeros(
        (total_q_tokens, num_heads, max_num_partitions), dtype=mx.float32
    )
    max_logits = mx.zeros(
        (total_q_tokens, num_heads, max_num_partitions), dtype=mx.float32
    )
    tmp_out = mx.zeros(
        (total_q_tokens, num_heads, max_num_partitions, kv_lora_rank),
        dtype=q_nope.dtype,
    )
    mx.eval(exp_sums, max_logits, tmp_out)

    ops = get_ops()
    _ensure_mla_library(ops)
    ops.mla_paged_attention_partitioned(
        out,
        exp_sums,
        max_logits,
        tmp_out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        cu_seqlens_q,
        block_size,
        scale,
        MLA_PARTITION_SIZE,
        max_num_partitions,
        heads_per_tg,
    )
    mx.synchronize()


# Decode-2pass kernel partition sizes (mirrored in mla.metal as
# instantiate_mla_2pass + the matching reduce specializations).
# 256 is kept as a bench knob between 128 and 512; auto-pick still
# returns {64, 128}.
_MLA_DECODE_2PASS_SIZES = (64, 128, 256, 512)


def _pick_mla_decode_2pass_partition(max_ctx: int) -> int:
    """Pick PARTITION_SIZE for the 2pass decode kernel.

    Smaller partition → more TGs → better GPU fill on small ctx. Larger
    partition → less reduce overhead at long ctx. Mirrors MLX's choice in
    `mlx/backend/metal/scaled_dot_product_attention.cpp::sdpa_vector_2pass`
    (devc=='s' branch): 64 by default, 128 once ctx > 1024."""
    if max_ctx <= 1024:
        return 64
    return 128


def metal_mla_paged_attention_decode_2pass(
    q_nope,  # [total_q_tokens, num_heads, kv_lora_rank]
    q_pe,  # [total_q_tokens, num_heads, qk_rope_head_dim]
    latent_cache,  # [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
    out,  # [total_q_tokens, num_heads, kv_lora_rank]
    block_tables,  # [num_seqs, max_blocks_per_seq], int32
    context_lens,  # [num_seqs], uint32
    cu_seqlens_q,  # [num_seqs + 1], int32  (only used for decode-only validation)
    scale: float,
    partition_size: int | None = None,
) -> None:
    """MLX sdpa_vector_2pass-style cross-head amortization for absorbed MLA
    decode (RFC #360, follow-up to the G-batched single-pass kernel).

    Each TG handles one (seq, ctx-partition) pair with 32*num_heads threads
    arranged as 32-lane × num_heads-head simdgroups. All heads in the TG
    read the same K cache tokens — the L1/L2 cache serves the H-1 repeats
    so total KV bandwidth is amortized H× across the whole launch.

    Same shape contract and decode-only constraints as
    `metal_mla_paged_attention`. Internally allocates `exp_sums` /
    `max_logits` / `tmp_out` scratch and dispatches the main kernel
    followed by the existing reduce kernel.

    `partition_size` defaults to a heuristic based on max ctx_len; pass an
    explicit value to override.
    """
    import mlx.core as mx
    import numpy as np

    if q_nope.shape[2] != latent_cache.shape[2] - q_pe.shape[2]:
        raise ValueError(
            f"MLA shape mismatch: q_nope.shape[2]={q_nope.shape[2]} must equal "
            f"latent_cache.shape[2] ({latent_cache.shape[2]}) - "
            f"q_pe.shape[2] ({q_pe.shape[2]})"
        )

    block_size = latent_cache.shape[1]

    mx.eval(out, q_nope, q_pe, latent_cache, block_tables, context_lens, cu_seqlens_q)

    cu_q = np.asarray(cu_seqlens_q)
    deltas = np.diff(cu_q)
    if np.any(deltas != 1):
        bad = int(np.argmax(deltas != 1))
        raise NotImplementedError(
            "MLA decode-2pass kernel supports decode only — one query token "
            f"per sequence. Got request {bad} with {int(deltas[bad])} query "
            "tokens."
        )

    ctx = np.asarray(context_lens)
    max_blocks_per_seq = int(block_tables.shape[1])
    required_blocks = (ctx + block_size - 1) // block_size
    if np.any(required_blocks > max_blocks_per_seq):
        bad = int(np.argmax(required_blocks > max_blocks_per_seq))
        raise ValueError(
            f"MLA decode-2pass: block_tables row width ({max_blocks_per_seq}) "
            f"too small for request {bad}: ctx_len={int(ctx[bad])} requires "
            f"{int(required_blocks[bad])} blocks at block_size={block_size}."
        )

    total_q_tokens = int(q_nope.shape[0])
    num_heads = int(q_nope.shape[1])
    kv_lora_rank = int(q_nope.shape[2])
    max_ctx = int(ctx.max())

    if partition_size is None:
        partition_size = _pick_mla_decode_2pass_partition(max_ctx)
    if partition_size not in _MLA_DECODE_2PASS_SIZES:
        raise ValueError(
            f"MLA decode-2pass: partition_size must be in "
            f"{_MLA_DECODE_2PASS_SIZES}; got {partition_size}"
        )
    if partition_size % block_size != 0:
        raise ValueError(
            f"MLA decode-2pass: partition_size ({partition_size}) must be "
            f"divisible by block_size ({block_size})."
        )

    max_num_partitions = max(1, (max_ctx + partition_size - 1) // partition_size)

    exp_sums = mx.zeros(
        (total_q_tokens, num_heads, max_num_partitions), dtype=mx.float32
    )
    max_logits = mx.zeros(
        (total_q_tokens, num_heads, max_num_partitions), dtype=mx.float32
    )
    tmp_out = mx.zeros(
        (total_q_tokens, num_heads, max_num_partitions, kv_lora_rank),
        dtype=q_nope.dtype,
    )
    mx.eval(exp_sums, max_logits, tmp_out)

    ops = get_ops()
    _ensure_mla_library(ops)
    ops.mla_paged_attention_decode_2pass(
        out,
        exp_sums,
        max_logits,
        tmp_out,
        q_nope,
        q_pe,
        latent_cache,
        block_tables,
        context_lens,
        block_size,
        scale,
        partition_size,
        max_num_partitions,
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

    # 5. Initialise GDN linear attention library
    gdn_src = _build_gdn_source()
    mod.init_gdn_library(gdn_src)

    # The MLA paged-attention library is loaded lazily on first use via
    # _ensure_mla_library so non-MLA paged-attention users do not pay the
    # experimental MLA shader compile cost, and a compile error in the
    # experimental MLA shader cannot block unrelated paged-attention paths.

    _ops_module = mod
    logger.info("Native paged-attention Metal kernels loaded")
    return mod


def _ensure_mla_library(mod: ModuleType) -> None:
    """Lazy-init the MLA shader library so non-MLA paged-attention users
    do not pay the experimental MLA shader compile cost; called from
    each MLA direct entrypoint before its first C++ dispatch.
    """
    global _mla_library_initialized
    if _mla_library_initialized:
        return
    mla_src = _build_mla_paged_attention_source()
    mod.init_mla_library(mla_src)
    _mla_library_initialized = True
    logger.info("MLA paged-attention Metal kernels loaded")
