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

from vllm_metal.metal.constants import PARTITION_SIZE

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
        _read_metal_source(_KERNELS_V2_DIR / "pagedattention_tiled.metal"),
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


def metal_mla_paged_attention(
    q_nope,  # [total_q_tokens, num_heads, kv_lora_rank]
    q_pe,  # [total_q_tokens, num_heads, qk_rope_head_dim]
    latent_cache,  # [num_blocks, block_size, kv_lora_rank + qk_rope_head_dim]
    block_tables,  # [num_seqs, max_blocks_per_seq], int32
    context_lens,  # [num_seqs], uint32
    cu_seqlens_q,  # [num_seqs + 1], int32
    scale: float,
    heads_per_tg: int = 1,
):
    """Paged Multi-head Latent Attention (RFC #360). Returns a lazy
    ``mx.array`` whose evaluation triggers the kernel dispatch.

    Q is expected to be already projected through ``embed_q`` (so
    q_nope is in kv_lora_rank space) and ``q_pe`` is RoPE-applied. The
    caller is responsible for ``unembed_out`` on the result to recover
    v_head_dim.

    The dispatch is wrapped in an MLX Primitive so it participates in
    MLX's lazy graph — no ``mx.eval`` / ``mx.synchronize`` boundary
    inside this entry. ``heads_per_tg`` (G) controls cross-head KV
    amortization: each threadgroup processes G consecutive query
    heads sharing the same latent KV; ``num_heads`` must be divisible
    by G. Currently instantiated for G ∈ {1, 2}.
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
