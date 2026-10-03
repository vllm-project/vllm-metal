# SPDX-License-Identifier: Apache-2.0
"""Kernel block-table conversion shared by target and draft attention."""

import mlx.core as mx

from vllm_metal.metal.constants import KERNEL_BLOCK_SIZES


def pick_kernel_block_size(cache_block_size: int) -> int:
    """Pick the largest kernel-supported block size that divides evenly."""
    for kbs in KERNEL_BLOCK_SIZES:
        if cache_block_size % kbs == 0:
            return kbs
    raise ValueError(
        f"Cache block_size={cache_block_size} is not divisible by any "
        f"supported kernel block size {KERNEL_BLOCK_SIZES}. "
        "Adjust --block-size (must be a multiple of 8)."
    )


def build_block_tables(
    raw_block_tables: list[list[int]],
    cache_block_size: int,
) -> tuple[mx.array, int]:
    """Build kernel-compatible block tables, translating if necessary.

    When ``cache_block_size`` exceeds the kernel's compiled block sizes,
    each vLLM block ``b`` is expanded into ``ratio`` kernel blocks
    ``[b*ratio, b*ratio+ratio)``.  The kernel applies the translated
    token stride when reading dense cache views.

    Returns:
        (block_tables, kernel_block_size)
    """
    if not raw_block_tables:
        return mx.zeros((0, 0), dtype=mx.int32), cache_block_size

    if cache_block_size in KERNEL_BLOCK_SIZES:
        # Fast path — no translation needed.
        max_blocks = max(len(bt) for bt in raw_block_tables)
        padded = [bt + [0] * (max_blocks - len(bt)) for bt in raw_block_tables]
        return mx.array(padded, dtype=mx.int32), cache_block_size

    # Hybrid path — translate large block_size to a kernel-compatible one.
    # Vectorized: each vLLM block b → [b*ratio, b*ratio+1, …, b*ratio+ratio-1].
    kernel_bs = pick_kernel_block_size(cache_block_size)
    ratio = cache_block_size // kernel_bs

    max_blocks = max(len(bt) for bt in raw_block_tables)
    padded = [bt + [0] * (max_blocks - len(bt)) for bt in raw_block_tables]
    bt_arr = mx.array(padded, dtype=mx.int32)  # [num_seqs, max_blocks]
    offsets = mx.arange(ratio, dtype=mx.int32)  # [ratio]
    # [num_seqs, max_blocks, 1] * ratio + [1, 1, ratio] → [num_seqs, max_blocks, ratio]
    expanded = (bt_arr[:, :, None] * ratio + offsets[None, None, :]).reshape(
        bt_arr.shape[0], -1
    )
    return expanded, kernel_bs
