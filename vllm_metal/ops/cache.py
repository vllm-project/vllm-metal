# SPDX-License-Identifier: Apache-2.0
"""Metal KV cache operations."""

from typing import Dict, List, Tuple

import torch

from vllm_metal.mlx import (
    mlx_copy_blocks,
    mlx_reshape_and_cache,
    mlx_swap_blocks,
    to_mlx,
    to_torch,
)


def reshape_and_cache(
    key: torch.Tensor,
    value: torch.Tensor,
    key_cache: torch.Tensor,
    value_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    kv_cache_dtype: str = "auto",
    k_scale: float = 1.0,
    v_scale: float = 1.0,
) -> None:
    """Reshape and cache key/value tensors using MLX.

    Args:
        key: Key tensor [num_tokens, num_kv_heads, head_size].
        value: Value tensor [num_tokens, num_kv_heads, head_size].
        key_cache: Key cache [num_blocks, block_size, num_kv_heads, head_size].
        value_cache: Value cache [num_blocks, block_size, num_kv_heads, head_size].
        slot_mapping: Slot mapping [num_tokens].
        kv_cache_dtype: Cache data type.
        k_scale: Key scale (for FP8).
        v_scale: Value scale (for FP8).
    """
    # Use PyTorch for cache updates since we need in-place operations
    # and the tensors are on MPS
    block_size = key_cache.shape[1]

    # Compute block indices and offsets
    block_indices = slot_mapping // block_size
    block_offsets = slot_mapping % block_size

    # Store using indexing
    num_tokens = key.shape[0]
    for i in range(num_tokens):
        block_idx = int(block_indices[i])
        offset = int(block_offsets[i])
        key_cache[block_idx, offset] = key[i]
        value_cache[block_idx, offset] = value[i]


def copy_blocks(
    key_caches: List[torch.Tensor],
    value_caches: List[torch.Tensor],
    block_mapping: torch.Tensor,
) -> None:
    """Copy cache blocks from source to destination.

    This is used for beam search and other operations that require
    duplicating cache contents.

    Args:
        key_caches: List of key cache tensors.
        value_caches: List of value cache tensors.
        block_mapping: Pairs of (src_block, dst_block) [num_pairs, 2].
    """
    if block_mapping.numel() == 0:
        return

    # Process each layer's cache
    for key_cache, value_cache in zip(key_caches, value_caches):
        # Use PyTorch for in-place operations
        for pair in block_mapping:
            src = int(pair[0])
            dst = int(pair[1])
            key_cache[dst].copy_(key_cache[src])
            value_cache[dst].copy_(value_cache[src])


def swap_blocks(
    src_cache: torch.Tensor,
    dst_cache: torch.Tensor,
    block_mapping: Dict[int, int],
) -> None:
    """Swap cache blocks between tensors.

    On Apple Silicon with unified memory, this is effectively a copy
    since CPU and GPU share the same memory.

    Args:
        src_cache: Source cache tensor.
        dst_cache: Destination cache tensor.
        block_mapping: Source to destination block mapping.
    """
    if not block_mapping:
        return

    for src, dst in block_mapping.items():
        dst_cache[dst].copy_(src_cache[src])
