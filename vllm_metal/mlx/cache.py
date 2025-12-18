# SPDX-License-Identifier: Apache-2.0
"""MLX KV cache operations for vLLM Metal backend."""

import mlx.core as mx


def mlx_reshape_and_cache(
    key: mx.array,
    value: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    slot_mapping: mx.array,
) -> tuple[mx.array, mx.array]:
    """Store key and value tensors into the cache at specified slots.

    This is the core operation for populating the KV cache during
    both prefill and decode phases.

    Args:
        key: Keys to store [num_tokens, num_kv_heads, head_dim].
        value: Values to store [num_tokens, num_kv_heads, head_dim].
        key_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim].
        value_cache: Value cache [num_blocks, block_size, num_kv_heads, head_dim].
        slot_mapping: Mapping from token index to cache slot [num_tokens].

    Returns:
        Updated (key_cache, value_cache) tuple.
    """
    num_tokens = key.shape[0]
    num_kv_heads = key.shape[1]
    head_dim = key.shape[2]
    block_size = key_cache.shape[1]

    # Compute block indices and offsets from slot mapping
    block_indices = slot_mapping // block_size
    block_offsets = slot_mapping % block_size

    # Flatten the cache for indexing
    num_blocks = key_cache.shape[0]
    key_cache_flat = key_cache.reshape(-1, num_kv_heads, head_dim)
    value_cache_flat = value_cache.reshape(-1, num_kv_heads, head_dim)

    # Compute flat indices
    flat_indices = block_indices * block_size + block_offsets

    # Update cache using scatter
    # MLX doesn't have direct scatter_nd, so we use a workaround
    for i in range(num_tokens):
        idx = int(flat_indices[i])
        key_cache_flat = key_cache_flat.at[idx].set(key[i])
        value_cache_flat = value_cache_flat.at[idx].set(value[i])

    # Reshape back
    key_cache = key_cache_flat.reshape(num_blocks, block_size, num_kv_heads, head_dim)
    value_cache = value_cache_flat.reshape(
        num_blocks, block_size, num_kv_heads, head_dim
    )

    return key_cache, value_cache


def mlx_reshape_and_cache_flash(
    key: mx.array,
    value: mx.array,
    key_cache: mx.array,
    value_cache: mx.array,
    slot_mapping: mx.array,
) -> tuple[mx.array, mx.array]:
    """Optimized reshape_and_cache using batched operations.

    This version is more efficient for larger batch sizes by
    avoiding the Python loop.

    Args:
        key: Keys to store [num_tokens, num_kv_heads, head_dim].
        value: Values to store [num_tokens, num_kv_heads, head_dim].
        key_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim].
        value_cache: Value cache [num_blocks, block_size, num_kv_heads, head_dim].
        slot_mapping: Mapping from token index to cache slot [num_tokens].

    Returns:
        Updated (key_cache, value_cache) tuple.
    """
    block_size = key_cache.shape[1]
    num_kv_heads = key.shape[1]
    head_dim = key.shape[2]

    # Compute block indices and offsets
    block_indices = slot_mapping // block_size
    block_offsets = slot_mapping % block_size

    # Use advanced indexing to update cache
    # This is more efficient for larger batches
    key_cache = key_cache.at[block_indices, block_offsets].set(key)
    value_cache = value_cache.at[block_indices, block_offsets].set(value)

    return key_cache, value_cache


def mlx_copy_blocks(
    key_cache: mx.array,
    value_cache: mx.array,
    block_mapping: mx.array,
) -> tuple[mx.array, mx.array]:
    """Copy cache blocks from source to destination.

    This is used for operations like beam search where cache contents
    need to be duplicated.

    Args:
        key_cache: Key cache [num_blocks, block_size, num_kv_heads, head_dim].
        value_cache: Value cache [num_blocks, block_size, num_kv_heads, head_dim].
        block_mapping: Pairs of (src_block, dst_block) [num_pairs, 2].

    Returns:
        Updated (key_cache, value_cache) tuple.
    """
    if block_mapping.shape[0] == 0:
        return key_cache, value_cache

    src_blocks = block_mapping[:, 0]
    dst_blocks = block_mapping[:, 1]

    # Copy blocks
    for i in range(block_mapping.shape[0]):
        src = int(src_blocks[i])
        dst = int(dst_blocks[i])
        key_cache = key_cache.at[dst].set(key_cache[src])
        value_cache = value_cache.at[dst].set(value_cache[src])

    return key_cache, value_cache


def mlx_swap_blocks(
    key_cache: mx.array,
    value_cache: mx.array,
    src_to_dst: dict[int, int],
) -> tuple[mx.array, mx.array]:
    """Swap cache blocks between GPU and CPU.

    On Apple Silicon with unified memory, this is effectively a no-op
    since GPU and CPU share the same memory. We just update the indices.

    Args:
        key_cache: Key cache tensor.
        value_cache: Value cache tensor.
        src_to_dst: Mapping of source block to destination block.

    Returns:
        Updated (key_cache, value_cache) tuple.
    """
    if not src_to_dst:
        return key_cache, value_cache

    # On unified memory, we just need to copy the blocks
    for src, dst in src_to_dst.items():
        key_cache = key_cache.at[dst].set(key_cache[src])
        value_cache = value_cache.at[dst].set(value_cache[src])

    return key_cache, value_cache


def mlx_create_kv_cache(
    num_blocks: int,
    block_size: int,
    num_kv_heads: int,
    head_dim: int,
    dtype: mx.Dtype = mx.float16,
) -> tuple[mx.array, mx.array]:
    """Create empty KV cache tensors.

    Args:
        num_blocks: Number of cache blocks.
        block_size: Number of tokens per block.
        num_kv_heads: Number of key-value heads.
        head_dim: Dimension of each head.
        dtype: Data type for the cache.

    Returns:
        Tuple of (key_cache, value_cache).
    """
    shape = (num_blocks, block_size, num_kv_heads, head_dim)
    key_cache = mx.zeros(shape, dtype=dtype)
    value_cache = mx.zeros(shape, dtype=dtype)
    return key_cache, value_cache
