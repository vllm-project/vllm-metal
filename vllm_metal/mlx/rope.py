# SPDX-License-Identifier: Apache-2.0
"""MLX rotary position embedding (RoPE) for vLLM Metal backend."""

import mlx.core as mx
import mlx.nn as nn


def mlx_rotary_embedding(
    x: mx.array,
    cos: mx.array,
    sin: mx.array,
    is_neox_style: bool = True,
) -> mx.array:
    """Apply rotary position embeddings to input tensor.

    This implements RoPE (Rotary Position Embedding) using MLX operations.
    Supports both standard and NeoX-style interleaving.

    Args:
        x: Input tensor [batch, seq_len, num_heads, head_dim] or
           [batch, num_heads, seq_len, head_dim].
        cos: Cosine values [seq_len, rotary_dim] or [1, 1, seq_len, rotary_dim].
        sin: Sine values [seq_len, rotary_dim] or [1, 1, seq_len, rotary_dim].
        is_neox_style: If True, use NeoX-style rotation (split in half).
                       If False, use interleaved rotation.

    Returns:
        Tensor with RoPE applied, same shape as input.
    """
    # Get dimensions
    head_dim = x.shape[-1]
    rotary_dim = cos.shape[-1]

    # Split into rotary and pass-through parts
    x_rot = x[..., :rotary_dim]
    x_pass = x[..., rotary_dim:] if rotary_dim < head_dim else None

    if is_neox_style:
        # NeoX style: split into two halves
        half_dim = rotary_dim // 2
        x1 = x_rot[..., :half_dim]
        x2 = x_rot[..., half_dim:]

        # Broadcast cos/sin to match x shape
        cos = _broadcast_for_rope(cos, x_rot)
        sin = _broadcast_for_rope(sin, x_rot)

        cos1 = cos[..., :half_dim]
        cos2 = cos[..., half_dim:]
        sin1 = sin[..., :half_dim]
        sin2 = sin[..., half_dim:]

        # Apply rotation
        out1 = x1 * cos1 - x2 * sin1
        out2 = x1 * sin2 + x2 * cos2
        x_rot = mx.concatenate([out1, out2], axis=-1)
    else:
        # Standard interleaved style
        cos = _broadcast_for_rope(cos, x_rot)
        sin = _broadcast_for_rope(sin, x_rot)

        # Rotate pairs of elements
        x_rot_shape = x_rot.shape
        x_rot = x_rot.reshape(*x_rot_shape[:-1], -1, 2)

        cos = cos.reshape(*cos.shape[:-1], -1, 2)
        sin = sin.reshape(*sin.shape[:-1], -1, 2)

        x0 = x_rot[..., 0]
        x1 = x_rot[..., 1]
        cos0 = cos[..., 0]
        sin0 = sin[..., 0]

        out0 = x0 * cos0 - x1 * sin0
        out1 = x0 * sin0 + x1 * cos0

        x_rot = mx.stack([out0, out1], axis=-1).reshape(x_rot_shape)

    # Concatenate rotary and pass-through parts
    if x_pass is not None:
        return mx.concatenate([x_rot, x_pass], axis=-1)
    return x_rot


def _broadcast_for_rope(freqs: mx.array, x: mx.array) -> mx.array:
    """Broadcast frequency tensor to match input shape.

    Args:
        freqs: Frequency tensor (cos or sin).
        x: Input tensor to match shape with.

    Returns:
        Broadcasted frequency tensor.
    """
    # Handle different input shapes
    while freqs.ndim < x.ndim:
        freqs = freqs[None, ...]

    # Ensure freqs can broadcast with x
    target_shape = list(x.shape)
    target_shape[-1] = freqs.shape[-1]

    return mx.broadcast_to(freqs, target_shape)


def mlx_apply_rope(
    query: mx.array,
    key: mx.array,
    positions: mx.array,
    head_dim: int,
    rotary_dim: int | None = None,
    base: float = 10000.0,
    is_neox_style: bool = True,
) -> tuple[mx.array, mx.array]:
    """Apply RoPE to query and key tensors.

    This is a convenience function that computes the rotary embeddings
    and applies them to both query and key tensors.

    Args:
        query: Query tensor.
        key: Key tensor.
        positions: Position indices for each token.
        head_dim: Dimension of each attention head.
        rotary_dim: Dimension to apply rotation to (default: head_dim).
        base: Base for the frequency computation.
        is_neox_style: Whether to use NeoX-style rotation.

    Returns:
        Tuple of (rotated_query, rotated_key).
    """
    if rotary_dim is None:
        rotary_dim = head_dim

    # Compute frequencies
    inv_freq = 1.0 / (base ** (mx.arange(0, rotary_dim, 2) / rotary_dim))

    # Compute position-dependent frequencies
    # positions: [batch, seq_len] or [seq_len]
    if positions.ndim == 1:
        positions = positions[None, :]

    # [batch, seq_len, rotary_dim/2]
    freqs = mx.outer(positions.reshape(-1), inv_freq).reshape(
        *positions.shape, rotary_dim // 2
    )

    # Duplicate for full rotary_dim
    freqs = mx.concatenate([freqs, freqs], axis=-1)

    cos = mx.cos(freqs)
    sin = mx.sin(freqs)

    # Apply to query and key
    rotated_query = mlx_rotary_embedding(query, cos, sin, is_neox_style)
    rotated_key = mlx_rotary_embedding(key, cos, sin, is_neox_style)

    return rotated_query, rotated_key


class MLXRotaryEmbedding(nn.Module):
    """MLX module for rotary position embeddings.

    This wraps the RoPE functionality in an MLX module for easier use
    in model definitions.
    """

    def __init__(
        self,
        dim: int,
        max_position_embeddings: int = 2048,
        base: float = 10000.0,
        is_neox_style: bool = True,
    ):
        """Initialize rotary embedding module.

        Args:
            dim: Dimension of the rotary embeddings.
            max_position_embeddings: Maximum sequence length.
            base: Base for frequency computation.
            is_neox_style: Whether to use NeoX-style rotation.
        """
        super().__init__()
        self.dim = dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style

        # Precompute inverse frequencies
        inv_freq = 1.0 / (base ** (mx.arange(0, dim, 2) / dim))
        self._inv_freq = inv_freq

    def __call__(
        self,
        query: mx.array,
        key: mx.array,
        positions: mx.array,
    ) -> tuple[mx.array, mx.array]:
        """Apply rotary embeddings to query and key.

        Args:
            query: Query tensor.
            key: Key tensor.
            positions: Position indices.

        Returns:
            Tuple of (rotated_query, rotated_key).
        """
        # Compute frequencies for positions
        if positions.ndim == 1:
            freqs = mx.outer(positions, self._inv_freq)
        else:
            freqs = positions[..., None] * self._inv_freq[None, :]

        # Duplicate for full dim
        freqs = mx.concatenate([freqs, freqs], axis=-1)

        cos = mx.cos(freqs)
        sin = mx.sin(freqs)

        rotated_query = mlx_rotary_embedding(query, cos, sin, self.is_neox_style)
        rotated_key = mlx_rotary_embedding(key, cos, sin, self.is_neox_style)

        return rotated_query, rotated_key
