# SPDX-License-Identifier: Apache-2.0
"""MLX operations for transformer inference."""

import math
from typing import Literal

import mlx.core as mx


def rms_norm(
    x: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Apply Root Mean Square Layer Normalization.

    Args:
        x: Input tensor of shape (..., hidden_size)
        weight: Learnable scale parameter of shape (hidden_size,)
        eps: Small constant for numerical stability

    Returns:
        Normalized tensor of same shape as input
    """
    # Compute RMS: sqrt(mean(x^2))
    rms = mx.sqrt(mx.mean(mx.square(x), axis=-1, keepdims=True) + eps)
    # Normalize and scale
    return (x / rms) * weight


def rotary_embedding(
    q: mx.array,
    k: mx.array,
    positions: mx.array,
    head_dim: int,
    rope_theta: float = 10000.0,
    rope_scaling: dict | None = None,
) -> tuple[mx.array, mx.array]:
    """Apply Rotary Position Embedding (RoPE) to queries and keys.

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch, seq_len, num_kv_heads, head_dim)
        positions: Position indices of shape (batch, seq_len)
        head_dim: Dimension of each attention head
        rope_theta: Base frequency for RoPE
        rope_scaling: Optional scaling configuration

    Returns:
        Tuple of (rotated_q, rotated_k) with same shapes as inputs
    """
    # Compute inverse frequencies
    dim = head_dim // 2
    inv_freq = 1.0 / (rope_theta ** (mx.arange(0, dim, dtype=mx.float32) / dim))

    # Apply rope scaling if configured
    if rope_scaling is not None:
        scale_type = rope_scaling.get("type", "linear")
        factor = rope_scaling.get("factor", 1.0)
        if scale_type == "linear":
            inv_freq = inv_freq / factor

    # Compute position embeddings
    # positions: (batch, seq_len) -> (batch, seq_len, 1)
    positions = positions[:, :, None].astype(mx.float32)
    # inv_freq: (dim,) -> (1, 1, dim)
    inv_freq = inv_freq[None, None, :]
    # freqs: (batch, seq_len, dim)
    freqs = positions * inv_freq

    # Compute sin and cos
    cos = mx.cos(freqs)
    sin = mx.sin(freqs)

    def apply_rope(x: mx.array, cos: mx.array, sin: mx.array) -> mx.array:
        # x: (batch, seq_len, num_heads, head_dim)
        # Split into even/odd indices
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
        # Expand cos/sin for broadcasting: (batch, seq_len, 1, dim)
        cos = cos[:, :, None, :]
        sin = sin[:, :, None, :]
        # Apply rotation
        rotated = mx.concatenate(
            [x1 * cos - x2 * sin, x1 * sin + x2 * cos],
            axis=-1,
        )
        return rotated

    q_rotated = apply_rope(q, cos, sin)
    k_rotated = apply_rope(k, cos, sin)

    return q_rotated, k_rotated


def attention(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    scale: float | None = None,
    mask: mx.array | None = None,
    attention_type: Literal["sdpa", "flash"] = "sdpa",
) -> mx.array:
    """Compute scaled dot-product attention.

    Args:
        q: Query tensor of shape (batch, seq_len, num_heads, head_dim)
        k: Key tensor of shape (batch, kv_len, num_kv_heads, head_dim)
        v: Value tensor of shape (batch, kv_len, num_kv_heads, head_dim)
        scale: Scaling factor (default: 1/sqrt(head_dim))
        mask: Optional attention mask
        attention_type: Type of attention to use

    Returns:
        Output tensor of shape (batch, seq_len, num_heads, head_dim)
    """
    batch, seq_len, num_heads, head_dim = q.shape
    _, kv_len, num_kv_heads, _ = k.shape

    if scale is None:
        scale = 1.0 / math.sqrt(head_dim)

    # Handle GQA: repeat k, v if num_heads != num_kv_heads
    if num_heads != num_kv_heads:
        num_groups = num_heads // num_kv_heads
        # Expand k, v: (batch, kv_len, num_kv_heads, head_dim)
        # -> (batch, kv_len, num_heads, head_dim)
        k = mx.repeat(k, num_groups, axis=2)
        v = mx.repeat(v, num_groups, axis=2)

    # Transpose to (batch, num_heads, seq_len, head_dim)
    q = mx.transpose(q, (0, 2, 1, 3))
    k = mx.transpose(k, (0, 2, 1, 3))
    v = mx.transpose(v, (0, 2, 1, 3))

    # Use MLX's fast_attention for better performance when available
    if attention_type == "sdpa":
        # Compute attention scores: (batch, num_heads, seq_len, kv_len)
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale

        if mask is not None:
            scores = scores + mask

        # Softmax and weighted sum
        weights = mx.softmax(scores, axis=-1)
        output = mx.matmul(weights, v)
    else:
        # Default to SDPA
        scores = mx.matmul(q, mx.transpose(k, (0, 1, 3, 2))) * scale
        if mask is not None:
            scores = scores + mask
        weights = mx.softmax(scores, axis=-1)
        output = mx.matmul(weights, v)

    # Transpose back: (batch, num_heads, seq_len, head_dim)
    # -> (batch, seq_len, num_heads, head_dim)
    output = mx.transpose(output, (0, 2, 1, 3))

    return output


def silu_and_mul(x: mx.array) -> mx.array:
    """Apply SiLU activation and gated multiplication.

    This is used in LLaMA-style feed-forward networks where the input
    is split in half, one half goes through SiLU, and they're multiplied.

    Args:
        x: Input tensor of shape (..., 2 * hidden_dim)

    Returns:
        Output tensor of shape (..., hidden_dim)
    """
    hidden_dim = x.shape[-1] // 2
    x1 = x[..., :hidden_dim]
    x2 = x[..., hidden_dim:]
    # SiLU(x) = x * sigmoid(x)
    return mx.sigmoid(x1) * x1 * x2


def gelu(x: mx.array, approximate: bool = True) -> mx.array:
    """Apply GELU activation function.

    Args:
        x: Input tensor
        approximate: Use tanh approximation (faster)

    Returns:
        Activated tensor
    """
    if approximate:
        return (
            0.5
            * x
            * (1.0 + mx.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))
        )
    return x * 0.5 * (1.0 + mx.erf(x / math.sqrt(2.0)))
