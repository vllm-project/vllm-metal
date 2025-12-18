# SPDX-License-Identifier: Apache-2.0
"""MLX normalization operations for vLLM Metal backend."""

import mlx.core as mx
import mlx.nn as nn


def mlx_rms_norm(
    x: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
) -> mx.array:
    """Apply RMS normalization using MLX.

    RMSNorm is a simplified version of LayerNorm that only normalizes
    by the root mean square, without centering. It's commonly used in
    modern LLMs like Llama, Mistral, etc.

    Args:
        x: Input tensor [..., hidden_size].
        weight: Learnable scale parameter [hidden_size].
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor, same shape as input.
    """
    # Compute RMS
    variance = mx.mean(x * x, axis=-1, keepdims=True)
    x_normed = x * mx.rsqrt(variance + eps)

    # Scale
    return x_normed * weight


def mlx_fused_add_rms_norm(
    x: mx.array,
    residual: mx.array,
    weight: mx.array,
    eps: float = 1e-6,
) -> tuple[mx.array, mx.array]:
    """Fused residual addition and RMS normalization.

    This combines the residual addition and normalization into a single
    operation for better efficiency.

    Args:
        x: Input tensor [..., hidden_size].
        residual: Residual tensor to add [..., hidden_size].
        weight: Learnable scale parameter [hidden_size].
        eps: Small constant for numerical stability.

    Returns:
        Tuple of (normalized_output, updated_residual).
    """
    # Add residual
    residual = residual + x

    # Compute RMS norm
    variance = mx.mean(residual * residual, axis=-1, keepdims=True)
    x_normed = residual * mx.rsqrt(variance + eps)

    # Scale
    output = x_normed * weight

    return output, residual


def mlx_layer_norm(
    x: mx.array,
    weight: mx.array,
    bias: mx.array | None = None,
    eps: float = 1e-5,
) -> mx.array:
    """Apply layer normalization using MLX.

    Standard LayerNorm with optional bias.

    Args:
        x: Input tensor [..., hidden_size].
        weight: Learnable scale parameter [hidden_size].
        bias: Optional learnable bias parameter [hidden_size].
        eps: Small constant for numerical stability.

    Returns:
        Normalized tensor, same shape as input.
    """
    # Compute mean and variance
    mean = mx.mean(x, axis=-1, keepdims=True)
    variance = mx.var(x, axis=-1, keepdims=True)

    # Normalize
    x_normed = (x - mean) * mx.rsqrt(variance + eps)

    # Scale and shift
    output = x_normed * weight
    if bias is not None:
        output = output + bias

    return output


class MLXRMSNorm(nn.Module):
    """MLX module for RMS normalization.

    This is compatible with MLX's module system and can be used
    in model definitions.
    """

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        """Initialize RMSNorm module.

        Args:
            hidden_size: Size of the hidden dimension.
            eps: Small constant for numerical stability.
        """
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((hidden_size,))

    def __call__(self, x: mx.array) -> mx.array:
        """Apply RMS normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        return mlx_rms_norm(x, self.weight, self.eps)


class MLXLayerNorm(nn.Module):
    """MLX module for layer normalization.

    This is compatible with MLX's module system and can be used
    in model definitions.
    """

    def __init__(
        self,
        hidden_size: int,
        eps: float = 1e-5,
        bias: bool = True,
    ):
        """Initialize LayerNorm module.

        Args:
            hidden_size: Size of the hidden dimension.
            eps: Small constant for numerical stability.
            bias: Whether to include learnable bias.
        """
        super().__init__()
        self.eps = eps
        self.weight = mx.ones((hidden_size,))
        self.bias = mx.zeros((hidden_size,)) if bias else None

    def __call__(self, x: mx.array) -> mx.array:
        """Apply layer normalization.

        Args:
            x: Input tensor.

        Returns:
            Normalized tensor.
        """
        return mlx_layer_norm(x, self.weight, self.bias, self.eps)
