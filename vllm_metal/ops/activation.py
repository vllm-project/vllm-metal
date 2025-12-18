# SPDX-License-Identifier: Apache-2.0
"""Metal activation operations."""

import torch

from vllm_metal.mlx import to_mlx, to_torch


def silu_and_mul(
    output: torch.Tensor,
    input: torch.Tensor,
) -> None:
    """SiLU activation with gated multiplication.

    Computes: silu(x[:, :d]) * x[:, d:]
    where d = input.shape[-1] // 2

    This is the GLU variant using SiLU (Swish) activation.

    Args:
        output: Output tensor [*, d].
        input: Input tensor [*, 2*d].
    """
    import mlx.core as mx

    # Convert to MLX
    x_mlx = to_mlx(input)

    # Split input
    d = x_mlx.shape[-1] // 2
    gate = x_mlx[..., :d]
    up = x_mlx[..., d:]

    # SiLU (Swish) activation: x * sigmoid(x)
    silu_gate = gate * mx.sigmoid(gate)

    # Multiply with up projection
    result = silu_gate * up

    # Convert back and copy to output
    result_torch = to_torch(result, device=output.device, dtype=output.dtype)
    output.copy_(result_torch)


def gelu_and_mul(
    output: torch.Tensor,
    input: torch.Tensor,
) -> None:
    """GELU activation with gated multiplication.

    Computes: gelu(x[:, :d]) * x[:, d:]
    where d = input.shape[-1] // 2

    This is the GLU variant using GELU activation.

    Args:
        output: Output tensor [*, d].
        input: Input tensor [*, 2*d].
    """
    import mlx.core as mx
    import mlx.nn as nn

    # Convert to MLX
    x_mlx = to_mlx(input)

    # Split input
    d = x_mlx.shape[-1] // 2
    gate = x_mlx[..., :d]
    up = x_mlx[..., d:]

    # GELU activation (using approximate formula for speed)
    gelu_gate = nn.gelu(gate)

    # Multiply with up projection
    result = gelu_gate * up

    # Convert back and copy to output
    result_torch = to_torch(result, device=output.device, dtype=output.dtype)
    output.copy_(result_torch)


def gelu_tanh_and_mul(
    output: torch.Tensor,
    input: torch.Tensor,
) -> None:
    """GELU (tanh approximation) activation with gated multiplication.

    Computes: gelu_tanh(x[:, :d]) * x[:, d:]
    where d = input.shape[-1] // 2

    Uses the tanh approximation of GELU for faster computation.

    Args:
        output: Output tensor [*, d].
        input: Input tensor [*, 2*d].
    """
    import mlx.core as mx
    import mlx.nn as nn

    # Convert to MLX
    x_mlx = to_mlx(input)

    # Split input
    d = x_mlx.shape[-1] // 2
    gate = x_mlx[..., :d]
    up = x_mlx[..., d:]

    # GELU with tanh approximation
    gelu_gate = nn.gelu_approx(gate)

    # Multiply with up projection
    result = gelu_gate * up

    # Convert back and copy to output
    result_torch = to_torch(result, device=output.device, dtype=output.dtype)
    output.copy_(result_torch)


def relu_and_mul(
    output: torch.Tensor,
    input: torch.Tensor,
) -> None:
    """ReLU activation with gated multiplication.

    Computes: relu(x[:, :d]) * x[:, d:]
    where d = input.shape[-1] // 2

    Args:
        output: Output tensor [*, d].
        input: Input tensor [*, 2*d].
    """
    import mlx.core as mx
    import mlx.nn as nn

    # Convert to MLX
    x_mlx = to_mlx(input)

    # Split input
    d = x_mlx.shape[-1] // 2
    gate = x_mlx[..., :d]
    up = x_mlx[..., d:]

    # ReLU activation
    relu_gate = nn.relu(gate)

    # Multiply with up projection
    result = relu_gate * up

    # Convert back and copy to output
    result_torch = to_torch(result, device=output.device, dtype=output.dtype)
    output.copy_(result_torch)
