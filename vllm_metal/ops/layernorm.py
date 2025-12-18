# SPDX-License-Identifier: Apache-2.0
"""Metal normalization operations."""

from typing import Optional, Tuple

import torch

from vllm_metal.mlx import (
    mlx_fused_add_rms_norm,
    mlx_rms_norm,
    to_mlx,
    to_torch,
)


def rms_norm(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> None:
    """RMS normalization using MLX.

    Args:
        output: Output tensor to write results.
        input: Input tensor [..., hidden_size].
        weight: Scale parameter [hidden_size].
        eps: Small constant for numerical stability.
    """
    import mlx.core as mx

    # Convert to MLX
    x_mlx = to_mlx(input)
    w_mlx = to_mlx(weight)

    # Compute RMS norm
    result = mlx_rms_norm(x_mlx, w_mlx, eps)

    # Convert back and copy to output
    result_torch = to_torch(result, device=output.device, dtype=output.dtype)
    output.copy_(result_torch)


def fused_add_rms_norm(
    input: torch.Tensor,
    residual: torch.Tensor,
    weight: torch.Tensor,
    eps: float = 1e-6,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Fused residual addition and RMS normalization using MLX.

    This combines residual addition and RMS normalization into a single
    operation for better efficiency.

    Args:
        input: Input tensor [..., hidden_size].
        residual: Residual tensor [..., hidden_size].
        weight: Scale parameter [hidden_size].
        eps: Small constant for numerical stability.

    Returns:
        Tuple of (normalized_output, updated_residual).
    """
    import mlx.core as mx

    # Convert to MLX
    x_mlx = to_mlx(input)
    res_mlx = to_mlx(residual)
    w_mlx = to_mlx(weight)

    # Compute fused add + RMS norm
    output_mlx, residual_mlx = mlx_fused_add_rms_norm(x_mlx, res_mlx, w_mlx, eps)

    # Convert back to PyTorch
    output = to_torch(output_mlx, device=input.device, dtype=input.dtype)
    residual_out = to_torch(residual_mlx, device=residual.device, dtype=residual.dtype)

    return output, residual_out


def layer_norm(
    output: torch.Tensor,
    input: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor] = None,
    eps: float = 1e-5,
) -> None:
    """Layer normalization using MLX.

    Args:
        output: Output tensor to write results.
        input: Input tensor [..., hidden_size].
        weight: Scale parameter [hidden_size].
        bias: Optional bias parameter [hidden_size].
        eps: Small constant for numerical stability.
    """
    from vllm_metal.mlx import mlx_layer_norm

    import mlx.core as mx

    # Convert to MLX
    x_mlx = to_mlx(input)
    w_mlx = to_mlx(weight)
    b_mlx = to_mlx(bias) if bias is not None else None

    # Compute layer norm
    result = mlx_layer_norm(x_mlx, w_mlx, b_mlx, eps)

    # Convert back and copy to output
    result_torch = to_torch(result, device=output.device, dtype=output.dtype)
    output.copy_(result_torch)
