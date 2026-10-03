# SPDX-License-Identifier: Apache-2.0
"""Use PyTorch's fused MPS RMSNorm through vLLM's operator registry.

vLLM's eager fallback decomposes RMSNorm into casts, a reduction and
multiplies, so it never selects PyTorch's fused RMSNorm kernel. Calling
the public PyTorch operator avoids those intermediate tensors and
dispatches without adding another kernel implementation.
"""

import torch
from vllm import ir


def _supports_rms_norm(x, weight, epsilon, variance_size=None):
    return (
        x.device.type == "mps"
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.numel() > 0
        and weight is not None
        and weight.dtype == x.dtype
        and weight.device == x.device
        and variance_size is None
    )


@ir.ops.rms_norm.register_impl("torch_mps", supports_args=_supports_rms_norm)
def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> torch.Tensor:
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, epsilon)


def _supports_add_rms_norm(x, x_residual, weight, epsilon, variance_size=None):
    return (
        _supports_rms_norm(x, weight, epsilon, variance_size)
        and x_residual.dtype == x.dtype
        and x_residual.device == x.device
        and x_residual.shape == x.shape
    )


@ir.ops.fused_add_rms_norm.register_impl(
    "torch_mps", supports_args=_supports_add_rms_norm
)
def fused_add_rms_norm(
    x: torch.Tensor,
    x_residual: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert weight is not None
    # vLLM computes variance from the unrounded FP32 residual sum.
    summed = x.float() + x_residual.float()
    # An explicit weight selects PyTorch's fused MPS kernel. Use FP32 ones
    # so we can cast before applying the learned weight, as vLLM requires.
    normalized = torch.nn.functional.rms_norm(
        summed,
        (summed.shape[-1],),
        torch.ones_like(weight, dtype=torch.float32),
        epsilon,
    )
    return normalized.to(x.dtype) * weight, summed.to(x.dtype)
