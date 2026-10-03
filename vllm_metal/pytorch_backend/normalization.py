# SPDX-License-Identifier: Apache-2.0
"""Use PyTorch's fused MPS RMSNorm through vLLM's operator registry."""

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
