# SPDX-License-Identifier: Apache-2.0
"""MPS RMSNorm implementations for vLLM's operator registry."""

from functools import cache

import torch
from vllm import ir


@cache
def _normalization_library():
    from vllm_metal import envs
    from vllm_metal.metal.build import prepare_metallib

    path = prepare_metallib(
        "normalization_kern", build_from_source=envs.VLLM_METAL_BUILD_FROM_SOURCE
    )
    return torch.mps.load_metallib(path)


@cache
def _add_rms_norm_kernel(dtype, weight_dtype=None):
    library = _normalization_library()
    name = "add_rms_norm_fp16" if dtype == torch.float16 else "add_rms_norm_bf16"
    if weight_dtype == torch.float32:
        name += "_weight_float"
    return library, getattr(library, name)


def _supports_rms_norm(x, weight, epsilon, variance_size=None):
    return (
        x.device.type == "mps"
        and x.dtype in (torch.float16, torch.bfloat16)
        and x.numel() > 0
        and weight is not None
        and weight.dtype in (x.dtype, torch.float32)
        and weight.device == x.device
        and variance_size is None
    )


def rms_norm(
    x: torch.Tensor,
    weight: torch.Tensor | None,
    epsilon: float,
    variance_size: int | None = None,
) -> torch.Tensor:
    # Omni also calls this function directly, bypassing the registry filter.
    if (
        weight is not None
        and weight.dtype == torch.float32
        and _supports_rms_norm(x, weight, epsilon, variance_size)
    ):
        if (
            x.is_contiguous()
            and weight.is_contiguous()
            and weight.ndim == 1
            and weight.shape[0] == x.shape[-1]
        ):
            output = torch.empty_like(x)
            suffix = "fp16" if x.dtype == torch.float16 else "bf16"
            kernel = getattr(
                _normalization_library(), f"rms_norm_{suffix}_weight_float"
            )
            kernel(
                x,
                weight,
                output,
                x.shape[-1],
                epsilon,
                threads=(256, x.numel() // x.shape[-1], 1),
                group_size=(256, 1, 1),
            )
            return output
        return torch.nn.functional.rms_norm(
            x.float(), (x.shape[-1],), weight, epsilon
        ).to(x.dtype)
    return torch.nn.functional.rms_norm(x, (x.shape[-1],), weight, epsilon)


ir.ops.rms_norm.register_impl("torch_mps", supports_args=_supports_rms_norm)(rms_norm)


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
    if (
        x.ndim == 2
        and x.is_contiguous()
        and x_residual.is_contiguous()
        and weight.ndim == 1
        and weight.shape[0] == x.shape[1]
        and weight.is_contiguous()
    ):
        output = torch.empty_like(x)
        residual = torch.empty_like(x)
        _, kernel = _add_rms_norm_kernel(x.dtype, weight.dtype)
        kernel(
            x,
            x_residual,
            weight,
            output,
            residual,
            x.shape[1],
            epsilon,
            threads=(256, x.shape[0], 1),
            group_size=(256, 1, 1),
        )
        return output, residual

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
    return (normalized.to(weight.dtype) * weight).to(x.dtype), summed.to(x.dtype)
