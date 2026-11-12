# SPDX-License-Identifier: Apache-2.0
"""PyTorch-stream launchers for precompiled shared Metal kernels."""

import importlib
import logging
from functools import cache

import torch

logger = logging.getLogger(__name__)


@cache
def _load_mps_module():
    from vllm_metal import envs
    from vllm_metal.metal.build import (
        build_mps,
        mps_artifact_is_stale,
        mps_output_path,
        mps_version_path,
    )

    if envs.VLLM_METAL_BUILD_FROM_SOURCE:
        build_mps()
    if not mps_output_path().exists() or not mps_version_path().exists():
        raise RuntimeError(
            "Prebuilt MPS extension is missing. Reinstall a vllm-metal wheel, "
            "or set VLLM_METAL_BUILD_FROM_SOURCE=1 to build from source."
        )
    built = mps_version_path().read_text().strip()
    if built != str(torch.__version__):
        raise RuntimeError(
            f"The MPS extension was built against PyTorch {built}, but "
            f"{torch.__version__} is installed. Reinstall a matching wheel, "
            "or set VLLM_METAL_BUILD_FROM_SOURCE=1 to rebuild it."
        )
    if mps_artifact_is_stale():
        raise RuntimeError(
            "Prebuilt MPS extension is stale. Run python -m vllm_metal.metal.build "
            "or set VLLM_METAL_BUILD_FROM_SOURCE=1 to rebuild it."
        )
    return importlib.import_module("vllm_metal.pytorch_backend._mps_ops")


@cache
def get_mps_ops():
    from vllm_metal import envs
    from vllm_metal.metal.build import NAX_METALLIB_NAME, prepare_metallib

    module = _load_mps_module()
    # The shared detector is compiled into each launcher; MPS never loads MLX.
    gpu_cores = module.gpu_core_count()
    build_from_source = envs.VLLM_METAL_BUILD_FROM_SOURCE
    path = prepare_metallib(
        "paged_attention_v2_kern", build_from_source=build_from_source
    )
    nax_path = ""
    if not envs.VLLM_METAL_DISABLE_NAX and module.nax_supported():
        try:
            nax_path = str(
                prepare_metallib(NAX_METALLIB_NAME, build_from_source=build_from_source)
            )
        except (OSError, RuntimeError) as exc:
            logger.warning("NAX unavailable; using the non-NAX fallback: %s", exc)
    return module.PagedAttention(str(path), nax_path, gpu_cores)


@cache
def _gdn_kernel(input_dtype, output_dtype):
    from vllm_metal import envs
    from vllm_metal.metal.build import prepare_metallib

    path = prepare_metallib(
        "gdn_kern", build_from_source=envs.VLLM_METAL_BUILD_FROM_SOURCE
    )
    library = torch.mps.load_metallib(path)
    types = {
        torch.float32: "float",
        torch.float16: "half",
        torch.bfloat16: "bfloat16_t",
    }
    name = "gdn_linear_attention_" + types[input_dtype]
    if input_dtype != torch.float32:
        name += "_state_float"
    if output_dtype != input_dtype:
        name += "_out_" + types[output_dtype]
    kernel = getattr(library, name + "_initial")
    if kernel.thread_execution_width != 32 or kernel.max_threads_per_threadgroup < 32:
        raise RuntimeError("GDN requires a 32-lane SIMD group")
    return library, kernel


def _validate_gdn(q, k, v, g, beta, state, cu_seqlens, slots, y, initial):
    if any(
        t.device.type != "mps"
        for t in (q, k, v, g, beta, state, cu_seqlens, slots, y, initial)
    ):
        raise RuntimeError("All GDN buffers must be on MPS")
    if q.dtype not in (torch.float32, torch.float16, torch.bfloat16) or any(
        not t.is_contiguous() or t.dtype != q.dtype for t in (q, k, v, g, beta)
    ):
        raise RuntimeError(
            "GDN Q/K/V/g/beta must be packed fp32/fp16/bf16 with the same dtype"
        )
    if not y.is_contiguous() or not (
        y.dtype == q.dtype
        or q.dtype == torch.float32
        and y.dtype in (torch.float16, torch.bfloat16)
    ):
        raise RuntimeError(
            "GDN Y must be packed, with the input dtype or fp32 inputs and fp16/bf16 output"
        )
    if (
        q.ndim != 3
        or k.shape != q.shape
        or v.ndim != 3
        or v.shape[0] != q.shape[0]
        or y.shape != v.shape
    ):
        raise RuntimeError("GDN requires Q/K [T,Hk,Dk] and V/Y [T,Hv,Dv]")
    tokens, hk, dk = q.shape
    _, hv, dv = v.shape
    if dk <= 0 or dk > 256 or dk % 32:
        raise RuntimeError(
            "GDN requires Dk to be a positive multiple of 32, at most 256"
        )
    if hk <= 0 or hv <= 0 or hv % hk or dv <= 0:
        raise RuntimeError("Invalid GDN heads or value dimension")
    if g.shape != (tokens, hv) or beta.shape != g.shape:
        raise RuntimeError("GDN g/beta must be [T,Hv]")
    if (
        state.dtype != torch.float32
        or state.ndim != 4
        or state.shape[0] == 0
        or state.shape[1:] != (hv, dv, dk)
        or state.stride()[1:] != (dv * dk, dk, 1)
        or state.stride(0) < hv * dv * dk
    ):
        raise RuntimeError(
            "GDN state must be fp32 [slots,Hv,Dv,Dk] with dense inner rows"
        )
    if (
        cu_seqlens.dtype != torch.int32
        or slots.dtype != torch.int32
        or cu_seqlens.ndim != 1
        or slots.ndim != 1
        or not cu_seqlens.is_contiguous()
        or not slots.is_contiguous()
        or cu_seqlens.numel() != slots.numel() + 1
    ):
        raise RuntimeError(
            "GDN metadata must be packed int32 cu_seqlens [B+1], slots [B]"
        )
    if (
        initial.dtype != torch.bool
        or initial.shape != slots.shape
        or not initial.is_contiguous()
    ):
        raise RuntimeError("GDN has_initial_state must be packed bool [B]")
    if (
        any(t.numel() > 2**31 - 1 for t in (q, k, v, g, beta, y))
        or max(slots.numel() * hv, hv * dv * dk) > 2**31 - 1
    ):
        raise RuntimeError(
            "GDN packed input/output or state row exceeds int32 indexing"
        )


def gdn_linear_attention(
    q, k, v, g, beta, state_pool, cu_seqlens, slot_mapping, y, has_initial_state
):
    """Enqueue GDN on PyTorch's MPS stream; mutate fp32 state and write Y.

    Q/K are packed [T,Hk,Dk], V/Y [T,Hv,Dv], g/beta [T,Hv]. Inputs share an
    fp32/fp16/bf16 dtype; Y matches it or is fp16/bf16 with fp32 inputs.
    Q/K must already be normalized and Q scaled. The
    shader consumes the decay exp(log_g), not the log-space gate.
    State is [slots,Hv,Dv,Dk], with dense inner rows and an optional slot gap.

    Metadata is packed int32 on MPS. Callers supply monotonic cu_seqlens from
    0 to T and distinct valid slots or PAD (-1); metadata values are not read
    back to the CPU. PAD and empty sequences leave state and Y untouched.
    has_initial_state is packed bool [B] on MPS; false starts from zero in
    the same launch without reading the old slot contents.
    Inputs, state and Y must not overlap. Autograd is not supported.
    """
    _validate_gdn(
        q, k, v, g, beta, state_pool, cu_seqlens, slot_mapping, y, has_initial_state
    )
    if slot_mapping.numel() == 0 or q.shape[0] == 0:
        return
    hk, dk = q.shape[1:]
    hv, dv = v.shape[1:]
    _, kernel = _gdn_kernel(q.dtype, y.dtype)
    kernel(
        q,
        k,
        v,
        g,
        beta,
        state_pool,
        cu_seqlens,
        slot_mapping,
        y,
        slot_mapping.numel(),
        hk,
        hv,
        dk,
        dv,
        state_pool.stride(0),
        has_initial_state,
        threads=(32 * dv, 1, slot_mapping.numel() * hv),
        group_size=(32, 1, 1),
        # Python ints default to int64; only the slot stride is int64 in the shader.
        arg_casts=dict.fromkeys(range(9, 14), "int32"),
    )
