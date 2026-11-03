# SPDX-License-Identifier: Apache-2.0
"""Experimental PyTorch-stream launcher; no new attention shaders."""

import importlib
import logging
from functools import cache

logger = logging.getLogger(__name__)


@cache
def _load_mps_module():
    import torch  # Load libtorch before importing the linked extension.

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
def get_mps_gdn_ops():
    from vllm_metal import envs
    from vllm_metal.metal.build import prepare_metallib

    module = _load_mps_module()
    path = prepare_metallib(
        "gdn_kern", build_from_source=envs.VLLM_METAL_BUILD_FROM_SOURCE
    )
    return module.GDNLinearAttention(str(path))


def gdn_linear_attention(
    q, k, v, g, beta, state_pool, cu_seqlens, slot_mapping, y, has_initial_state
):
    """Enqueue GDN on PyTorch's MPS stream; mutate fp32 state and write Y.

    Q/K are packed [T,Hk,Dk], V/Y [T,Hv,Dv], g/beta [T,Hv], all the same
    fp32/fp16/bf16 dtype. Q/K must already be normalized and Q scaled. The
    shader consumes the decay exp(log_g), not the log-space gate.
    State is [slots,Hv,Dv,Dk], with dense inner rows and an optional slot gap.

    Metadata is packed int32 on MPS. Callers supply monotonic cu_seqlens from
    0 to T and distinct valid slots or PAD (-1); metadata values are not read
    back to the CPU. PAD and empty sequences leave state and Y untouched.
    has_initial_state is packed bool [B] on MPS; false starts from zero in
    the same launch without reading the old slot contents.
    Inputs, state and Y must not overlap. Autograd is not supported.
    """
    get_mps_gdn_ops().gdn_linear_attention(
        q, k, v, g, beta, state_pool, cu_seqlens, slot_mapping, y, has_initial_state
    )
