# SPDX-License-Identifier: Apache-2.0
"""Experimental PyTorch-stream launcher; no new attention shaders."""

import logging
from functools import cache
from pathlib import Path

logger = logging.getLogger(__name__)


@cache
def _load_mps_module():
    from torch.utils.cpp_extension import load

    from vllm_metal.metal.constants import PA_WINDOW_MAX_HEAD_SIZE, PA_WINDOW_ROWS

    root = Path(__file__).resolve().parent
    return load(
        name="vllm_metal_mps_ops",
        sources=[str(root / "mps_ops.mm")],
        extra_cflags=[
            "-O3",
            f"-DVLLM_METAL_PA_WINDOW_ROWS={PA_WINDOW_ROWS}",
            f"-DVLLM_METAL_PA_WINDOW_MAX_HEAD={PA_WINDOW_MAX_HEAD_SIZE}",
        ],
        extra_ldflags=[
            "-framework",
            "Metal",
            "-framework",
            "Foundation",
            "-framework",
            "IOKit",
            "-framework",
            "CoreFoundation",
        ],
    )


@cache
def get_mps_ops():
    from vllm_metal import envs
    from vllm_metal.metal.build import NAX_METALLIB_NAME, prepare_metallib

    module = _load_mps_module()
    # The shared detector is compiled into each launcher; MPS never loads MLX.
    gpu_cores = module.detected_gpu_core_count()
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
