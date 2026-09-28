# SPDX-License-Identifier: Apache-2.0
"""Experimental PyTorch-stream launcher; no new attention shaders."""

import re
import subprocess
from functools import cache
from pathlib import Path


@cache
def get_mps_ops():
    from torch.utils.cpp_extension import load

    from vllm_metal.metal import get_ops

    # Reuse the existing artifact validation/build policy and hardware gate.
    metal_ops = get_ops()
    root = Path(__file__).resolve().parent
    module = load(
        name="vllm_metal_mps_ops",
        sources=[str(root / "mps_ops.mm")],
        extra_cflags=["-O3"],
        extra_ldflags=["-framework", "Metal", "-framework", "Foundation"],
    )
    registry = subprocess.check_output(
        ["ioreg", "-r", "-c", "AGXAccelerator", "-d", "1"], text=True
    )
    cores = re.search(r'"gpu-core-count"\s*=\s*(\d+)', registry)
    metal_dir = root.parent / "metal"
    return module.PagedAttention(
        str(metal_dir / "paged_attention_v2_kern.metallib"),
        str(metal_dir / "paged_attention_nax_kern.metallib")
        if metal_ops.nax_ready()
        else "",
        int(cores[1]) if cores else 14,
    )
