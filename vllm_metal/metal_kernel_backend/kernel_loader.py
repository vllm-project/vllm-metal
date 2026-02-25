# SPDX-License-Identifier: Apache-2.0
"""Load the HuggingFace paged-attention Metal kernel.

Uses ``kernels.get_kernel()`` to fetch the community paged-attention shader
which provides ``reshape_and_cache`` and ``paged_attention_v1`` ops that run
natively on Apple Metal (MPS).

The latest HF builds (Jan 2026+) are compiled with Metal language version 4.0
which requires macOS 16 (Tahoe).  On macOS 15 (Sequoia) and earlier, we pin
to the Nov 2025 build (commit ``8968951``) which targets Metal 3.2.
"""

from __future__ import annotations

import logging
import platform
from typing import Any

logger = logging.getLogger(__name__)

_kernel: Any = None

# Latest HF build requires Metal 4.0 (macOS 16+).  This older revision
# was built with Metal 3.2 and works on macOS 15 and earlier.
_MACOS15_COMPAT_REVISION = "8968951"


def _needs_compat_revision() -> bool:
    """Return True when the current macOS only supports Metal < 4.0."""
    ver = platform.mac_ver()[0]
    if not ver:
        return False
    major = int(ver.split(".")[0])
    return major <= 15


def get_paged_attention_ops() -> Any:
    """Return the loaded paged-attention kernel module.

    The module exposes at minimum:
        - ``reshape_and_cache(...)``
        - ``paged_attention_v1(...)``

    The kernel is loaded once and cached for subsequent calls.
    """
    global _kernel
    if _kernel is None:
        try:
            from kernels import get_kernel
        except ImportError:
            raise ImportError(
                "Paged attention requires the 'kernels' package. "
                "Install it with:  pip install 'vllm-metal[paged]'"
            ) from None

        revision = _MACOS15_COMPAT_REVISION if _needs_compat_revision() else None
        _kernel = get_kernel("kernels-community/paged-attention", revision=revision)
        if revision:
            logger.info(
                "Loaded HF paged-attention Metal kernel (compat revision %s)",
                revision,
            )
        else:
            logger.info("Loaded HF paged-attention Metal kernel")
    return _kernel
