# SPDX-License-Identifier: Apache-2.0
"""Load the HuggingFace paged-attention Metal kernel.

Uses ``kernels.get_kernel()`` to fetch the community paged-attention shader
which provides ``reshape_and_cache`` and ``paged_attention_v1`` ops that run
natively on Apple Metal (MPS).
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger(__name__)

_kernel: Any = None


def get_paged_attention_ops() -> Any:
    """Return the loaded paged-attention kernel module.

    The module exposes at minimum:
        - ``reshape_and_cache(...)``
        - ``paged_attention_v1(...)``

    The kernel is loaded once and cached for subsequent calls.
    """
    global _kernel
    if _kernel is None:
        from kernels import get_kernel

        _kernel = get_kernel("kernels-community/paged-attention")
        logger.info("Loaded HF paged-attention Metal kernel")
    return _kernel
