# SPDX-License-Identifier: Apache-2.0
"""Metal utility functions for vLLM Metal plugin."""

import logging

logger = logging.getLogger(__name__)


def set_wired_limit() -> None:
    """Set Metal wired memory limit for optimal GPU performance.

    Pins model weights in GPU-accessible memory to prevent memory paging
    and GPU stalls during inference.

    See: https://github.com/ml-explore/mlx-lm/pull/652
    """
    try:
        import mlx.core as mx

        device_info = mx.metal.device_info()
        max_wired = int(device_info.get("max_recommended_working_set_size", 0))
        if max_wired > 0:
            if hasattr(mx, "set_wired_limit"):
                mx.set_wired_limit(max_wired)
            elif hasattr(mx.metal, "set_wired_limit"):
                mx.metal.set_wired_limit(max_wired)
            logger.info(f"Set Metal wired_limit to {max_wired / (1024**3):.1f} GB")
    except Exception as e:
        logger.warning(f"Failed to set wired_limit: {e}")
