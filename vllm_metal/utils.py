# SPDX-License-Identifier: Apache-2.0
"""Metal utility functions for vLLM Metal plugin."""

import logging
import os
from pathlib import Path

logger = logging.getLogger(__name__)


def get_model_download_path(model_repo_name: str) -> str:
    """
    Get the path to the model, downloading from ModelScope if configured, otherwise will pass the model_repo_name.

    Args:
        model_repo_name: Model repo name from HuggingFace or ModelScope

    Returns:
        Local folder path (string) of repo snapshot

    Example:

    model_repo_name example: mlx-community/Qwen2.5-0.5B-4bit

    Spin up the server:

    ```bash
    export VLLM_USE_MODELSCOPE=True && export VLLM_METAL_MODELSCOPE_CACHE=/path/of/model/cache/to/be/downloaded &&  vllm-metal -m mlx-community/Qwen2.5-0.5B-4bit --host 0.0.0.0 --port 8000 --log-level debug
    ```
    """
    if Path(model_repo_name).exists():
        return model_repo_name

    if os.environ.get("VLLM_USE_MODELSCOPE", "False").lower() == "true":
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            model_cache_dir = os.environ.get("VLLM_METAL_MODELSCOPE_CACHE")

            logger.info(f"Downloading model {model_repo_name} from ModelScope...")
            model_path = snapshot_download(model_repo_name, cache_dir=model_cache_dir)
            logger.info(f"Model downloaded to {model_path}")
            return str(model_path)
        except ImportError:
            logger.warning(
                "modelscope not installed, falling back to default loader (HuggingFace)"
            )
        except Exception as e:
            logger.warning(f"Failed to download from ModelScope: {e}")

    # Fallback: Let mlx_lm or mlx_vlm handle the download natively from HuggingFace
    return model_repo_name


def set_wired_limit() -> None:
    """
    Set Metal wired memory limit for optimal GPU performance.

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
