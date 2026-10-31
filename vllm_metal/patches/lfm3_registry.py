# SPDX-License-Identifier: Apache-2.0
"""LFM3 scheduler metadata; mlx-vlm supplies the executable model."""

from typing import TYPE_CHECKING

from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.model_executor.models.lfm2 import Lfm2ForCausalLM

if TYPE_CHECKING:
    from vllm.config import VllmConfig


# TODO: Once the pinned vLLM registers Lfm3ForCausalLM with native hybrid
# capabilities and ShortConv shape/dtype/copy hooks, remove this shim and its
# LFM3 branch in compat.register_model_compat. Keep the shared hook and
# metal_model_metadata entry point while Flint or other fallbacks need them.
# Re-run batched and prefix-cache LFM3 parity against the native registration.
class Lfm3ForCausalLM(Lfm2ForCausalLM):
    """Reuse LFM2's hybrid capabilities and short-convolution state policy."""

    def __init__(self, *, vllm_config: "VllmConfig", prefix: str = "") -> None:
        raise NotImplementedError("LFM3 execution requires the Metal MLX model runner.")

    @classmethod
    def get_mamba_state_shape_from_config(
        cls, vllm_config: "VllmConfig"
    ) -> tuple[tuple[int, int]]:
        config = vllm_config.model_config.hf_text_config
        return MambaStateShapeCalculator.short_conv_state_shape(
            tp_world_size=vllm_config.parallel_config.tensor_parallel_size,
            intermediate_size=config.hidden_size,
            conv_kernel=config.conv_L_cache,
        )
