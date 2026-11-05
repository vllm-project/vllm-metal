# SPDX-License-Identifier: Apache-2.0
"""Bridge vLLM's GPU-runner-only DSpark check to the native Metal runner."""

from functools import wraps
from typing import Any

from vllm_metal.patches.v1_runner_guard import allow_v1_runner_feature


def _serves_dspark(vllm_config: Any) -> bool:
    return (
        vllm_config.parallel_config.worker_cls == "vllm_metal.v1.worker.MetalWorker"
        and vllm_config.speculative_config is not None
        and vllm_config.speculative_config.method == "dspark"
    )


def enable_dspark_for_metal_runner() -> None:
    """Keep V1 validation except the DSpark ban for our own worker.

    vLLM 0.30's V1 GPU runner has no DSpark implementation. MetalWorker uses
    its own runner and proposer. Remove this bridge when vLLM lets out-of-tree
    runners declare speculative-method support instead of applying GPU rules.
    Install from platform config validation, after VllmConfig is fully imported.
    """
    allow_v1_runner_feature("dspark speculative decoding", _serves_dspark)
    allow_v1_runner_feature("adaptive draft verification", _serves_dspark)
    _allow_metal_adaptive_verification()


def _allow_metal_adaptive_verification() -> None:
    from vllm.config import VllmConfig

    original = VllmConfig._validate_adaptive_verification
    if getattr(original, "_metal_dspark_adaptive", False):
        return

    @wraps(original)
    def validate(self: Any) -> None:
        if (
            not _serves_dspark(self)
            or not self.speculative_config.enable_adaptive_verification
        ):
            return original(self)
        # Metal returns admitted prefixes on the CPU before scheduling. It
        # does not use the GPU runner's placeholder boundaries or CUDA graphs.
        if self.lora_config is not None:
            raise ValueError(
                "Adaptive verification is not currently compatible with LoRA"
            )
        if self.parallel_config.pipeline_parallel_size > 1:
            raise ValueError(
                "Adaptive verification is not currently compatible with pipeline parallelism"
            )

    validate._metal_dspark_adaptive = True  # type: ignore[attr-defined]
    VllmConfig._validate_adaptive_verification = validate
