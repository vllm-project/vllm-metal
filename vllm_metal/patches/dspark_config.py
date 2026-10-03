# SPDX-License-Identifier: Apache-2.0
"""Bridge vLLM's GPU-runner-only DSpark check to the native Metal runner."""

from functools import wraps


def enable_dspark_for_metal_runner() -> None:
    """Keep V1 validation except the DSpark ban for our own worker.

    vLLM 0.30's V1 GPU runner has no DSpark implementation. MetalWorker uses
    its own runner and proposer. Remove this bridge when vLLM lets out-of-tree
    runners declare speculative-method support instead of applying GPU rules.
    Install from platform config validation, after VllmConfig is fully imported.
    """
    from vllm.config import VllmConfig

    original = VllmConfig._get_v1_model_runner_unsupported_features
    if getattr(original, "_metal_dspark", False):
        return

    @wraps(original)
    def unsupported_features(self: VllmConfig) -> list[str]:
        unsupported = original(self)
        if (
            self.parallel_config.worker_cls == "vllm_metal.v1.worker.MetalWorker"
            and self.speculative_config is not None
            and self.speculative_config.method == "dspark"
        ):
            return [
                item for item in unsupported if item != "dspark speculative decoding"
            ]
        return unsupported

    unsupported_features._metal_dspark = True  # type: ignore[attr-defined]
    VllmConfig._get_v1_model_runner_unsupported_features = unsupported_features
