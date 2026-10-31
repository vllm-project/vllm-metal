# SPDX-License-Identifier: Apache-2.0
"""Bridge vLLM's GPU runner validation for the Metal backends.

Remove when upstream delegates the runner's hardware requirements to platforms.
"""

from functools import wraps


def patch_metal_model_runner_validation() -> None:
    """Keep upstream feature checks without requiring Triton on Metal."""
    from vllm.config import VllmConfig

    # vllm#63373 renames the validator and its feature list together.
    names = {
        "_validate_v2_model_runner": "_get_v2_model_runner_unsupported_features",
        "_validate_model_runner": "_get_model_runner_unsupported_features",
    }
    validator_name = next((n for n in names if hasattr(VllmConfig, n)), None)
    if validator_name is None:
        raise RuntimeError(
            "Expected VllmConfig._validate_v2_model_runner or "
            "VllmConfig._validate_model_runner; vllm-metal needs updating "
            "for this vLLM version."
        )
    original = getattr(VllmConfig, validator_name)
    features_name = names[validator_name]
    if getattr(original, "_metal_mrv2", False):
        return

    @wraps(original)
    def validate(config: VllmConfig) -> None:
        from vllm.platforms import current_platform

        from vllm_metal.platform import MetalPlatform

        if not isinstance(current_platform, MetalPlatform):
            return original(config)
        # Metal backends replace Triton operations, but retain feature checks.
        unsupported = getattr(config, features_name)()
        if unsupported:
            raise ValueError(
                f"Model Runner V2 on Metal does not support: {', '.join(unsupported)}"
            )

    validate._metal_mrv2 = True  # type: ignore[attr-defined]
    setattr(VllmConfig, validator_name, validate)
