# SPDX-License-Identifier: Apache-2.0
"""STT-specific scheduler and request policy at the platform boundary."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams

# Nominal memory reported to vLLM scheduler for STT models.
# No KV cache is actually allocated; this just passes minimum-memory checks.
STT_SCHED_AVAILABLE_BYTES = 1 << 30  # 1 GiB

# Block size reported to vLLM for STT models (minimal, no real KV cache).
STT_SCHED_BLOCK_BYTES = 1

# Nominal head size for the placeholder KV spec used only to satisfy
# vLLM scheduler initialization for STT models.
STT_SCHED_NOMINAL_HEAD_SIZE = 64


class _ModelConfigLike(Protocol):
    model: str
    tokenizer: str | None


class _SchedulerConfigLike(Protocol):
    async_scheduling: bool


def apply_stt_scheduler_policy(
    model_config: _ModelConfigLike, scheduler_config: _SchedulerConfigLike
) -> None:
    """Apply STT scheduler compatibility policy for Metal runtime.

    STT requests are processed as one-shot execute calls, so async scheduling
    (which expects decode-phase queuing) must be disabled.
    """
    if not model_config.tokenizer:
        model_config.tokenizer = model_config.model
    if scheduler_config.async_scheduling:
        scheduler_config.async_scheduling = False


def reject_non_greedy_stt_sampling(params: SamplingParams) -> None:
    """Reject a transcription request the one-shot STT decode cannot honour."""
    # Imported here so this module stays importable without vLLM's sampling
    # stack, which pulls in torch and mlx.
    from vllm.exceptions import VLLMValidationError
    from vllm.sampling_params import SamplingType

    if params.sampling_type is SamplingType.GREEDY:
        return
    raise VLLMValidationError(
        "vllm-metal transcribes with greedy decoding, so speech-to-text "
        "requests require temperature=0.",
        parameter="temperature",
        value=params.temperature,
    )
