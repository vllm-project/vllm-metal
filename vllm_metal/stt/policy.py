# SPDX-License-Identifier: Apache-2.0
"""STT-specific scheduler policy at the platform boundary."""

from __future__ import annotations

from typing import Protocol

# Nominal memory reported to vLLM scheduler for STT models.
# No KV cache is actually allocated; this just passes minimum-memory checks.
STT_SCHED_AVAILABLE_BYTES = 1 << 30  # 1 GiB

# Block size reported to vLLM for STT models (minimal, no real KV cache).
STT_SCHED_BLOCK_BYTES = 1


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
