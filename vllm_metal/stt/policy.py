# SPDX-License-Identifier: Apache-2.0
"""STT-specific scheduler policy at the platform boundary."""

from __future__ import annotations

from typing import Protocol


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
