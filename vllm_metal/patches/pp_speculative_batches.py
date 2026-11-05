# SPDX-License-Identifier: Apache-2.0
"""Keep one batch in flight when drafting under synchronous pipeline parallelism."""

from typing import Any


def run_one_batch_when_drafting_under_pp() -> None:
    """Make ``VllmConfig.max_concurrent_batches`` 1 for speculative decoding
    under synchronous pipeline parallelism.

    With ``pipeline_parallel_size`` batches in flight, vLLM's engine takes a
    step's drafts in ``post_step`` before it processes that step's sampled
    tokens, so the next ``schedule()`` gives a request its drafts without its
    sampled token. With one batch in flight the sampled token lands first.
    Install from a general plugin: the engine core process reads the property
    when it starts, and the platform's config check runs in the API process
    only.
    """
    from vllm.config import VllmConfig

    original = VllmConfig.max_concurrent_batches.fget
    if getattr(original, "_metal_one_batch_when_drafting", False):
        return

    def max_concurrent_batches(self: Any) -> int:
        if (
            self.speculative_config is not None
            and self.parallel_config.pipeline_parallel_size > 1
            and not self.scheduler_config.async_scheduling
        ):
            return 1
        return original(self)

    max_concurrent_batches._metal_one_batch_when_drafting = True  # type: ignore[attr-defined]
    VllmConfig.max_concurrent_batches = property(max_concurrent_batches)  # type: ignore[method-assign,assignment]
