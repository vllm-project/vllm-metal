# SPDX-License-Identifier: Apache-2.0
"""Logits processors used by Metal's torch sampler bridge.

Owns the minimal processor interface and the ``LogitsProcessors``
container that the vendored sampler drives, mirroring the removed V1
``vllm.v1.sample.logits_processor`` API surface vllm-metal used
(``is_argmax_invariant()`` splitting into ``argmax_invariant`` /
``non_argmax_invariant`` lists, vllm-project/vllm#63373).
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Protocol

import torch


class LogitsProcessor(Protocol):
    """The per-step interface the sampler drives a processor through."""

    def is_argmax_invariant(self) -> bool: ...

    def update_state(self, batch_update: object | None) -> None: ...

    def apply(self, logits: torch.Tensor) -> torch.Tensor: ...


class LogitsProcessors:
    """Encapsulates initialized logitsproc objects."""

    def __init__(self, logitsprocs: Iterable[LogitsProcessor] | None = None) -> None:
        self.argmax_invariant: list[LogitsProcessor] = []
        self.non_argmax_invariant: list[LogitsProcessor] = []
        if logitsprocs:
            for logitproc in logitsprocs:
                (
                    self.argmax_invariant
                    if logitproc.is_argmax_invariant()
                    else self.non_argmax_invariant
                ).append(logitproc)


class BatchMinPLogitsProcessor:
    """Per-step ``min_p`` mask for ``SamplingBatch``."""

    def __init__(self, min_p: torch.Tensor) -> None:
        self.min_p = min_p.unsqueeze(-1)

    def is_argmax_invariant(self) -> bool:
        """Apply after temperature and before top-k/top-p."""
        return True

    def update_state(self, batch_update: object | None) -> None:
        """No-op because ``SamplingBatch`` is rebuilt per step."""

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        threshold = probs.amax(dim=-1, keepdim=True).mul_(self.min_p)
        return logits.masked_fill(probs < threshold, float("-inf"))
