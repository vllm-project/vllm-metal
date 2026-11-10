# SPDX-License-Identifier: Apache-2.0
"""Logits processors used by Metal's torch sampler bridge.

Owns the minimal processor interface, the ``BatchUpdate`` it receives, and
the ``LogitsProcessors`` container that the vendored sampler drives,
mirroring the removed V1 ``vllm.v1.sample.logits_processor`` API surface
vllm-metal used (``is_argmax_invariant()`` splitting into
``argmax_invariant`` / ``non_argmax_invariant`` lists,
vllm-project/vllm#63373). ``MoveDirectionality`` and ``BatchUpdate`` are
vendored from vllm b46907cce0f7 (== v0.33.0)
vllm/v1/sample/logits_processor/interface.py (Apache-2.0).
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from enum import Enum, auto
from typing import Protocol

import torch
from vllm import SamplingParams


class MoveDirectionality(Enum):
    # One-way i1->i2 req move within batch
    UNIDIRECTIONAL = auto()
    # Two-way i1<->i2 req swap within batch
    SWAP = auto()


# Batch indices of any removed requests.
RemovedRequest = int

# (index, params, prompt_tok_ids, output_tok_ids) tuples for new
# requests added to the batch.
AddedRequest = tuple[int, SamplingParams, list[int] | None, list[int]]

# (index 1, index 2, directionality) tuples representing
# one-way moves or two-way swaps of requests in batch
MovedRequest = tuple[int, int, MoveDirectionality]


@dataclass(frozen=True)
class BatchUpdate:
    """Persistent batch state change info for logitsprocs."""

    batch_size: int  # Current num reqs in batch

    # Metadata for requests added to, removed from, and moved
    # within the persistent batch.
    #
    # Key assumption: the `output_tok_ids` list (which is an element of each
    # tuple in `added`) is a reference to the request's running output tokens
    # list; via this reference, the logits processors always see the latest
    # list of generated output tokens.
    #
    # NOTE:
    # * Added or moved requests may replace existing requests with the same
    #   index.
    # * Operations should be processed in the following order:
    #   - removed, added, moved
    removed: Sequence[RemovedRequest]
    added: Sequence[AddedRequest]
    moved: Sequence[MovedRequest]


class LogitsProcessor(Protocol):
    """The per-step interface the sampler drives a processor through."""

    def is_argmax_invariant(self) -> bool: ...

    def update_state(self, batch_update: BatchUpdate | None) -> None: ...

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

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        """No-op because ``SamplingBatch`` is rebuilt per step."""

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        threshold = probs.amax(dim=-1, keepdim=True).mul_(self.min_p)
        return logits.masked_fill(probs < threshold, float("-inf"))
