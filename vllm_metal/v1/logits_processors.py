# SPDX-License-Identifier: Apache-2.0
"""Logits processors used by Metal's torch sampler bridge."""

import torch
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor


class BatchMinPLogitsProcessor(LogitsProcessor):
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
