# SPDX-License-Identifier: Apache-2.0
"""vLLM logits processors for the Metal v1 torch sampler path.

``SamplingBatch`` hands these to ``SamplingMetadata.logitsprocs`` so that
``Sampler`` applies them in upstream order, rather than masking logits
before ``Sampler.forward``.
"""

import torch
from vllm.v1.sample.logits_processor import BatchUpdate, LogitsProcessor


class BatchMinPLogitsProcessor(LogitsProcessor):
    """Per-row ``min_p`` mask for the torch sampler path.

    vLLM's own ``MinPLogitsProcessor`` tracks a persistent batch through
    ``BatchUpdate`` deltas; the Metal ``SamplingBatch`` is rebuilt from
    scratch each step, so this variant takes the row values directly.

    Registered as argmax-invariant so ``Sampler`` applies it after
    temperature and before top-k/top-p — the upstream order. Masking ahead
    of ``Sampler.forward`` instead would threshold un-scaled probabilities
    and leave ``-inf`` in the raw logprobs the sampler reports.
    """

    def __init__(self, min_p: torch.Tensor) -> None:
        self.min_p = min_p.unsqueeze(-1)

    def is_argmax_invariant(self) -> bool:
        """Min-p never impacts greedy sampling."""
        return True

    def update_state(self, batch_update: BatchUpdate | None) -> None:
        """No-op: the batch is rebuilt per step, so there are no deltas."""

    def apply(self, logits: torch.Tensor) -> torch.Tensor:
        probs = torch.softmax(logits, dim=-1)
        threshold = probs.amax(dim=-1, keepdim=True).mul_(self.min_p)
        return logits.masked_fill(probs < threshold, float("-inf"))
