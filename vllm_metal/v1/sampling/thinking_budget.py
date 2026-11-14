# SPDX-License-Identifier: Apache-2.0
"""Minimal protocol for thinking-token-budget state.

The vendored sampler drives the holder through exactly these calls
(``has_tracked_requests`` gate, ``update_state`` bookkeeping, and the
``apply_to_logits`` mask). Upstream's concrete holder lives in a module
vllm-project/vllm#63373 removes, so a thinking-budget implementation
only needs to satisfy this protocol; vllm-metal passes ``None`` today.
"""

from __future__ import annotations

from typing import Protocol

import torch


class ThinkingBudgetState(Protocol):
    def has_tracked_requests(self) -> bool: ...

    def update_state(
        self,
        output_token_ids: list[list[int]],
        spec_token_ids: list[list[int]] | None,
        repeat_indices: torch.Tensor | None = None,
    ) -> None: ...

    def apply_to_logits(
        self,
        logits: torch.Tensor,
        predict_bonus_token: bool,
        spec_token_ids: list[list[int]] | None,
    ) -> torch.Tensor: ...
