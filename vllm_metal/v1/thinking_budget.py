# SPDX-License-Identifier: Apache-2.0
"""Thinking token budget for the Metal sampler.

Mirrors vLLM Model Runner V2 (``vllm/v1/worker/gpu/sample/thinking_budget.py``):
the forced token is a function of the request's token ids alone. Find the last
reasoning-start marker; when no natural reasoning-end marker follows it and at
least ``thinking_token_budget`` tokens come after it, force the next token of
the reasoning-end sequence. Reasoning tokens already in a resumed prompt count
against the budget, and a tail that already holds a prefix of the end sequence
continues from the next marker token.

Metal rebuilds its sampling batch every step from the full token ids, so no
per-request state is kept; MRv2 caches marker positions only to avoid
rescanning on the GPU.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

# Same fill value as vLLM, large enough to survive penalties and temperature.
_FORCED_LOGIT = 1.0e9


@dataclass(frozen=True)
class ThinkingBudgetMarkers:
    """Token ids of the reasoning markers from vLLM's ``ReasoningConfig``."""

    start: tuple[int, ...]
    end: tuple[int, ...]
    natural_end: tuple[int, ...]

    @classmethod
    def from_reasoning_config(
        cls, reasoning_config: Any | None
    ) -> "ThinkingBudgetMarkers | None":
        """Return the markers, or ``None`` when budgets cannot be enforced.

        As in MRv2, all three marker sequences are required.
        """
        if reasoning_config is None:
            return None
        start = reasoning_config.reasoning_start_token_ids or []
        end = reasoning_config.reasoning_end_token_ids or []
        natural_end = reasoning_config.natural_reasoning_end_token_ids or []
        if not (start and end and natural_end):
            return None
        return cls(tuple(start), tuple(end), tuple(natural_end))


def _rfind(token_ids: np.ndarray, pattern: tuple[int, ...]) -> int:
    """Start index of the last occurrence of ``pattern``, or -1."""
    num_starts = token_ids.shape[0] - len(pattern) + 1
    if num_starts <= 0:
        return -1
    match = token_ids[:num_starts] == pattern[0]
    for offset in range(1, len(pattern)):
        match &= token_ids[offset : offset + num_starts] == pattern[offset]
    (positions,) = np.nonzero(match)
    return int(positions[-1]) if positions.size else -1


def forced_end_token(
    token_ids: Sequence[int],
    budget: int,
    markers: ThinkingBudgetMarkers,
) -> int | None:
    """Return the token to force next, or ``None`` to sample freely."""
    tokens = np.asarray(token_ids, dtype=np.int64)
    last_start = _rfind(tokens, markers.start)
    if last_start < 0:
        return None
    if _rfind(tokens, markers.natural_end) >= last_start:
        return None
    num_reasoning_tokens = tokens.shape[0] - (last_start + len(markers.start))
    if num_reasoning_tokens < budget:
        return None

    end_prefix_len = 0
    for prefix_len in range(1, min(len(markers.end) - 1, tokens.shape[0]) + 1):
        if tuple(tokens[-prefix_len:].tolist()) == markers.end[:prefix_len]:
            end_prefix_len = prefix_len
    return markers.end[end_prefix_len]


def forced_end_tokens(
    budgets: Sequence[int | None],
    prompt_token_id_lists: Sequence[list[int]],
    output_token_id_lists: Sequence[list[int]],
    markers: ThinkingBudgetMarkers,
) -> dict[int, int]:
    """Map batch row to forced token for the rows over their budget."""
    forced: dict[int, int] = {}
    for row, budget in enumerate(budgets):
        if budget is None:
            continue
        token = forced_end_token(
            prompt_token_id_lists[row] + output_token_id_lists[row],
            budget,
            markers,
        )
        if token is not None:
            forced[row] = token
    return forced


class ForcedEndTokens(ThinkingBudgetStateHolder):
    """Precomputed forced tokens handed to vLLM's ``Sampler``.

    The sampler applies its thinking-budget holder after penalties and before
    sampling, with raw logprobs already taken, so forcing here keeps the same
    order as upstream. All state lives in the precomputed ``row -> token`` map.
    """

    def __init__(self, forced: dict[int, int]) -> None:  # noqa: D107
        # The upstream constructor builds incremental state this class
        # replaces, so it is deliberately not called.
        self._forced = forced

    def has_tracked_requests(self) -> bool:
        return bool(self._forced)

    def update_state(self, *args: Any, **kwargs: Any) -> None:
        return None

    def apply_to_logits(
        self,
        logits: torch.Tensor,
        predict_bonus_token: bool,
        spec_token_ids: list[list[int]] | None,
    ) -> torch.Tensor:
        rows = torch.tensor(list(self._forced), dtype=torch.long)
        tokens = torch.tensor(list(self._forced.values()), dtype=torch.long)
        logits[rows.to(logits.device), tokens.to(logits.device)] = _FORCED_LOGIT
        return logits
