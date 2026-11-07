# SPDX-License-Identifier: Apache-2.0
"""Calibrated prefix admission for greedy DSpark verification."""

from __future__ import annotations

import json
import math
from bisect import bisect_right
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any


def _positive(values: Any, count: int) -> tuple[float, ...]:
    if (
        not isinstance(values, list)
        or len(values) != count
        or any(
            type(value) not in (int, float) or not math.isfinite(value) or value <= 0
            for value in values
        )
    ):
        raise ValueError(f"DSpark profile requires {count} finite positive values")
    return tuple(float(value) for value in values)


@dataclass(frozen=True)
class VerificationCosts:
    context: int
    draft_ms: tuple[float, ...]
    # Row b-1 prices a batch of b requests with 0..b*K admitted draft tokens.
    verify_ms: tuple[tuple[float, ...], ...]


@dataclass(frozen=True)
class AdaptiveVerificationProfile:
    temperatures: tuple[float, ...]
    costs: tuple[VerificationCosts, ...]

    @classmethod
    def load(
        cls,
        path: str,
        *,
        identity: dict[str, Any],
        width: int,
        max_num_seqs: int,
    ) -> AdaptiveVerificationProfile:
        data = json.loads(Path(path).read_text())
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Unsupported DSpark adaptive profile version")
        if data.get("identity") != identity:
            raise ValueError(
                "DSpark adaptive profile does not match this model/runtime"
            )
        temperatures = _positive(data.get("temperatures"), width)
        tables = data.get("costs")
        if not isinstance(tables, list) or not tables:
            raise ValueError("DSpark profile requires measured verification costs")
        costs = []
        previous = -1
        for table in tables:
            if not isinstance(table, dict):
                raise ValueError("Invalid DSpark cost table")
            context = table.get("context")
            if type(context) is not int or context <= previous:
                raise ValueError(
                    "DSpark profile contexts must increase from zero or above"
                )
            draft = _positive(table.get("draft_ms"), max_num_seqs)
            verify = table.get("verify_ms")
            if not isinstance(verify, list) or len(verify) != max_num_seqs:
                raise ValueError(
                    "DSpark profile must cover every configured batch size"
                )
            costs.append(
                VerificationCosts(
                    context,
                    draft,
                    tuple(
                        _positive(row, batch * width + 1)
                        for batch, row in enumerate(verify, start=1)
                    ),
                )
            )
            previous = context
        return cls(temperatures, tuple(costs))

    def admit(
        self,
        logits: Sequence[Sequence[float]],
        capacities: Sequence[int],
        context: float,
    ) -> list[int]:
        """Choose balanced prefixes using measured draft-and-target time.

        Each conditional confidence contributes its prefix survival probability.
        The cost table measures balanced lengths, so grow a common prefix and
        allocate its next position by confidence. Search every feasible budget:
        Metal dispatch costs need not be convex.
        This chooses work only; the unchanged target verifier decides acceptance.
        """
        batch, width = len(capacities), len(self.temperatures)
        if not 1 <= batch <= len(self.costs[0].draft_ms) or len(logits) != batch:
            raise ValueError("DSpark adaptive batch is outside its measured profile")
        if not math.isfinite(context) or context < 0:
            raise ValueError("DSpark adaptive context must be finite and nonnegative")
        probabilities = []
        for row, capacity in zip(logits, capacities, strict=True):
            if (
                len(row) != width
                or type(capacity) is not int
                or not 0 <= capacity <= width
            ):
                raise ValueError(
                    "DSpark adaptive capacity does not match the draft block"
                )
            survival = 1.0
            prefix = []
            for value, temperature in zip(row, self.temperatures, strict=True):
                if not math.isfinite(value):
                    raise ValueError("DSpark confidence logits must be finite")
                # Stable sigmoid; unlike clipping it preserves calibrated tails.
                scaled = value / temperature
                exponential = math.exp(-abs(scaled))
                probability = (
                    1 / (1 + exponential)
                    if scaled >= 0
                    else exponential / (1 + exponential)
                )
                survival *= probability
                prefix.append(survival)
            probabilities.append(prefix)
        contexts = [table.context for table in self.costs]
        upper = min(bisect_right(contexts, context), len(contexts) - 1)
        lower = max(upper - 1, 0)
        lo, hi = self.costs[lower], self.costs[upper]
        mix = (
            0.0
            if lo.context == hi.context
            else min(1.0, max(0.0, (context - lo.context) / (hi.context - lo.context)))
        )
        draft = lo.draft_ms[batch - 1] * (1 - mix) + hi.draft_ms[batch - 1] * mix
        costs = [
            draft + a * (1 - mix) + b * mix
            for a, b in zip(
                lo.verify_ms[batch - 1], hi.verify_ms[batch - 1], strict=True
            )
        ]
        admitted = [0] * batch
        best = admitted.copy()
        expected, best_rate, budget = float(batch), batch / costs[0], 0
        for position in range(min(capacities) + 1):
            ranked = sorted(
                (-row[position], request)
                for request, (row, capacity) in enumerate(
                    zip(probabilities, capacities, strict=True)
                )
                if position < capacity
            )
            for negative_survival, request in ranked:
                admitted[request] += 1
                expected -= negative_survival
                budget += 1
                rate = expected / costs[budget]
                if rate > best_rate:
                    best_rate, best = rate, admitted.copy()
        return best
