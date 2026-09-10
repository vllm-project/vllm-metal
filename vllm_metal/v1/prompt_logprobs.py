# SPDX-License-Identifier: Apache-2.0
"""Prompt logprobs accounting and tensor construction for the Metal runner."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import mlx.core as mx
import torch
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.sampler import Sampler

from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch

_LOGITS_LOGPROBS_MODES = ("raw_logits", "processed_logits")


@dataclass(frozen=True, slots=True)
class PromptLogprobsWindow:
    """Which rows of one prefill chunk produce prompt logprobs.

    The chunk forwards prompt positions ``[start_pos, start_pos + num_tokens)``.
    The row at position ``p`` predicts prompt token ``p + 1``, so the first
    ``num_logits`` rows of the chunk score prompt tokens
    ``[start_pos + 1, start_pos + 1 + num_logits)``; the row that predicts the
    first sampled token is not a prompt logprob.  ``completes`` is True when
    this chunk reaches the end of the prompt, which is when the accumulated
    tensors are handed to the engine.
    """

    start_pos: int
    num_logits: int
    completes: bool

    @property
    def first_target(self) -> int:
        return self.start_pos + 1


def prompt_logprobs_window(
    start_pos: int, num_tokens: int, prompt_len: int
) -> PromptLogprobsWindow:
    """Mirror of vLLM's ``_get_prompt_logprobs_dict`` chunk accounting."""
    num_remaining = prompt_len - (start_pos + 1)
    if num_tokens <= num_remaining:
        # A chunk with more prompt tokens to come; every row scores a prompt
        # token.  The == case has nothing left to score afterwards but the
        # engine still expects delivery on the completing step.
        return PromptLogprobsWindow(start_pos, num_tokens, completes=False)
    return PromptLogprobsWindow(start_pos, max(num_remaining, 0), completes=True)


class PromptLogprobsAccumulator:
    """Per-request ``LogprobsTensors`` filled one prefill chunk at a time."""

    def __init__(self, *, prompt_len: int, num_logprobs: int) -> None:
        if prompt_len < 1:
            raise ValueError("prompt_len must be at least 1")
        self.prompt_len = prompt_len
        self.num_logprobs = num_logprobs
        self.tensors = LogprobsTensors.empty_cpu(prompt_len - 1, num_logprobs + 1)

    def fill(self, window: PromptLogprobsWindow, chunk: LogprobsTensors) -> None:
        """Copy one chunk's rows into positions ``window.start_pos ...``."""
        start = window.start_pos
        end = start + window.num_logits
        self.tensors.logprob_token_ids[start:end].copy_(chunk.logprob_token_ids)
        self.tensors.logprobs[start:end].copy_(chunk.logprobs)
        self.tensors.selected_token_ranks[start:end].copy_(chunk.selected_token_ranks)


class PromptLogprobsTracker:
    """Active prompt-logprobs requests keyed by request id.

    ``SamplingParams.prompt_logprobs`` stays on the request for its full
    lifetime, but vLLM expects prompt logprobs exactly once.  The runner
    registers each new request here, clears it after delivery, and keeps
    in-progress chunks across prompt-stage preemption.
    """

    def __init__(self) -> None:
        self._active: dict[str, int] = {}
        self._in_progress: dict[str, PromptLogprobsAccumulator] = {}

    def register(self, req_id: str, num_logprobs: int) -> None:
        """Track one request until its prompt logprobs are delivered."""
        self._active[req_id] = num_logprobs

    def wants(self, req_id: str) -> bool:
        """Whether *req_id* still needs prompt logprobs."""
        return req_id in self._active

    def wants_any(self, req_ids: Iterable[str]) -> bool:
        """Whether any request still needs prompt logprobs."""
        return any(req_id in self._active for req_id in req_ids)

    def observe_chunk(
        self,
        req_id: str,
        prompt_token_ids: list[int],
        start_pos: int,
        num_tokens: int,
        chunk_logits: mx.array,
        logprobs_mode: str,
    ) -> LogprobsTensors | None:
        """Score one prefill chunk's logits rows against the prompt.

        ``chunk_logits`` holds one row per forwarded chunk position
        (``num_tokens`` rows); only the leading ``window.num_logits`` rows
        target prompt tokens and are scored.  Returns the completed tensors
        when this chunk reaches the end of the prompt.
        """
        prompt_len = len(prompt_token_ids)
        window = prompt_logprobs_window(
            start_pos=start_pos, num_tokens=num_tokens, prompt_len=prompt_len
        )
        accumulator = self._in_progress.get(req_id)
        if accumulator is None:
            accumulator = PromptLogprobsAccumulator(
                prompt_len=prompt_len,
                num_logprobs=_resolve_num_logprobs(
                    self._active[req_id], int(chunk_logits.shape[-1])
                ),
            )
            self._in_progress[req_id] = accumulator
        if window.num_logits > 0:
            targets = prompt_token_ids[
                window.first_target : window.first_target + window.num_logits
            ]
            chunk = gather_prompt_logprobs(
                chunk_logits[: window.num_logits],
                targets,
                accumulator.num_logprobs,
                logprobs_mode=logprobs_mode,
            )
            accumulator.fill(window, chunk)
        if not window.completes:
            return None
        del self._in_progress[req_id]
        self._active.pop(req_id, None)
        return accumulator.tensors

    def discard(self, req_ids: set[str] | list[str]) -> None:
        """Drop prompt-logprobs state for finished or aborted requests."""
        for req_id in req_ids:
            self._active.pop(req_id, None)
            self._in_progress.pop(req_id, None)


def full_prompt_logprobs(
    logits_rows: mx.array,
    prompt_token_ids: list[int],
    num_logprobs: int,
    logprobs_mode: str,
) -> LogprobsTensors:
    """One-shot prompt logprobs when the whole prompt ran in one forward.

    ``logits_rows`` holds one row per prompt position. Rows past the scored
    positions are ignored; the last row predicts the first sampled token.
    """
    prompt_len = len(prompt_token_ids)
    window = prompt_logprobs_window(
        start_pos=0, num_tokens=prompt_len, prompt_len=prompt_len
    )
    num_logprobs = _resolve_num_logprobs(num_logprobs, int(logits_rows.shape[-1]))
    accumulator = PromptLogprobsAccumulator(
        prompt_len=prompt_len, num_logprobs=num_logprobs
    )
    if window.num_logits > 0:
        targets = prompt_token_ids[1 : 1 + window.num_logits]
        chunk = gather_prompt_logprobs(
            logits_rows[: window.num_logits],
            targets,
            num_logprobs,
            logprobs_mode=logprobs_mode,
        )
        accumulator.fill(window, chunk)
    return accumulator.tensors


def _resolve_num_logprobs(num_logprobs: int, vocab_size: int) -> int:
    return vocab_size if num_logprobs == -1 else num_logprobs


def gather_prompt_logprobs(
    logits_rows: mx.array,
    target_token_ids: list[int],
    num_logprobs: int,
    logprobs_mode: str = "raw_logprobs",
) -> LogprobsTensors:
    """Score ``target_token_ids`` against ``logits_rows`` (one row per target).

    Uses vLLM's own ``Sampler.compute_logprobs`` / ``gather_logprobs`` so the
    returned tensors carry the same layout as the sampled-token logprobs:
    column 0 is the target token, the rest are the top-``num_logprobs``
    alternatives, and ``selected_token_ranks`` holds the target's 1-based rank.
    """
    if logits_rows.ndim != 2 or logits_rows.shape[0] != len(target_token_ids):
        raise ValueError(
            "logits_rows must be (num_targets, vocab); got shape "
            f"{tuple(logits_rows.shape)} for {len(target_token_ids)} targets"
        )
    mx.eval(logits_rows)
    logits = mlx_to_torch(logits_rows.astype(mx.float32), device="cpu")
    scores = (
        logits
        if logprobs_mode in _LOGITS_LOGPROBS_MODES
        else Sampler.compute_logprobs(logits)
    )
    targets = torch.tensor(target_token_ids, dtype=torch.int64)
    return Sampler.gather_logprobs(scores, num_logprobs, targets)
