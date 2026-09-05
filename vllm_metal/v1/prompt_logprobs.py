# SPDX-License-Identifier: Apache-2.0
"""Prompt logprobs for the Metal runner.

vLLM's ``SamplingParams.prompt_logprobs`` (and the OpenAI ``echo`` +
``logprobs`` combination the server maps onto it) asks for the log
probability the model assigns to each prompt token given its prefix, plus the
top-k alternatives at that position.  The engine expects one
``LogprobsTensors`` per request covering positions ``1 .. prompt_len - 1``,
delivered on the step that finishes the prompt (``prompt_logprobs_dict`` in
``ModelRunnerOutput``); chunked prefill fills it slice by slice.  This module
holds the accounting and the tensor construction so the paged and non-paged
runner paths share one contract.
"""

from __future__ import annotations

from dataclasses import dataclass

import mlx.core as mx
import torch
from vllm.v1.outputs import LogprobsTensors
from vllm.v1.sample.sampler import Sampler

from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch


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
    *, start_pos: int, num_tokens: int, prompt_len: int
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
        if chunk.logprob_token_ids.shape[0] != window.num_logits:
            raise ValueError(
                "prompt logprobs chunk has "
                f"{chunk.logprob_token_ids.shape[0]} rows, window expects "
                f"{window.num_logits}"
            )
        if end > self.prompt_len - 1:
            raise ValueError(
                f"prompt logprobs window [{start}, {end}) exceeds the "
                f"{self.prompt_len - 1} scored prompt positions"
            )
        self.tensors.logprob_token_ids[start:end].copy_(chunk.logprob_token_ids)
        self.tensors.logprobs[start:end].copy_(chunk.logprobs)
        self.tensors.selected_token_ranks[start:end].copy_(chunk.selected_token_ranks)


class PromptLogprobsTracker:
    """In-progress prompt-logprobs accumulators keyed by request id.

    One instance lives on the runner.  Every prefill chunk of a request whose
    ``SamplingParams.prompt_logprobs`` is set is observed exactly once, with
    the logits rows the model produced for that chunk; the tracker scores the
    rows that target prompt tokens and returns the request's completed
    ``LogprobsTensors`` on the chunk that finishes the prompt (``None``
    before that).  State survives preemption on purpose — a resumed request
    re-runs its prompt chunks and overwrites the same positions — and is
    dropped via :meth:`discard` when the engine finishes the request.
    """

    def __init__(self) -> None:
        self._in_progress: dict[str, PromptLogprobsAccumulator] = {}

    @staticmethod
    def wants(sampling_params: object) -> bool:
        """Whether *sampling_params* asks for prompt logprobs."""
        return getattr(sampling_params, "prompt_logprobs", None) is not None

    def observe_chunk(
        self,
        req_id: str,
        *,
        prompt_token_ids: list[int],
        start_pos: int,
        num_tokens: int,
        chunk_logits: mx.array,
        num_logprobs: int,
    ) -> LogprobsTensors | None:
        """Score one prefill chunk's logits rows against the prompt.

        ``chunk_logits`` holds one row per forwarded chunk position
        (``num_tokens`` rows); only the leading ``window.num_logits`` rows
        target prompt tokens and are scored.  Returns the completed tensors
        when this chunk reaches the end of the prompt.
        """
        if chunk_logits.ndim != 2 or chunk_logits.shape[0] != num_tokens:
            raise ValueError(
                f"chunk_logits must be ({num_tokens}, vocab); got shape "
                f"{tuple(chunk_logits.shape)}"
            )
        prompt_len = len(prompt_token_ids)
        window = prompt_logprobs_window(
            start_pos=start_pos, num_tokens=num_tokens, prompt_len=prompt_len
        )
        accumulator = self._in_progress.get(req_id)
        if accumulator is None:
            accumulator = PromptLogprobsAccumulator(
                prompt_len=prompt_len, num_logprobs=num_logprobs
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
            )
            accumulator.fill(window, chunk)
        if not window.completes:
            return None
        del self._in_progress[req_id]
        return accumulator.tensors

    def discard(self, req_ids: set[str] | list[str]) -> None:
        """Drop in-progress state for finished or aborted requests."""
        for req_id in req_ids:
            self._in_progress.pop(req_id, None)


def full_prompt_logprobs(
    logits_rows: mx.array,
    prompt_token_ids: list[int],
    num_logprobs: int,
) -> LogprobsTensors:
    """One-shot prompt logprobs when the whole prompt ran in one forward.

    ``logits_rows`` holds one row per prompt position (the non-paged path
    forwards the full prompt in a single chunk); rows past the scored
    positions — the last row predicts the first sampled token — are ignored.
    """
    prompt_len = len(prompt_token_ids)
    window = prompt_logprobs_window(
        start_pos=0, num_tokens=prompt_len, prompt_len=prompt_len
    )
    accumulator = PromptLogprobsAccumulator(
        prompt_len=prompt_len, num_logprobs=num_logprobs
    )
    if window.num_logits > 0:
        targets = prompt_token_ids[1 : 1 + window.num_logits]
        chunk = gather_prompt_logprobs(
            logits_rows[: window.num_logits], targets, num_logprobs
        )
        accumulator.fill(window, chunk)
    return accumulator.tensors


def gather_prompt_logprobs(
    logits_rows: mx.array,
    target_token_ids: list[int],
    num_logprobs: int,
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
    logprobs = Sampler.compute_logprobs(logits)
    targets = torch.tensor(target_token_ids, dtype=torch.int64)
    return Sampler.gather_logprobs(logprobs, num_logprobs, targets)
