# SPDX-License-Identifier: Apache-2.0
"""Shared sampling-feature admission policy for the Metal backends.

One place that decides which sampling features a request or batch uses,
consumed by the MLX native-path gates (:class:`~vllm_metal.v1.sampling_batch.SamplingBatch`),
the greedy-only spec-decode gate, and the MLX score-option and MPS
request-control checks in ``platform.validate_request``. Pure Python: rows
are duck-typed ``SamplingParams``-shaped objects, so this module imports no
MLX, Torch, or model code at module scope and stays importable before vLLM
selects a platform.

RFC #1314's feature x backend x sync/deferred direction without its
undecided pieces (no RNG migration, no generic framework). Each function
documents its own contract; ``tests/test_sampling_policy.py`` pins the
distinctions between callers (API vs tokenized bad_words, prompt vs output
scores, seeded rows, MPS speculation and Omni repetition, pending vs
admitted min_tokens).
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.sampling_params import SamplingParams

# Below this temperature vLLM's sampler argmaxes, so the native paths may
# too. SamplingParams clamps 0 < t < 0.01 to 0.01, so normally only exactly
# zero lands here; the epsilon keeps mutated or cloned rows honest.
GREEDY_TEMPERATURE_EPS = 1e-5


def is_greedy_temperature(params: SamplingParams) -> bool:
    """Whether the row's temperature selects argmax sampling."""
    return params.temperature < GREEDY_TEMPERATURE_EPS


def has_penalties(params: SamplingParams) -> bool:
    """Whether any frequency/presence/repetition penalty is non-neutral."""
    return (
        params.frequency_penalty != 0.0
        or params.presence_penalty != 0.0
        or params.repetition_penalty != 1.0
    )


def needs_sample_logprobs(params: SamplingParams) -> bool:
    """Whether the row asks for sample logprobs (top-k or specific tokens).

    ``num_logprobs`` covers both ``logprobs`` and ``logprob_token_ids``.
    Prompt logprobs are gathered before sampling and do not count.
    """
    return params.num_logprobs is not None


def needs_full_vocab_sample_logprobs(params: SamplingParams) -> bool:
    """Whether the row asks for sample logprobs over the whole vocabulary.

    vLLM admits ``logprobs=-1`` when ``--max-logprobs`` is -1 or at least the
    vocabulary size. The MLX runner cannot serve it.
    """
    return params.logprobs == -1


def has_token_constraints(params: SamplingParams) -> bool:
    """Whether the row carries hard token constraints.

    Only the tokenized forms constrain sampling: ``allowed_token_ids`` and
    ``bad_words_token_ids`` (API ``bad_words`` is tokenized into the latter
    once a tokenizer is available).
    """
    return bool(params.allowed_token_ids) or bool(params.bad_words_token_ids)


def has_logit_bias(params: SamplingParams) -> bool:
    return bool(params.logit_bias)


def has_structured_outputs(params: SamplingParams) -> bool:
    return getattr(params, "structured_outputs", None) is not None


def min_tokens_pending(params: SamplingParams, output_length: int) -> bool:
    """Whether the row still needs vLLM's min-tokens stop-token mask.

    The processor masks stop tokens only while a row has generated fewer
    than ``min_tokens`` tokens, so rows past the minimum can sample
    natively again.
    """
    return output_length < params.min_tokens


def any_min_tokens_pending(
    sampling_params_list: Sequence[SamplingParams],
    output_lengths: Sequence[int] | None,
) -> bool:
    """Whether any row still needs vLLM's min-tokens stop-token mask.

    ``None`` counts every row as having generated nothing. The reduction
    short-circuits on the first pending row, so an over-long
    ``output_lengths`` list is only rejected by the zip-strict check when
    no earlier row is pending.
    """
    if output_lengths is None:
        output_lengths = [0] * len(sampling_params_list)
    return any(
        min_tokens_pending(params, output_length)
        for params, output_length in zip(
            sampling_params_list, output_lengths, strict=True
        )
    )


def params_allow_native_greedy(
    sampling_params_list: Sequence[SamplingParams],
    output_lengths: Sequence[int] | None = None,
) -> bool:
    """Whether MLX argmax matches *sampling_params_list* exactly.

    Greedy temperature with no candidate filters (top-k/top-p), no
    penalties, no sample logprobs, no token constraints, no logit bias, and
    no row still pending its min-tokens mask. ``None`` counts every row as
    having generated nothing.
    """
    return not any_min_tokens_pending(sampling_params_list, output_lengths) and all(
        is_greedy_temperature(params)
        and params.top_k <= 0
        and params.top_p == 1.0
        and not has_penalties(params)
        and not needs_sample_logprobs(params)
        and not has_token_constraints(params)
        and not has_logit_bias(params)
        for params in sampling_params_list
    )


def params_allow_native_random(
    sampling_params_list: Sequence[SamplingParams],
    output_lengths: Sequence[int] | None = None,
    *,
    allow_seed: bool = False,
) -> bool:
    """Whether the native MLX sampler supports *sampling_params_list*.

    Requests must use plain temperature/top-k/top-p/min-p sampling; the
    mask is built per row, so rows may mix greedy (temperature 0) with
    random sampling and differ in top-k/top-p/min-p. Seeded rows need
    ``allow_seed`` (their keys are derived from the seed and output
    position). Requests with sample logprobs, structured outputs, or
    ``logit_bias`` stay on the torch path. An empty batch has no rows to
    draw for, so it is never eligible.
    """
    if not sampling_params_list:
        return False
    return not any_min_tokens_pending(sampling_params_list, output_lengths) and all(
        (allow_seed or params.seed is None)
        and not has_penalties(params)
        and not needs_sample_logprobs(params)
        and not has_structured_outputs(params)
        and not has_token_constraints(params)
        and not has_logit_bias(params)
        for params in sampling_params_list
    )


def unsupported_mlx_scoring_option(
    params: SamplingParams,
) -> tuple[str, str] | None:
    """``(parameter, description)`` of the first score request MLX rejects.

    Checked at admission so the request gets a 400 instead of reaching the
    runner, where ``logprobs=-1`` raises inside ``sample_tokens`` and is
    fatal to the engine. MPS serves both through MRv2.
    """
    if params.prompt_logprob_start:
        return "prompt_logprob_start", "prompt_logprob_start"
    if needs_full_vocab_sample_logprobs(params):
        return "logprobs", "logprobs=-1"
    return None


def unsupported_request_controls(
    params: SamplingParams,
    *,
    speculative: bool = False,
    allow_repetition_penalty: bool = False,
) -> list[str]:
    """Names of sampling controls the MPS backend cannot apply, in order.

    These are the logits-processor-backed controls; model-specific
    restrictions (diffusion, pooling) stay with their owners. ``bad_words``
    is the API field, read before tokenization fills
    ``bad_words_token_ids``. Under MPS speculation any non-zero temperature
    is unverifiable, so it is rejected exactly (not by the greedy
    epsilon). ``min_tokens`` is rejected at admission, before any output
    length exists to make it pending-aware. The MPS sampler has no penalty
    processors, so all three penalties are listed; repetition_penalty is
    exempted with ``allow_repetition_penalty`` for Omni models, which apply
    it in model code (the MLX backend instead serves penalized requests
    through the torch fallback sampler).
    """
    controls = [
        name
        for name, enabled in (
            ("temperature", params.temperature != 0 and speculative),
            ("presence_penalty", params.presence_penalty != 0),
            ("frequency_penalty", params.frequency_penalty != 0),
            (
                "repetition_penalty",
                params.repetition_penalty != 1 and not allow_repetition_penalty,
            ),
            ("allowed_token_ids", bool(params.allowed_token_ids)),
            ("bad_words", bool(params.bad_words)),
            ("logit_bias", bool(params.logit_bias)),
            ("min_tokens", params.min_tokens > 0),
        )
        if enabled
    ]
    controls.extend(
        name
        for name in ("logprob_token_ids", "structured_outputs")
        if getattr(params, name, None) is not None
    )
    return controls
