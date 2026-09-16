# SPDX-License-Identifier: Apache-2.0
"""Exact stochastic proposal and verification for DSpark.

Pure functions over MLX arrays plus the per-request records they need. The
proposer samples every draft token of a non-greedy request from a normalized
float32 distribution ``q`` built from the drafter's block logits, the Markov
step bias and the request's own target sampling transforms (temperature,
top-k and top-p with the Metal sampler's mask semantics), and keeps that exact
``q``. Verification builds the target distribution ``p`` from the target
logits with the same transforms, accepts draft ``x`` with probability
``min(1, p(x) / q(x))``, samples the first rejected position from the
normalized positive residual ``max(p - q, 0)`` and samples the bonus token
from ``p`` after full acceptance. Every random draw comes from the owning
request's own streams (proposal, acceptance, target), so scheduling,
cancellation or reordering of another request never consumes it.

Finite precision is explicit: distributions are float32 and sampled by inverse
CDF over a float32 cumulative sum against a uniform drawn in float64 and
rounded to float32. A token with zero mass can never be selected, and a
cumulative sum that falls short of the uniform because of rounding selects the
last token with positive mass. A residual whose float32 mass is at most
``RESIDUAL_MASS_EPS`` means ``p`` and ``q`` are indistinguishable at that
precision, so the position samples from ``p`` instead (the reference
evaluator's rule). Greedy requests keep the argmax draft and the exact
greedy verifier; they hold no distributions.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, cast

import mlx.core as mx
import numpy as np

from vllm_metal.v1.sampling_batch import GREEDY_TEMPERATURE_EPS, SamplingBatch

RESIDUAL_MASS_EPS = 1e-8
PROPOSAL_PRECISION = (
    "float32 softmax of (drafter logits + Markov step bias) / temperature "
    "after the Metal top-k/top-p mask; inverse-CDF sampling on the float32 "
    "cumulative sum against a float64 uniform rounded to float32"
)
_SEED_MODULUS = 2**63


@dataclass(frozen=True, slots=True)
class SamplingTransforms:
    """The target sampling transforms a request applies to its logits."""

    temperature: float
    top_k: int
    top_p: float

    @classmethod
    def from_params(cls, params: Any) -> SamplingTransforms:
        return cls(float(params.temperature), int(params.top_k), float(params.top_p))

    @property
    def greedy(self) -> bool:
        return self.temperature < GREEDY_TEMPERATURE_EPS


def transformed_distribution(
    logits: mx.array, transforms: SamplingTransforms, *, vocab_size: int
) -> mx.array:
    """``[..., V']`` logits to ``[..., vocab_size]`` float32 distributions.

    Applies exactly what the Metal sampler applies for the request: divide by
    the temperature, mask to the top-k/top-p candidate set with
    :meth:`SamplingBatch._top_k_top_p_masked_logits` (vLLM's tie and boundary
    semantics), then a float32 softmax. Padded lm_head columns beyond
    ``vocab_size`` are dropped first so they can never carry mass.
    """
    if transforms.greedy:
        raise ValueError("greedy requests have no sampling distribution")
    if transforms.temperature <= 0.0 or not np.isfinite(transforms.temperature):
        raise ValueError("temperature must be positive and finite")
    if not 0.0 < transforms.top_p <= 1.0:
        raise ValueError("top_p must lie in (0, 1]")
    rows = logits[..., :vocab_size].astype(mx.float32) / transforms.temperature
    masked = SamplingBatch._top_k_top_p_masked_logits(
        rows, transforms.top_k, transforms.top_p
    )
    return mx.softmax(masked, axis=-1)


def batched_transformed_distribution(
    logits: mx.array, transforms: Sequence[SamplingTransforms], *, vocab_size: int
) -> mx.array:
    """Per-row :func:`transformed_distribution` for ``[R, V']`` logits."""
    if logits.ndim != 2 or logits.shape[0] != len(transforms):
        raise ValueError("one transform per logits row is required")
    groups: dict[tuple[int, float], list[int]] = {}
    for index, item in enumerate(transforms):
        if item.greedy:
            raise ValueError("greedy requests have no sampling distribution")
        groups.setdefault((item.top_k, item.top_p), []).append(index)
    if len(groups) == 1:
        (top_k, top_p), _ = next(iter(groups.items()))
        temperatures = mx.array(
            [item.temperature for item in transforms], dtype=mx.float32
        )[:, None]
        rows = logits[:, :vocab_size].astype(mx.float32) / temperatures
        masked = SamplingBatch._top_k_top_p_masked_logits(rows, top_k, top_p)
        return mx.softmax(masked, axis=-1)
    parts = [
        transformed_distribution(logits[index], item, vocab_size=vocab_size)
        for index, item in enumerate(transforms)
    ]
    return mx.stack(parts, axis=0)


def sample_from_distribution(probs: mx.array, uniforms: mx.array) -> mx.array:
    """Inverse-CDF sample of every row: ``[R, V]`` probs, ``[R]`` uniforms.

    Returns lazy ``int32`` token ids. Row ``r`` selects the first index whose
    cumulative mass exceeds ``uniforms[r] * total_mass``, so zero-mass tokens
    are never selected; when rounding leaves the whole cumulative sum below the
    scaled uniform, the last token with positive mass is selected.
    """
    if probs.ndim != 2 or uniforms.ndim != 1 or probs.shape[0] != uniforms.shape[0]:
        raise ValueError("probs must be [rows, vocab] with one uniform per row")
    vocab = probs.shape[-1]
    cumulative = mx.cumsum(probs, axis=-1)
    threshold = uniforms.astype(mx.float32)[:, None] * cumulative[:, -1:]
    index = mx.sum(cumulative <= threshold, axis=-1)
    positive = probs > 0
    last_positive = vocab - 1 - mx.argmax(positive[:, ::-1], axis=-1)
    return mx.where(index >= vocab, last_positive, index).astype(mx.int32)


def residual_distribution(target: mx.array, draft: mx.array) -> mx.array:
    """Normalized ``max(target - draft, 0)`` per row, or ``target`` when empty."""
    residual = mx.maximum(target - draft, 0.0)
    mass = mx.sum(residual, axis=-1, keepdims=True)
    fallback = mass <= RESIDUAL_MASS_EPS
    residual = mx.where(fallback, target, residual)
    mass = mx.where(fallback, mx.sum(target, axis=-1, keepdims=True), mass)
    return residual / mass


def acceptance_probabilities(
    target: mx.array, draft: mx.array, token_ids: Sequence[int]
) -> tuple[mx.array, mx.array]:
    """``min(1, p(x)/q(x))`` per drafted position and the ``q(x)`` it used.

    ``target`` and ``draft`` are ``[K, V]``; a zero ``q(x)`` is reported so the
    caller can reject the record (a token sampled from ``q`` cannot have zero
    mass), and it yields probability zero rather than a division by zero.
    """
    if target.ndim != 2 or draft.shape != target.shape:
        raise ValueError("target and draft distributions must both be [K, V]")
    if len(token_ids) != target.shape[0]:
        raise ValueError("one draft token per distribution row is required")
    columns = mx.array(list(token_ids), dtype=mx.int32)[:, None]
    p_at = mx.take_along_axis(target, columns, axis=-1)[:, 0]
    q_at = mx.take_along_axis(draft, columns, axis=-1)[:, 0]
    ratio = mx.where(q_at > 0, p_at / mx.where(q_at > 0, q_at, 1.0), 0.0)
    return mx.minimum(ratio, 1.0), q_at


def first_rejection(acceptance: Sequence[float], uniforms: Sequence[float]) -> int:
    """Index of the first rejected draft, or ``len(acceptance)`` if none."""
    if len(acceptance) != len(uniforms):
        raise ValueError("one uniform per drafted position is required")
    for index, (probability, uniform) in enumerate(
        zip(acceptance, uniforms, strict=True)
    ):
        if not uniform < probability:
            return index
    return len(acceptance)


def verify_rows(
    target: mx.array,
    draft: mx.array,
    token_ids: Sequence[int],
    acceptance_uniforms: Sequence[float],
    target_uniform: float,
) -> list[int]:
    """One request's verification with explicit uniforms (the reference path).

    ``target`` holds ``K + 1`` distributions (drafted positions then the bonus
    row), ``draft`` the ``K`` proposal distributions. Returns the accepted
    prefix plus the recovered token at the first rejection, or plus the bonus
    token after full acceptance.
    """
    width = len(token_ids)
    if draft.shape[0] != width or target.shape[0] != width + 1:
        raise ValueError("verification needs K draft rows and K + 1 target rows")
    if width == 0:
        raise ValueError("verification needs at least one drafted token")
    probabilities, q_at = acceptance_probabilities(target[:width], draft, token_ids)
    mx.eval(probabilities, q_at)
    if any(value <= 0.0 for value in cast("list[float]", q_at.tolist())):
        raise ValueError("a drafted token has zero proposal mass")
    rejected = first_rejection(
        cast("list[float]", probabilities.tolist()), list(acceptance_uniforms)
    )
    uniform = mx.array([target_uniform], dtype=mx.float32)
    if rejected < width:
        distribution = residual_distribution(
            target[rejected : rejected + 1], draft[rejected : rejected + 1]
        )
    else:
        distribution = target[width : width + 1]
    token = sample_from_distribution(distribution, uniform)
    mx.eval(token)
    return [*token_ids[:rejected], int(token.item())]


class RequestRandomStreams:
    """Three independent per-request streams: proposal, acceptance, target.

    A seeded request derives all three from its seed alone, and each draws only
    that request's own width, so no co-scheduled request consumes from a stream
    that is not its own. An unseeded request derives them from the engine seed
    and its admission ordinal, or from operating-system entropy when the engine
    has no seed.

    At a fixed draft width the same seed therefore reproduces the same draws
    whatever else is in the batch. Under the adaptive planner it does not: the
    width itself comes from ``decide(active_requests=...)``, so how far a stream
    advances on a step depends on how many requests were co-scheduled. Reproducing
    a sampled run byte-for-byte needs the fixed mode.
    """

    __slots__ = ("acceptance", "proposal", "target")

    def __init__(self, sequence: np.random.SeedSequence) -> None:
        proposal, acceptance, target = sequence.spawn(3)
        self.proposal = np.random.Generator(np.random.PCG64(proposal))
        self.acceptance = np.random.Generator(np.random.PCG64(acceptance))
        self.target = np.random.Generator(np.random.PCG64(target))

    @classmethod
    def for_request(
        cls, seed: int | None, *, engine_seed: int | None, ordinal: int
    ) -> RequestRandomStreams:
        if seed is not None:
            return cls(np.random.SeedSequence(int(seed) % _SEED_MODULUS))
        if engine_seed is None:
            return cls(np.random.SeedSequence())
        return cls(np.random.SeedSequence([int(engine_seed) % _SEED_MODULUS, ordinal]))


@dataclass(slots=True)
class DSparkProposal:
    """One request's scheduled proposal and its provenance.

    ``owner`` is the runner's ``RequestState`` for this request generation;
    ``anchor_position`` is the absolute position of ``anchor_token`` (the last
    committed token when the block was drafted), ``token_ids`` are the drafted
    tokens for positions ``anchor_position + 1 ...`` and ``confidence`` the
    drafter's raw confidence logit per drafted position. A stochastic proposal
    also keeps ``distributions``, the float32 ``[len(token_ids), vocab]`` rows
    the tokens were sampled from, with its ``transforms`` and ``streams``; a
    greedy proposal leaves those ``None``.
    """

    owner: Any
    anchor_position: int
    anchor_token: int
    token_ids: list[int]
    distributions: mx.array | None = None
    transforms: SamplingTransforms | None = None
    streams: RequestRandomStreams | None = None
    confidence: list[float] | None = None
    precision: str = PROPOSAL_PRECISION

    @property
    def stochastic(self) -> bool:
        return self.distributions is not None

    def matches(self, owner: Any, anchor_position: int, anchor_token: int) -> bool:
        return (
            self.owner is owner
            and self.anchor_position == anchor_position
            and self.anchor_token == anchor_token
        )


__all__ = [
    "PROPOSAL_PRECISION",
    "RESIDUAL_MASS_EPS",
    "DSparkProposal",
    "RequestRandomStreams",
    "SamplingTransforms",
    "acceptance_probabilities",
    "batched_transformed_distribution",
    "first_rejection",
    "residual_distribution",
    "sample_from_distribution",
    "transformed_distribution",
    "verify_rows",
]
