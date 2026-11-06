# SPDX-License-Identifier: Apache-2.0
"""Shared MLX scoring for sampled tokens and teacher-forced prompt tokens."""

from collections.abc import Sequence
from dataclasses import dataclass

import mlx.core as mx
import numpy as np
from vllm.v1.outputs import LogprobsLists, LogprobsTensors

from vllm_metal.tensor_bridge import mlx_to_torch


@dataclass(frozen=True)
class NativeLogprobs:
    token_ids: mx.array
    scores: mx.array
    ranks: mx.array

    def tolists(self) -> LogprobsLists:
        mx.eval(self.token_ids, self.scores, self.ranks)
        return LogprobsLists(
            np.array(self.token_ids), np.array(self.scores), np.array(self.ranks)
        )

    def to_tensors(self) -> LogprobsTensors:
        # Only the gathered result crosses the bridge; source logits stay on MLX.
        mx.eval(self.token_ids, self.scores, self.ranks)
        return LogprobsTensors(
            mlx_to_torch(self.token_ids, device="cpu"),
            mlx_to_torch(self.scores, device="cpu"),
            mlx_to_torch(self.ranks, device="cpu"),
        )


def _topk_token_ids(values: mx.array, k: int) -> mx.array:
    """Find k IDs by sorting only promising chunks of a large vocabulary."""
    batch_size, vocab_size = values.shape
    chunk_size = 256
    full_vocab = vocab_size // chunk_size * chunk_size
    # Keep the full sort unless candidate selection cuts its width substantially.
    if vocab_size < 4096 or k * chunk_size > full_vocab // 4:
        return mx.argsort(-values, axis=-1)[:, :k].astype(mx.int32)

    chunks = values[:, :full_vocab].reshape(batch_size, -1, chunk_size)
    chunk_maxima = mx.max(chunks, axis=-1)
    chunk_ids = mx.argsort(-chunk_maxima, axis=-1)[:, :k].astype(mx.int32)
    candidate_ids = (
        chunk_ids[:, :, None] * chunk_size
        + mx.arange(chunk_size, dtype=mx.int32)[None, None, :]
    ).reshape(batch_size, -1)
    # Include the real tail rather than padding it: -inf padding could otherwise
    # be returned when k exceeds the number of finite scores.
    if full_vocab < vocab_size:
        tail = mx.broadcast_to(
            mx.arange(full_vocab, vocab_size, dtype=mx.int32)[None, :],
            (batch_size, vocab_size - full_vocab),
        )
        candidate_ids = mx.concatenate([candidate_ids, tail], axis=-1)
    candidates = mx.take_along_axis(values, candidate_ids, axis=-1)
    order = mx.argsort(-candidates, axis=-1)[:, :k]
    # Each omitted chunk is bounded by k chosen chunk maxima, so its scores
    # cannot displace the top k. Ties may choose any k IDs, as in torch.topk.
    return mx.take_along_axis(candidate_ids, order, axis=-1)


def gather_native_logprobs(
    logits: mx.array,
    target_token_ids: mx.array,
    num_logprobs: int | Sequence[int],
    *,
    logits_mode: bool = False,
    requested_token_ids: dict[int, list[int]] | None = None,
) -> NativeLogprobs:
    """Gather target + top-k/specific scores and vLLM's inclusive tie ranks.

    Specific IDs override top-k only on their own row. Short rows are padded
    with token 0 / -inf; the target always occupies column zero. Tied top-k
    IDs follow MLX's sort order, which need not match torch's unstable top-k.
    """
    batch_size, vocab_size = logits.shape
    counts = (
        [num_logprobs] * batch_size
        if isinstance(num_logprobs, int)
        else list(num_logprobs)
    )
    requested = requested_token_ids or {}
    top_k = max((n for row, n in enumerate(counts) if row not in requested), default=0)
    if min(counts, default=0) < 0 or top_k > vocab_size:
        raise ValueError("num_logprobs must be between zero and the vocabulary size")
    widths = [
        len(requested[row]) if row in requested else n for row, n in enumerate(counts)
    ]
    width = max(widths, default=0)
    values = logits.astype(mx.float32)
    if not logits_mode:
        # Subtract the maximum first to retain small score differences even
        # when every logit has a large common offset.
        shifted = values - mx.max(values, axis=-1, keepdims=True)
        values = shifted - mx.logsumexp(shifted, axis=-1, keepdims=True)
    top_ids = (
        _topk_token_ids(values, top_k)
        if top_k
        else mx.zeros((batch_size, 0), dtype=mx.int32)
    )
    alternatives = mx.pad(top_ids, [(0, 0), (0, width - top_k)])
    for row, ids in requested.items():
        alternatives[row] = mx.array(ids + [0] * (width - len(ids)), dtype=mx.int32)
    ids = mx.concatenate(
        [target_token_ids.astype(mx.int32)[:, None], alternatives], axis=-1
    )
    scores = mx.take_along_axis(values, ids, axis=-1)
    valid = mx.arange(width + 1)[None, :] <= mx.array(widths)[:, None]
    scores = mx.where(valid, scores, -mx.inf)
    ids = mx.where(valid, ids, 0)
    # vLLM ranks ties by counting >=, including all tokens tied with the target.
    ranks = mx.sum(values >= scores[:, :1], axis=-1).astype(mx.int32)
    return NativeLogprobs(ids, scores, ranks)
