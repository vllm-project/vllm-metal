# SPDX-License-Identifier: Apache-2.0
"""Allowed-token and bad-word masks for the MLX sampler."""

from collections.abc import Sequence

import mlx.core as mx
from vllm.sampling_params import SamplingParams


def apply_native_constraints(
    logits: mx.array,
    params: Sequence[SamplingParams],
    output_token_ids: Sequence[list[int]],
) -> mx.array:
    """Build one device mask from host request metadata, without reading logits.

    Bad-word prefixes match committed output only, as in vLLM's sampler.
    A single-token word always blocks its token; longer words block only the
    final token when the output suffix matches the preceding tokens.
    """
    if not any(sp.allowed_token_ids or sp.bad_words_token_ids for sp in params):
        return logits
    blocked = mx.broadcast_to(
        mx.array([bool(sp.allowed_token_ids) for sp in params])[:, None],
        logits.shape,
    )
    allowed_rows: list[int] = []
    allowed_ids: list[int] = []
    banned_rows: list[int] = []
    banned_ids: list[int] = []
    for row, (sp, output_ids) in enumerate(zip(params, output_token_ids, strict=True)):
        if sp.allowed_token_ids:
            allowed_rows.extend([row] * len(sp.allowed_token_ids))
            allowed_ids.extend(sp.allowed_token_ids)
        for word in sp.bad_words_token_ids or ():
            prefix_len = len(word) - 1
            if prefix_len == 0 or (
                prefix_len <= len(output_ids) and output_ids[-prefix_len:] == word[:-1]
            ):
                banned_rows.append(row)
                banned_ids.append(word[-1])
    if allowed_ids:
        blocked[mx.array(allowed_rows), mx.array(allowed_ids)] = False
    if banned_ids:
        blocked[mx.array(banned_rows), mx.array(banned_ids)] = True
    return mx.where(blocked, -mx.inf, logits)
