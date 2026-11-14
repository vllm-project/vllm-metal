# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Longest-suffix n-gram lookup for prompt-lookup speculative decoding.

Adapted from vLLM v0.33.0 ``vllm/v1/spec_decode/ngram_proposer.py``, which vLLM
removed together with Model Runner V1 (vllm-project/vllm#63373). The KMP kernel
is unchanged. The batch wrapper is reduced to what the Metal proposer uses: it
matches every row it is given on the calling thread, which is what upstream's
one-thread cap amounted to.

Replace this copy once upstream provides a CPU-compatible matcher with the same
matching rules and no GPU model runner dependency.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from numba import njit

if TYPE_CHECKING:
    from vllm.config import VllmConfig


class NgramMatcher:
    def __init__(self, vllm_config: VllmConfig) -> None:
        spec = vllm_config.speculative_config
        assert spec is not None
        assert spec.prompt_lookup_min is not None
        assert spec.prompt_lookup_max is not None
        self.min_n = spec.prompt_lookup_min
        self.max_n = spec.prompt_lookup_max
        self.max_model_len = vllm_config.model_config.max_model_len
        # Compile the kernel now so the first decode step does not pay for it.
        self.propose(1, np.zeros(1, dtype=np.int32), np.zeros((1, 1), dtype=np.int32))

    def propose(
        self,
        num_speculative_tokens: int,
        num_tokens_no_spec: np.ndarray,
        token_ids_cpu: np.ndarray,
    ) -> list[list[int]]:
        """Return up to ``num_speculative_tokens`` draft tokens for each row.

        Row ``i`` holds one request's committed history in
        ``token_ids_cpu[i, : num_tokens_no_spec[i]]``.
        """
        return [
            _find_longest_matched_ngram_and_propose_tokens(
                token_ids_cpu[i, :num_tokens],
                self.min_n,
                self.max_n,
                self.max_model_len,
                num_speculative_tokens,
            ).tolist()
            for i, num_tokens in enumerate(num_tokens_no_spec)
        ]


@njit
def _find_longest_matched_ngram_and_propose_tokens(
    origin_tokens: np.ndarray,
    min_ngram: int,
    max_ngram: int,
    max_model_len: int,
    k: int,
) -> np.ndarray:
    """Find the longest n-gram which matches the suffix of the given tokens
    whose length is within [min_ngram, max_ngram] (inclusive).

    If found, we will extract k right after the matched ngram.
    """
    # Do not generate draft tokens is context is shorter than minimum n-gram
    total_token = origin_tokens.shape[0]
    if total_token < min_ngram:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Do not generate draft tokens beyond the max model length.
    k = min(k, max_model_len - total_token)
    if k <= 0:
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip tokens, and the goal become to find longest ngram
    # on the rightmost position which matches the prefix with
    # length [min_n, max_n] (inclusive).
    tokens = origin_tokens[::-1]

    # Longest prefix (not including itself) which is a suffix of
    # the current position.
    #   lps[i] = max{v, where tokens[0:v] == tokens[i+1-v:i+1]}
    #
    # As ngram is capped by max_ngram to save memory, we only need to
    # store lps for the first max_ngram prefix.
    lps = np.zeros(max_ngram, dtype=np.int32)

    longest_ngram = 0
    position = 0

    # lps[0] always equal to 0, we start with index 1
    prev_lps = 0
    i = 1
    while i < total_token:
        # tokens[:prev_lps] is the longest prefix as a suffix of tokens[:i]
        if tokens[prev_lps] == tokens[i]:
            # Token match: tokens[:prev_lps+1] is the longest prefix as
            # a suffix of tokens[:i+1]
            prev_lps += 1
            # Check if we found a longer valid ngram.
            #
            # Update position when longest_ngram matched prev_lps,
            # as we want to get the target n-gram of the earliest position
            # in the original tokens (i.e.
            # latest position in the reversed tokens)
            if prev_lps >= longest_ngram:
                longest_ngram = prev_lps
                position = i
            if i < max_ngram:
                # Store LPS for the first max_ngram prefix
                lps[i] = prev_lps
            if prev_lps == max_ngram:
                # When prev_lps reached max_ngram, update prev_lps
                # to lps[max_ngram-1] to avoid matching ngram
                # longer than max_ngram
                prev_lps = lps[max_ngram - 1]
            i += 1
        elif prev_lps != 0:
            # Token mismatch: try the second-longest prefix
            # among all suffix of tokens[:i],
            # which is the longest prefix of tokens[:prev_lps]
            prev_lps = lps[prev_lps - 1]
        else:
            # Token mismatch, and no more prefix (except empty string)
            # as a suffix of tokens[:i]
            i += 1

    if longest_ngram < min_ngram:
        # No valid ngram is found
        return np.empty((0,), dtype=origin_tokens.dtype)

    # Flip the position back, so in origin_tokens,
    # origin_tokens[total_token-1-position:total_token-1-position+longest_ngram]
    # is the matched ngram, so we should start drafting tokens from
    # total_token-1-position+longest_ngram
    start_position = total_token - 1 - position + longest_ngram
    k = min(k, total_token - start_position)
    return origin_tokens[start_position : start_position + k]
