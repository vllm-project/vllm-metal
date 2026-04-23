# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness of paged prefix caching (issue #182).

Sends a batch of prompts that share a long prefix (>= one KV block of
16 tokens) twice through ``vllm.LLM`` with prefix caching enabled.  The
first ``generate`` populates the cache; the second triggers cache hits
that walk the model_runner's ``start_pos > 0`` path because the upstream
scheduler reports ``num_computed_tokens > 0``.

Two assertions (in code order):
  1. Cache-hit reach — at least one prefill in the second pass is issued
     with ``start_pos > 0``.  Verified by spying on ``prepare_unified``
     (which receives per-prefill ``start_pos`` tuples).  Fails fast if
     the cache-hit branch was never taken.
  2. Determinism — second pass produces identical tokens to the first
     (greedy, same prompts).  A broken cache-hit path that still
     reaches the branch surfaces as a token mismatch.

The LLM body runs in a spawned child process (``multiprocessing`` with
the ``spawn`` start method) so Metal device init happens in a fresh
interpreter.  Required on Metal because:
  - ``fork`` inherits the parent's Metal context and segfaults
    (Metal is not fork-safe).
  - Running in the parent pytest process alongside the cache-off
    baseline fixture in ``test_paged_deterministic`` causes
    ``kv_budget=0`` — MLX wired buffers aren't released by Python gc.
"""

from __future__ import annotations

import multiprocessing as mp
import os

import pytest

from tests.test_paged_deterministic import (
    DEFAULT_PAGED_MEMORY_FRACTION,
    DEFAULT_USE_PAGED_ATTENTION,
    MODEL_NAME,
)

# Long shared prefix (~30 tokens — comfortably more than the 16-token
# Metal block size, so the upstream scheduler hashes at least one block
# and prefix cache lookups can succeed).
SHARED_PREFIX = (
    "Once upon a time in a far away kingdom there was a great king "
    "named Aragorn who ruled the land of Gondor with wisdom and "
    "grace, and his people loved him dearly because "
)
PROMPTS = [
    SHARED_PREFIX + "the queen agreed.",
    SHARED_PREFIX + "the people prospered.",
    SHARED_PREFIX + "the ravens watched silently.",
]
MAX_TOKENS = 10


def _setenv_default(key: str, default: str) -> None:
    if os.environ.get(key) is None:
        os.environ[key] = default


def _run_prefix_cache_correctness() -> None:
    """Body of the e2e test — runs in a spawned child process."""
    _setenv_default("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _setenv_default("VLLM_METAL_USE_PAGED_ATTENTION", DEFAULT_USE_PAGED_ATTENTION)
    _setenv_default("VLLM_METAL_MEMORY_FRACTION", DEFAULT_PAGED_MEMORY_FRACTION)

    if os.environ.get("VLLM_METAL_USE_PAGED_ATTENTION", "0") != "1":
        return  # non-paged path: nothing to test

    from vllm import LLM, SamplingParams

    import vllm_metal.paged_attention_common as pac

    seen_start_pos: list[int] = []
    orig_prepare = pac.prepare_unified

    def patched_prepare(decode_requests, prefill_requests, block_size, buffers=None):
        for _, _, start_pos in prefill_requests:
            seen_start_pos.append(start_pos)
        return orig_prepare(decode_requests, prefill_requests, block_size, buffers)

    pac.prepare_unified = patched_prepare

    try:
        llm = LLM(
            model=MODEL_NAME,
            max_model_len=512,
            max_num_seqs=1,
            enable_prefix_caching=True,
        )
        sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
        out_first = llm.generate(PROMPTS, sp)
        prime_count = len(seen_start_pos)
        out_second = llm.generate(PROMPTS, sp)
    finally:
        pac.prepare_unified = orig_prepare

    # Cache-hit reach: at least one prefill in the second pass must
    # advance past the cached prefix.
    second_pass = seen_start_pos[prime_count:]
    if not any(sp > 0 for sp in second_pass):
        raise AssertionError(
            f"Second pass should issue at least one prefill with start_pos "
            f"> 0 (cache-hit path), but saw start_pos values: {second_pass}"
        )

    # Determinism: greedy decode of the same prompts must produce the
    # same tokens whether or not the prefix was served from cache.
    mismatches = []
    for o1, o2 in zip(out_first, out_second, strict=True):
        toks1 = list(o1.outputs[0].token_ids)
        toks2 = list(o2.outputs[0].token_ids)
        if toks1 != toks2:
            mismatches.append(
                f"  prompt: {o1.prompt!r}\n"
                f"    first  pass tokens: {toks1}\n"
                f"    second pass tokens: {toks2}"
            )
    if mismatches:
        raise AssertionError(
            "Cache-hit path produced different tokens than the priming pass:\n"
            + "\n".join(mismatches)
        )


@pytest.mark.slow
def test_prefix_cache_hit_path_correctness() -> None:
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_run_prefix_cache_correctness)
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise AssertionError(
            f"Prefix-cache e2e test failed in spawned child "
            f"(exit code: {proc.exitcode})"
        )
