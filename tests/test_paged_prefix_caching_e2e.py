# SPDX-License-Identifier: Apache-2.0
"""End-to-end correctness of paged prefix caching (issue #182).

Fires the deterministic-test prompts twice through ``vllm.LLM`` with
prefix caching enabled.  The first pass primes the cache; the second
pass exercises the model_runner's ``start_pos > 0`` path because the
upstream scheduler reports ``num_computed_tokens > 0``.  The asserted
token sequence is the existing cache-off golden, so a broken cache-hit
path surfaces as a token mismatch.

The LLM body runs in a spawned child process (``multiprocessing`` with
the ``spawn`` start method) so Metal device init happens in a fresh
interpreter.  This is required on the Metal platform because:
  - ``fork`` inherits the parent's Metal context and segfaults in the
    child (Metal is not fork-safe).
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
)


def _setenv_default(key: str, default: str) -> None:
    if os.environ.get(key) is None:
        os.environ[key] = default


def _run_prefix_cache_correctness() -> None:
    """Body of the e2e test — runs in a spawned child process.

    Imports happen lazily inside the child so vllm / MLX init is not
    inherited from the parent process.
    """
    _setenv_default("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    _setenv_default("VLLM_METAL_USE_PAGED_ATTENTION", DEFAULT_USE_PAGED_ATTENTION)
    _setenv_default("VLLM_METAL_MEMORY_FRACTION", DEFAULT_PAGED_MEMORY_FRACTION)

    if os.environ.get("VLLM_METAL_USE_PAGED_ATTENTION", "0") != "1":
        return  # non-paged path: nothing to test

    from vllm import LLM, SamplingParams

    from tests.test_paged_deterministic import (
        GOLDEN_MLX,
        GOLDEN_PAGED,
        MAX_TOKENS,
        MODEL_NAME,
        PROMPTS,
    )

    llm = LLM(
        model=MODEL_NAME,
        max_model_len=512,
        max_num_seqs=1,
        enable_prefix_caching=True,
    )
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    llm.generate(PROMPTS, sp)  # prime the cache
    outputs = llm.generate(PROMPTS, sp)  # cache hits expected
    by_prompt = {o.prompt: o for o in outputs}

    mismatches = []
    for prompt in PROMPTS:
        output = by_prompt[prompt]
        token_ids = list(output.outputs[0].token_ids)
        mlx_expected = GOLDEN_MLX[prompt]
        paged_expected = GOLDEN_PAGED[prompt]
        if token_ids != mlx_expected and token_ids != paged_expected:
            mismatches.append(
                f"  {prompt!r}\n"
                f"    got:        {token_ids}\n"
                f"    mlx golden: {mlx_expected}\n"
                f"    pgd golden: {paged_expected}"
            )

    if mismatches:
        raise AssertionError(
            "Prefix-cached output matched neither golden set for some prompts:\n"
            + "\n".join(mismatches)
        )


@pytest.mark.slow
def test_prefix_cached_matches_golden() -> None:
    ctx = mp.get_context("spawn")
    proc = ctx.Process(target=_run_prefix_cache_correctness)
    proc.start()
    proc.join()
    if proc.exitcode != 0:
        raise AssertionError(
            f"Prefix-cache e2e test failed in spawned child "
            f"(exit code: {proc.exitcode})"
        )
