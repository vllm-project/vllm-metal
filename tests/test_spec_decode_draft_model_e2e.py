# SPDX-License-Identifier: Apache-2.0
"""End-to-end gate for draft-model speculative decoding on Metal.

With draft == target == ``Qwen/Qwen3-0.6B`` under greedy decoding, every draft
token equals the target's own argmax, so all drafts are accepted and
speculative decoding must reproduce plain greedy decoding token-for-token.

The plain-greedy *paged* output is exactly what ``test_paged_deterministic``
pins: ``GOLDEN_MLX`` for five prompts and a documented paged fallback for the
sixth. This gate asserts that draft-model SD reproduces those same tokens.

Process isolation: build exactly ONE engine per process (target + draft model +
two paged caches). Run on its own:

    PYTHONPATH=$PWD VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      python -m pytest tests/test_spec_decode_draft_model_e2e.py -v -s -m slow

Numerical note: the verify forward runs the K+1 verification rows through the
QKV/MLP/LM-head GEMMs with M=K+1 (vs M=1 for plain decode); per the documented
MLX batch/M-size FP behavior this can shift one argmax. The gate asserts exact
identity to the greedy goldens first; the per-prompt paged fallback covers the
one prompt already known to sit on a ULP tie.
"""

from __future__ import annotations

import os

import pytest
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 10
NUM_SPECULATIVE_TOKENS = 3

PROMPTS = [
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
    "Machine learning is",
]

# Plain-greedy reference, shared with test_paged_deterministic. SD with the same
# model as draft and target must reproduce these greedy tokens.
# fmt: off
GOLDEN_MLX = {
    "The capital of France is":                   [12095, 13, 576, 6722, 315, 9625, 374, 1083, 279, 6722],
    "The weather today is not":                   [1661, 13, 576, 9315, 374, 220, 17, 15, 12348, 13],
    "One plus one equals":                        [825, 11, 825, 5519, 825, 16819, 1378, 13, 2055, 11],
    "The largest planet in our solar system is":  [1112, 30, 362, 13, 43562, 425, 13, 48976, 356, 13],
    "Water boils at a temperature of":            [220, 16, 15, 15, 30937, 13, 3555, 374, 279, 9315],
    "Machine learning is":                        [264, 7988, 5392, 429, 702, 13791, 1506, 279, 2070, 315],
}

# Prompts whose top-2 logits at the divergence step are within ULP; the paged
# path can legitimately pick a different argmax than MLX. Same documented
# exceptions as test_paged_deterministic.
GOLDEN_PAGED_FALLBACK = {
    "One plus one equals":                        [825, 13, 3776, 5519, 825, 16819, 1378, 13, 3776, 5519],
    "The capital of France is":                   [12095, 13, 576, 6722, 315, 15344, 374, 21718, 13, 576],
}
# fmt: on


@pytest.fixture(autouse=True, scope="module")
def _set_env():
    """Force the paged path (spec-decode verify requires it) with headroom."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        mp.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        if os.environ.get("VLLM_METAL_MEMORY_FRACTION") is None:
            mp.setenv("VLLM_METAL_MEMORY_FRACTION", "0.2")
        yield


@pytest.fixture(scope="module")
def sd_engine():
    """One draft-model SD engine (max_num_seqs lets prompts co-schedule).

    A single engine serves both the single-stream gate (prompts fed one at a
    time => batch-size-1 dynamics) and the continuous-batching test (all
    prompts in one ``generate`` call), so only one engine + draft model lives
    in the process at a time.
    """
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=512,
        max_num_seqs=len(PROMPTS),
        enable_prefix_caching=False,
        async_scheduling=False,
        speculative_config={
            "method": "draft_model",
            "model": MODEL_NAME,
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        },
    )

    vllm_config = llm.llm_engine.vllm_config
    spec_cfg = vllm_config.speculative_config
    assert spec_cfg is not None and spec_cfg.uses_draft_model(), (
        "expected vLLM to resolve method=draft_model for a Qwen3 draft"
    )
    assert vllm_config.scheduler_config.async_scheduling is False, (
        "draft-model SD on Metal requires synchronous scheduling"
    )

    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    from vllm_metal.v1.draft_model_proposer import DraftModelProposer

    assert isinstance(runner._drafter, DraftModelProposer), (
        f"expected DraftModelProposer drafter, got {type(runner._drafter)!r}"
    )
    return llm


def _greedy(llm: LLM, prompts: list[str]) -> dict[str, list[int]]:
    sp = SamplingParams(temperature=0, max_tokens=MAX_TOKENS)
    return {o.prompt: list(o.outputs[0].token_ids) for o in llm.generate(prompts, sp)}


class TestDraftModelSpecDecode:
    @pytest.mark.slow
    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_single_stream_matches_greedy(self, sd_engine, prompt):
        # One prompt at a time => batch-size-1 dynamics, so SD must reproduce
        # plain greedy decoding token-for-token (up to the documented ULP tie).
        token_ids = _greedy(sd_engine, [prompt])[prompt]

        mlx_expected = GOLDEN_MLX[prompt]
        fallback = GOLDEN_PAGED_FALLBACK.get(prompt)

        print(f"\n  prompt: {prompt!r}")
        print(f"  ids:    {token_ids}")

        if token_ids == mlx_expected:
            print("  result: MATCHED greedy ground truth (SD == plain greedy)")
            return
        if fallback is not None and token_ids == fallback:
            print("  result: MATCHED paged fallback (documented ULP divergence)")
            return

        diverge = next(
            (
                i
                for i in range(min(len(token_ids), len(mlx_expected)))
                if token_ids[i] != mlx_expected[i]
            ),
            min(len(token_ids), len(mlx_expected)),
        )
        raise AssertionError(
            f"Draft-model SD output for {prompt!r} did not reproduce greedy "
            f"decoding (first divergence at index {diverge}).\n"
            f"Got:            {token_ids}\n"
            f"Expected (MLX): {mlx_expected}\n"
            + (f"Fallback:       {fallback}\n" if fallback is not None else "")
        )

    @pytest.mark.slow
    def test_continuous_batching_drafts_all_requests(self, sd_engine):
        # All prompts co-scheduled in one batch exercises the batched draft
        # loop (mixed-length ingest + K-step decode across requests). Every
        # request must complete full-length (drafts accepted, no stall) and
        # begin with the same greedy first token as single-stream. Tokens may
        # diverge later by batch-size FP (documented), so we do not require
        # full identity against the batch-size-1 goldens here.
        batched = _greedy(sd_engine, list(PROMPTS))

        assert set(batched) == set(PROMPTS)
        for prompt in PROMPTS:
            ids = batched[prompt]
            golden = GOLDEN_MLX[prompt]
            matched = next(
                (i for i in range(min(len(ids), len(golden))) if ids[i] != golden[i]),
                min(len(ids), len(golden)),
            )
            print(f"\n  prompt: {prompt!r}")
            print(f"  ids:    {ids}  (prefix match vs greedy: {matched}/{len(golden)})")
            assert len(ids) == MAX_TOKENS, (
                f"{prompt!r}: expected {MAX_TOKENS} tokens (drafts should be "
                f"accepted under draft==target greedy), got {len(ids)}"
            )
            assert ids[0] == golden[0], (
                f"{prompt!r}: first committed token {ids[0]} != greedy {golden[0]}"
            )

    @pytest.mark.slow
    def test_long_prompt_crosses_block_boundary(self, sd_engine):
        # A long prompt makes drafting cross KV-cache block boundaries
        # (block_size=16) during decode — the case that crashed when the draft
        # reused the target's exact-fit block_ids instead of owning its blocks.
        # Must complete full-length without an IndexError.
        long_prompt = (
            "Explain, step by step and in detail, how a transformer language "
            "model turns a sequence of input tokens into output logits: token "
            "and positional embeddings, multi-head self-attention with key and "
            "value caching, residual connections and layer normalization, the "
            "feed-forward network, and the output projection. Then explain why "
            "key-value caching is essential for efficient autoregressive "
            "decoding of long sequences."
        )
        max_tokens = 24  # > block_size, so decode crosses at least one boundary
        sp = SamplingParams(temperature=0, max_tokens=max_tokens)
        ids = list(sd_engine.generate([long_prompt], sp)[0].outputs[0].token_ids)
        print(f"\n  long-prompt completion len: {len(ids)}")
        assert len(ids) == max_tokens, (
            f"expected {max_tokens} tokens, got {len(ids)} — draft likely "
            "crossed a KV block boundary (own-block-allocator regression)"
        )
