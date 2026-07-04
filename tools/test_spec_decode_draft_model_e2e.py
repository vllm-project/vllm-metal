# SPDX-License-Identifier: Apache-2.0
"""End-to-end gate for draft-model speculative decoding on Metal.

With draft == target == ``Qwen/Qwen3-0.6B`` under greedy decoding, every draft
token equals the target's own argmax, so all drafts are accepted and
speculative decoding must reproduce plain greedy decoding token-for-token. The
greedy goldens are the same ones ``test_paged_deterministic`` pins.

Process isolation: build exactly ONE engine per process (target + draft model +
two paged caches). Run on its own:

    PYTHONPATH=$PWD VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      python -m pytest tests/test_spec_decode_draft_model_e2e.py -v -s -m slow

Numerical note: the verify forward runs the K+1 verification rows through the
QKV/MLP/LM-head GEMMs with M=K+1 (vs M=1 for plain decode); per the documented
MLX batch/M-size FP behavior this can shift one argmax, so two prompts have a
documented paged fallback.
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
    """One draft-model SD engine (target + draft + two paged caches)."""
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=512,
        max_num_seqs=1,
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
    def test_single_stream_lossless_and_fully_accepted(self, sd_engine, prompt):
        """Greedy draft-model SD must (a) reproduce plain greedy token-for-token
        and (b) accept every draft. Output alone is not enough: verification
        corrects an inert drafter, so acceptance is the only proof the draft
        model actually ran and proposed the target's tokens.
        """
        runner = sd_engine.llm_engine.model_executor.driver_worker.model_runner
        controller = runner._spec_decode_controller
        original_verify = controller.verify_greedy
        counts = {"accepted": 0, "drafted": 0}

        def counting_verify(logits, decode_reqs, decode_segments, *, logitsprocs):
            # Each verified row is (accepted drafts) + 1 trailing token (bonus on
            # full accept, else the target's correction), so accepted = len - 1.
            result = original_verify(
                logits, decode_reqs, decode_segments, logitsprocs=logitsprocs
            )
            for segment, output_ids in zip(decode_segments, result, strict=True):
                counts["drafted"] += len(segment.draft_token_ids)
                counts["accepted"] += len(output_ids) - 1
            return result

        with pytest.MonkeyPatch.context() as mp:
            mp.setattr(controller, "verify_greedy", counting_verify)
            token_ids = _greedy(sd_engine, [prompt])[prompt]

        # (a) lossless vs greedy goldens (two prompts sit on a documented ULP tie)
        expected = GOLDEN_MLX[prompt]
        fallback = GOLDEN_PAGED_FALLBACK.get(prompt)
        assert token_ids in (expected, fallback), (
            f"SD output for {prompt!r} did not reproduce greedy decoding.\n"
            f"  got:      {token_ids}\n  expected: {expected}"
            + (f"\n  fallback: {fallback}" if fallback else "")
        )

        # (b) every draft accepted => the drafter genuinely proposed the target's
        # tokens (a shortfall means stale draft KV or a wrong block table)
        assert counts["drafted"] > 0, "no drafts proposed — inert drafter"
        assert counts["accepted"] == counts["drafted"], (
            f"single-stream must accept every draft, got "
            f"{counts['accepted']}/{counts['drafted']}"
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
        assert len(ids) == max_tokens, (
            f"expected {max_tokens} tokens, got {len(ids)} — draft likely "
            "crossed a KV block boundary (own-block-allocator regression)"
        )
