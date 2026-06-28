# SPDX-License-Identifier: Apache-2.0
"""End-to-end gate for n-gram speculative decoding on Metal.

N-gram drafting is lossless by construction: the verify half (greedy argmax over
the target's own logits) only ever emits the target's tokens, so SD output must
reproduce plain greedy decoding token-for-token. The greedy goldens are the same
ones ``test_paged_deterministic`` and the draft-model e2e test pin.

Output equality alone is not proof the drafter ran — verification corrects an inert
drafter, so a broken proposer would still produce the right tokens at zero speedup.
``test_ngram_accepts_on_repetitive_prompt`` adds the missing proof: a strongly
repetitive prompt where the n-gram matcher must find continuations, and we assert
the verify half *accepted* drafts.

Process isolation: build exactly ONE engine per process (target + n-gram proposer,
no draft model, no second cache). Run on its own:

    PYTHONPATH=$PWD VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      python -m pytest tools/test_ngram_spec_decode_e2e.py -v -s -m slow
"""

from __future__ import annotations

import os

import pytest
from vllm import LLM, SamplingParams

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 10
NUM_SPECULATIVE_TOKENS = 3
PROMPT_LOOKUP_MIN = 2
PROMPT_LOOKUP_MAX = 3

PROMPTS = [
    "The capital of France is",
    "The weather today is not",
    "One plus one equals",
    "The largest planet in our solar system is",
    "Water boils at a temperature of",
    "Machine learning is",
]

# Plain-greedy reference, shared with test_paged_deterministic and the draft-model
# e2e test. N-gram SD must reproduce these greedy tokens.
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

# Strongly repetitive prompt: the n-gram matcher reliably finds the "1 2 3 4"
# cycle and proposes its continuation, so the verify half accepts real drafts.
# REPETITIVE_GOLDEN is the plain-greedy (no-SD) decode, generated once and pinned
# so the losslessness check is SD-vs-greedy (not SD-vs-SD). Regenerate by running
# the same prompt under SamplingParams(temperature=0) with no speculative_config
# if the tokenizer/model changes.
REPETITIVE_PROMPT = "1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4 1 2 3 4"
REPETITIVE_MAX_TOKENS = 24
# fmt: off
REPETITIVE_GOLDEN = [
    220, 16, 220, 17, 220, 18, 220, 19, 220, 16, 220, 17,
    220, 18, 220, 19, 220, 16, 220, 17, 220, 18, 220, 19,
]
# fmt: on


@pytest.fixture(autouse=True, scope="module")
def _set_env():
    """Force the paged path (spec-decode verify requires it) with headroom.

    0.6 (not 0.2 like the draft-model e2e test) so the KV budget stays positive
    on an 8 GB M1: there the metal limit is ~5.7 GB, and after the ~1.2 GB model
    plus ~0.5 GB forward overhead, 0.2 leaves a negative KV budget. N-gram needs
    only one model (no draft), so 0.6 fits comfortably. Override the env var for
    larger machines.
    """
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        mp.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        if os.environ.get("VLLM_METAL_MEMORY_FRACTION") is None:
            mp.setenv("VLLM_METAL_MEMORY_FRACTION", "0.6")
        yield


@pytest.fixture(scope="module")
def sd_engine():
    """One n-gram SD engine (target + n-gram proposer, no draft model)."""
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=512,
        max_num_seqs=1,
        enable_prefix_caching=False,
        async_scheduling=False,
        speculative_config={
            "method": "ngram",
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
            "prompt_lookup_min": PROMPT_LOOKUP_MIN,
            "prompt_lookup_max": PROMPT_LOOKUP_MAX,
        },
    )

    vllm_config = llm.llm_engine.vllm_config
    spec_cfg = vllm_config.speculative_config
    assert spec_cfg is not None and spec_cfg.method == "ngram", (
        "expected vLLM to resolve method=ngram"
    )
    assert vllm_config.scheduler_config.async_scheduling is False, (
        "n-gram SD on Metal requires synchronous scheduling"
    )

    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    from vllm_metal.v1.ngram_proposer import NgramProposer

    assert isinstance(runner._drafter, NgramProposer), (
        f"expected NgramProposer drafter, got {type(runner._drafter)!r}"
    )
    return llm


def _greedy(llm: LLM, prompts: list[str], max_tokens: int) -> dict[str, list[int]]:
    sp = SamplingParams(temperature=0, max_tokens=max_tokens)
    return {o.prompt: list(o.outputs[0].token_ids) for o in llm.generate(prompts, sp)}


def _generate_with_acceptance_counting(llm: LLM, prompt: str, max_tokens: int):
    """Run greedy decode while counting drafted vs accepted draft tokens."""
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    controller = runner._spec_decode_controller
    original_verify = controller.verify_greedy
    counts = {"accepted": 0, "drafted": 0}

    def counting_verify(logits, decode_reqs, decode_segments, *, logitsprocs):
        # Each verified row is (accepted drafts) + 1 trailing token (bonus on full
        # accept, else the target's correction), so accepted = len - 1.
        result = original_verify(
            logits, decode_reqs, decode_segments, logitsprocs=logitsprocs
        )
        for segment, output_ids in zip(decode_segments, result, strict=True):
            counts["drafted"] += len(segment.draft_token_ids)
            counts["accepted"] += len(output_ids) - 1
        return result

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(controller, "verify_greedy", counting_verify)
        token_ids = _greedy(llm, [prompt], max_tokens)[prompt]

    return token_ids, counts


class TestNgramSpecDecode:
    @pytest.mark.slow
    @pytest.mark.parametrize("prompt", PROMPTS)
    def test_single_stream_lossless(self, sd_engine, prompt):
        """Greedy n-gram SD must reproduce plain greedy decoding token-for-token.

        N-gram drafts may be rejected for natural prompts (no repeated pattern),
        so unlike the draft-model test we do not require full acceptance here —
        losslessness is the invariant. Acceptance is proven separately on a
        repetitive prompt below.
        """
        token_ids = _greedy(sd_engine, [prompt], MAX_TOKENS)[prompt]

        expected = GOLDEN_MLX[prompt]
        fallback = GOLDEN_PAGED_FALLBACK.get(prompt)
        assert token_ids in (expected, fallback), (
            f"SD output for {prompt!r} did not reproduce greedy decoding.\n"
            f"  got:      {token_ids}\n  expected: {expected}"
            + (f"\n  fallback: {fallback}" if fallback else "")
        )

    @pytest.mark.slow
    def test_ngram_accepts_on_repetitive_prompt(self, sd_engine):
        """On a repetitive prompt the n-gram matcher must propose tokens that the
        target accepts. A zero-acceptance result means the drafter never ran or
        the array marshalling is wrong — verification would still emit correct
        output, so acceptance is the only proof the proposer is live.
        """
        token_ids, counts = _generate_with_acceptance_counting(
            sd_engine, REPETITIVE_PROMPT, REPETITIVE_MAX_TOKENS
        )

        assert counts["drafted"] > 0, "no drafts proposed — inert n-gram drafter"
        assert counts["accepted"] > 0, (
            f"n-gram proposed {counts['drafted']} tokens but none were accepted; "
            "the repetitive pattern should yield accepted drafts"
        )
        # Losslessness: SD output equals the pinned plain-greedy (no-SD) decode.
        assert token_ids == REPETITIVE_GOLDEN, (
            "n-gram SD diverged from greedy decoding on the repetitive prompt.\n"
            f"  sd:      {token_ids}\n  greedy:  {REPETITIVE_GOLDEN}"
        )


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v", "-s", "-m", "slow"]))
