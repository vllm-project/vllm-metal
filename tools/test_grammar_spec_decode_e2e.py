# SPDX-License-Identifier: Apache-2.0
"""End-to-end gate for grammar-forced speculative decoding on Metal.

Grammar-forced drafting is lossless by construction: the verify half (greedy
argmax over the target's own logits, after the grammar bitmask has been applied
to every verification row) only ever emits the target's tokens, so SD output must
reproduce plain greedy decoding token-for-token.

Output equality alone is not proof the drafter ran — verification corrects an
inert drafter, so a broken proposer would still produce the right tokens at zero
speedup. ``test_accepts_on_structured_request`` supplies the missing proof by
asserting the verify half actually *accepted* drafts.

This proposer also needs a check the other three do not:
``test_inert_without_a_grammar`` pins the claim that a request carrying no
grammar produces no drafts at all. That is what keeps an unconstrained request on
Metal's one-row decode fast path, and it is not a detail — tool calling is *not*
grammar-constrained by default (a structural tag needs
``VLLM_ENFORCE_STRICT_TOOL_CALLING=1`` and ``"strict": true`` on a tool), so the
no-grammar path is the common one and must cost nothing.

Process isolation: build exactly ONE engine per process (target + grammar
proposer, no draft model, no second cache). Run on its own:

    PYTHONPATH=$PWD VLLM_ENABLE_V1_MULTIPROCESSING=0 \
      python -m pytest tools/test_grammar_spec_decode_e2e.py -v -s -m slow
"""

from __future__ import annotations

import json
import os

import pytest
from vllm import LLM, SamplingParams
from vllm.sampling_params import StructuredOutputsParams

MODEL_NAME = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 48
NUM_SPECULATIVE_TOKENS = 8

SCHEMA = {
    "type": "object",
    "properties": {
        "name": {"type": "string", "enum": ["get_weather", "get_forecast"]},
        "arguments": {
            "type": "object",
            "properties": {
                "location": {"type": "string"},
                "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
            },
            "required": ["location", "unit"],
        },
    },
    "required": ["name", "arguments"],
}
SCHEMA_JSON = json.dumps(SCHEMA)

PROMPT = "Weather in Paris in celsius. Reply with a JSON tool call."
UNCONSTRAINED_PROMPT = "The capital of France is"

# Plain-greedy (no-SD) decode of PROMPT under SCHEMA, pinned so the losslessness
# check is SD-vs-greedy rather than SD-vs-SD. Regenerate with the same prompt and
# schema under SamplingParams(temperature=0) and no speculative_config if the
# model or tokenizer changes.
# fmt: off
GREEDY_GOLDEN = [
    4913, 606, 788, 330, 455, 69364, 497, 330, 16370, 788, 5212, 2527,
    788, 330, 59604, 497, 330, 3843, 788, 330, 66, 40247, 30975, 151645,
]
# fmt: on


@pytest.fixture(autouse=True, scope="module")
def _set_env():
    """Force the paged path (spec-decode verify requires it) with headroom."""
    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
        mp.setenv("VLLM_METAL_USE_PAGED_ATTENTION", "1")
        if os.environ.get("VLLM_METAL_MEMORY_FRACTION") is None:
            mp.setenv("VLLM_METAL_MEMORY_FRACTION", "0.6")
        yield


@pytest.fixture(scope="module")
def sd_engine():
    """One grammar-forced SD engine (target + proposer, no draft model)."""
    llm = LLM(
        model=MODEL_NAME,
        max_model_len=512,
        max_num_seqs=1,
        enable_prefix_caching=False,
        async_scheduling=False,
        # disable_any_whitespace is load-bearing: with free whitespace between
        # JSON tokens the grammar forces almost nothing and this gate would pass
        # vacuously with near-zero drafts.
        structured_outputs_config={
            "backend": "xgrammar",
            "disable_any_whitespace": True,
        },
        speculative_config={
            "method": "custom_class",
            "model": "vllm_metal.v1.grammar_proposer.GrammarProposer",
            "num_speculative_tokens": NUM_SPECULATIVE_TOKENS,
        },
    )

    vllm_config = llm.llm_engine.vllm_config
    spec_cfg = vllm_config.speculative_config
    assert spec_cfg is not None and spec_cfg.method == "custom_class", (
        "expected vLLM to resolve method=custom_class"
    )
    assert vllm_config.scheduler_config.async_scheduling is False, (
        "grammar-forced SD on Metal requires synchronous scheduling"
    )

    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    from vllm_metal.v1.grammar_proposer import GrammarProposer

    assert isinstance(runner._drafter, GrammarProposer), (
        f"expected GrammarProposer drafter, got {type(runner._drafter)!r}"
    )
    return llm


def _structured_params(max_tokens: int) -> SamplingParams:
    return SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        structured_outputs=StructuredOutputsParams(json=SCHEMA_JSON),
    )


def _generate_with_acceptance_counting(
    llm: LLM, prompt: str, params: SamplingParams
) -> tuple[list[int], dict[str, int]]:
    """Run a greedy decode while counting drafted vs accepted draft tokens."""
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    controller = runner._spec_decode_controller
    original_verify = controller.verify_greedy
    counts = {"accepted": 0, "drafted": 0}

    def counting_verify(logits, decode_reqs, decode_segments):
        # Each verified row is (accepted drafts) + 1 trailing token (bonus on
        # full accept, else the target's correction), so accepted = len - 1.
        result = original_verify(logits, decode_reqs, decode_segments)
        for segment, output_ids in zip(decode_segments, result, strict=True):
            counts["drafted"] += len(segment.draft_token_ids)
            counts["accepted"] += len(output_ids) - 1
        return result

    with pytest.MonkeyPatch.context() as mp:
        mp.setattr(controller, "verify_greedy", counting_verify)
        outputs = llm.generate([prompt], params)

    return list(outputs[0].outputs[0].token_ids), counts


class TestGrammarSpecDecode:
    @pytest.mark.slow
    def test_single_stream_lossless(self, sd_engine):
        """Grammar-forced SD must reproduce plain greedy decoding exactly."""
        outputs = sd_engine.generate([PROMPT], _structured_params(MAX_TOKENS))
        token_ids = list(outputs[0].outputs[0].token_ids)

        assert token_ids == GREEDY_GOLDEN, (
            "SD output did not reproduce greedy decoding.\n"
            f"  got:      {token_ids}\n  expected: {GREEDY_GOLDEN}"
        )

    @pytest.mark.slow
    def test_output_is_schema_valid(self, sd_engine):
        """The bitmask is applied to draft rows too, so output stays in-grammar."""
        outputs = sd_engine.generate([PROMPT], _structured_params(MAX_TOKENS))
        payload = json.loads(outputs[0].outputs[0].text)
        assert payload["name"] in ("get_weather", "get_forecast")
        assert set(payload["arguments"]) == {"location", "unit"}

    @pytest.mark.slow
    def test_accepts_on_structured_request(self, sd_engine):
        """The proposer must land drafts the target accepts.

        A zero-acceptance result means the drafter never ran or its forced-string
        walk is wrong — verification would still emit correct output, so
        acceptance is the only proof the proposer is live.
        """
        token_ids, counts = _generate_with_acceptance_counting(
            sd_engine, PROMPT, _structured_params(MAX_TOKENS)
        )

        assert counts["drafted"] > 0, "no drafts proposed — inert grammar drafter"
        assert counts["accepted"] > 0, (
            f"proposed {counts['drafted']} tokens but none were accepted; a "
            "grammar-forced draft is legal by construction and should land"
        )
        assert token_ids == GREEDY_GOLDEN, "acceptance run diverged from greedy"

    @pytest.mark.slow
    def test_inert_without_a_grammar(self, sd_engine):
        """A request with no grammar must produce no drafts at all.

        This is what keeps unconstrained traffic on Metal's one-row decode fast
        path, where a wrong draft would cost roughly 2x. It matters in practice:
        tool calling is not grammar-constrained by default, so this is the common
        path, and the proposer must not tax it.
        """
        _, counts = _generate_with_acceptance_counting(
            sd_engine,
            UNCONSTRAINED_PROMPT,
            SamplingParams(temperature=0, max_tokens=16),
        )

        assert counts["drafted"] == 0, (
            f"proposer drafted {counts['drafted']} tokens for a request with no "
            "grammar; it must return None so the step stays one row wide"
        )
