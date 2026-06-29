#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Visual acceptance-rate smoke check for n-gram speculative decoding.

Runs a fixed set of four prompts through ``Qwen/Qwen3-0.6B`` with n-gram SD
enabled and reports per-prompt (and aggregate) ``drafted / accepted / rate``.
This is a *regression guard*, not an assertion-based test: the baseline
rates depend on the model, tokenizer, and n-gram parameters, so drift is
expected when any of those change. The existing ``test_ngram_spec_decode_e2e``
pins the lossless-decoding invariant and the acceptance-on-repetitive-prompt
invariant with strict asserts; this script complements it by exercising a
broader mix of prompt shapes (highly repetitive, structured, natural prose).

Expected baseline (indicative, not asserted):

    Prompt          Drafted     Accepted    Rate
    getter_setter   ~132        ~60         ~45%
    echo_numbers    ~150        ~150        ~100%
    json_repeat     ~147        ~81         ~55%
    natural_prose   ~180        ~118        ~66%
    Overall         ~609        ~409        ~67%

Run:

    PYTHONPATH=$PWD VLLM_ENABLE_V1_MULTIPROCESSING=0 \
        python tools/smoke_ngram_sd_acceptance.py
"""

from __future__ import annotations

import os
import sys

# Environment mirrors tools/test_ngram_spec_decode_e2e.py: paged attention
# (required by the verify half), single-process driver, ~0.6 memory fraction
# for the 8 GB M1. Override VLLM_METAL_MEMORY_FRACTION on larger machines.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("VLLM_METAL_USE_PAGED_ATTENTION", "1")
os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", "0.6")

from vllm import LLM, SamplingParams  # noqa: E402

MODEL = "Qwen/Qwen3-0.6B"
MAX_TOKENS = 200
NUM_SPECULATIVE_TOKENS = 3
PROMPT_LOOKUP_MIN = 2
PROMPT_LOOKUP_MAX = 3

PROMPTS: dict[str, str] = {
    "getter_setter": (
        "Write Python getter and setter methods for the following fields: "
        "name, age, email, phone, address, city, state, zip_code, country. "
        "Follow this exact pattern for each field:\n\n"
        "    def get_name(self):\n"
        "        return self._name\n\n"
        "    def set_name(self, name):\n"
        "        self._name = name\n\n"
        "Write all getter/setter pairs now:"
    ),
    "echo_numbers": (
        "1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 "
        "1 2 3 4 5 1 2 3 4 5 1 2 3 4 5 1 2 3 4 5"
    ),
    "json_repeat": (
        "Generate a JSON array of 10 users. Each user has the same structure: "
        '{"name": "...", "age": ..., "email": "..."}. '
        'Here is the first entry: {"name": "Alice", "age": 30, '
        '"email": "alice@example.com"}. '
        "Now generate all 10:"
    ),
    "natural_prose": "Explain the theory of general relativity in simple terms.",
}


def _build_engine() -> LLM:
    return LLM(
        model=MODEL,
        max_model_len=4096,
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


def _count_acceptance(
    engine: LLM, prompt: str, max_tokens: int
) -> tuple[list[int], dict[str, int]]:
    """Run greedy decode while counting drafted vs accepted draft tokens.

    Each verified row contributes ``len(output_ids) - 1`` accepted draft
    tokens (one trailing slot is the target's bonus/correction token) and
    ``len(segment.draft_token_ids)`` drafted tokens.
    """
    runner = engine.llm_engine.model_executor.driver_worker.model_runner
    controller = runner._spec_decode_controller
    original_verify = controller.verify_greedy
    counts = {"accepted": 0, "drafted": 0}

    def counting_verify(logits, decode_reqs, decode_segments, *, logitsprocs):
        result = original_verify(
            logits, decode_reqs, decode_segments, logitsprocs=logitsprocs
        )
        for segment, output_ids in zip(decode_segments, result, strict=True):
            counts["drafted"] += len(segment.draft_token_ids)
            counts["accepted"] += len(output_ids) - 1
        return result

    controller.verify_greedy = counting_verify  # type: ignore[method-assign]
    try:
        sp = SamplingParams(temperature=0, max_tokens=max_tokens)
        out = engine.generate([prompt], sp)[0]
        token_ids = list(out.outputs[0].token_ids)
    finally:
        controller.verify_greedy = original_verify  # type: ignore[method-assign]

    return token_ids, counts


def _print_table(rows: list[tuple[str, dict[str, int]]]) -> None:
    total_draft = sum(c["drafted"] for _, c in rows)
    total_accepted = sum(c["accepted"] for _, c in rows)

    name_w = max(max(len(name) for name, _ in rows), len("Prompt"), len("Overall"))
    print()
    print(f"{'Prompt':<{name_w}}  {'Drafted':>8}  {'Accepted':>9}  {'Rate':>6}")
    print(f"{'-' * name_w}  {'-' * 8}  {'-' * 9}  {'-' * 6}")
    for name, counts in rows:
        drafted = counts["drafted"]
        accepted = counts["accepted"]
        rate = (accepted / drafted * 100) if drafted else 0.0
        print(f"{name:<{name_w}}  {drafted:>8}  {accepted:>9}  {rate:>5.1f}%")
    if rows:
        overall = (total_accepted / total_draft * 100) if total_draft else 0.0
        print(f"{'-' * name_w}  {'-' * 8}  {'-' * 9}  {'-' * 6}")
        print(
            f"{'Overall':<{name_w}}  {total_draft:>8}  {total_accepted:>9}  {overall:>5.1f}%"
        )


def main() -> int:
    print(
        f"Building engine: {MODEL} with n-gram SD "
        f"(num_speculative_tokens={NUM_SPECULATIVE_TOKENS}, "
        f"lookup=[{PROMPT_LOOKUP_MIN}, {PROMPT_LOOKUP_MAX}])"
    )
    engine = _build_engine()

    spec_cfg = engine.llm_engine.vllm_config.speculative_config
    assert spec_cfg is not None and spec_cfg.method == "ngram"

    rows: list[tuple[str, dict[str, int]]] = []
    for name, prompt in PROMPTS.items():
        _tokens, counts = _count_acceptance(engine, prompt, MAX_TOKENS)
        rows.append((name, counts))

    _print_table(rows)
    return 0


if __name__ == "__main__":
    sys.exit(main())
