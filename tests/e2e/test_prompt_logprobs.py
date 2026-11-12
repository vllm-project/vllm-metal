# SPDX-License-Identifier: Apache-2.0
"""Prompt logprobs through the paged engine."""

from __future__ import annotations

import pytest


@pytest.mark.slow
def test_prompt_logprobs_end_to_end_paged():
    """Bare ``prompt_logprobs`` returns one real entry per prompt token (#680).

    Covers both reported symptoms at the engine level: the silent ``[None]``
    (the dict below used to stay empty) and the data the OpenAI ``echo`` +
    ``logprobs`` path indexes into (its 500 came from these tensors never
    existing).  ``max_num_batched_tokens`` forces chunked prefill so the
    accounting crosses chunk boundaries.
    """
    from vllm import LLM, SamplingParams

    with pytest.MonkeyPatch.context() as mp:
        mp.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

        llm = LLM(
            model="Qwen/Qwen3-0.6B",
            max_model_len=512,
            max_num_seqs=2,
            max_num_batched_tokens=16,
            enable_chunked_prefill=True,
            gpu_memory_utilization=0.3,
        )
        prompt = (
            "The three most important properties of a distributed cache "
            "are consistency, availability, and partition tolerance."
        )
        sp = SamplingParams(temperature=0, max_tokens=4, prompt_logprobs=1, logprobs=1)
        [output] = llm.generate([prompt], sp)

        prompt_len = len(output.prompt_token_ids)
        assert prompt_len > 16, "prompt must span multiple prefill chunks"
        prompt_logprobs = output.prompt_logprobs
        assert prompt_logprobs is not None
        assert len(prompt_logprobs) == prompt_len
        assert prompt_logprobs[0] is None
        for position, (token_id, entry) in enumerate(
            zip(output.prompt_token_ids[1:], prompt_logprobs[1:], strict=True), 1
        ):
            assert entry is not None, f"position {position} silently empty"
            assert token_id in entry, f"prompt token missing at {position}"
            assert entry[token_id].logprob <= 0.0
