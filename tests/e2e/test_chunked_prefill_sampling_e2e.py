# SPDX-License-Identifier: Apache-2.0
"""Seeded chunked prefill with real scheduling, sampling, and draft KV writes.

Run explicitly with ``pytest -m slow tests/e2e/test_chunked_prefill_sampling_e2e.py``.
"""

import os

import pytest


def _run_chunked_prefill_sampling(with_draft):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import torch
    from vllm import LLM, SamplingParams

    model = "Qwen/Qwen3-0.6B"
    llm = LLM(
        model=model,
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        block_size=16,
        num_gpu_blocks_override=64,
        gpu_memory_utilization=0.2,
        enable_prefix_caching=False,
        async_scheduling=False,
        speculative_config=(
            {"method": "draft_model", "model": model, "num_speculative_tokens": 3}
            if with_draft
            else None
        ),
    )
    engine = llm.llm_engine
    runner = engine.model_executor.driver_worker.model_runner
    original_sample = runner._sample_paged_batch
    stats = {"intermediate": 0, "mixed": 0, "draft_ingests": 0, "final": 0}

    def sample(*args, **kwargs):
        state = runner._execute_model_state
        tracked = [
            (pr, pr.generator.get_state())
            for pr in state.prefill_reqs
            if pr.generator is not None
        ]
        result = original_sample(*args, **kwargs)
        batch, _ = result
        for pr, before in tracked:
            if pr.prompt_len is not None:
                stats["final"] += 1
                assert not torch.equal(pr.generator.get_state(), before)
                continue
            stats["intermediate"] += 1
            stats["mixed"] += bool(state.decode_reqs)
            assert torch.equal(pr.generator.get_state(), before), (
                "An intermediate chunk consumed the request's sampling RNG"
            )
            output_idx = batch.req_id_to_index[pr.req_id]
            assert batch.sampled_tokens[output_idx] == []
            assert batch.sample_logprobs[output_idx] is None
            assert runner._paged_request_seq_lens[pr.req_id] == (
                pr.start_pos + len(pr.token_ids)
            )
            if with_draft:
                assert runner._drafter._draft_seq_lens[pr.req_id] == (
                    pr.start_pos + len(pr.token_ids)
                )
                stats["draft_ingests"] += 1
        return result

    runner._sample_paged_batch = sample

    def drain():
        finished = {}
        for _ in range(100):
            if not engine.has_unfinished_requests():
                return finished
            for output in engine.step():
                if output.finished:
                    finished[output.request_id] = output.outputs[0]
        raise AssertionError("Engine did not finish within 100 steps")

    try:
        prompt = llm.get_tokenizer().encode(
            "Explain how a computer works and why memory matters. " * 20
        )[:90]
        params = SamplingParams(
            temperature=0.8,
            seed=7,
            max_tokens=8,
            ignore_eos=True,
            logprobs=2,
            prompt_logprobs=1,
        )
        solo = llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)[0]
        assert len(solo.prompt_logprobs) == len(prompt)
        assert len(solo.outputs[0].logprobs) == 8

        engine.add_request(
            "decode",
            {"prompt_token_ids": prompt[:4]},
            SamplingParams(temperature=0, max_tokens=24, ignore_eos=True),
        )
        engine.step()
        engine.add_request("chunked", {"prompt_token_ids": prompt}, params)
        outputs = drain()

        assert list(outputs["chunked"].token_ids) == list(solo.outputs[0].token_ids)
        assert len(outputs["chunked"].logprobs) == 8
        assert len(outputs["decode"].token_ids) == 24
        assert stats["intermediate"] >= 4
        assert stats["mixed"] >= 2
        assert stats["final"] == 2
        if with_draft:
            assert stats["draft_ingests"] == stats["intermediate"]
    finally:
        engine.engine_core.shutdown()


@pytest.mark.slow
@pytest.mark.parametrize("with_draft", [False, True])
def test_chunked_prefill_preserves_seeded_sampling_e2e(
    with_draft, run_in_spawn_process
):
    run_in_spawn_process(_run_chunked_prefill_sampling, with_draft)
