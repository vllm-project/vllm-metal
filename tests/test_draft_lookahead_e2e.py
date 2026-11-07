# SPDX-License-Identifier: Apache-2.0
"""Real-model draft lookahead under scheduler pressure and cancellation.

Run explicitly with ``pytest -m slow tests/test_draft_lookahead_e2e.py``.
Each case owns one engine in a spawned process; Metal is not fork-safe.
"""

import os

import pytest

from tests.scheduler_helpers import (
    assert_v2_model_runner,
    track_preemptions_and_resumptions,
)


def _run_lookahead_lifecycle(prefix_caching, target_model="Qwen/Qwen3-0.6B"):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=target_model,
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        block_size=16,
        num_gpu_blocks_override=10,
        gpu_memory_utilization=0.2,
        enable_prefix_caching=prefix_caching,
        async_scheduling=False,
        speculative_config={
            "method": "draft_model",
            "model": "Qwen/Qwen3-0.6B",
            "num_speculative_tokens": 3,
        },
    )
    engine = llm.llm_engine
    scheduler = engine.engine_core.engine_core.scheduler
    runner = engine.model_executor.driver_worker.model_runner
    proposer = runner._drafter
    stats = {"preemptions": 0, "resumptions": 0, "drafted": 0, "accepted": 0}
    released = set()
    original_verify = runner._spec_decode_controller.verify
    original_release = proposer.release_requests

    def verify(logits, requests, segments, *args, **kwargs):
        outputs = original_verify(logits, requests, segments, *args, **kwargs)
        for segment, tokens in zip(segments, outputs, strict=True):
            stats["drafted"] += len(segment.draft_token_ids)
            stats["accepted"] += len(tokens) - 1
        return outputs

    def release(request_ids):
        released.update(request_ids)
        return original_release(request_ids)

    def drain():
        finished = {}
        for _ in range(500):
            if not engine.has_unfinished_requests():
                return finished
            for output in engine.step():
                if output.finished:
                    finished[output.request_id] = list(output.outputs[0].token_ids)
        raise AssertionError("Scheduler made insufficient progress within 500 steps")

    try:
        tokenizer = llm.get_tokenizer()
        # Distinct prefixes prevent sharing from hiding allocation pressure.
        prompts = [
            tokenizer.encode("Explain how a computer works. " * 20)[:63],
            tokenizer.encode("Describe why plants grow. " * 20)[:61],
        ]
        sampling = SamplingParams(temperature=0, max_tokens=48, ignore_eos=True)
        baseline = []
        for prompt in prompts:
            result = llm.generate(
                [{"prompt_token_ids": prompt}], sampling, use_tqdm=False
            )[0]
            baseline.append(list(result.outputs[0].token_ids))
        assert all(len(tokens) == 48 for tokens in baseline)
        assert llm.reset_prefix_cache()
        free_before = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()

        assert_v2_model_runner(llm)
        track_preemptions_and_resumptions(scheduler, stats)
        runner._spec_decode_controller.verify = verify
        proposer.release_requests = release
        for i, prompt in enumerate(prompts):
            engine.add_request(f"pressure-{i}", {"prompt_token_ids": prompt}, sampling)
        pressured = drain()
        assert [pressured[f"pressure-{i}"] for i in range(2)] == baseline
        assert stats["preemptions"] > 0 and stats["resumptions"] > 0
        assert stats["drafted"] > 0 and stats["accepted"] > 0
        assert (
            scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_before
        )

        # Abort only after actual speculative KV exists, then reuse the public
        # request ID with a different prompt. The runner must clear the old KV
        # validity state before the replacement request drafts.
        old_id = engine.add_request(
            "reused", {"prompt_token_ids": prompts[0]}, sampling
        )
        for _ in range(20):
            engine.step()
            if old_id in proposer._spec_kv_writes:
                break
        assert old_id in proposer._spec_kv_writes
        engine.abort_request(["reused"])
        new_id = engine.add_request(
            "reused", {"prompt_token_ids": prompts[1]}, sampling
        )
        engine.step()
        if old_id != new_id:
            assert old_id not in proposer._draft_seq_lens
            assert old_id not in proposer._spec_kv_writes
        assert drain()["reused"] == baseline[1]
        assert old_id in released
        assert (
            scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_before
        )
        # The final speculative step must fit the engine's context limit,
        # even though the draft checkpoint supports a much longer context.
        limit_prompt = tokenizer.encode("Explain how a computer works. " * 30)[:125]
        limited = llm.generate(
            [{"prompt_token_ids": limit_prompt}], sampling, use_tqdm=False
        )[0].outputs[0]
        assert len(limited.token_ids) == 3
        assert limited.finish_reason == "length"
        assert (
            scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free_before
        )
        return stats
    finally:
        engine.engine_core.shutdown()


@pytest.mark.slow
@pytest.mark.parametrize("prefix_caching", [False, True])
def test_draft_lookahead_preemption_and_cancellation_e2e(
    prefix_caching, run_in_spawn_process
):
    run_in_spawn_process(_run_lookahead_lifecycle, prefix_caching)
