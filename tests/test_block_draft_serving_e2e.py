# SPDX-License-Identifier: Apache-2.0
"""Qualified checkpoint: exact greedy parity, pressure, cancellation, and limits.

Run explicitly with ``pytest -m slow tests/test_block_draft_serving_e2e.py``.
"""

import json
import math
import os

import pytest

DRAFT_MODELS = {
    "dflash": "z-lab/Qwen3-4B-DFlash-b16",
    "dspark": "deepseek-ai/dspark_qwen3_4b_block7",
}
DRAFT_WIDTHS = {"dflash": 3, "dspark": 7}


def _spawn_env(verify_window):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_METAL_SPEC_VERIFY_WINDOW"] = "1" if verify_window else "0"


def _block_draft_llm(**overrides):
    """LLM with the shared block-draft test configuration, tuned by *overrides*."""
    from vllm import LLM

    return LLM(
        **{
            "model": "mlx-community/Qwen3-4B-4bit",
            "max_model_len": 128,
            "max_num_seqs": 2,
            "max_num_batched_tokens": 32,
            "block_size": 16,
            "num_gpu_blocks_override": 10,
            "gpu_memory_utilization": 0.25,
            "enable_prefix_caching": False,
            "async_scheduling": False,
            **overrides,
        }
    )


def _serve(mode, baseline_path, verify_window):
    _spawn_env(verify_window)
    from vllm import SamplingParams
    from vllm.sampling_params import StructuredOutputsParams

    spec = (
        None
        if mode == "target"
        else {
            "method": mode,
            "model": DRAFT_MODELS[mode],
            "num_speculative_tokens": DRAFT_WIDTHS[mode],
        }
    )
    llm = _block_draft_llm(speculative_config=spec)
    engine = llm.llm_engine
    runner = engine.model_executor.driver_worker.model_runner
    scheduler = engine.engine_core.engine_core.scheduler
    tokenizer = llm.get_tokenizer()
    prompts = [
        tokenizer.encode("Explain how a computer works. " * 20)[:n]
        for n in (13, 15, 16, 63)
    ] + [
        tokenizer.encode("Describe why plants grow. " * 20)[:61],
        tokenizer.encode("Explain how a computer works. " * 20)[:33],
    ]
    # Preserve the original boundary/pressure budgets. The extra 33-token
    # prompt supplies the 16-token fallback below; longer continuations can
    # hit the existing BF16 target batch-shape near ties (also on upstream).
    budgets = [32, 32, 32, 48, 48, 16]
    sampling = SamplingParams(temperature=0, max_tokens=48, ignore_eos=True)
    eos_prompt = tokenizer.apply_chat_template(
        [{"role": "user", "content": "Reply with just the word OK."}],
        tokenize=True,
        return_dict=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )

    def generate(prompt, params=sampling):
        return list(
            llm.generate([{"prompt_token_ids": prompt}], params, use_tqdm=False)[0]
            .outputs[0]
            .token_ids
        )

    def drain():
        outputs = {}
        for _ in range(500):
            if not engine.has_unfinished_requests():
                return outputs
            for output in engine.step():
                if output.finished:
                    outputs[output.request_id] = list(output.outputs[0].token_ids)
        raise AssertionError("Scheduler failed to make progress")

    try:
        if mode == "target":
            eos = generate(eos_prompt, SamplingParams(temperature=0, max_tokens=32))
            baseline_path.write_text(
                json.dumps(
                    {
                        "eos": eos,
                        "outputs": [
                            generate(
                                prompt,
                                SamplingParams(
                                    temperature=0,
                                    max_tokens=budgets[i],
                                    ignore_eos=True,
                                ),
                            )
                            for i, prompt in enumerate(prompts)
                        ],
                    }
                )
            )
            return
        reference = json.loads(baseline_path.read_text())
        baseline = reference["outputs"]
        proposer = runner._drafter
        assert proposer.cache.storage is runner.paged_attention_runtime.storage
        stats = {
            "drafted": 0,
            "accepted": 0,
            "rejected": 0,
            "preemptions": 0,
            "resumed": 0,
        }
        verify, schedule = (
            runner._spec_decode_controller.verify_greedy,
            scheduler.schedule,
        )

        def record_verify(logits, requests, segments):
            result = verify(logits, requests, segments)
            for segment, tokens in zip(segments, result, strict=True):
                stats["drafted"] += len(segment.draft_token_ids)
                stats["accepted"] += len(tokens) - 1
                stats["rejected"] += len(segment.draft_token_ids) - len(tokens) + 1
            return result

        def record_schedule(*args, **kwargs):
            result = schedule(*args, **kwargs)
            stats["preemptions"] += len(result.preempted_req_ids)
            stats["resumed"] += len(result.scheduled_cached_reqs.resumed_req_ids)
            return result

        runner._spec_decode_controller.verify_greedy = record_verify
        scheduler.schedule = record_schedule
        # Boundary cases stay within 32 outputs; pressure/cancellation cases
        # continue for 48. Longer repetitive continuations can hit BF16 ties
        # between punctuation tokens across different target batch shapes.
        actual = [
            generate(
                prompt,
                SamplingParams(temperature=0, max_tokens=budgets[i], ignore_eos=True),
            )
            for i, prompt in enumerate(prompts)
        ]
        baseline_path.with_name(f"{mode}-actual.json").write_text(json.dumps(actual))
        assert actual == baseline, [
            (i, j, a, b)
            for i, (observed, expected) in enumerate(zip(actual, baseline, strict=True))
            for j, (a, b) in enumerate(zip(observed, expected, strict=True))
            if a != b
        ][:3]
        free = scheduler.kv_cache_manager.block_pool.get_num_free_blocks()
        for i in (3, 4):
            engine.add_request(str(i), {"prompt_token_ids": prompts[i]}, sampling)
        pressured = drain()
        assert [pressured[str(i)] for i in (3, 4)] == baseline[3:5]
        assert all(value > 0 for value in stats.values()), stats
        assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free

        # Abort after actual drafting; reuse the public ID with a different
        # prompt. Freed pages cannot retain committed-feature validity.
        old_id = engine.add_request(
            "reused", {"prompt_token_ids": prompts[3]}, sampling
        )
        drafted = stats["drafted"]
        for _ in range(30):
            engine.step()
            if stats["drafted"] > drafted:
                break
        assert stats["drafted"] > drafted
        engine.abort_request(["reused"])
        new_id = engine.add_request(
            "reused", {"prompt_token_ids": prompts[4]}, sampling
        )
        engine.step()
        if old_id != new_id:
            assert old_id not in proposer._valid_ends
        assert drain()["reused"] == baseline[4]
        assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free

        limit_prompt = tokenizer.encode("Explain how a computer works. " * 30)[:125]
        limited = generate(limit_prompt)
        assert len(limited) == 3
        assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
        print(f"{mode} lifecycle:", stats, flush=True)
        drafted = stats["drafted"]
        constrained = (
            llm.generate(
                ["Answer yes or no: is water wet?"],
                SamplingParams(
                    temperature=0,
                    max_tokens=8,
                    structured_outputs=StructuredOutputsParams(choice=["yes", "no"]),
                ),
                use_tqdm=False,
            )[0]
            .outputs[0]
            .text
        )
        assert constrained in ("yes", "no")
        assert stats["drafted"] == drafted

        # Finishing inside a speculative block must not leak extra tokens or
        # leave pages unavailable for the next request.
        for budget in (1, 2, 3, 5):
            assert (
                generate(
                    prompts[0],
                    SamplingParams(temperature=0, max_tokens=budget, ignore_eos=True),
                )
                == baseline[0][:budget]
            )
            assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
        stop_index = next(
            i for i in range(4, 16) if baseline[0][i] not in baseline[0][:i]
        )
        stop = baseline[0][stop_index]
        stopped = llm.generate(
            [{"prompt_token_ids": prompts[0]}],
            SamplingParams(
                temperature=0, max_tokens=32, ignore_eos=True, stop_token_ids=[stop]
            ),
            use_tqdm=False,
        )[0].outputs[0]
        assert list(stopped.token_ids) == baseline[0][: stop_index + 1]
        assert stopped.finish_reason == "stop" and stopped.stop_reason == stop
        eos = llm.generate(
            [{"prompt_token_ids": eos_prompt}],
            SamplingParams(temperature=0, max_tokens=32),
            use_tqdm=False,
        )[0].outputs[0]
        assert list(eos.token_ids) == reference["eos"]
        assert eos.finish_reason == "stop"
        assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free

        mixed = {"decode_prefill": False, "spec_plain": False}
        sample = runner._sample_paged_batch

        def record_mixed(*args, **kwargs):
            state = runner._execute_model_state
            segments = state.decode_segments
            mixed["decode_prefill"] |= bool(segments and state.prefill_reqs)
            mixed["spec_plain"] |= any(s.draft_token_ids for s in segments) and any(
                not s.draft_token_ids for s in segments
            )
            return sample(*args, **kwargs)

        runner._sample_paged_batch = record_mixed
        for temperature in (0, 0.7):
            mixed.update(decode_prefill=False, spec_plain=False)
            drafted = stats["drafted"]
            # The 33-token fallback spans two prefill chunks, then decodes
            # while the greedy request is still live even with K=7 drafting.
            results = llm.generate(
                [{"prompt_token_ids": prompts[i]} for i in (0, 5)],
                [
                    SamplingParams(temperature=0, max_tokens=32, ignore_eos=True),
                    SamplingParams(
                        temperature=temperature,
                        seed=7,
                        max_tokens=16,
                        ignore_eos=True,
                        logprobs=1,
                    ),
                ],
                use_tqdm=False,
            )
            assert list(results[0].outputs[0].token_ids) == baseline[0]
            fallback = results[1].outputs[0]
            assert len(fallback.token_ids) == len(fallback.logprobs) == 16
            assert all(
                math.isfinite(row[token].logprob)
                for token, row in zip(
                    fallback.token_ids, fallback.logprobs, strict=True
                )
            )
            if temperature == 0:
                assert list(fallback.token_ids) == baseline[5][:16]
            assert all(mixed.values()), mixed
            assert stats["drafted"] > drafted
            assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
    finally:
        engine.engine_core.shutdown()


@pytest.mark.slow
@pytest.mark.parametrize("method", ["dflash", "dspark"])
@pytest.mark.parametrize("verify_window", [False, True])
def test_block_draft_serving_parity_and_lifecycle(
    tmp_path, run_in_spawn_process, verify_window, method
):
    baseline = tmp_path / "target.json"
    run_in_spawn_process(_serve, "target", baseline, verify_window, label="target")
    run_in_spawn_process(_serve, method, baseline, verify_window, label=method)
