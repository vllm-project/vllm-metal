# SPDX-License-Identifier: Apache-2.0
"""Qualified checkpoint: exact greedy parity, pressure, cancellation, and limits.

Run explicitly with ``pytest -m slow tests/test_block_draft_serving_e2e.py``.
"""

import json
import math
import os

import pytest

from tests.scheduler_helpers import (
    assert_v2_model_runner,
    track_preemptions_and_resumptions,
)

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


def _serve(
    mode, baseline_path, verify_window, draft_quantization=None, draft_checkpoint=None
):
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
    if spec is not None and draft_checkpoint is not None:
        spec.update(model=draft_checkpoint[0], revision=draft_checkpoint[1])
    llm = _block_draft_llm(
        speculative_config=spec,
        additional_config={"dspark_draft_quantization": draft_quantization}
        if draft_quantization is not None
        else {},
    )
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
        if draft_quantization is not None:
            import mlx.nn as nn

            assert isinstance(proposer.draft_model.lm_head, nn.QuantizedLinear)
        assert proposer.cache.storage is runner.paged_attention_runtime.storage
        stats = {
            "drafted": 0,
            "accepted": 0,
            "rejected": 0,
            "preemptions": 0,
            "resumed": 0,
            "injected_rejection": 0,
        }
        propose = proposer.propose
        inject_rejection = False

        def check_propose(ctx):
            nonlocal inject_rejection
            result = propose(ctx)
            if result is None:
                return result
            for req_id, drafts in zip(
                result.req_ids, result.draft_token_ids, strict=True
            ):
                state = ctx.request_states[req_id]
                limit = state.sampling_params.max_tokens
                assert (
                    limit is None or len(drafts) + 1 <= limit - state.generated_tokens
                )
            if inject_rejection:
                # Exercise correction even when this checkpoint predicts the
                # repetitive workload perfectly, as in the width-change test.
                state = ctx.request_states[result.req_ids[0]]
                prompt_index = prompts.index(state.token_ids[: state.prompt_len])
                correct = baseline[prompt_index][state.generated_tokens]
                result.draft_token_ids[0][0] = (
                    correct + 1
                ) % proposer.model.config.vocab_size
                inject_rejection = False
                stats["injected_rejection"] += 1
            return result

        proposer.propose = check_propose
        verify = runner._spec_decode_controller.verify_greedy

        def record_verify(logits, requests, segments):
            result = verify(logits, requests, segments)
            states = dict(requests)
            for segment, tokens in zip(segments, result, strict=True):
                state = states[segment.req_id]
                limit = state.sampling_params.max_tokens
                assert limit is None or len(tokens) <= limit - state.generated_tokens
                stats["drafted"] += len(segment.draft_token_ids)
                stats["accepted"] += len(tokens) - 1
                stats["rejected"] += len(segment.draft_token_ids) - len(tokens) + 1
            return result

        runner._spec_decode_controller.verify_greedy = record_verify
        assert_v2_model_runner(llm)
        track_preemptions_and_resumptions(scheduler, stats, resumption_key="resumed")
        # The prefill's first output can finish the request by itself.
        assert (
            generate(
                prompts[0], SamplingParams(temperature=0, max_tokens=1, ignore_eos=True)
            )
            == baseline[0][:1]
        )
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
        inject_rejection = True
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
            drafted = stats["drafted"]
            assert (
                generate(
                    prompts[0],
                    SamplingParams(temperature=0, max_tokens=budget, ignore_eos=True),
                )
                == baseline[0][:budget]
            )
            if budget <= 2:
                assert stats["drafted"] == drafted
            assert scheduler.kv_cache_manager.block_pool.get_num_free_blocks() == free
        # A short request must not truncate its neighbor's proposal or inherit
        # that neighbor's remaining budget in the shared draft batch.
        results = llm.generate(
            [{"prompt_token_ids": prompt} for prompt in prompts[:2]],
            [
                SamplingParams(temperature=0, max_tokens=budget, ignore_eos=True)
                for budget in (3, 9)
            ],
            use_tqdm=False,
        )
        assert [list(output.outputs[0].token_ids) for output in results] == [
            baseline[0][:3],
            baseline[1][:9],
        ]
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
@pytest.mark.parametrize(
    "method,draft_quantization,draft_checkpoint",
    [
        ("dflash", None, None),
        ("dspark", None, None),
        ("dspark", "q4", None),
        pytest.param(
            "dspark",
            None,
            (
                "RedHatAI/Qwen3-4B-speculator.dspark",
                "bdffaa5f011da5a1a58ad841a1f6ced3e87f2f23",
            ),
            id="speculators",
        ),
    ],
)
@pytest.mark.parametrize("verify_window", [False, True])
def test_block_draft_serving_parity_and_lifecycle(
    tmp_path,
    run_in_spawn_process,
    verify_window,
    method,
    draft_quantization,
    draft_checkpoint,
):
    baseline = tmp_path / "target.json"
    run_in_spawn_process(_serve, "target", baseline, verify_window, label="target")
    run_in_spawn_process(
        _serve,
        method,
        baseline,
        verify_window,
        draft_quantization,
        draft_checkpoint,
        label=method,
    )


def _sleep_round_trip(method, draft, revision, quantization, caching, path):
    _spawn_env(True)
    import mlx.core as mx
    from vllm import SamplingParams

    llm = _block_draft_llm(
        revision="4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25",
        speculative_config={
            "method": method,
            "model": draft,
            "revision": revision,
            "num_speculative_tokens": DRAFT_WIDTHS[method],
        },
        additional_config={"dspark_draft_quantization": quantization}
        if quantization
        else {},
        enable_sleep_mode=True,
        enable_prefix_caching=caching,
        num_gpu_blocks_override=256,
        gpu_memory_utilization=0.5,
        max_model_len=256,
    )
    engine = llm.llm_engine
    runner = engine.model_executor.driver_worker.model_runner
    storage = runner._paged_attention_runtime.storage
    assert runner._drafter.cache.storage is storage
    assert_v2_model_runner(llm)
    prompts = [
        {
            "prompt": "A student saves a document, closes the program, and opens "
            "the same document the next day. Explain how a computer stores and "
            "retrieves that file, including the different roles of memory and storage."
        },
        {
            "prompt": "Two identical plants receive the same amount of water, "
            "but one grows near a window and the other stays in a dark room. "
            "Explain why sunlight affects their growth and describe photosynthesis."
        },
    ]
    params = SamplingParams(temperature=0, max_tokens=32, ignore_eos=True)
    counts = {"drafted": 0, "accepted": 0, "target_decisions": 0}
    verify = runner._spec_decode_controller.verify_greedy

    def record_verify(logits, requests, segments):
        result = verify(logits, requests, segments)
        targets = mx.argmax(logits[0], axis=-1).tolist()
        for segment, tokens in zip(segments, result, strict=True):
            assert (
                tokens == targets[segment.start_row : segment.start_row + len(tokens)]
            )
            counts["drafted"] += len(segment.draft_token_ids)
            counts["accepted"] += len(tokens) - 1
            counts["target_decisions"] += len(tokens)
        return result

    runner._spec_decode_controller.verify_greedy = record_verify

    def generate(inputs):
        before = counts.copy()
        outputs = llm.generate(inputs, params, use_tqdm=False)
        # In-process generate returns before finished-request cleanup reaches
        # the worker. Drain it as the continuous engine loop would before pause.
        assert engine.step() == []
        return {
            "tokens": [list(o.outputs[0].token_ids) for o in outputs],
            "counters": {k: counts[k] - before[k] for k in counts},
            "cached_tokens": sum(o.num_cached_tokens for o in outputs),
        }

    records = {"storage_bytes": storage.nbytes, "rounds": []}
    try:
        cold = generate(prompts)
        warm = generate(prompts)
        assert cold["counters"]["drafted"] > 0
        records.update(cold=cold, warm=warm)
        if caching:
            assert warm["cached_tokens"] > 0
        for level in (1, 0):
            before = mx.get_active_memory()
            llm.sleep(level=level)
            if level == 0:
                llm.release_kv_cache_memory()
            freed = before - mx.get_active_memory()
            assert storage.released
            assert freed >= storage.nbytes
            assert llm.wake_up(tags=["kv_cache"] if level == 0 else None)
            assert not storage.released
            after_cold = generate(prompts)
            after_warm = generate(prompts)
            assert after_cold == cold
            assert after_warm == warm
            records["rounds"].append(
                {
                    "level": level,
                    "freed_bytes": freed,
                    "cold": after_cold,
                    "warm": after_warm,
                }
            )

        # A kept request is re-prefilled after wake, not resumed from discarded KV.
        assert llm.reset_prefix_cache()
        uninterrupted = generate(prompts[:1])
        records["uninterrupted"] = uninterrupted
        for release in (False, True):
            label = "kept_sleep" if release else "kept_recompute"
            assert llm.reset_prefix_cache()
            engine.add_request(
                label,
                prompts[0],
                SamplingParams(temperature=0, max_tokens=32, ignore_eos=True),
            )
            for _ in range(50):
                output = engine.step()
                if output and len(output[0].outputs[0].token_ids) >= 2:
                    assert not output[0].finished
                    records[label + "_prefix"] = list(output[0].outputs[0].token_ids)
                    break
            else:
                raise AssertionError("Request did not start decoding")
            llm.sleep(level=int(release), mode="keep")
            if not release:
                assert llm.reset_prefix_cache(reset_running_requests=True)
            assert storage.released == release
            assert llm.wake_up()
            for _ in range(100):
                output = engine.step()
                if output and output[0].finished:
                    records[label] = list(output[0].outputs[0].token_ids)
                    break
            else:
                raise AssertionError("Kept request did not finish after wake")
            assert engine.step() == []
        # Re-prefill can perturb low-precision near ties. The matched control
        # recomputes too, but never releases the backing allocation.
        assert records["kept_sleep_prefix"] == records["kept_recompute_prefix"]
        assert records["kept_sleep"] == records["kept_recompute"]
        records["kept_matches_uninterrupted"] = (
            records["kept_sleep"] == uninterrupted["tokens"][0]
        )
        records["all_target_decisions"] = counts["target_decisions"]
    finally:
        path.write_text(json.dumps(records, indent=2))
        engine.engine_core.shutdown()


@pytest.mark.slow
@pytest.mark.parametrize("caching", [False, True])
@pytest.mark.parametrize(
    "method,draft,revision,quantization",
    [
        (
            "dflash",
            DRAFT_MODELS["dflash"],
            "b74e3a329c4d963783143b1e970d95b002be72bd",
            None,
        ),
        (
            "dspark",
            DRAFT_MODELS["dspark"],
            "3457dff1417cb84927f6098a5fcb7cee85c934b7",
            None,
        ),
        (
            "dspark",
            DRAFT_MODELS["dspark"],
            "3457dff1417cb84927f6098a5fcb7cee85c934b7",
            "q4",
        ),
        (
            "dspark",
            "RedHatAI/Qwen3-4B-speculator.dspark",
            "bdffaa5f011da5a1a58ad841a1f6ced3e87f2f23",
            None,
        ),
    ],
)
def test_block_draft_sleep_round_trip(
    tmp_path, run_in_spawn_process, method, draft, revision, quantization, caching
):
    run_in_spawn_process(
        _sleep_round_trip,
        method,
        draft,
        revision,
        quantization,
        caching,
        tmp_path / "sleep.json",
        label="sleep",
    )
