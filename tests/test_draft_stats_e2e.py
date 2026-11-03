# SPDX-License-Identifier: Apache-2.0
"""Exercise public draft statistics and acceptance through real engines."""

import json
import os

import pytest


def _run_stats_rpc(engine_multiprocessing):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = str(int(engine_multiprocessing))
    from vllm import LLM, SamplingParams

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=128,
        max_num_seqs=2,
        max_num_batched_tokens=32,
        block_size=16,
        num_gpu_blocks_override=10,
        gpu_memory_utilization=0.2,
        enable_prefix_caching=True,
        async_scheduling=False,
        speculative_config={
            "method": "draft_model",
            "model": "Qwen/Qwen3-0.6B",
            "max_model_len": 64,
            "num_speculative_tokens": 3,
            "num_speculative_tokens_per_batch_size": [[1, 1, 3], [2, 2, 1]],
        },
    )
    try:
        expected = {
            "num_context_limit_fallback_requests": 0,
            "min_draft_tokens": 1,
            "max_model_len": 64,
        }
        assert llm.collective_rpc("get_draft_model_stats", timeout=60) == [expected]
        tokenizer = llm.get_tokenizer()
        prompts = [
            tokenizer.encode("Explain how a computer works. " * 20)[:55],
            tokenizer.encode("Describe why plants grow. " * 20)[:53],
        ]
        params = SamplingParams(temperature=0, max_tokens=48, ignore_eos=True)
        for count in (2, 4):
            results = llm.generate(
                [{"prompt_token_ids": prompt} for prompt in prompts],
                params,
                use_tqdm=False,
            )
            assert all(len(result.outputs[0].token_ids) == 48 for result in results)
            expected["num_context_limit_fallback_requests"] = count
            snapshot = llm.collective_rpc("get_draft_model_stats", timeout=60)
            assert snapshot == [expected]
            snapshot[0]["num_context_limit_fallback_requests"] = -1
            assert llm.collective_rpc("get_draft_model_stats", timeout=60) == [expected]
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.mark.slow
@pytest.mark.parametrize("engine_multiprocessing", [False, True])
def test_draft_model_stats_rpc_e2e(engine_multiprocessing, run_in_spawn_process):
    run_in_spawn_process(_run_stats_rpc, engine_multiprocessing)


def _counters(llm):
    from vllm.v1.metrics.reader import Counter, Vector

    metrics = {metric.name: metric for metric in llm.get_metrics()}
    result = {}
    for name in ("num_drafts", "num_draft_tokens", "num_accepted_tokens"):
        metric = metrics[f"vllm:spec_decode_{name}"]
        assert isinstance(metric, Counter)
        result[name] = metric.value
    metric = metrics["vllm:spec_decode_num_accepted_tokens_per_pos"]
    assert isinstance(metric, Vector)
    result["per_position"] = metric.values
    return result


def _record_phase(path, records, phase, before, after, outputs):
    record = {
        "before": before,
        "after": after,
        "tokens": [list(output.outputs[0].token_ids) for output in outputs],
        "prompt_tokens": [output.prompt_token_ids for output in outputs],
        "cached_tokens": [output.num_cached_tokens for output in outputs],
        "passed": False,
    }
    records["phases"][phase] = record
    path.write_text(json.dumps(records, indent=2) + "\n")
    return record


def _acceptance_engine(batch_size, speculative_config):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    import mlx.core as mx
    from mlx.utils import tree_flatten
    from vllm import LLM

    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        revision="c1899de289a04d12100db370d81485cdf75e47ca",
        # Isolate cache reuse from BF16 rounding around tied logits.
        dtype="float32",
        max_model_len=256,
        max_num_seqs=batch_size,
        max_num_batched_tokens=32,
        block_size=16,
        num_gpu_blocks_override=129,
        gpu_memory_utilization=0.2,
        enable_prefix_caching=True,
        async_scheduling=False,
        disable_log_stats=False,
        speculative_config=speculative_config,
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    models = [runner.model]
    if speculative_config["method"] == "draft_model":
        models.append(runner._drafter._model)
    # Native MLX-LM loaders retain checkpoint weight dtypes. dtype="float32"
    # alone configures the KV cache; convert the fixture's weights explicitly.
    for model in models:
        model.set_dtype(mx.float32)
        mx.eval(model.parameters())
        assert all(p.dtype == mx.float32 for _, p in tree_flatten(model.parameters()))
    return llm


def _run_same_model_acceptance(batch_size, path):
    from vllm import SamplingParams

    from tools.parity_prompts import PROMPTS

    llm = _acceptance_engine(
        batch_size,
        {
            "method": "draft_model",
            "model": "Qwen/Qwen3-0.6B",
            "revision": "c1899de289a04d12100db370d81485cdf75e47ca",
            "num_speculative_tokens": 3,
        },
    )

    records = {"kind": "same-model", "batch_size": batch_size, "phases": {}}
    try:
        prompts = PROMPTS[15:23]
        # One initial token followed by sixteen complete verification rounds.
        params = SamplingParams(temperature=0, max_tokens=65, ignore_eos=True)
        for phase in ("cold", "cached"):
            before = _counters(llm)
            outputs = []
            for start in range(0, len(prompts), batch_size):
                outputs.extend(
                    llm.generate(
                        prompts[start : start + batch_size], params, use_tqdm=False
                    )
                )
            after = _counters(llm)
            drafts = after["num_drafts"] - before["num_drafts"]
            drafted = after["num_draft_tokens"] - before["num_draft_tokens"]
            accepted = after["num_accepted_tokens"] - before["num_accepted_tokens"]
            per_position = [
                end - begin
                for begin, end in zip(
                    before["per_position"], after["per_position"], strict=True
                )
            ]
            record = _record_phase(path, records, phase, before, after, outputs)
            assert len(outputs) == len(prompts)
            assert all(len(output.outputs[0].token_ids) == 65 for output in outputs)
            if phase == "cached":
                assert any(output.num_cached_tokens > 0 for output in outputs)
                assert all(
                    output.num_cached_tokens
                    == ((len(output.prompt_token_ids) - 1) // 16) * 16
                    for output in outputs
                )
            assert 0 < drafts <= drafted <= 3 * drafts
            assert 0 <= accepted <= drafted
            assert len(per_position) == 3 and sum(per_position) == accepted
            assert drafts >= per_position[0] >= per_position[1] >= per_position[2]
            acceptance = accepted / drafted
            eal = 1 + accepted / drafts
            rates = [count / drafts for count in per_position]
            print(
                f"B={batch_size} {phase}: {acceptance=}, {eal=}, {rates=}", flush=True
            )
            # Same-checkpoint greedy drafting should accept almost every token.
            assert acceptance >= 0.98
            assert eal >= 0.98 * 4
            assert all(rate >= 0.98 for rate in rates)
            record["passed"] = True
            path.write_text(json.dumps(records, indent=2) + "\n")
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.mark.slow
@pytest.mark.parametrize("batch_size", [1, 2])
def test_same_model_acceptance_after_prefix_hit(
    batch_size, tmp_path, run_in_spawn_process
):
    """A cached prefix must preserve near-perfect FP32 same-model acceptance."""
    run_in_spawn_process(
        _run_same_model_acceptance, batch_size, tmp_path / "metrics.json"
    )


def _run_controlled_recovery(batch_size, path):
    from vllm import SamplingParams
    from vllm.v1.outputs import DraftTokenIds

    from tools.parity_prompts import PROMPTS

    llm = _acceptance_engine(
        batch_size, {"method": "ngram", "num_speculative_tokens": 3}
    )
    runner = llm.llm_engine.model_executor.driver_worker.model_runner
    params = SamplingParams(temperature=0, max_tokens=37, ignore_eos=True)
    vocab_size = llm.get_tokenizer().vocab_size
    records = {"kind": "controlled", "batch_size": batch_size, "phases": {}}

    def generate():
        return llm.generate(PROMPTS[15:17], params, use_tqdm=False)

    try:
        # N-gram owns no draft KV. Replace only its proposals, leaving target
        # forwards, paged KV, scheduling, verification and metrics unchanged.
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(runner._drafter, "propose", lambda ctx: None)
            baseline = generate()
            reference = {
                tuple(output.prompt_token_ids): list(output.outputs[0].token_ids)
                for output in baseline
            }
            records["reference"] = [
                {"prompt_tokens": list(prompt), "tokens": tokens}
                for prompt, tokens in reference.items()
            ]
            path.write_text(json.dumps(records, indent=2) + "\n")
            assert all(len(tokens) == 37 for tokens in reference.values())
            assert llm.reset_prefix_cache()
            # Each rejection is followed by a full acceptance to test recovery.
            accepted_by_offset = {}
            offset = 1
            for accepted in [0, 3, 1, 3, 2, 3] * 2:
                accepted_by_offset[offset] = accepted
                offset += accepted + 1
            assert offset == 37
            proposals = []

            def propose(ctx):
                req_ids, drafts = [], []
                for (
                    req_id,
                    state,
                ) in runner._spec_decode_controller.draft_eligible_requests(
                    ctx.decode_reqs,
                    ctx.decode_token_ids,
                    ctx.prefill_reqs,
                    ctx.prefill_token_ids,
                    ctx.prefill_result_modes,
                    ctx.request_states,
                ):
                    offset = state.generated_tokens
                    if offset == 37:
                        continue
                    tokens = reference[tuple(state.token_ids[: state.prompt_len])]
                    # Check recovery at every round, not just the final output.
                    assert state.token_ids[state.prompt_len :] == tokens[:offset]
                    accepted = accepted_by_offset[offset]
                    draft = tokens[offset : offset + 3]
                    if accepted < 3:
                        draft[accepted] = (draft[accepted] + 1) % vocab_size
                    proposals.append(
                        {
                            "request": req_id,
                            "offset": offset,
                            "accept": accepted,
                            "tokens": draft,
                        }
                    )
                    req_ids.append(req_id)
                    drafts.append(draft)
                return DraftTokenIds(req_ids=req_ids, draft_token_ids=drafts)

            patch.setattr(runner._drafter, "propose", propose)
            for phase in ("cold", "cached"):
                proposals.clear()
                before = _counters(llm)
                outputs = generate()
                after = _counters(llm)
                record = _record_phase(path, records, phase, before, after, outputs)
                record["proposals"] = list(proposals)
                path.write_text(json.dumps(records, indent=2) + "\n")
                assert len(outputs) == 2
                assert record["tokens"] == [
                    list(o.outputs[0].token_ids) for o in baseline
                ]
                assert len(proposals) == 24
                assert after["num_drafts"] - before["num_drafts"] == 24
                assert after["num_draft_tokens"] - before["num_draft_tokens"] == 72
                assert (
                    after["num_accepted_tokens"] - before["num_accepted_tokens"] == 48
                )
                assert [
                    end - begin
                    for begin, end in zip(
                        before["per_position"], after["per_position"], strict=True
                    )
                ] == [20, 16, 12]
                if phase == "cached":
                    assert any(record["cached_tokens"])
                    assert record["cached_tokens"] == [
                        ((len(tokens) - 1) // 16) * 16
                        for tokens in record["prompt_tokens"]
                    ]
                record["passed"] = True
                path.write_text(json.dumps(records, indent=2) + "\n")
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.mark.slow
@pytest.mark.parametrize("batch_size", [1, 2])
def test_controlled_rejection_recovers_after_prefix_hit(
    batch_size, tmp_path, run_in_spawn_process
):
    """Rejection at every draft position must preserve tokens and public counters."""
    run_in_spawn_process(
        _run_controlled_recovery, batch_size, tmp_path / "metrics.json"
    )
