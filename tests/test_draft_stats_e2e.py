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


def _run_same_model_acceptance(batch_size, path):
    from vllm import LLM, SamplingParams
    from vllm.v1.metrics.reader import Counter, Vector

    from tools.parity_prompts import PROMPTS

    model = "Qwen/Qwen3-0.6B"
    revision = "c1899de289a04d12100db370d81485cdf75e47ca"
    llm = LLM(
        model=model,
        revision=revision,
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
        speculative_config={
            "method": "draft_model",
            "model": model,
            "revision": revision,
            "num_speculative_tokens": 3,
        },
    )

    def counters():
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

    records = {}
    try:
        prompts = PROMPTS[15:23]
        # One initial token followed by sixteen complete verification rounds.
        params = SamplingParams(temperature=0, max_tokens=65, ignore_eos=True)
        for phase in ("cold", "cached"):
            before = counters()
            outputs = []
            for start in range(0, len(prompts), batch_size):
                outputs.extend(
                    llm.generate(
                        prompts[start : start + batch_size], params, use_tqdm=False
                    )
                )
            after = counters()
            drafts = after["num_drafts"] - before["num_drafts"]
            drafted = after["num_draft_tokens"] - before["num_draft_tokens"]
            accepted = after["num_accepted_tokens"] - before["num_accepted_tokens"]
            per_position = [
                end - begin
                for begin, end in zip(
                    before["per_position"], after["per_position"], strict=True
                )
            ]
            records[phase] = {
                "before": before,
                "after": after,
                "tokens": [list(output.outputs[0].token_ids) for output in outputs],
                "prompt_tokens": [output.prompt_token_ids for output in outputs],
                "cached_tokens": [output.num_cached_tokens for output in outputs],
            }
            path.write_text(json.dumps(records, indent=2) + "\n")
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
