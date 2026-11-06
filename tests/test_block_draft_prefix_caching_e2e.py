# SPDX-License-Identifier: Apache-2.0
"""Pinned checkpoint qualification for scheduler-owned block-draft prefixes."""

import json
import time
from importlib.metadata import version

import pytest

from tests.scheduler_helpers import (
    assert_v2_model_runner,
    track_preemptions_and_resumptions,
)
from tests.test_block_draft_serving_e2e import _block_draft_llm, _spawn_env

# Use the same checkpoint revisions as the serving/continuation qualification.
CHECKPOINTS = {
    "dflash-4b": (
        "mlx-community/Qwen3-4B-4bit",
        "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25",
        "z-lab/Qwen3-4B-DFlash-b16",
        "b74e3a329c4d963783143b1e970d95b002be72bd",
    ),
    "dspark-4b": (
        "mlx-community/Qwen3-4B-4bit",
        "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25",
        "deepseek-ai/dspark_qwen3_4b_block7",
        "3457dff1417cb84927f6098a5fcb7cee85c934b7",
    ),
    "dspark-speculators": (
        "mlx-community/Qwen3-4B-4bit",
        "4dcb3d101c2a062e5c1d4bb173588c54ea6c4d25",
        "RedHatAI/Qwen3-4B-speculator.dspark",
        "bdffaa5f011da5a1a58ad841a1f6ced3e87f2f23",
    ),
}


def _first_window_decisions(case):
    return sorted(
        (f["position"], f["anchor"], f["first_proposal"], f["target_next"])
        for f in case["first_windows"]
    )


def _serve(pair, verify_window, caching, path):
    _spawn_env(verify_window)
    from vllm import SamplingParams

    target, revision, draft, draft_revision = CHECKPOINTS[pair]
    method = pair.split("-")[0]
    spec = {
        "method": method,
        "model": draft,
        "revision": draft_revision,
        "num_speculative_tokens": 7 if method == "dspark" else 3,
    }
    if method == "dspark":
        spec["dspark_draft_topk"] = 64
    llm = _block_draft_llm(
        model=target,
        revision=revision,
        speculative_config=spec,
        enable_prefix_caching=caching,
        # Leave room for weights and profiling on a 32-GiB machine. Cache
        # pressure is fixed by num_gpu_blocks_override=10 in both arms.
        gpu_memory_utilization=0.5,
    )
    engine = llm.llm_engine
    runner = engine.model_executor.driver_worker.model_runner
    scheduler = engine.engine_core.engine_core.scheduler
    proposer = runner._drafter
    assert proposer.enable_prefix_caching == caching
    tokenizer = llm.get_tokenizer()
    computer = tokenizer.encode("Explain how a computer works. " * 20)
    plants = tokenizer.encode("Describe why plants grow. " * 20)
    greedy = SamplingParams(temperature=0, max_tokens=8, ignore_eos=True)
    stats = {
        "prefill_tokens": 0,
        "cache_hits": 0,
        "cached_tokens": 0,
        "drafted": 0,
        "drafted_after_hit": 0,
        "rejected": 0,
        "preemptions": 0,
        "resumed": 0,
    }
    hit_ids = set()
    pending_first = {}
    first_windows = []
    reject_next = False
    propose = proposer.propose
    verify = runner._spec_decode_controller.verify

    def record_propose(ctx):
        nonlocal reject_next
        for prefill in ctx.prefill_reqs:
            stats["prefill_tokens"] += len(prefill.token_ids)
            if prefill.req_id not in proposer._valid_ends:
                hit_ids.discard(prefill.req_id)
                if prefill.start_pos:
                    hit_ids.add(prefill.req_id)
                    stats["cache_hits"] += 1
                    stats["cached_tokens"] += prefill.start_pos
        result = propose(ctx)
        if result is not None:
            for i, prefill in enumerate(ctx.prefill_reqs):
                if (
                    ctx.prefill_result_modes[i] != "intermediate"
                    and prefill.req_id in result.req_ids
                ):
                    pending_first[prefill.req_id] = {
                        "position": prefill.start_pos + len(prefill.token_ids),
                        "anchor": ctx.prefill_token_ids[i],
                        "cache_hit": prefill.req_id in hit_ids,
                    }
        if reject_next and result is not None:
            # Repetitive prompts can accept every natural proposal. Force one
            # wrong first candidate to qualify rejection without relying on a
            # particular checkpoint's error rate. The target must correct it.
            assert result.draft_token_ids[0][0] != tokenizer.eos_token_id
            result.draft_token_ids[0][0] = tokenizer.eos_token_id
            reject_next = False
        return result

    def record_verify(logits, requests, segments, *args, **kwargs):
        result = verify(logits, requests, segments, *args, **kwargs)
        for segment, tokens in zip(segments, result, strict=True):
            count = len(segment.draft_token_ids)
            stats["drafted"] += count
            stats["rejected"] += count - len(tokens) + 1
            if segment.req_id in hit_ids:
                stats["drafted_after_hit"] += count
            if count and segment.req_id in pending_first:
                first = pending_first.pop(segment.req_id)
                assert segment.cache_start_pos == first["position"], first
                assert segment.input_token_ids[0] == first["anchor"], first
                first_windows.append(
                    {
                        **first,
                        "first_proposal": segment.draft_token_ids[0],
                        "target_next": tokens[0],
                    }
                )
        return result

    proposer.propose = record_propose
    runner._spec_decode_controller.verify = record_verify
    assert_v2_model_runner(llm)
    track_preemptions_and_resumptions(scheduler, stats, resumption_key="resumed")
    records = {}

    def generate(prompts, params=greedy):
        return [
            list(output.outputs[0].token_ids)
            for output in llm.generate(
                [{"prompt_token_ids": p} for p in prompts], params, use_tqdm=False
            )
        ]

    def measure(name, fn):
        before = stats.copy()
        first_start = len(first_windows)
        start = time.perf_counter()
        outputs = fn()
        records[name] = {
            "outputs": outputs,
            "seconds": time.perf_counter() - start,
            "first_windows": first_windows[first_start:],
            **{key: value - before[key] for key, value in stats.items()},
        }
        return records[name]

    def drain():
        results = {}
        for _ in range(500):
            if not engine.has_unfinished_requests():
                return results
            for output in engine.step():
                if output.finished:
                    results[output.request_id] = list(output.outputs[0].token_ids)
        raise AssertionError("Scheduler failed to make progress")

    pool = scheduler.kv_cache_manager.block_pool
    free = pool.get_num_free_blocks()
    try:
        measure("cold", lambda: generate([computer[:47]]))
        warm = measure("warm", lambda: generate([computer[:47]]))
        assert warm["outputs"] == records["cold"]["outputs"]
        for case in (records["cold"], warm):
            assert len(case["first_windows"]) == 1, case
        # #721's one-token shift can preserve output equality via corrections.
        # Compare proposals too: unlike its same-model control, a separately
        # trained block drafter is allowed to reject on this continuation.
        assert _first_window_decisions(warm) == _first_window_decisions(records["cold"])
        if caching:
            assert warm["cache_hits"] > 0 and warm["drafted_after_hit"] > 0, warm
            assert warm["prefill_tokens"] < records["cold"]["prefill_tokens"], warm
        reject_next = True
        rejected = measure("rejected_prefix", lambda: generate([computer[:47]]))
        assert not reject_next and rejected["rejected"] > 0, rejected
        assert rejected["outputs"] == records["cold"]["outputs"]
        if caching:
            assert rejected["cache_hits"] and rejected["drafted_after_hit"], rejected
        measure("shared", lambda: generate([computer[:47], computer[:63]]))

        # Target-only fallbacks still have to populate usable draft prefixes.
        assert llm.reset_prefix_cache()
        before = stats["drafted"]
        generate(
            [computer[:47]],
            SamplingParams(temperature=0.8, seed=7, max_tokens=1, logprobs=1),
        )
        assert stats["drafted"] == before
        fallback = measure("fallback_prefix", lambda: generate([computer[:47]]))
        if caching:
            assert fallback["cache_hits"] and fallback["drafted_after_hit"], fallback

        # Abort after actual drafting, then reuse the public ID with a different
        # prompt. Neither request-local coverage nor freed page contents may leak.
        assert llm.reset_prefix_cache()
        engine.add_request(
            "reused",
            {"prompt_token_ids": computer[:63]},
            SamplingParams(temperature=0, max_tokens=48, ignore_eos=True),
        )
        before = stats["drafted"]
        for _ in range(30):
            engine.step()
            if stats["drafted"] > before:
                break
        assert stats["drafted"] > before
        engine.abort_request(["reused"])
        engine.add_request("reused", {"prompt_token_ids": plants[:61]}, greedy)
        measure("reused_id", drain)
        assert pool.get_num_free_blocks() == free

        assert llm.reset_prefix_cache()
        pressure = measure(
            "pressure",
            lambda: generate(
                [computer[:63], plants[:61]],
                SamplingParams(temperature=0, max_tokens=48, ignore_eos=True),
            ),
        )
        assert pressure["preemptions"] and pressure["resumed"], pressure
        assert pool.get_num_free_blocks() == free
        assert stats["rejected"] > 0, stats
    finally:
        path.write_text(
            json.dumps(
                {
                    "checkpoints": CHECKPOINTS[pair],
                    "prefix_caching": caching,
                    "verify_window": verify_window,
                    "speculative_config": spec,
                    "versions": {
                        name: version(name) for name in ("vllm", "mlx", "mlx-lm")
                    },
                    "cases": records,
                }
            )
        )
        engine.engine_core.shutdown()


@pytest.mark.slow
@pytest.mark.parametrize("pair", CHECKPOINTS)
@pytest.mark.parametrize("verify_window", [False, True])
def test_block_draft_prefix_caching(
    tmp_path, run_in_spawn_process, pair, verify_window
):
    cold, warm = tmp_path / "disabled.json", tmp_path / "enabled.json"
    for enabled, path in ((False, cold), (True, warm)):
        run_in_spawn_process(
            _serve,
            pair,
            verify_window,
            enabled,
            path,
            timeout=600,
            label=f"{pair}, prefix_caching={enabled}",
        )
    expected, actual = (json.loads(p.read_text())["cases"] for p in (cold, warm))
    for name in expected:
        assert actual[name]["outputs"] == expected[name]["outputs"], name
        assert _first_window_decisions(actual[name]) == _first_window_decisions(
            expected[name]
        ), name
