# SPDX-License-Identifier: Apache-2.0
"""GGMLModelRunner bookkeeping tests against a fake engine (no Metal, no model)."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
from vllm.sampling_params import SamplingParams
from vllm.v1.core.sched.output import CachedRequestData, NewRequestData, SchedulerOutput
from vllm.v1.kv_cache_interface import KVCacheConfig, KVCacheGroupSpec

from vllm_metal.ggml.model_runner import KV_LAYER_NAME, GGMLModelRunner

VOCAB = 32
BLOCK = 4


class FakeEngine:
    """Records forward() calls; logits make token t predict (t + 1) % VOCAB."""

    def __init__(self, with_state: bool = True) -> None:
        self.calls: list[dict] = []
        self.cache: tuple[int, int, int] | None = None
        self._info = {
            "arch": "fake",
            "backend": "fake",
            "vocab_size": VOCAB,
            "weight_bytes": 1 << 20,
            "kv_bytes_per_token": 2 * 2 * (8 * 2 + 16 * 1),
            "state_bytes_per_slot": 4096 if with_state else 0,
            "kv_layers": [
                {"layer": 0, "num_kv_heads": 2, "head_dim": 8, "sliding_window": None},
                {"layer": 1, "num_kv_heads": 1, "head_dim": 16, "sliding_window": 4},
            ],
            "state_layers": [{"layer": 2, "conv_elems": 8, "ssm_elems": 16}]
            if with_state
            else [],
        }

    def info(self) -> dict:
        return self._info

    def memory(self) -> tuple[int, int]:
        return (8 << 30, 16 << 30)

    def init_cache(self, num_blocks: int, block_size: int, slots: int) -> int:
        self.cache = (num_blocks, block_size, slots)
        return num_blocks * block_size * self._info["kv_bytes_per_token"]

    def forward(self, tokens, q, c, bt, slots, reset, n_logits, out) -> int:
        self.calls.append(
            {
                "tokens": tokens.tolist(),
                "q": q.tolist(),
                "ctx": c.tolist(),
                "bt": bt.tolist(),
                "slots": slots.tolist(),
                "reset": reset.tolist(),
                "n_logits": n_logits.tolist(),
            }
        )
        rows, t = 0, 0
        for i, n in enumerate(q.tolist()):
            if n_logits[i]:
                out[rows] = -10.0
                out[rows, (int(tokens[t + n - 1]) + 1) % VOCAB] = 10.0
                rows += 1
            t += n
        return rows


def make_runner(
    with_state: bool = True, max_num_seqs: int = 4
) -> tuple[GGMLModelRunner, FakeEngine]:
    cfg = SimpleNamespace(
        model_config=SimpleNamespace(
            model="fake", revision=None, get_vocab_size=lambda: VOCAB, max_model_len=64
        ),
        cache_config=SimpleNamespace(block_size=BLOCK, gpu_memory_utilization=0.5),
        scheduler_config=SimpleNamespace(
            max_num_seqs=max_num_seqs, max_num_batched_tokens=16
        ),
    )
    runner = GGMLModelRunner(cfg)  # type: ignore[arg-type]
    engine = FakeEngine(with_state)
    runner._attach_engine(engine)
    spec = runner.get_kv_cache_spec()[KV_LAYER_NAME]
    runner.initialize_kv_cache(
        KVCacheConfig(
            num_blocks=32,
            kv_cache_tensors=[],
            kv_cache_groups=[
                KVCacheGroupSpec(layer_names=[KV_LAYER_NAME], kv_cache_spec=spec)
            ],
        )
    )
    return runner, engine


def new_req(
    req_id: str, prompt: list[int], blocks: list[int], computed: int = 0
) -> NewRequestData:
    return NewRequestData(
        req_id=req_id,
        prompt_token_ids=prompt,
        mm_features=[],
        sampling_params=SamplingParams(temperature=0.0),
        pooling_params=None,
        block_ids=(blocks,),
        num_computed_tokens=computed,
        lora_request=None,
    )


def sched(
    scheduled: dict[str, int],
    *,
    new: list[NewRequestData] | None = None,
    cached: dict[str, tuple[int, list[int] | None]] | None = None,
    resumed: set[str] | None = None,
    finished: set[str] | None = None,
    preempted: set[str] | None = None,
    num_output: dict[str, int] | None = None,
) -> SchedulerOutput:
    cached = cached or {}
    ids = list(cached)
    return SchedulerOutput(
        scheduled_new_reqs=new or [],
        scheduled_cached_reqs=CachedRequestData(
            req_ids=ids,
            resumed_req_ids=resumed or set(),
            new_token_ids=[],
            all_token_ids={},
            new_block_ids=[
                (cached[r][1],) if cached[r][1] is not None else None for r in ids
            ],
            num_computed_tokens=[cached[r][0] for r in ids],
            num_output_tokens=[(num_output or {}).get(r, 10**6) for r in ids],
        ),
        num_scheduled_tokens=scheduled,
        total_num_scheduled_tokens=sum(scheduled.values()),
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=finished or set(),
        free_encoder_mm_hashes=[],
        preempted_req_ids=preempted or set(),
        has_structured_output_requests=False,
    )


def step(runner: GGMLModelRunner, so: SchedulerOutput):
    assert runner.execute_model(so) is None
    return runner.sample_tokens(None)


def test_kv_spec_is_single_layer_matching_engine_bytes() -> None:
    runner, _ = make_runner()
    spec = runner.get_kv_cache_spec()[KV_LAYER_NAME]
    assert spec.page_size_bytes == BLOCK * runner.info["kv_bytes_per_token"]
    assert runner.get_cache_block_size_bytes() == spec.page_size_bytes


def test_initialize_reserves_state_slot_zero() -> None:
    runner, engine = make_runner(max_num_seqs=4)
    assert engine.cache == (32, BLOCK, 5)
    assert sorted(runner._free_slots) == [1, 2, 3, 4]


def test_empty_step_returns_output_directly() -> None:
    runner, engine = make_runner()
    out = runner.execute_model(sched({}))
    assert out is not None and out.req_ids == []
    assert engine.calls == []


def test_chunked_prefill_then_decode() -> None:
    runner, engine = make_runner()
    prompt = [1, 2, 3, 4, 5]
    # chunk 1: 3 tokens, no sample
    out = step(runner, sched({"a": 3}, new=[new_req("a", prompt, [7, 9])]))
    assert out.sampled_token_ids == [[]]
    c = engine.calls[-1]
    assert (c["tokens"], c["q"], c["ctx"], c["reset"], c["n_logits"]) == (
        [1, 2, 3],
        [3],
        [3],
        [1],
        [0],
    )
    assert c["bt"] == [[7, 9]]
    slot = c["slots"][0]
    assert slot > 0
    # chunk 2: finishes the prompt -> samples 6
    out = step(runner, sched({"a": 2}, cached={"a": (3, None)}))
    assert out.sampled_token_ids == [[6]]
    c = engine.calls[-1]
    assert (c["tokens"], c["ctx"], c["reset"], c["n_logits"], c["slots"]) == (
        [4, 5],
        [5],
        [0],
        [1],
        [slot],
    )
    # decode: feeds the sampled token, gets a new block
    out = step(runner, sched({"a": 1}, cached={"a": (5, [11])}))
    assert out.sampled_token_ids == [[7]]
    c = engine.calls[-1]
    assert (c["tokens"], c["ctx"], c["bt"]) == ([6], [6], [[7, 9, 11]])
    assert runner.requests["a"].token_ids == [1, 2, 3, 4, 5, 6, 7]


def test_decodes_packed_first_output_in_scheduler_order() -> None:
    runner, engine = make_runner()
    step(runner, sched({"d": 2}, new=[new_req("d", [3, 4], [1])]))
    so = sched(
        {"p": 3, "d": 1}, new=[new_req("p", [10, 11, 12], [2])], cached={"d": (2, None)}
    )
    out = step(runner, so)
    c = engine.calls[-1]
    assert c["q"] == [1, 3]  # decode first
    assert c["tokens"] == [5, 10, 11, 12]
    assert out.req_ids == ["p", "d"]
    assert out.sampled_token_ids == [[13], [6]]
    assert out.req_id_to_index == {"p": 0, "d": 1}


def test_finish_and_preempt_recycle_state_slots() -> None:
    runner, engine = make_runner(max_num_seqs=2)
    step(
        runner,
        sched(
            {"a": 2, "b": 2}, new=[new_req("a", [1, 2], [1]), new_req("b", [3, 4], [2])]
        ),
    )
    slot_a, slot_b = engine.calls[-1]["slots"]
    assert runner._free_slots == []
    # finish a -> slot reused by c
    step(runner, sched({"c": 1}, new=[new_req("c", [5], [3])], finished={"a"}))
    assert engine.calls[-1]["slots"] == [slot_a]
    # preempt b -> slot released; resume -> fresh slot + reset + new blocks
    runner.execute_model(sched({}, preempted={"b"}))
    assert slot_b in runner._free_slots
    out = step(runner, sched({"b": 3}, cached={"b": (0, [8])}, resumed={"b"}))
    c = engine.calls[-1]
    assert c["reset"] == [1] and c["bt"] == [[8]] and c["tokens"] == [3, 4, 5]
    assert out.sampled_token_ids == [[6]]


def test_discarded_tokens_are_truncated() -> None:
    runner, engine = make_runner()
    step(runner, sched({"a": 2}, new=[new_req("a", [1, 2], [1])]))
    step(runner, sched({"a": 1}, cached={"a": (2, None)}))
    assert runner.requests["a"].token_ids == [1, 2, 3, 4]
    # scheduler says only 1 output token survived
    step(runner, sched({"a": 1}, cached={"a": (2, None)}, num_output={"a": 1}))
    assert engine.calls[-1]["tokens"] == [3]


def test_rejects_spec_decode() -> None:
    runner, _ = make_runner()
    so = sched({"a": 2}, new=[new_req("a", [1, 2], [1])])
    so.scheduled_spec_decode_tokens = {"a": [1]}
    with pytest.raises(NotImplementedError):
        runner.execute_model(so)


def test_memory_budget_accounts_weights_state_and_activations() -> None:
    runner, engine = make_runner(max_num_seqs=4)
    avail = runner.determine_available_memory()
    # profile run executed a single max_num_batched_tokens prefill
    assert engine.calls[-1]["q"] == [16]
    total = 16 << 30
    state = 5 * engine.info()["state_bytes_per_slot"]
    expected_max = int(total * 0.5) - engine.info()["weight_bytes"] - state
    assert 0 < avail <= expected_max


def test_no_state_model_does_not_need_slots() -> None:
    runner, engine = make_runner(with_state=False)
    assert engine.cache is not None and engine.cache[2] == 1
    step(runner, sched({"a": 1}, new=[new_req("a", [1], [1])]))
    assert engine.calls[-1]["slots"] == [0]
    np.testing.assert_equal(runner._free_slots, [])
