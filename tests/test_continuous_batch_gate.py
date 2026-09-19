# SPDX-License-Identifier: Apache-2.0
"""CPU-only contracts for the continuous-arrival collector's observer.

Run with ``PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest --noconftest`` to avoid the
repository-wide model-backend fixtures. These tests need only pytest and Python.
"""

from __future__ import annotations

import importlib.util
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

ROOT = Path(__file__).resolve().parents[1]
TOOL = ROOT / "tools/continuous_batch_gate.py"
spec = importlib.util.spec_from_file_location("continuous_batch_gate", TOOL)
gate = importlib.util.module_from_spec(spec)
spec.loader.exec_module(gate)


class FakeTokenizer:
    def encode(self, text, *, add_special_tokens):
        assert add_special_tokens is False
        return list(text.encode())


def output(**overrides):
    fields = {
        "scheduled_new_reqs": [],
        "scheduled_cached_reqs": SimpleNamespace(resumed_req_ids=set()),
        "preempted_req_ids": None,
        "num_scheduled_tokens": {},
        "finished_req_ids": set(),
    }
    fields.update(overrides)
    return SimpleNamespace(**fields)


@pytest.fixture
def make_observer():
    installed = []

    def make(*, outputs=None, forward=None, populate=None, finish=None, requests=None):
        scheduler = SimpleNamespace(
            schedule=Mock(side_effect=outputs or [output()]),
            finish_requests=finish or Mock(return_value=[]),
            running=[],
            requests=requests or {},
            processed_step_seq=7,
            sched_step_seq=9,
        )
        runtime = SimpleNamespace(populate_step_context=populate or Mock())
        runner = SimpleNamespace(
            _paged_attention_runtime=runtime,
            _start_paged_forward=forward or Mock(),
        )
        report = {"steps": [], "abort_events": []}
        records = {}
        observer = gate.Observer(
            SimpleNamespace(scheduler=scheduler), runner, report, records
        )
        observer.now = Mock(return_value=1.0)
        originals = {
            "schedule": scheduler.schedule,
            "finish_requests": scheduler.finish_requests,
            "forward": runner._start_paged_forward,
            "populate": runtime.populate_step_context,
        }
        observer.install()
        installed.append(observer)
        return SimpleNamespace(
            observer=observer,
            scheduler=scheduler,
            runner=runner,
            runtime=runtime,
            report=report,
            records=records,
            originals=originals,
        )

    yield make
    for observer in installed:
        observer.restore()


def test_import_does_not_load_model_backends():
    script = """
import builtins
import importlib.util
import sys

original_import = builtins.__import__
def guarded_import(name, *args, **kwargs):
    if name.split('.')[0] in {'mlx', 'vllm', 'torch'}:
        raise AssertionError('collector imported backend: ' + name)
    return original_import(name, *args, **kwargs)
builtins.__import__ = guarded_import
spec = importlib.util.spec_from_file_location('collector', sys.argv[1])
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
assert not any(name.split('.')[0] in {'mlx', 'vllm', 'torch'} for name in sys.modules)
assert callable(module.workload) and callable(module.Observer)
"""
    result = subprocess.run(
        [sys.executable, "-B", "-c", script, str(TOOL)],
        text=True,
        capture_output=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("block", [544, 784])
@pytest.mark.parametrize("scenario", ["mixed", "pressure"])
def test_workload_has_the_complete_fixed_arrival_and_output_plan(block, scenario):
    if scenario == "mixed":
        expected = [
            (8 * block + 1, 256, 0.0, None),
            (128, block + 128, 0.0, None),
            (2 * block + 1, 32, 0.03, None),
            (4 * block + 17, 128, 0.08, None),
            (12 * block + 1, 96, 0.12, "prefill"),
            (block + 1, 256, 0.18, "decode"),
            (96, 16, 0.25, None),
            (6 * block + 1, block + 64, 0.40, None),
            (8 * block + 1, 64, 0.70, None),
            (256, 48, 1.00, None),
            (2 * block + 17, 128, 1.30, None),
            (128, 16, 1.60, None),
        ]
    else:
        expected = [
            (8 * block + 1, 256, 0.0, None),
            (6 * block + 17, 384, 0.0, None),
            (10 * block + 1, 256, 0.04, None),
            (4 * block + 1, 128, 0.08, None),
            (128, 32, 0.20, None),
            (7 * block + 1, 192, 0.40, None),
            (2 * block + 17, 64, 0.80, None),
            (96, 16, 1.20, None),
        ]
    plan = gate.workload(FakeTokenizer(), block, scenario)
    assert [
        (
            len(r["prompt_token_ids"]),
            r["max_tokens"],
            r["arrival_offset_s"],
            r["cancel_stage"],
        )
        for r in plan
    ] == expected
    assert [r["request_id"] for r in plan] == [
        f"{scenario}-{index:02d}" for index in range(len(expected))
    ]
    assert plan == gate.workload(FakeTokenizer(), block, scenario)
    assert len({tuple(r["prompt_token_ids"]) for r in plan}) == len(plan)
    for request in plan:
        assert request["ignore_eos"] is True
        assert request["sampling"] == {"temperature": 0, "logprobs": None, "seed": 0}


def test_three_mixed_waves_have_unique_requests_and_complete_shifted_plans():
    tokenizer = FakeTokenizer()
    first_wave = gate.workload(tokenizer, 544, "mixed")
    plan = gate.workload(tokenizer, 544, "mixed", waves=3, wave_interval=2.0)
    assert len(plan) == 36
    assert [r["request_id"] for r in plan] == [
        f"mixed-{index:02d}" for index in range(36)
    ]
    assert len({tuple(r["prompt_token_ids"]) for r in plan}) == 36
    arrivals = [r["arrival_offset_s"] for r in plan]
    assert arrivals == sorted(arrivals)
    for wave in range(3):
        requests = plan[wave * 12 : (wave + 1) * 12]
        assert [r["cancel_stage"] for r in requests if r["cancel_stage"]] == [
            "prefill",
            "decode",
        ]
        for request, template in zip(requests, first_wave, strict=True):
            assert len(request["prompt_token_ids"]) == len(template["prompt_token_ids"])
            assert request["max_tokens"] == template["max_tokens"]
            assert request["sampling"] == template["sampling"]
            assert request["ignore_eos"] is True
            assert request["cancel_stage"] == template["cancel_stage"]
            assert (
                request["arrival_offset_s"] == wave * 2.0 + template["arrival_offset_s"]
            )


def test_workload_can_allow_configured_eos():
    plan = gate.workload(FakeTokenizer(), 544, "pressure", ignore_eos=False)
    assert plan
    assert all(request["ignore_eos"] is False for request in plan)
    assert [len(r["prompt_token_ids"]) for r in plan] == [
        len(r["prompt_token_ids"])
        for r in gate.workload(FakeTokenizer(), 544, "pressure")
    ]


class _ChatTokenizer(FakeTokenizer):
    chat_template = "{{ messages }}"

    def apply_chat_template(
        self, messages, *, tokenize, add_generation_prompt, **kwargs
    ):
        assert tokenize is True
        assert add_generation_prompt is True
        assert messages[0]["role"] == "user"
        return [11, 22, 33]


def test_chat_template_coerces_numpy_int_tokens():
    class Int64Tokenizer(_ChatTokenizer):
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, **kwargs
        ):
            del messages, tokenize, add_generation_prompt, kwargs
            return [type("I64", (int,), {})(9), type("I64", (int,), {})(10)]

    plan = gate.workload(FakeTokenizer(), 544, "pressure")
    gate.apply_chat_template_to_plan(Int64Tokenizer(), plan)
    assert all(item["prompt_token_ids"][:2] == [9, 10] for item in plan)
    assert all(
        type(token) is int for item in plan for token in item["prompt_token_ids"][:2]
    )


def test_chat_template_accepts_numpy_token_ids():
    class ArrayTokenizer(_ChatTokenizer):
        def apply_chat_template(
            self, messages, *, tokenize, add_generation_prompt, **kwargs
        ):
            del messages, tokenize, add_generation_prompt, kwargs
            return type("Arr", (), {"tolist": staticmethod(lambda: [7, 8])})()

    plan = gate.workload(FakeTokenizer(), 544, "pressure")
    gate.apply_chat_template_to_plan(ArrayTokenizer(), plan)
    assert all(item["prompt_token_ids"][:2] == [7, 8] for item in plan)


def test_chat_template_padding_is_stable_across_calls():
    tokenizer = _ChatTokenizer()
    first = gate.workload(tokenizer, 544, "pressure", ignore_eos=False)
    second = gate.workload(tokenizer, 544, "pressure", ignore_eos=False)
    gate.apply_chat_template_to_plan(tokenizer, first)
    gate.apply_chat_template_to_plan(tokenizer, second)
    assert [item["prompt_token_ids"] for item in first] == [
        item["prompt_token_ids"] for item in second
    ]


def test_chat_template_rewrite_preserves_planned_lengths():
    tokenizer = _ChatTokenizer()
    plan = gate.workload(tokenizer, 544, "pressure", ignore_eos=False)
    lengths = [len(item["prompt_token_ids"]) for item in plan]
    gate.apply_chat_template_to_plan(tokenizer, plan)
    assert [len(item["prompt_token_ids"]) for item in plan] == lengths
    assert all(item["prompt_token_ids"][:3] == [11, 22, 33] for item in plan)


def test_chat_template_rejects_prompts_shorter_than_the_template():
    tokenizer = _ChatTokenizer()
    tokenizer.apply_chat_template = lambda *a, **k: list(range(200))  # noqa: ARG005
    plan = gate.workload(FakeTokenizer(), 544, "pressure")
    with pytest.raises(ValueError, match="chat template has 200 tokens"):
        gate.apply_chat_template_to_plan(tokenizer, plan)


@pytest.mark.parametrize(
    ("args", "kwargs"),
    [
        ((), {}),
        ((False,), {}),
        ((True,), {}),
        ((), {"throttle_prefills": False}),
        ((), {"throttle_prefills": True}),
    ],
)
def test_schedule_preserves_throttle_call_and_returns_original_output(
    make_observer, args, kwargs
):
    scheduled = output(num_scheduled_tokens={"internal": 17})
    h = make_observer(outputs=[scheduled])
    h.observer.ids["internal"] = "client"
    assert h.scheduler.schedule(*args, **kwargs) is scheduled
    h.originals["schedule"].assert_called_once_with(*args, **kwargs)
    step = h.report["steps"][0]
    assert step["scheduled_tokens"] == {"client": 17}
    assert step["processed_step_seq"] == 7
    assert step["sched_step_seq"] == 9
    assert h.observer.by_output[id(scheduled)] is step


def test_schedule_records_external_labels_and_keeps_internal_identity(make_observer):
    scheduled = output(
        scheduled_new_reqs=[SimpleNamespace(req_id="i-new")],
        scheduled_cached_reqs=SimpleNamespace(resumed_req_ids={"i-resumed"}),
        preempted_req_ids={"i-preempted"},
        num_scheduled_tokens={"i-new": 544, "unmapped": 1},
        finished_req_ids={"i-finished"},
        state_cache_budget_stats={"quota_stalls": 0},
    )
    h = make_observer(outputs=[scheduled])
    h.observer.ids.update(
        {f"i-{name}": name for name in ("new", "resumed", "preempted", "finished")}
    )
    h.observer.internal_ids.update({v: k for k, v in h.observer.ids.items()})
    reverse_before = dict(h.observer.internal_ids)
    h.scheduler.running = [
        SimpleNamespace(request_id="i-resumed"),
        SimpleNamespace(request_id="unmapped"),
    ]
    h.scheduler.schedule()
    step = h.report["steps"][0]
    assert step["admitted_req_ids"] == ["new"]
    assert step["preempted_req_ids"] == ["preempted"]
    assert step["resumed_req_ids"] == ["resumed"]
    assert step["preempted_internal_req_ids"] == ["i-preempted"]
    assert step["resumed_internal_req_ids"] == ["i-resumed"]
    assert step["scheduler_running_ids"] == ["resumed", "unmapped"]
    assert step["finished_req_ids"] == ["finished"]
    assert step["scheduled_tokens"] == {"new": 544, "unmapped": 1}
    assert step["budget"] == {"quota_stalls": 0}
    assert h.observer.internal_ids == reverse_before
    assert h.observer.external("unmapped") == "unmapped"


def test_forward_records_real_order_and_populate_takes_read_only_snapshots(
    make_observer,
):
    context = SimpleNamespace(state_group_slot_mappings=[[8, 2], [9, 3]])
    positions = [[17], [31], [0, 1], [544, 545]]
    req_ids = ["i-b", "i-a", "i-d", "i-c"]
    populate_result, forward_result, marker, batch = (object() for _ in range(4))
    populate = Mock(return_value=populate_result)

    def forward(actual_batch, actual_prefill, actual_decode, actual_output):
        assert actual_batch is batch
        assert actual_prefill is prefill
        assert actual_decode is decode
        assert actual_output is scheduled
        assert h.observer.current is h.report["steps"][0]
        assert (
            h.runtime.populate_step_context(
                marker, ctx=context, req_ids=req_ids, step_positions=positions
            )
            is populate_result
        )
        return forward_result

    h = make_observer(forward=forward, populate=populate)
    h.observer.ids.update({f"i-{name}": name for name in "abcd"})
    scheduled = h.scheduler.schedule()
    decode = [("i-b", object()), ("i-a", object())]
    prefill = [SimpleNamespace(req_id="i-d"), SimpleNamespace(req_id="i-c")]
    mappings_before, positions_before, ids_before = deepcopy(
        (context.state_group_slot_mappings, positions, req_ids)
    )
    assert (
        h.runner._start_paged_forward(batch, prefill, decode, scheduled)
        is forward_result
    )
    step = h.report["steps"][0]
    assert step["batch_request_ids"] == ["b", "a", "d", "c"]
    assert step["batch_internal_req_ids"] == req_ids
    assert step["phases"] == {
        "b": "decode",
        "a": "decode",
        "d": "prefill",
        "c": "prefill",
    }
    assert h.observer.last_phases == step["phases"]
    assert step["forward_submitted"] is True
    assert h.observer.current is None
    populate.assert_called_once_with(
        marker, ctx=context, req_ids=req_ids, step_positions=positions
    )
    assert populate.call_args.kwargs["ctx"] is context
    assert populate.call_args.kwargs["req_ids"] is req_ids
    assert populate.call_args.kwargs["step_positions"] is positions
    assert context.state_group_slot_mappings == mappings_before
    assert positions == positions_before
    assert req_ids == ids_before
    assert [r.req_id for r in prefill] == ["i-d", "i-c"]
    assert [r for r, _ in decode] == ["i-b", "i-a"]
    context.state_group_slot_mappings[0][0] = 99
    positions[0][0] = 99
    req_ids[0] = "changed"
    assert step["state_slot_mappings"] == mappings_before
    assert step["step_positions"] == positions_before
    assert step["state_request_ids"] == ["b", "a", "d", "c"]


def test_populate_outside_forward_is_transparent(make_observer):
    result = object()
    h = make_observer(populate=Mock(return_value=result))
    # Outside a forward there need not even be a context for the observer to read.
    assert h.runtime.populate_step_context("opaque", unrelated=17) is result
    h.originals["populate"].assert_called_once_with("opaque", unrelated=17)
    assert h.report["steps"] == []


def test_forward_exception_propagates_and_clears_current_step(make_observer):
    failure = ValueError("real forward failed")
    h = make_observer(forward=Mock(side_effect=failure))
    scheduled = h.scheduler.schedule()
    with pytest.raises(ValueError, match="real forward failed") as caught:
        h.runner._start_paged_forward(object(), [], [], scheduled)
    assert caught.value is failure
    assert h.observer.current is None
    assert "forward_submitted" not in h.report["steps"][0]


def test_only_actual_finished_requests_are_aborted_and_carried_to_next_step(
    make_observer,
):
    live = SimpleNamespace(
        request_id="i-live",
        num_computed_tokens=544,
        num_in_flight_tokens=1,
        num_prompt_tokens=128,
    )
    status = SimpleNamespace(name="FINISHED_ABORTED")
    attempted = ["i-live", "i-already-done", "i-missing"]
    finished = [live]

    def finish(request_ids, finished_status):
        assert request_ids is attempted
        assert finished_status is status
        h.scheduler.requests.pop("i-live")
        live.num_computed_tokens = live.num_in_flight_tokens = 0
        h.observer.now.return_value = 2.5
        return finished

    h = make_observer(
        outputs=[output(), output(), output()], finish=finish, requests={"i-live": live}
    )
    h.observer.ids.update(
        {"i-live": "live", "i-already-done": "done", "i-missing": "missing"}
    )
    h.records.update(
        {
            "live": {"finish_reason": None},
            "done": {"finish_reason": "length", "finished_at_s": 0.5},
        }
    )
    scheduled = h.scheduler.schedule()
    h.runner._start_paged_forward(object(), [], [("i-live", object())], scheduled)
    assert h.scheduler.finish_requests(attempted, status) is finished
    event = {
        "request_id": "live",
        "internal_request_id": "i-live",
        "at_s": 2.5,
        "phase": "decode",
        "computed": 544,
        "in_flight": 1,
        "prompt_tokens": 128,
    }
    assert h.report["abort_events"] == [event]
    assert h.observer.pending_cancels == [event]
    assert h.records["live"] == {"finish_reason": "abort", "finished_at_s": 2.5}
    assert h.records["done"] == {"finish_reason": "length", "finished_at_s": 0.5}
    assert "missing" not in h.records
    assert h.report["steps"][0]["cancelled_requests"] == []
    h.observer.now.return_value = 3.0
    h.scheduler.schedule()
    assert h.report["steps"][1]["cancelled_requests"] == [event]
    assert h.report["steps"][0]["at_s"] < event["at_s"] < h.report["steps"][1]["at_s"]
    assert h.observer.pending_cancels == []
    h.observer.now.return_value = 4.0
    h.scheduler.schedule()
    assert h.report["steps"][2]["cancelled_requests"] == []
    assert [step["sequence"] for step in h.report["steps"]] == [1, 2, 3]


@pytest.mark.parametrize("status_name", ["FINISHED_ABORTED", "FINISHED_LENGTH_CAPPED"])
def test_no_actual_abort_means_no_cancel_event(make_observer, status_name):
    request = SimpleNamespace(
        request_id="i-done",
        num_computed_tokens=5,
        num_in_flight_tokens=0,
        num_prompt_tokens=4,
    )
    returned = [] if status_name == "FINISHED_ABORTED" else [request]
    h = make_observer(finish=Mock(return_value=returned), requests={"i-done": request})
    h.records["i-done"] = {"finish_reason": "length"}
    assert (
        h.scheduler.finish_requests(
            ["i-done", "missing"], SimpleNamespace(name=status_name)
        )
        is returned
    )
    assert h.report["abort_events"] == []
    assert h.observer.pending_cancels == []
    assert h.records["i-done"] == {"finish_reason": "length"}


def test_restore_recovers_every_original_instance_method(make_observer):
    h = make_observer()
    other = make_observer()
    other_schedule = other.scheduler.schedule
    h.scheduler.schedule()
    h.observer.restore()
    assert h.scheduler.schedule is h.originals["schedule"]
    assert h.scheduler.finish_requests is h.originals["finish_requests"]
    assert h.runner._start_paged_forward is h.originals["forward"]
    assert h.runtime.populate_step_context is h.originals["populate"]
    assert other.scheduler.schedule is other_schedule
    h.originals["schedule"].side_effect = None
    h.scheduler.schedule(True)
    assert len(h.report["steps"]) == 1
    h.originals["schedule"].assert_called_with(True)
