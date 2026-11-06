# SPDX-License-Identifier: Apache-2.0
"""Model-free checks for scheduler lifecycle instrumentation."""

from types import SimpleNamespace

import pytest

from tests.scheduler_helpers import (
    assert_v2_model_runner,
    track_preemptions_and_resumptions,
)


def _scheduler_output(*, new=(), running=(), preempted=(), finished=()):
    return SimpleNamespace(
        preempted_req_ids=set(preempted),
        num_scheduled_tokens=dict.fromkeys((*new, *running), 1),
        finished_req_ids=set(finished),
    )


@pytest.mark.parametrize(
    "steps",
    [
        pytest.param(
            [
                ({"new": ("req",)}, 0, 0),
                ({"preempted": ("req",)}, 1, 0),
                ({}, 1, 0),
                ({"new": ("other",)}, 1, 0),
                ({"new": ("req",)}, 1, 1),
                ({"running": ("req",)}, 1, 1),
                ({"running": ("req",)}, 1, 1),
                ({"preempted": ("req",)}, 2, 1),
                ({"new": ("req",)}, 2, 2),
            ],
            id="delayed-and-repeated-resume",
        ),
        pytest.param(
            [
                ({"new": ("req",)}, 0, 0),
                ({"preempted": ("req",)}, 1, 0),
                ({"finished": ("req",)}, 1, 0),
                ({}, 1, 0),
                ({"new": ("req",)}, 1, 0),
            ],
            id="finished-while-preempted",
        ),
    ],
)
def test_track_preemptions_and_resumptions(steps):
    outputs = [_scheduler_output(**fields) for fields, _, _ in steps]
    scheduler = SimpleNamespace(schedule=iter(outputs).__next__)
    stats = {"preemptions": 0, "resumptions": 0}
    track_preemptions_and_resumptions(scheduler, stats)

    for output, (_, preemptions, resumptions) in zip(outputs, steps, strict=True):
        assert scheduler.schedule() is output
        assert stats == {"preemptions": preemptions, "resumptions": resumptions}


def test_custom_resumption_key():
    outputs = [
        _scheduler_output(preempted=("req",)),
        _scheduler_output(new=("req",)),
    ]
    scheduler = SimpleNamespace(schedule=iter(outputs).__next__)
    stats = {"preemptions": 0, "resumed": 0}
    track_preemptions_and_resumptions(scheduler, stats, resumption_key="resumed")

    scheduler.schedule()
    scheduler.schedule()

    assert stats == {"preemptions": 1, "resumed": 1}


@pytest.mark.parametrize("use_v2", [False, True], ids=["rejected", "accepted"])
def test_assert_v2_model_runner(use_v2):
    llm = SimpleNamespace(
        llm_engine=SimpleNamespace(
            vllm_config=SimpleNamespace(use_v2_model_runner=use_v2)
        )
    )
    if use_v2:
        assert_v2_model_runner(llm)
    else:
        with pytest.raises(AssertionError, match="Model Runner V2"):
            assert_v2_model_runner(llm)
