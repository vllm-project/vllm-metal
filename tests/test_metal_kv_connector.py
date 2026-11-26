# SPDX-License-Identifier: Apache-2.0
"""The runner drives KV transfers through MRv2's ``ActiveKVConnector``."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput
from vllm.v1.worker.gpu.kv_connector import NO_OP_KV_CONNECTOR, ActiveKVConnector

import vllm_metal.v1.model_runner as mr
from tests.stub_runner import make_stub_runner


def test_without_a_connector_the_driver_is_a_no_op(monkeypatch):
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: False)

    assert make_stub_runner()._kv_connector() is NO_OP_KV_CONNECTOR


def test_a_step_before_registration_fails(monkeypatch):
    """A connector step needs the caches registered first."""
    monkeypatch.setattr(mr, "has_kv_transfer_group", lambda: True)

    with pytest.raises(RuntimeError, match="before register_kv_connector_caches"):
        make_stub_runner()._kv_connector()


def test_a_fresh_connector_runs_a_whole_step(spy_group):
    """Upstream's step methods run over a registered KVCacheStorage."""
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        num_invalid_spec_tokens=None,
        num_spec_tokens_to_schedule=0,
    )
    scheduler_output.kv_connector_metadata = SimpleNamespace()
    spy_group.step = "1"
    storage = object()

    driver = ActiveKVConnector(SimpleNamespace(), storage)  # type: ignore[arg-type]

    output = driver.no_forward(scheduler_output)

    assert spy_group.registered[-1] is storage
    assert spy_group.events == ["preempt:1", "open:1", "close:1"]
    assert spy_group.finish_forward_calls == 1
    assert output.kv_connector_output.finished_recving == {"r-1"}
