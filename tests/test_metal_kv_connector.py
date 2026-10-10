# SPDX-License-Identifier: Apache-2.0
"""MetalKVConnector keeps upstream's step methods working without torch caches."""

from __future__ import annotations

from types import SimpleNamespace

from vllm.v1.core.sched.output import CachedRequestData, SchedulerOutput

import vllm_metal.v1.kv_connector as metal_kv_connector
from vllm_metal.v1.kv_connector import MetalKVConnector


def _group(registered: list) -> SimpleNamespace:
    return SimpleNamespace(
        register_kv_caches=registered.append,
        set_host_xfer_buffer_ops=lambda op: None,
    )


def test_does_not_register_torch_caches(monkeypatch):
    """The runner registers KVCacheStorage itself."""
    registered: list = []
    group = _group(registered)
    monkeypatch.setattr(metal_kv_connector, "get_kv_transfer_group", lambda: group)

    MetalKVConnector(SimpleNamespace())

    assert registered == []


def test_a_fresh_connector_runs_a_whole_step(spy_group):
    """Upstream's step methods run on a fresh instance."""
    scheduler_output = SchedulerOutput(
        scheduled_new_reqs=[],
        scheduled_cached_reqs=CachedRequestData.make_empty(),
        num_scheduled_tokens={},
        total_num_scheduled_tokens=0,
        scheduled_spec_decode_tokens={},
        scheduled_encoder_inputs={},
        num_common_prefix_blocks=[],
        finished_req_ids=set(),
        free_encoder_mm_hashes=[],
        num_invalid_spec_tokens=None,
        num_spec_tokens_to_schedule=0,
    )
    scheduler_output.kv_connector_metadata = SimpleNamespace()
    spy_group.step = "1"

    output = MetalKVConnector(SimpleNamespace()).no_forward(scheduler_output)

    assert spy_group.events == ["preempt:1", "open:1", "close:1"]
    assert spy_group.finish_forward_calls == 1
    assert output.kv_connector_output.finished_recving == {"r-1"}
