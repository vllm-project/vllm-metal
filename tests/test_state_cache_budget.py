# SPDX-License-Identifier: Apache-2.0
"""A fixed state budget must fund real KV capacity without double billing."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch
from vllm.config import VllmConfig
from vllm.v1.attention.backends.utils import record_kv_cache_layout
from vllm.v1.core.kv_cache_utils import get_kv_cache_config_from_groups
from vllm.v1.kv_cache_interface import FullAttentionSpec, KVCacheGroupSpec, MambaSpec

from vllm_metal.attention.caches.placement import KV_CACHE_LAYOUT
from vllm_metal.state_budget import (
    STATE_BUDGET_ASYNC_SCHEDULER,
    STATE_BUDGET_SCHEDULER,
    StateCacheBudget,
    configure_state_cache_budget,
    resolve_state_cache_budget,
    state_cache_budget_bytes,
    state_cache_budget_from_kv_config,
    state_cache_rows_per_request,
    state_cache_scratch_allowance,
)
from vllm_metal.v1.cache_policy import WorkerCachePlanner, _state_budget_for_runner

MIB = 1024**2
GIB = 1024**3
GDN_LAYER_BYTES = 3_207_168
STATE_ROW_BYTES = 16 * GDN_LAYER_BYTES
KV_BLOCK_BYTES = 16 * 784 * 4096


def config(**kwargs):
    base = SimpleNamespace(
        additional_config={"state_cache_budget_mib": 2048},
        max_concurrent_batches=2,
        speculative_config=None,
        kv_transfer_config=None,
        ec_transfer_config=None,
        cache_config=SimpleNamespace(
            block_size=784,
            mamba_cache_mode="align",
            enable_prefix_caching=True,
            prefix_match_unit=None,
            gpu_memory_utilization=0.75,
        ),
        parallel_config=SimpleNamespace(
            tensor_parallel_size=1,
            pipeline_parallel_size=1,
            data_parallel_size=1,
            distributed_executor_backend="uni",
        ),
        scheduler_config=SimpleNamespace(
            scheduler_cls=None,
            async_scheduling=True,
            max_num_seqs=4,
            max_num_batched_tokens=1568,
        ),
    )
    base.__dict__.update(kwargs)
    return base


@pytest.mark.parametrize("bad", [0, -1, True, False, 1.5, "2048", None])
def test_invalid_budget_cannot_silently_enable_or_disable(bad):
    cfg = config(additional_config={"state_cache_budget_mib": bad})
    with pytest.raises(ValueError, match="positive integer"):
        state_cache_budget_bytes(cfg)


def test_no_option_preserves_scheduler_and_no_reservation():
    cfg = config(additional_config={})
    original = cfg.scheduler_config.scheduler_cls = object()
    configure_state_cache_budget(cfg)
    assert cfg.scheduler_config.scheduler_cls is original
    assert resolve_state_cache_budget(cfg, STATE_ROW_BYTES) is None


@pytest.mark.parametrize("async_enabled", [False, True, None])
def test_scheduler_selection_preserves_async(async_enabled):
    cfg = config()
    cfg.scheduler_config.async_scheduling = async_enabled
    configure_state_cache_budget(cfg)
    expected = (
        STATE_BUDGET_SCHEDULER
        if async_enabled is False
        else STATE_BUDGET_ASYNC_SCHEDULER
    )
    assert cfg.scheduler_config.scheduler_cls == expected
    configure_state_cache_budget(cfg)
    assert cfg.scheduler_config.scheduler_cls == expected


@pytest.mark.parametrize(
    ("section", "field", "value"),
    [
        (None, "speculative_config", object()),
        (None, "kv_transfer_config", object()),
        (None, "ec_transfer_config", object()),
        ("parallel_config", "tensor_parallel_size", 2),
        ("parallel_config", "pipeline_parallel_size", 2),
        ("parallel_config", "data_parallel_size", 2),
        ("parallel_config", "distributed_executor_backend", "mp"),
        ("cache_config", "prefix_match_unit", 16),
        ("cache_config", "kv_offloading_size", 1),
        ("scheduler_config", "policy", "priority"),
    ],
)
def test_unsupported_lifecycles_fail_before_allocation(section, field, value):
    cfg = config()
    setattr(getattr(cfg, section) if section else cfg, field, value)
    with pytest.raises(NotImplementedError, match="does not yet support"):
        configure_state_cache_budget(cfg)


def test_cannot_override_foreign_scheduler():
    cfg = config()
    cfg.scheduler_config.scheduler_cls = "third.party.Scheduler"
    with pytest.raises(ValueError, match="custom scheduler"):
        configure_state_cache_budget(cfg)


def test_budget_rounds_down_to_whole_physical_rows():
    budget = StateCacheBudget.from_bytes(2048 * MIB, STATE_ROW_BYTES)
    assert budget.capacity == 41
    assert budget.allocated_bytes == 41 * STATE_ROW_BYTES <= budget.requested_bytes
    with pytest.raises(ValueError, match="smaller than one"):
        StateCacheBudget.from_bytes(STATE_ROW_BYTES - 1, STATE_ROW_BYTES)


def test_scheduler_striping_counts_sixteen_pools_not_forty_eight_layers():
    cfg = VllmConfig()
    cfg.additional_config = {"state_cache_budget_mib": 2048}
    record_kv_cache_layout(cfg.cache_config, KV_CACHE_LAYOUT)
    attention = FullAttentionSpec(
        block_size=784, num_kv_heads=4, head_size=256, dtype=torch.bfloat16
    )
    state = MambaSpec(
        block_size=784,
        shapes=((3, 10240), (48, 128, 128)),
        dtypes=(torch.bfloat16, torch.float32),
        page_size_padded=attention.page_size_bytes,
        mamba_cache_mode="align",
    )
    assert state.state_content_size_bytes == GDN_LAYER_BYTES
    groups = [
        KVCacheGroupSpec([f"attention.{i}" for i in range(16)], attention),
        *[
            KVCacheGroupSpec([f"state.{g}.{i}" for i in range(16)], state)
            for g in range(3)
        ],
    ]
    kv = get_kv_cache_config_from_groups(cfg, groups, KV_BLOCK_BYTES * 100)
    budget = state_cache_budget_from_kv_config(cfg, kv)
    assert budget == StateCacheBudget.from_bytes(2048 * MIB, STATE_ROW_BYTES)


def test_fixed_budget_becomes_additional_real_kv_blocks(monkeypatch):
    cfg = config()
    runner = SimpleNamespace(
        vllm_config=cfg,
        is_hybrid=True,
        cache_config=cfg.cache_config,
        hybrid_runtime_plan=SimpleNamespace(
            family=SimpleNamespace(label="gdn"),
            layers=SimpleNamespace(num_state=48, num_attention=16),
        ),
        hybrid_align_state_bytes_per_block=lambda: STATE_ROW_BYTES,
        hybrid_align_growth_bytes_per_block=lambda: GDN_LAYER_BYTES,
        draft_scratch_reserve_bytes=lambda: 0,
    )
    worker = SimpleNamespace(
        vllm_config=cfg,
        model_runner=runner,
        get_cache_block_size_bytes=lambda: KV_BLOCK_BYTES,
    )
    planner = WorkerCachePlanner(worker)
    monkeypatch.setattr(planner, "_metal_limit_bytes", lambda: 64 * GIB)
    monkeypatch.setattr(planner, "get_model_memory_usage", lambda: 16 * GIB)
    bounded = planner._paged_attention_plan(overhead=2 * GIB)
    cfg.additional_config = {}
    unbounded = planner._paged_attention_plan(overhead=2 * GIB)
    assert bounded.per_block_bytes == KV_BLOCK_BYTES
    assert (
        unbounded.per_block_bytes == KV_BLOCK_BYTES + STATE_ROW_BYTES + GDN_LAYER_BYTES
    )
    budget = bounded.state_cache_budget
    assert budget is not None
    scratch = bounded.state_cache_scratch
    assert scratch is not None
    assert scratch.total_bytes == 4_310_433_792
    assert bounded.kv_budget + budget.allocated_bytes + scratch.total_bytes == 30 * GIB
    assert bounded.num_blocks == bounded.kv_budget // KV_BLOCK_BYTES
    assert bounded.num_blocks > unbounded.num_blocks
    assert (
        bounded.num_blocks * KV_BLOCK_BYTES
        + budget.allocated_bytes
        + scratch.total_bytes
        <= 30 * GIB
    )
    assert unbounded.state_cache_scratch is None
    assert "runtime overhead is estimated" in bounded.format_breakdown()
    assert "state_scratch_allowance=4.31GB" in bounded.format_breakdown()
    assert "additional to profile, not a total peak bound" in bounded.format_breakdown()


@pytest.mark.parametrize(
    "batches,max_requests,max_tokens,expected_requests,expected_copies",
    [
        (2, 4, 1568, 4, 7),  # default async, constrained by the state quota
        (1, 4, 1568, 4, 4),  # synchronous scheduling keeps one in-flight batch
        (2, 2, 1568, 2, 7),  # explicit smaller max_num_seqs
        (2, 100, 2, 2, 7),  # every scheduled request needs at least one token
        (2, 100, 1568, 4, 7),  # C=41 and 9 working rows/request limit concurrency
    ],
)
def test_state_scratch_tracks_actual_concurrency_bounds(
    batches, max_requests, max_tokens, expected_requests, expected_copies
):
    cfg = config(max_concurrent_batches=batches)
    cfg.scheduler_config.max_num_seqs = max_requests
    cfg.scheduler_config.max_num_batched_tokens = max_tokens
    budget = StateCacheBudget.from_bytes(2048 * MIB, STATE_ROW_BYTES)
    scratch = state_cache_scratch_allowance(cfg, budget, 3)
    assert scratch.in_flight_batches == batches
    assert scratch.active_requests == expected_requests
    assert scratch.payload_copies == expected_copies
    # Three rows cover all 48 logical GDN layers, not just 16 shared pools.
    assert scratch.bytes_per_request == 48 * GDN_LAYER_BYTES
    assert (
        scratch.total_bytes
        == expected_copies * expected_requests * 48 * GDN_LAYER_BYTES
    )


def test_state_scratch_is_independent_of_spare_stable_checkpoint_rows():
    cfg = config()
    small = StateCacheBudget.from_bytes(2048 * MIB, STATE_ROW_BYTES)
    large = StateCacheBudget.from_bytes(4096 * MIB, STATE_ROW_BYTES)
    assert small.capacity < large.capacity
    assert state_cache_scratch_allowance(
        cfg, small, 3
    ) == state_cache_scratch_allowance(cfg, large, 3)


def test_non_gdn_budget_fails_instead_of_reducing_kv_reserve(monkeypatch):
    cfg = config()
    runner = SimpleNamespace(vllm_config=cfg, is_hybrid=False, hybrid_runtime_plan=None)
    worker = SimpleNamespace(vllm_config=cfg, model_runner=runner)
    planner = WorkerCachePlanner(worker)
    monkeypatch.setattr(planner, "_metal_limit_bytes", lambda: 64 * GIB)
    monkeypatch.setattr(planner, "get_model_memory_usage", lambda: 16 * GIB)
    worker.get_cache_block_size_bytes = Mock(return_value=KV_BLOCK_BYTES)
    with pytest.raises(NotImplementedError, match="GDN align"):
        planner._paged_attention_plan(overhead=2 * GIB)


@pytest.mark.parametrize("batches,groups,expected", [(1, 3, 6), (2, 3, 9), (1, 1, 2)])
def test_working_reserve_is_source_plus_one_row_per_in_flight_batch(
    batches, groups, expected
):
    cfg = config(max_concurrent_batches=batches)
    assert state_cache_rows_per_request(cfg, groups) == expected


def test_one_request_synchronous_floor_is_six_rows_for_qwen38_geometry():
    runner_kw = {
        "is_hybrid": True,
        "cache_config": config().cache_config,
        "hybrid_runtime_plan": SimpleNamespace(
            family=SimpleNamespace(label="gdn"),
            layers=SimpleNamespace(num_state=48, num_attention=16),
        ),
        "hybrid_align_state_bytes_per_block": lambda: STATE_ROW_BYTES,
    }
    too_small = config(
        additional_config={"state_cache_budget_mib": 293}, max_concurrent_batches=1
    )
    with pytest.raises(ValueError, match=r"at least 6 rows \(294 MiB\)"):
        _state_budget_for_runner(SimpleNamespace(vllm_config=too_small, **runner_kw))
    floor = config(
        additional_config={"state_cache_budget_mib": 294}, max_concurrent_batches=1
    )
    budget = _state_budget_for_runner(SimpleNamespace(vllm_config=floor, **runner_kw))
    assert budget is not None
    assert budget.capacity == 6
    assert budget.allocated_bytes == 6 * STATE_ROW_BYTES


def test_four_synchronous_requests_need_twenty_four_rows():
    cfg = config(max_concurrent_batches=1)
    cfg.scheduler_config.max_num_seqs = 4
    rows = state_cache_rows_per_request(cfg, 3)
    assert rows == 6
    needed = 4 * rows * STATE_ROW_BYTES
    mib = (needed + MIB - 1) // MIB
    assert mib == 1175
    budget = StateCacheBudget.from_bytes(mib * MIB, STATE_ROW_BYTES)
    assert budget.capacity == 24
    assert budget.capacity // rows == 4
    assert StateCacheBudget.from_bytes(1174 * MIB, STATE_ROW_BYTES).capacity == 23


def test_budget_below_inflight_working_reserve_fails_before_pool_allocation():
    cfg = config(additional_config={"state_cache_budget_mib": 400})
    runner = SimpleNamespace(
        vllm_config=cfg,
        is_hybrid=True,
        cache_config=cfg.cache_config,
        hybrid_runtime_plan=SimpleNamespace(
            family=SimpleNamespace(label="gdn"),
            layers=SimpleNamespace(num_state=48, num_attention=16),
        ),
        hybrid_align_state_bytes_per_block=lambda: STATE_ROW_BYTES,
    )
    with pytest.raises(ValueError, match=r"at least 9 rows \(441 MiB\)"):
        _state_budget_for_runner(runner)
