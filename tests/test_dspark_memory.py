# SPDX-License-Identifier: Apache-2.0
"""DSpark resource admission, real buffer storage and recovery contracts."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import mlx.core as mx
import pytest

from tests.test_dspark_loader import checkpoint
from tests.test_dspark_proposer import _features, _proposer, _seed, _state
from tests.test_v1_worker import TestPagedAttentionPlanDiagnostics as PlanDiagnostics
from vllm_metal.v1.dspark.loader import load_drafter
from vllm_metal.v1.dspark.memory import DSparkMemoryPlan
from vllm_metal.v1.dspark.model import ContextArena, CtxCache
from vllm_metal.v1.dspark.paging import PAGED_BLOCK_SIZE


def test_append_reuses_chunks_and_rollback_hides_old_suffix(monkeypatch):
    cache = CtxCache(capacity=300)
    values = mx.arange(300 * 8, dtype=mx.float32).reshape(1, 2, 300, 4)
    cache.append(values[:, :, :250], -values[:, :, :250])
    mx.eval(cache.k, cache.v)
    allocated = cache.allocated_bytes
    concatenate = mx.concatenate

    def no_growth(*args, **kwargs):
        pytest.fail("an append within capacity must not concatenate the context")

    monkeypatch.setattr(mx, "concatenate", no_growth)
    cache.append(values[:, :, 250:256], -values[:, :, 250:256])
    mx.eval(cache.k, cache.v)
    assert cache.allocated_bytes == allocated
    assert mx.array_equal(cache.k, values[:, :, :256]).item()
    cache.trim_to(10)
    cache.append(values[:, :, 20:23], -values[:, :, 20:23])
    mx.eval(cache.k, cache.v)
    assert cache.k.shape[2] == cache.v.shape[2] == 13
    assert mx.array_equal(cache.k[:, :, :10], values[:, :, :10]).item()
    assert mx.array_equal(cache.k[:, :, 10:], values[:, :, 20:23]).item()
    monkeypatch.setattr(mx, "concatenate", concatenate)
    cache.append(values[:, :, 13:], -values[:, :, 13:])
    mx.eval(cache.k, cache.v)
    assert cache.length == 300
    assert cache.allocated_bytes == 2 * values.nbytes
    with pytest.raises(ValueError, match="reserved capacity"):
        cache.append(values[:, :, :1], values[:, :, :1])
    assert cache.length == 300


@pytest.mark.parametrize("dtype", [mx.bfloat16, mx.float16, mx.float32])
def test_memory_plan_matches_actual_full_context_storage(dtype):
    proposer = _proposer()
    proposer._drafter.set_dtype(dtype)
    plan = DSparkMemoryPlan.build(
        proposer._config,
        itemsize=mx.array(0, dtype=dtype).itemsize,
        max_num_seqs=2,
        max_model_len=257,
        max_num_batched_tokens=64,
    )
    caches = [
        proposer._drafter.make_ctx_cache(plan.max_context_tokens) for _ in range(2)
    ]
    # Input dtype intentionally differs from the BF16/FP16 draft precision.
    features = _features(list(range(plan.max_context_tokens)))[None]
    for row in caches:
        proposer._drafter.update_context(features, 0, row)
        mx.eval([(cache.k, cache.v) for cache in row])
        assert all(cache.k.dtype == cache.v.dtype == dtype for cache in row)
    # The arena the proposer allocates for this plan is exactly the reserved
    # context bytes: every slot at full capacity plus the block's scratch.
    config = proposer._config
    arenas = [
        ContextArena(
            slots=plan.max_contexts,
            kv_heads=config.n_kv_heads,
            capacity=plan.max_context_tokens,
            block_size=config.block_size,
            head_dim=config.attn_head_dim,
            dtype=dtype,
        )
        for _ in proposer._drafter.layers
    ]
    assert sum(arena.nbytes for arena in arenas) == plan.context_bytes
    assert plan.block_size == config.block_size


def test_context_slot_exhaustion_releases_and_recovers():
    proposer = _proposer(max_num_seqs=1)
    first, second = _state([1, 2, 3]), _state([4, 5, 6])
    assert _seed(proposer, first, "a") is not None
    assert _seed(proposer, second, "b") is None
    assert proposer._contexts["b"].disabled_reason == "context capacity exhausted"
    assert not proposer._contexts["b"].caches
    proposer.release_requests({"a"})
    # Only a recompute from zero can recover the skipped feature prefix.
    assert _seed(proposer, second, "b") is not None
    assert proposer._contexts["b"].covered_end == 2


def test_memory_pressure_rejects_before_allocation_then_recovers(monkeypatch):
    proposer = _proposer()
    current = 1234
    monkeypatch.setattr(mx, "get_active_memory", lambda: current)
    # Context slots are reserved up front in the arena, so the live check
    # that admission applies is the workspace the draft step will need.
    budget = current + proposer.memory_plan.workspace_bytes
    proposer._memory_budget_bytes = budget - 1
    owner = _state([1, 2, 3])
    assert _seed(proposer, owner) is None
    assert proposer._contexts["r"].disabled_reason == "context memory budget exhausted"
    assert not proposer._contexts["r"].caches
    assert proposer._arena[0].free_slots == proposer.memory_plan.max_contexts
    proposer._memory_budget_bytes = budget
    assert _seed(proposer, owner) is not None
    slot_bytes = proposer.memory_plan.context_bytes // proposer.memory_plan.max_contexts
    assert (
        sum(cache.allocated_bytes for cache in proposer._contexts["r"].caches)
        == slot_bytes
    )
    assert proposer._arena[0].free_slots == proposer.memory_plan.max_contexts - 1


@pytest.mark.parametrize(
    "error",
    [
        MemoryError("injected"),
        RuntimeError("[metal::malloc] Resource limit (injected)"),
    ],
)
@pytest.mark.parametrize("phase", ["ingest", "draft"])
def test_allocation_failure_drops_partial_context_and_next_request_recovers(
    monkeypatch, phase, error
):
    proposer = _proposer()
    owner = _state([1, 2, 3])
    assert _seed(proposer, owner, "old") is not None
    name = "update_context" if phase == "ingest" else "backbone"
    original = getattr(proposer._drafter, name)

    def fail(*args, **kwargs):
        if phase == "ingest":
            # Fail after real writes, not before any state exists.
            original(*args, **kwargs)
        raise error

    monkeypatch.setattr(proposer._drafter, name, fail)
    assert _seed(proposer, _state([4, 5, 6]), "failed") is None
    assert not proposer._contexts
    monkeypatch.setattr(proposer._drafter, name, original)
    assert _seed(proposer, _state([7, 8, 9]), "new") is not None
    assert proposer._contexts["new"].covered_end == 2


def test_scheduler_span_cannot_exceed_reserved_workspace():
    proposer = _proposer()
    proposer.memory_plan = replace(proposer.memory_plan, max_step_tokens=2)
    with pytest.raises(ValueError, match="reserved scheduler token bound"):
        _seed(proposer, _state([1, 2, 3, 4]))
    assert not proposer._contexts


def test_target_kv_plan_subtracts_complete_dspark_reservation(monkeypatch):
    proposer = _proposer()
    runner = SimpleNamespace(
        is_hybrid=False,
        draft_scratch_reserve_bytes=lambda: 0,
        _dspark_memory_plan=proposer.memory_plan,
    )
    planner = PlanDiagnostics()._make_planner(
        runner, gpu_memory_utilization=0.5, per_block_bytes=4096
    )
    monkeypatch.setattr(planner, "_metal_limit_bytes", lambda: 8_000_000_000)
    monkeypatch.setattr(planner, "get_model_memory_usage", lambda: 1_000_000_000)
    plain = planner._paged_attention_plan(overhead=100_000_000)
    planner._worker.vllm_config.speculative_config = SimpleNamespace(method="dspark")
    spec = planner._paged_attention_plan(overhead=100_000_000)
    # The arena is allocated at load and therefore inside the measured model
    # memory; the planner subtracts the capture and workspace that remain.
    plan = proposer.memory_plan
    assert plan.planning_reserve_bytes == plan.reserve_bytes - plan.context_bytes
    assert plain.kv_budget - spec.kv_budget == plan.planning_reserve_bytes
    assert (
        spec.num_blocks * spec.per_block_bytes
        + spec.model_memory
        + spec.overhead
        + plan.planning_reserve_bytes
        <= spec.usable_metal
    )
    runner._dspark_memory_plan = None
    with pytest.raises(RuntimeError, match="before target KV allocation"):
        planner._paged_attention_plan(overhead=100_000_000)


def test_loader_budget_and_resolved_identity_fail_before_reading_weights(
    tmp_path, monkeypatch
):
    model, _ = checkpoint(tmp_path)

    def no_read(*args, **kwargs):
        pytest.fail("startup admission must precede weight data reads")

    monkeypatch.setattr(mx, "load", no_read)
    with pytest.raises(ValueError, match="startup memory budget"):
        load_drafter(str(tmp_path), memory_budget_bytes=1)
    with pytest.raises(ValueError, match="resolved draft ModelConfig"):
        load_drafter(str(tmp_path), expected_config=replace(model.config, block_size=5))


def test_memory_plan_honors_configured_context_cap():
    proposer = _proposer()
    common = {"itemsize": 2, "max_model_len": 256, "max_num_batched_tokens": 32}
    build = DSparkMemoryPlan.build
    assert (
        build(proposer._config, max_num_seqs=8, max_contexts=2, **common).max_contexts
        == 2
    )
    assert (
        build(proposer._config, max_num_seqs=8, max_contexts=64, **common).max_contexts
        == 8
    )
    assert build(proposer._config, max_num_seqs=64, **common).max_contexts == 32
    # Each slot holds the full context plus the drafted block's scratch K/V.
    block = proposer._config.block_size
    assert (
        build(proposer._config, max_num_seqs=8, max_contexts=2, **common).context_bytes
        == 2
        * (256 + block)
        * build(proposer._config, max_num_seqs=8, **common).kv_bytes_per_token
    )
    with pytest.raises(ValueError, match="at least one"):
        build(proposer._config, max_num_seqs=8, max_contexts=0, **common)


class TestTheMemoryPlanSizesForTheBackendItWillGet:
    """The plan must reserve for the context backend the proposer will actually build.

    `planning_reserve_bytes` is not a private drafter number: `cache_policy` subtracts it
    from what the TARGET model's KV cache may use. A drafter that reserves for a rewrite
    it cannot perform takes those blocks away from the target and hands back nothing.
    """

    KWARGS = {
        "itemsize": 2,
        "max_num_seqs": 16,
        "max_model_len": 4096,
        "max_num_batched_tokens": 2048,
    }

    def _plans(self, monkeypatch):
        from tests.test_dspark_contracts import draft_hf_config
        from vllm_metal.v1.dspark.config import DSparkConfig

        # The toy config's head dim has no kernel, so the pool would be refused; the
        # plan is pure arithmetic and only needs a head size the kernels are built for.
        config = replace(
            DSparkConfig.from_dict(draft_hf_config().to_dict()), head_dim=64
        )
        monkeypatch.delenv("VLLM_METAL_DSPARK_PAGED_CONTEXT", raising=False)
        arena = DSparkMemoryPlan.build(config, **self.KWARGS)
        monkeypatch.setenv("VLLM_METAL_DSPARK_PAGED_CONTEXT", "1")
        paged = DSparkMemoryPlan.build(config, **self.KWARGS)
        return arena, paged

    def test_only_the_arena_reserves_a_transient_copy_of_itself(self, monkeypatch):
        arena, paged = self._plans(monkeypatch)
        # Every other workspace term is a function of the config and the scheduler
        # limits alone, so the entire difference between the two is that copy.
        second_arena = (
            arena.max_contexts * arena.max_context_tokens * arena.kv_bytes_per_token
        )
        assert second_arena > 0
        assert arena.workspace_bytes - paged.workspace_bytes == second_arena

    def test_the_target_cache_gets_those_bytes_back(self, monkeypatch):
        arena, paged = self._plans(monkeypatch)
        saved = arena.planning_reserve_bytes - paged.planning_reserve_bytes
        assert saved == (
            arena.max_contexts * arena.max_context_tokens * arena.kv_bytes_per_token
        )
        # not a rounding difference: most of what the drafter asks the target to give up
        assert saved > 0.4 * arena.planning_reserve_bytes

    def test_the_saving_at_the_shipped_drafter_geometry(self, monkeypatch):
        """The number the PR claims, at the drafter it claims it for.

        The pinned Qwen3 DSpark drafter is five layers, eight KV heads, head dim 128,
        bf16. At 16 requests x 4096 tokens that is 20 KiB of context per token.
        """
        from tests.test_dspark_contracts import draft_hf_config
        from vllm_metal.v1.dspark.config import DSparkConfig

        config = replace(
            DSparkConfig.from_dict(draft_hf_config().to_dict()),
            head_dim=128,
            num_hidden_layers=5,
            num_key_value_heads=8,
            num_attention_heads=32,
        )
        monkeypatch.delenv("VLLM_METAL_DSPARK_PAGED_CONTEXT", raising=False)
        arena = DSparkMemoryPlan.build(config, **self.KWARGS)
        monkeypatch.setenv("VLLM_METAL_DSPARK_PAGED_CONTEXT", "1")
        paged = DSparkMemoryPlan.build(config, **self.KWARGS)
        assert arena.kv_bytes_per_token == 20480  # 20 KiB per token across five layers
        saved = arena.planning_reserve_bytes - paged.planning_reserve_bytes
        assert saved == 16 * 4096 * 20480  # 1.34 GB returned to the target KV cache

    def test_the_pool_still_houses_the_arena_worst_case(self, monkeypatch):
        arena, paged = self._plans(monkeypatch)
        assert paged.paged_blocks > 0
        assert arena.paged_blocks == 0
        # whole pages plus the sink: never short, and never more than one page per
        # context over
        slack = paged.context_bytes - arena.context_bytes
        assert -paged.kv_bytes_per_token * paged.block_size <= slack
        assert slack <= (paged.max_contexts + 1) * PAGED_BLOCK_SIZE * (
            paged.kv_bytes_per_token
        )
