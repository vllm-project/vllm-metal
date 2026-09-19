# SPDX-License-Identifier: Apache-2.0
"""Real scheduler/core FIFO pressure regressions; no model or GPU allocations."""

from __future__ import annotations

import hashlib
import pickle
from collections import defaultdict, deque
from concurrent.futures import Future
from queue import Queue
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.config import (
    CacheConfig,
    ObservabilityConfig,
    ParallelConfig,
    SchedulerConfig,
)
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
from vllm.v1.engine.core import EngineCore
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

import vllm_metal.v1.state_budget_scheduler as budget_module
from vllm_metal.state_budget import StateCacheBudget

BLOCK = 16


def _hash(value):
    return hashlib.sha256(pickle.dumps(value)).digest()


@pytest.fixture(autouse=True)
def _initialize_hash():
    init_none_hash(_hash)


def _scheduler(
    monkeypatch, *, depth=2, blocks=128, groups=3, sequences=4, capacity=None
):
    chunk = 4 * BLOCK
    state = MambaSpec(
        block_size=BLOCK,
        shapes=((4, 4),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_prefill_checkpoint_blocks=0,
    )
    kv = KVCacheConfig(
        num_blocks=blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(
                    block_size=BLOCK, num_kv_heads=1, head_size=2, dtype=torch.float32
                ),
            ),
            *(KVCacheGroupSpec([f"state_{i}"], state) for i in range(groups)),
        ],
        prefix_cache_retention_interval=0,
    )
    cache = CacheConfig(
        block_size=BLOCK,
        mamba_block_size=BLOCK,
        mamba_cache_mode="align",
        enable_prefix_caching=True,
    )
    cache.num_gpu_blocks = blocks
    config = SimpleNamespace(
        scheduler_config=SchedulerConfig(
            max_model_len=32 * BLOCK,
            is_encoder_decoder=False,
            max_num_batched_tokens=chunk,
            max_num_seqs=sequences,
            async_scheduling=depth > 1,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=BLOCK,
        ),
        cache_config=cache,
        parallel_config=ParallelConfig(),
        observability_config=ObservabilityConfig(),
        model_config=SimpleNamespace(
            uses_mrope=False,
            uses_xdrope=False,
            is_encoder_decoder=False,
            is_diffusion=False,
            max_model_len=32 * BLOCK,
            enable_return_routed_experts=False,
            return_sampling_mask=False,
        ),
        lora_config=None,
        kv_events_config=None,
        is_mm_encoder_only=False,
        kv_transfer_config=None,
        ec_transfer_config=None,
        ec_manager_config=SimpleNamespace(get_encoder_cache_manager_obj=lambda: None),
        speculative_config=None,
        num_speculative_tokens=0,
        num_lookahead_tokens=0,
        max_in_flight_tokens=depth * chunk,
        max_concurrent_batches=depth,
        use_v2_model_runner=False,
    )
    if capacity is None:
        capacity = (depth + 1) * groups * sequences
    monkeypatch.setattr(
        budget_module,
        "state_cache_budget_from_kv_config",
        lambda *_: StateCacheBudget(capacity * 64, 64, capacity, capacity * 64),
    )
    scheduler_type = (
        budget_module.StateBudgetAsyncScheduler
        if depth > 1
        else budget_module.StateBudgetScheduler
    )
    return scheduler_type(
        config,
        kv,
        structured_output_manager=SimpleNamespace(should_advance=lambda *a, **k: False),
        block_size=BLOCK,
        hash_block_size=BLOCK,
        mm_registry=SimpleNamespace(supports_multimodal_inputs=lambda _: False),
    )


def _request(name, prompt=8 * BLOCK + 1, output=16, offset=0):
    return Request(
        request_id=name,
        prompt_token_ids=list(range(offset, offset + prompt)),
        sampling_params=SamplingParams(
            temperature=0, max_tokens=output, ignore_eos=True
        ),
        pooling_params=None,
        block_hasher=get_request_block_hasher(BLOCK, _hash),
    )


class _Result(Future):
    def __init__(self, output, on_result):
        super().__init__()
        self.set_result(output)
        self.on_result = on_result

    def result(self, timeout=None):
        if self.on_result is not None:
            callback, self.on_result = self.on_result, None
            callback()
        return super().result(timeout)


class _Executor:
    """Only model work is synthetic; core selects and consumes the real FIFO."""

    def __init__(self, scheduler):
        self.scheduler = scheduler
        self.steps = []
        self.result_steps = []
        self.sample_counts = defaultdict(int)
        self.pending_sample = None
        self.before_result = None

    def execute_model(self, output, non_block=True):
        assert non_block
        self.steps.append(output)
        ids = list(output.num_scheduled_tokens)
        sampled = []
        for rid in ids:
            if self.scheduler.requests[rid].is_prefill_chunk:
                sampled.append([])
            else:
                self.sample_counts[rid] += 1
                sampled.append([100 + self.sample_counts[rid]])
        result = ModelRunnerOutput(
            req_ids=ids,
            req_id_to_index={rid: i for i, rid in enumerate(ids)},
            sampled_token_ids=sampled,
        )

        def delivered():
            self.result_steps.append(output)
            if self.before_result is not None:
                self.before_result(output)

        future = _Result(result, delivered)
        self.pending_sample = future
        return future

    def sample_tokens(self, _grammar, non_block=False):
        assert self.pending_sample is not None
        return self.pending_sample if non_block else self.pending_sample.result()


def _core(scheduler, depth):
    # Bypass only engine construction/model loading. Scheduling, FIFO handling,
    # abort processing and result delivery use the unmodified EngineCore methods.
    core = object.__new__(EngineCore)
    core.scheduler = scheduler
    core.model_executor = _Executor(scheduler)
    core.vllm_config = scheduler.vllm_config
    core.log_stats = False
    core.batch_queue_size = depth
    core.batch_queue = deque(maxlen=depth) if depth > 1 else None
    core.is_ec_consumer = True
    core.is_pooling_model = False
    core.aborts_queue = Queue()
    core.async_scheduling = depth > 1
    core.check_for_draft_tokens = False
    return core


def _step(core):
    return core.step_with_batch_queue() if core.batch_queue is not None else core.step()


def _drain(core, limit=1000):
    delivered = defaultdict(list)
    for _ in range(limit):
        if not core.scheduler.has_requests() and not core.batch_queue:
            break
        outputs, _ = _step(core)
        for batch in (outputs or {}).values():
            for output in batch.outputs:
                delivered[output.request_id].extend(output.new_token_ids)
    else:
        pytest.fail("real EngineCore FIFO did not drain")
    assert not core.scheduler.requests
    assert not core.scheduler.deferred_frees
    assert not core.batch_queue
    assert not core.scheduler.finished_req_ids
    assert core.scheduler.processed_step_seq == core.scheduler.sched_step_seq
    assert core.model_executor.result_steps == core.model_executor.steps
    return delivered


@pytest.mark.parametrize("depth", [1, 2, 3])
def test_real_core_pressure_finishes_without_preempting_pending_output(
    monkeypatch, depth
):
    scheduler = _scheduler(monkeypatch, depth=depth, blocks=24)
    requests = [
        _request(f"r{i}", prompt=(8 + i) * BLOCK + 1, offset=i * 1000) for i in range(4)
    ]
    for request in requests:
        scheduler.add_request(request)
    core = _core(scheduler, depth)
    preempt = scheduler._preempt_request
    preemptions = []

    def observed_preempt(request, *args, **kwargs):
        preemptions.append(request.request_id)
        assert request.num_in_flight_tokens == 0
        return preempt(request, *args, **kwargs)

    monkeypatch.setattr(scheduler, "_preempt_request", observed_preempt)
    delivered = _drain(core)
    for request in requests:
        assert request.is_finished()
        # At depth three vLLM can finish/remove a request before consuming an
        # extra sampled result. Its detached object keeps the old count; the
        # real FIFO/fences above must still drain and no extra token is emitted.
        assert delivered[request.request_id] == list(range(101, 117))
    assert preemptions  # Genuine KV pressure, not only a harmless small pool.
    stats = scheduler._state_quota.stats()
    assert (
        stats["pressure_wait_steps"] > 0
        if depth > 1
        else stats["pressure_wait_steps"] == 0
    )
    sequences = [step.metal_state_cache.sequence for step in core.model_executor.steps]
    assert sequences == list(range(1, len(sequences) + 1))
    assert scheduler.max_num_scheduled_tokens == 4 * BLOCK


@pytest.mark.parametrize("phase", ["prefill", "decode"])
def test_core_abort_keeps_rows_pinned_until_last_result_and_delivers_cleanup(
    monkeypatch, phase
):
    scheduler = _scheduler(monkeypatch, depth=3, groups=1, sequences=1, capacity=4)
    request = _request("cancel", prompt=6 * BLOCK + 1, output=16)
    scheduler.add_request(request)
    core = _core(scheduler, 3)
    quota = scheduler._state_quota
    observed = {}

    def cancel_pending(_output):
        if observed or request.is_prefill_chunk != (phase == "prefill"):
            return
        assert request.num_in_flight_tokens > 0
        observed["ids"] = set(dict(quota.resident_blocks))
        observed["fence"] = scheduler.sched_step_seq
        core.aborts_queue.put([request.request_id])

    core.model_executor.before_result = cancel_pending
    for _ in range(30):
        _step(core)
        if observed:
            break
    assert observed and request.status == RequestStatus.FINISHED_ABORTED
    assert scheduler.processed_step_seq < observed["fence"]
    assert scheduler.deferred_frees
    assert all(quota.block_pool.blocks[bid].ref_cnt > 0 for bid in observed["ids"])
    newcomer = _request("new", prompt=2 * BLOCK + 1, offset=2000)
    scheduler.add_request(newcomer)
    delivered = _drain(core)
    assert delivered["new"] == list(range(101, 117))
    metadata_steps = [
        s for s in core.model_executor.steps if "cancel" in s.finished_req_ids
    ]
    assert len(metadata_steps) == 1
    assert any(s.total_num_scheduled_tokens == 0 for s in core.model_executor.steps)
    assert all(s.metal_state_cache is not None for s in metadata_steps)
    stats = quota.stats()
    assert stats["pressure_wait_global_steps"] == 0
    if phase == "prefill":
        assert stats["pressure_wait_state_steps"] > 0
    else:
        assert stats["quota_stalls"] > 0


def test_finished_metadata_reaches_executor_after_last_normal_output(monkeypatch):
    scheduler = _scheduler(monkeypatch, groups=1, sequences=1)
    request = _request("one", prompt=1, output=1)
    scheduler.add_request(request)
    core = _core(scheduler, 2)
    assert _drain(core)["one"] == [101]
    cleanup = [s for s in core.model_executor.steps if "one" in s.finished_req_ids]
    assert len(cleanup) == 1
    assert cleanup[0].total_num_scheduled_tokens == 0
    assert cleanup[0].metal_state_cache.sequence > 1


def test_existing_tail_needs_no_new_block_even_with_no_free_global_blocks(monkeypatch):
    scheduler = _scheduler(monkeypatch, groups=1, sequences=1)
    request = _request("tail", prompt=1)
    scheduler.add_request(request)
    first = scheduler.schedule()
    assert first.total_num_scheduled_tokens == 1
    pool = scheduler.kv_cache_manager.block_pool
    held = pool.get_new_blocks(pool.get_num_free_blocks())
    try:
        assert scheduler.processed_step_seq < scheduler.sched_step_seq
        assert not scheduler._should_wait_for_cache_fence()
        second = scheduler.schedule()
        assert second.total_num_scheduled_tokens == 1
        assert not second.preempted_req_ids
    finally:
        pool.free_blocks(held)


def test_ample_free_blocks_keep_real_async_queue_filled(monkeypatch):
    scheduler = _scheduler(monkeypatch, depth=3, sequences=1)
    request = _request("ample")
    scheduler.add_request(request)
    core = _core(scheduler, 3)
    assert _step(core)[0] is None
    assert _step(core)[0] is None
    assert len(core.batch_queue) == 2
    assert scheduler.processed_step_seq == 0
    assert all(s.total_num_scheduled_tokens > 0 for s in core.model_executor.steps)
    _drain(core)
    assert scheduler._state_quota.stats()["pressure_wait_steps"] == 0


@pytest.mark.parametrize(
    "groups,threshold", [(1, 0), (3, BLOCK // 2), (3, BLOCK), (1, 2 * BLOCK)]
)
def test_running_bound_covers_real_allocation_after_core_skips_earlier_request(
    monkeypatch, groups, threshold
):
    scheduler = _scheduler(monkeypatch, groups=groups, sequences=2)
    scheduler.scheduler_config.long_prefill_token_threshold = threshold
    first, second = _request("skip"), _request("run", offset=1000)
    scheduler.add_request(first)
    scheduler.add_request(second)
    scheduler.schedule()
    # The upstream loop skips this owner without spending the shared budget.
    # The later request must retain its independent conservative allowance.
    first.next_decode_eligible_step = scheduler.current_step + 2
    before_tokens = [
        (r.num_computed_tokens, r.num_in_flight_tokens) for r in (first, second)
    ]
    pool = scheduler.kv_cache_manager.block_pool
    before_free = pool.get_num_free_blocks()
    global_bound, state_bound = scheduler._running_cache_allocation_needs()
    assert [
        (r.num_computed_tokens, r.num_in_flight_tokens) for r in (first, second)
    ] == before_tokens
    assert pool.get_num_free_blocks() == before_free
    allocations = []
    allocate = pool.get_new_blocks

    def record_allocation(count):
        allocations.append(count)
        return allocate(count)

    monkeypatch.setattr(pool, "get_new_blocks", record_allocation)
    before_generation = scheduler._state_quota._generation
    output = scheduler.schedule()
    assert first.request_id not in output.num_scheduled_tokens
    assert output.num_scheduled_tokens[second.request_id] > 0
    assert sum(allocations) <= global_bound
    assert scheduler._state_quota._generation - before_generation <= state_bound


def test_no_outstanding_fence_does_not_wait_on_an_exhausted_pool(monkeypatch):
    scheduler = _scheduler(monkeypatch, groups=1, sequences=1)
    request = _request("ordinary")
    scheduler.add_request(request)
    core = _core(scheduler, 1)
    _step(core)
    assert scheduler.processed_step_seq == scheduler.sched_step_seq
    pool = scheduler.kv_cache_manager.block_pool
    held = pool.get_new_blocks(pool.get_num_free_blocks())
    try:
        assert not scheduler._should_wait_for_cache_fence()
        output = scheduler.schedule()
        assert request.request_id in output.preempted_req_ids
        assert scheduler._state_quota.stats()["pressure_wait_steps"] == 0
    finally:
        pool.free_blocks(held)


def test_exactly_enough_free_blocks_does_not_wait(monkeypatch):
    scheduler = _scheduler(monkeypatch, groups=1, sequences=1)
    request = _request("boundary")
    scheduler.add_request(request)
    scheduler.schedule()
    needed, _ = scheduler._running_cache_allocation_needs()
    assert needed > 0
    pool = scheduler.kv_cache_manager.block_pool
    held = pool.get_new_blocks(pool.get_num_free_blocks() - needed)
    try:
        assert not scheduler._should_wait_for_cache_fence()
        output = scheduler.schedule()
        assert output.total_num_scheduled_tokens > 0
        assert not output.preempted_req_ids
    finally:
        pool.free_blocks(held)


def test_completed_old_state_is_reclaimed_before_waiting_for_global_space(monkeypatch):
    scheduler = _scheduler(monkeypatch, depth=3, groups=1, sequences=1)
    request = _request("completed-source")
    scheduler.add_request(request)
    pending = [scheduler.schedule() for _ in range(3)]
    for output in pending[:2]:
        scheduler.update_from_output(
            output,
            ModelRunnerOutput(
                req_ids=[request.request_id],
                req_id_to_index={request.request_id: 0},
                sampled_token_ids=[[]],
            ),
        )
    manager = scheduler._state_quota.coordinator.single_type_managers[1]
    old, source, in_flight = manager.req_to_blocks[request.request_id]
    assert old.ref_cnt == source.ref_cnt == in_flight.ref_cnt == 1
    assert request.num_computed_tokens - request.num_in_flight_tokens == 2 * BLOCK
    pool = scheduler.kv_cache_manager.block_pool
    needed, _ = scheduler._running_cache_allocation_needs()
    held = pool.get_new_blocks(pool.get_num_free_blocks() - needed + 1)
    try:
        assert not scheduler._should_wait_for_cache_fence()
        assert old.ref_cnt == 0
        assert manager.req_to_blocks[request.request_id][0].is_null
        assert source.ref_cnt == in_flight.ref_cnt == 1
        assert pool.get_num_free_blocks() >= needed
    finally:
        pool.free_blocks(held)


def test_wait_restores_token_limit_even_if_upstream_schedule_raises(monkeypatch):
    scheduler = _scheduler(monkeypatch, groups=1, sequences=1)
    scheduler.add_request(_request("waiting"))
    scheduler.schedule()
    maximum = scheduler.max_num_scheduled_tokens
    pool = scheduler.kv_cache_manager.block_pool
    held = pool.get_new_blocks(pool.get_num_free_blocks())

    def fail(self, throttle_prefills=False):
        assert self.max_num_scheduled_tokens == 0
        assert throttle_prefills is True
        raise RuntimeError("synthetic upstream failure")

    monkeypatch.setattr(budget_module.AsyncScheduler, "schedule", fail)
    try:
        with pytest.raises(RuntimeError, match="synthetic upstream failure"):
            scheduler.schedule(True)
        assert scheduler.max_num_scheduled_tokens == maximum
    finally:
        pool.free_blocks(held)


def test_idle_checkpoint_rows_are_evictable_not_pinned_quota(monkeypatch):
    scheduler = _scheduler(monkeypatch, groups=1, sequences=1, capacity=3)
    old = _request("old", prompt=BLOCK + 1, output=1)
    scheduler.add_request(old)
    _drain(_core(scheduler, 2))
    scheduler.add_request(_request("old-2", prompt=BLOCK + 1, output=1, offset=500))
    _drain(_core(scheduler, 2))
    quota = scheduler._state_quota
    assert len(quota.resident_blocks) == 2
    assert all(
        quota.block_pool.blocks[bid].ref_cnt == 0 for bid, _ in quota.resident_blocks
    )
    current = _request("current", offset=1000)
    scheduler.add_request(current)
    scheduler.schedule()
    assert scheduler.processed_step_seq < scheduler.sched_step_seq
    assert not scheduler._should_wait_for_cache_fence()
    assert scheduler.schedule().total_num_scheduled_tokens > 0
    assert quota.evicted_checkpoints > 0


@pytest.mark.parametrize("subclass", [False, True])
def test_unsupported_manager_is_rejected_before_adapter_or_pool_mutation(
    monkeypatch, subclass
):
    scheduler = _scheduler(monkeypatch, groups=1)
    delegate = scheduler._state_quota._delegate

    class UnknownAttention(FullAttentionManager):
        pass

    unknown = object.__new__(UnknownAttention) if subclass else object()
    managers = delegate.coordinator.single_type_managers
    monkeypatch.setattr(
        delegate.coordinator, "single_type_managers", (unknown, *managers[1:])
    )
    before = [
        (block.block_id, block.ref_cnt, block.block_hash)
        for block in delegate.block_pool.blocks
    ]
    adapter = object.__new__(budget_module.StateQuotaKVCacheManager)
    with pytest.raises(ValueError, match="other cache manager types are unsupported"):
        adapter.__init__(delegate, 3)
    assert "_delegate" not in vars(adapter)
    assert [
        (block.block_id, block.ref_cnt, block.block_hash)
        for block in delegate.block_pool.blocks
    ] == before
