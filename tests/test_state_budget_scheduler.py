# SPDX-License-Identifier: Apache-2.0
"""Exercise quota admission against vLLM's real block/prefix managers, no model."""

from __future__ import annotations

import hashlib
import inspect
import pickle
from collections import deque
from dataclasses import fields
from types import SimpleNamespace

import pytest
import torch
from vllm import SamplingParams
from vllm.v1.core.kv_cache_manager import KVCacheManager
from vllm.v1.core.kv_cache_utils import get_request_block_hasher, init_none_hash
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import FCFSRequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)
from vllm.v1.request import Request, RequestStatus, StreamingUpdate

from vllm_metal.state_budget import StateCacheBudget, StateCacheStep
from vllm_metal.v1.state_budget_scheduler import (
    StateBudgetAsyncScheduler,
    StateBudgetScheduler,
    StateBudgetSchedulerOutput,
    StateQuotaKVCacheManager,
)

BLOCK = 16


def _hash(value):
    return hashlib.sha256(pickle.dumps(value)).digest()


@pytest.fixture(autouse=True)
def _initialize_hash():
    init_none_hash(_hash)


def _config(
    groups=1,
    blocks=128,
    retention=None,
    *,
    partial=False,
    checkpoints=0,
    block_size=BLOCK,
    max_model_len=1024,
    max_in_flight_tokens=None,
):
    state = MambaSpec(
        block_size=block_size,
        shapes=((4, 4),),
        dtypes=(torch.float32,),
        mamba_cache_mode="align",
        num_prefill_checkpoint_blocks=checkpoints,
    )
    config = KVCacheConfig(
        num_blocks=blocks,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(
                ["attention"],
                FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=1,
                    head_size=2,
                    dtype=torch.float32,
                ),
            ),
            *(KVCacheGroupSpec([f"state_{i}"], state) for i in range(groups)),
        ],
        prefix_cache_retention_interval=retention,
    )
    return KVCacheManager(
        config,
        max_model_len=max_model_len,
        scheduler_block_size=block_size,
        hash_block_size=block_size // 2 if partial else block_size,
        max_in_flight_tokens=2 * block_size
        if max_in_flight_tokens is None
        else max_in_flight_tokens,
    )


def _request(name, tokens=128, *, offset=0, block_size=BLOCK):
    return Request(
        request_id=name,
        prompt_token_ids=list(range(offset, offset + tokens)),
        sampling_params=SamplingParams(temperature=0, max_tokens=128),
        pooling_params=None,
        block_hasher=get_request_block_hasher(block_size, _hash),
    )


def _advance(manager, request, count=BLOCK):
    manager.new_step_starts()
    blocks = manager.allocate_slots(request, count)
    assert blocks is not None
    request.status = RequestStatus.RUNNING
    request.num_computed_tokens += count
    return blocks


def _state_ids(manager, request):
    return {
        block.block_id
        for group in manager.coordinator.single_type_managers[1:]
        for block in group.req_to_blocks[request.request_id]
        if not block.is_null
    }


def test_long_request_recycles_only_old_checkpoints_across_multiple_groups():
    quota = StateQuotaKVCacheManager(_config(groups=2), capacity=4)
    request = _request("long", tokens=256)
    retired = set()
    for _ in range(12):
        before = dict(quota.resident_blocks)
        _advance(quota, request)
        resident = dict(quota.resident_blocks)
        assert len(resident) <= 4
        assert _state_ids(quota, request) <= resident.keys()
        retired.update(before.keys() - resident.keys())
    assert request.num_computed_tokens == 192
    assert quota.evicted_checkpoints > 0
    assert retired
    # Their former prefix entries cannot return an unbacked state.
    for block_id in retired - dict(quota.resident_blocks).keys():
        assert quota.block_pool.blocks[block_id].block_hash is None
        assert block_id not in quota.block_pool.cached_block_hashes_by_block


def test_prefill_skips_null_blocks_in_state_quota():
    quota = StateQuotaKVCacheManager(_config(groups=2), capacity=4)
    request = _request("chunked", tokens=256)
    _advance(quota, request, 8 * BLOCK)
    assert len(quota.resident_blocks) == 2
    assert len(quota.coordinator.single_type_managers[1].req_to_blocks["chunked"]) == 8
    _advance(quota, request, 4 * BLOCK)
    assert len(quota.resident_blocks) == 4


@pytest.mark.parametrize("groups,concurrent,requests", [(1, 1, 2), (3, 2, 2)])
def test_working_bound_allows_every_admitted_request_to_advance(
    groups, concurrent, requests
):
    capacity = (concurrent + 1) * groups * requests
    quota = StateQuotaKVCacheManager(
        _config(groups=groups, blocks=512), capacity=capacity
    )
    running = [
        _request(f"request-{i}", tokens=512, offset=1000 * i) for i in range(requests)
    ]
    for step in range(10):
        # Match the executor's bounded queue: before scheduling a new batch,
        # drain one old batch once all concurrent slots have been occupied.
        if step >= concurrent:
            for request in running:
                request.num_in_flight_tokens -= BLOCK
        quota.new_step_starts()
        for request in running:
            assert quota.allocate_slots(request, BLOCK) is not None
            request.status = RequestStatus.RUNNING
            request.num_computed_tokens += BLOCK
            request.num_in_flight_tokens += BLOCK
        assert len(quota.resident_blocks) <= capacity
        assert quota.quota_stalls == 0
    assert all(request.num_computed_tokens == 10 * BLOCK for request in running)


@pytest.mark.parametrize("capacity", [9, 10])
def test_sparse_long_prefill_fits_working_reserve_with_two_inflight_batches(capacity):
    block_size, chunk, prompt, decode = 784, 1568, 156801, 32
    quota = StateQuotaKVCacheManager(
        _config(
            groups=3,
            blocks=218,
            retention=0,
            block_size=block_size,
            max_model_len=163840,
            max_in_flight_tokens=2 * chunk,
        ),
        capacity=capacity,
    )
    quota.max_resident_requests = 1
    request = _request("long-sparse", tokens=prompt, block_size=block_size)
    pending = deque()
    counts = [chunk] * 100 + [1] + [1] * (decode - 1)
    max_owned = max_resident = 0

    for count in counts:
        if len(pending) == 2:
            request.num_in_flight_tokens -= pending.popleft()
        completed = request.num_computed_tokens - request.num_in_flight_tokens
        completed_source = max(0, (completed - 1) // block_size)
        protected = {
            block.block_id
            for manager in quota._state_managers.values()
            for index, block in enumerate(manager.req_to_blocks[request.request_id])
            if not block.is_null and index >= completed_source
        }
        quota.new_step_starts()
        allocated = quota.allocate_slots(request, count)
        assert allocated is not None
        owned = _state_ids(quota, request)
        assert protected <= owned  # Completed source and in-flight rows survive.
        max_owned = max(max_owned, len(owned))
        max_resident = max(max_resident, len(quota.resident_blocks))
        assert len(quota.resident_blocks) <= 9
        request.status = RequestStatus.RUNNING
        request.num_computed_tokens += count
        request.num_in_flight_tokens += count
        pending.append(count)
        if request.num_computed_tokens >= prompt:
            request.append_output_token_ids(100)

    while pending:
        request.num_in_flight_tokens -= pending.popleft()
    assert request.num_computed_tokens == prompt + decode - 1
    assert request.num_preemptions == 0
    assert quota.quota_stalls == quota.admission_stalls == 0
    assert max_owned == max_resident == 9


def test_sparse_completed_rows_release_before_a_failed_kv_allocation():
    quota = StateQuotaKVCacheManager(
        _config(groups=3, blocks=20, retention=0, max_in_flight_tokens=4 * BLOCK),
        capacity=9,
    )
    request = _request("sparse", tokens=10 * BLOCK)
    chunk = 2 * BLOCK
    pending = deque()
    for _ in range(3):
        if len(pending) == 2:
            request.num_in_flight_tokens -= pending.popleft()
        _advance(quota, request, chunk)
        request.num_in_flight_tokens += chunk
        pending.append(chunk)
    expired = {
        manager.req_to_blocks[request.request_id][1].block_id
        for manager in quota._state_managers.values()
    }
    protected = _state_ids(quota, request) - expired
    held = quota.block_pool.get_new_blocks(quota.block_pool.get_num_free_blocks())
    request.num_in_flight_tokens -= pending.popleft()
    generations = quota._generation
    quota.new_step_starts()

    # No KV room even after safely retiring the first three state rows.
    # Failed allocation must still release expired rows without touching the
    # completed source at position 3 or in-flight destination at position 5.
    assert quota.allocate_slots(request, chunk) is None
    assert quota._generation == generations
    assert _state_ids(quota, request) == protected
    assert set(dict(quota.resident_blocks)) == protected
    assert all(quota.block_pool.blocks[block].ref_cnt == 0 for block in expired)
    assert all(quota.block_pool.blocks[block].ref_cnt == 1 for block in protected)
    for manager in quota._state_managers.values():
        assert manager.req_to_blocks[request.request_id][1].is_null

    quota.block_pool.free_blocks(held)
    assert quota.allocate_slots(request, chunk) is not None
    assert protected <= _state_ids(quota, request)
    assert len(quota.resident_blocks) == 9


def test_writing_same_block_preserves_its_generation():
    quota = StateQuotaKVCacheManager(_config(), capacity=2)
    request = _request("same-block")
    _advance(quota, request, 1)
    before = quota.resident_blocks
    _advance(quota, request, 1)
    assert quota.resident_blocks == before


def test_lookup_sources_are_protected_before_upstream_touch():
    quota = StateQuotaKVCacheManager(_config(), capacity=2)
    old = _request("producer", tokens=2 * BLOCK)
    _advance(quota, old)
    quota.free(old)
    source_id = next(iter(dict(quota.resident_blocks)))
    assert quota.block_pool.blocks[source_id].ref_cnt == 0
    consumer = _request("consumer", tokens=2 * BLOCK)
    quota.new_step_starts()
    hits, count, _ = quota.get_computed_blocks(consumer)
    assert count == BLOCK
    result = quota.allocate_slots(
        consumer, BLOCK, num_new_computed_tokens=count, new_computed_blocks=hits
    )
    assert result is not None
    assert source_id in dict(quota.resident_blocks)
    assert quota.block_pool.blocks[source_id].ref_cnt == 1
    assert len(quota.resident_blocks) == 2


def test_full_quota_cannot_evict_the_untouched_hit_source():
    quota = StateQuotaKVCacheManager(_config(), capacity=1)
    producer = _request("producer", tokens=2 * BLOCK)
    _advance(quota, producer)
    quota.free(producer)
    source_id = next(iter(dict(quota.resident_blocks)))
    consumer = _request("consumer", tokens=2 * BLOCK)
    quota.new_step_starts()
    hits, count, _ = quota.get_computed_blocks(consumer)
    assert count == BLOCK
    assert (
        quota.allocate_slots(
            consumer, BLOCK, num_new_computed_tokens=count, new_computed_blocks=hits
        )
        is None
    )
    assert quota.block_pool.blocks[source_id].block_hash is not None
    assert quota.block_pool.blocks[source_id].ref_cnt == 0
    assert quota.quota_stalls == 1


def test_eviction_removes_all_prefix_aliases_without_freeing_live_rows():
    quota = StateQuotaKVCacheManager(_config(), capacity=2)
    old = _request("old", tokens=2 * BLOCK)
    _advance(quota, old)
    quota.free(old)
    cold = next(iter(dict(quota.resident_blocks)))
    # A block can own several hash aliases even though it occupies one row.
    original_hash = quota.block_pool.blocks[cold].block_hash
    assert original_hash is not None
    alias = type(original_hash)(b"extra-prefix-alias")
    quota.block_pool._insert_block_hash(
        alias, quota.block_pool.blocks[cold], num_tokens=BLOCK
    )
    live = _request("live", offset=1000)
    _advance(quota, live)
    live_ids = _state_ids(quota, live)
    newcomer = _request("new", offset=2000)
    _advance(quota, newcomer)
    assert (
        quota.block_pool.cached_block_hash_to_block.get_one_block(original_hash) is None
    )
    assert quota.block_pool.cached_block_hash_to_block.get_one_block(alias) is None
    assert cold not in quota.block_pool.cached_block_hashes_by_block
    assert live_ids <= dict(quota.resident_blocks).keys()


def test_failed_kv_allocation_does_not_reserve_new_state_rows():
    quota = StateQuotaKVCacheManager(_config(blocks=6), capacity=2)
    held = quota.block_pool.get_new_blocks(4)
    request = _request("blocked")
    assert quota.allocate_slots(request, BLOCK) is None
    assert quota.resident_blocks == ()
    quota.block_pool.free_blocks(held)
    assert quota.allocate_slots(request, BLOCK) is not None
    assert len(quota.resident_blocks) == 1


def test_same_step_new_checkpoint_stays_ineligible():
    quota = StateQuotaKVCacheManager(_config(), capacity=3)
    producer = _request("producer", tokens=2 * BLOCK)
    _advance(quota, producer)
    consumer = _request("consumer", tokens=2 * BLOCK)
    hits, count, _ = quota.get_computed_blocks(consumer)
    assert count == BLOCK
    before = quota.resident_blocks
    assert (
        quota.allocate_slots(
            consumer, BLOCK, num_new_computed_tokens=count, new_computed_blocks=hits
        )
        is None
    )
    assert quota.resident_blocks == before
    quota.new_step_starts()
    assert (
        quota.allocate_slots(
            consumer, BLOCK, num_new_computed_tokens=count, new_computed_blocks=hits
        )
        is not None
    )


def test_all_pinned_waits_for_real_scheduler_fence_then_makes_progress():
    quota = StateQuotaKVCacheManager(_config(), capacity=1)
    first = _request("first")
    _advance(quota, first)
    first.last_sched_seq = 1
    source = next(iter(dict(quota.resident_blocks)))

    # Exercise vLLM's real delayed-free protocol used by the budget scheduler.
    scheduler = object.__new__(Scheduler)
    scheduler.kv_cache_manager = quota
    scheduler.defer_block_free = True
    scheduler.processed_step_seq = 0
    scheduler.sched_step_seq = 1
    scheduler.deferred_frees = deque()
    scheduler._free_request_blocks(first)
    assert quota.block_pool.blocks[source].ref_cnt > 0
    waiting = _request("waiting", offset=1000)
    quota.new_step_starts()
    assert quota.allocate_slots(waiting, BLOCK) is None
    scheduler._drain_deferred_frees()
    assert quota.block_pool.blocks[source].ref_cnt > 0

    scheduler.processed_step_seq = 1
    scheduler._drain_deferred_frees()
    assert quota.block_pool.blocks[source].ref_cnt == 0
    assert quota.allocate_slots(waiting, BLOCK) is not None
    assert len(quota.resident_blocks) == 1
    assert quota.evicted_checkpoints == 1


def _stream_scheduler(quota, *requests):
    """Real core lifecycle hooks, with only model/encoder work omitted."""
    scheduler = object.__new__(StateBudgetScheduler)
    scheduler._state_quota = quota
    scheduler.kv_cache_manager = quota
    scheduler.policy = SchedulingPolicy.FCFS
    scheduler.log_stats = False
    scheduler.running = []
    scheduler.num_waiting_for_streaming_input = 0
    scheduler.waiting = FCFSRequestQueue()
    scheduler.skipped_waiting = FCFSRequestQueue()
    scheduler.requests = {request.request_id: request for request in requests}
    scheduler.max_num_running_reqs = 1
    quota.max_resident_requests = 1
    scheduler.defer_block_free = True
    scheduler.sched_step_seq = scheduler.processed_step_seq = 2
    scheduler.deferred_frees = deque()
    scheduler.connector = scheduler.ec_connector = None
    scheduler.encoder_cache_manager = SimpleNamespace(free=lambda _: None)
    scheduler._inflight_prefills = set()
    scheduler.finished_req_ids = set()
    scheduler.finished_req_ids_dict = None
    return scheduler


def test_queued_stream_continuation_resumes_before_new_admission():
    """The next queued input must not lose its existing working reservation."""
    quota = StateQuotaKVCacheManager(_config(), capacity=3)
    continuation = _request("stream", tokens=2 * BLOCK)
    _advance(quota, continuation)
    _advance(quota, continuation)
    continuation.resumable = True
    continuation.streaming_queue = deque(
        [
            StreamingUpdate(
                mm_features=None,
                prompt_token_ids=list(range(2 * BLOCK, 3 * BLOCK)),
                max_tokens=128,
                arrival_time=10,
                sampling_params=continuation.sampling_params,
            )
        ]
    )
    newcomer = _request("newcomer", offset=1000)
    scheduler = _stream_scheduler(quota, continuation, newcomer)
    scheduler.waiting.add_request(newcomer)
    assert not scheduler._handle_stopped_request(continuation)
    assert continuation.status == RequestStatus.WAITING
    assert scheduler.num_waiting_for_streaming_input == 0
    assert [r.request_id for r in scheduler.waiting] == ["newcomer", "stream"]
    assert len(_state_ids(quota, continuation)) == 2

    # A row would fit now, but the newcomer would steal the continuation's
    # working reservation and block both requests on the next copy-forward.
    assert quota.allocate_slots(newcomer, BLOCK) is None
    assert quota.admission_stalls == 1
    queue = scheduler._select_waiting_queue_for_scheduling()
    assert queue is scheduler.waiting
    assert queue.pop_request() is continuation
    _advance(quota, continuation)
    assert continuation.num_computed_tokens == 3 * BLOCK
    assert len(quota.resident_blocks) <= quota.capacity
    scheduler.running.append(continuation)
    scheduler.finish_requests("stream", RequestStatus.FINISHED_ABORTED)
    assert not quota.request_has_state("stream")
    assert scheduler._select_waiting_queue_for_scheduling().peek_request() is newcomer
    _advance(quota, newcomer)
    assert newcomer.num_computed_tokens == BLOCK


def test_later_stream_update_recovers_owner_from_skipped_waiting():
    quota = StateQuotaKVCacheManager(_config(), capacity=3)
    continuation = _request("stream", tokens=2 * BLOCK)
    _advance(quota, continuation)
    _advance(quota, continuation)
    continuation.resumable = True
    continuation.streaming_queue = deque()
    newcomer = _request("newcomer", offset=1000)
    scheduler = _stream_scheduler(quota, continuation, newcomer)
    scheduler.waiting.add_request(newcomer)
    assert not scheduler._handle_stopped_request(continuation)
    assert continuation.status == RequestStatus.WAITING_FOR_STREAMING_REQ
    assert scheduler.num_waiting_for_streaming_input == 1
    assert quota.allocate_slots(newcomer, BLOCK) is None
    before = quota.resident_blocks

    incoming = _request("stream", tokens=BLOCK, offset=2 * BLOCK)
    incoming.resumable = True
    scheduler.add_request(incoming)
    assert continuation.status == RequestStatus.WAITING
    assert scheduler.num_waiting_for_streaming_input == 0
    assert quota.resident_blocks == before
    queue = scheduler._select_waiting_queue_for_scheduling()
    assert queue is scheduler.skipped_waiting
    assert queue.pop_request() is continuation
    _advance(quota, continuation)
    assert continuation.num_computed_tokens == 3 * BLOCK
    assert quota.resident_request_ids() == {"stream"}


def test_owner_reserve_survives_temporary_removal_from_waiting_queues():
    quota = StateQuotaKVCacheManager(_config(), capacity=3)
    continuation = _request("stream")
    _advance(quota, continuation)
    continuation.resumable = True
    continuation.status = RequestStatus.WAITING
    newcomer = _request("newcomer", offset=1000)
    scheduler = _stream_scheduler(quota, continuation, newcomer)
    # Core may temporarily keep skipped work in a local step_skipped_waiting
    # queue.  The request's actual block ownership still reserves its place.
    scheduler.waiting.add_request(newcomer)
    assert scheduler._select_waiting_queue_for_scheduling().peek_request() is newcomer
    assert quota.allocate_slots(newcomer, BLOCK) is None
    assert quota.admission_stalls == 1
    assert quota.quota_stalls == 0
    scheduler.skipped_waiting.add_request(continuation)
    queue = scheduler._select_waiting_queue_for_scheduling()
    assert queue.pop_request() is continuation
    _advance(quota, continuation)


def test_aborted_stream_releases_admission_but_keeps_inflight_rows_fenced():
    quota = StateQuotaKVCacheManager(_config(), capacity=3)
    request = _request("stream")
    _advance(quota, request)
    request.num_in_flight_tokens = BLOCK
    _advance(quota, request)
    # One earlier result has completed before a third step is scheduled.
    _advance(quota, request)
    request.num_in_flight_tokens = 2 * BLOCK
    assert len(quota.resident_blocks) == 3
    request.resumable = True
    request.streaming_queue = deque()
    request.last_sched_seq = 3
    newcomer = _request("newcomer", offset=1000)
    scheduler = _stream_scheduler(quota, request, newcomer)
    scheduler.sched_step_seq = 3
    scheduler.processed_step_seq = 1
    scheduler.running.append(request)
    scheduler.finish_requests("stream", RequestStatus.FINISHED_ABORTED)
    assert not quota.request_has_state("stream")
    assert not quota.resident_request_ids()
    assert scheduler.deferred_frees

    # No admission owner remains, but the outstanding GPU rows are still
    # charged.  The ordinary result/fence path can drain without admission.
    assert quota.allocate_slots(newcomer, BLOCK) is None
    assert quota.admission_stalls == 0
    assert quota.quota_stalls == 1
    scheduler._drain_deferred_frees()
    assert scheduler.deferred_frees
    scheduler.processed_step_seq = 3
    scheduler._drain_deferred_frees()
    assert not scheduler.deferred_frees
    _advance(quota, newcomer)
    _advance(quota, newcomer)
    assert newcomer.num_computed_tokens == 2 * BLOCK
    assert quota.resident_request_ids() == {"newcomer"}


def test_aborting_paused_stream_releases_counter_and_owner():
    quota = StateQuotaKVCacheManager(_config(), capacity=3)
    request = _request("stream")
    _advance(quota, request)
    request.resumable = True
    request.streaming_queue = deque()
    scheduler = _stream_scheduler(quota, request)
    assert not scheduler._handle_stopped_request(request)
    assert scheduler.num_waiting_for_streaming_input == 1
    scheduler.finish_requests("stream", RequestStatus.FINISHED_ABORTED)
    assert scheduler.num_waiting_for_streaming_input == 0
    assert not scheduler.skipped_waiting
    assert not quota.resident_request_ids()
    _advance(quota, _request("newcomer", offset=1000))


def test_fresh_stream_does_not_jump_ordinary_fcfs_admission():
    quota = StateQuotaKVCacheManager(_config(), capacity=3)
    first = _request("first")
    fresh_stream = _request("new-stream", offset=1000)
    fresh_stream.resumable = True
    scheduler = _stream_scheduler(quota, first, fresh_stream)
    scheduler.waiting.add_request(first)
    scheduler.waiting.add_request(fresh_stream)
    assert scheduler._select_waiting_queue_for_scheduling().peek_request() is first


def test_uncached_free_rows_retire_without_spending_checkpoint_quota():
    quota = StateQuotaKVCacheManager(_config(), capacity=2)
    request = _request("partial-prompt", tokens=BLOCK - 1)
    _advance(quota, request, BLOCK - 1)
    quota.free(request)
    quota.retire_uncached()
    assert quota.resident_blocks == ()
    assert quota.retired_uncached == 1
    assert quota.evicted_checkpoints == 0


def test_reallocated_state_block_gets_a_new_generation():
    quota = StateQuotaKVCacheManager(_config(blocks=3), capacity=1)
    first = _request("first", tokens=BLOCK - 1)
    _advance(quota, first, BLOCK - 1)
    before = dict(quota.resident_blocks)
    quota.free(first)
    quota.retire_uncached()
    # The next request reuses the old state ID for attention.  A third request
    # then switches it back to state, which must not resurrect its first lease.
    second = _request("second", tokens=BLOCK - 1, offset=1000)
    second_blocks = _advance(quota, second, BLOCK - 1)
    assert second_blocks.blocks[0][0].block_id in before
    assert before.keys().isdisjoint(dict(quota.resident_blocks))
    quota.free(second)
    quota.retire_uncached()
    third = _request("third", tokens=BLOCK - 1, offset=2000)
    _advance(quota, third, BLOCK - 1)
    after = dict(quota.resident_blocks)
    assert before.keys() == after.keys()
    assert next(iter(after.values())) > next(iter(before.values()))
    assert quota.reused_block_ids == 1


@pytest.mark.parametrize("partial,checkpoints", [(True, 0), (False, 1)])
def test_unaccounted_allocation_shapes_fail_before_use(partial, checkpoints):
    with pytest.raises(ValueError, match="full-block prefix hits"):
        StateQuotaKVCacheManager(
            _config(partial=partial, checkpoints=checkpoints), capacity=4
        )


def test_external_allocation_is_rejected_before_mutation():
    quota = StateQuotaKVCacheManager(_config(), capacity=4)
    free = quota.block_pool.get_num_free_blocks()
    with pytest.raises(ValueError, match="external cache loads"):
        quota.allocate_slots(
            _request("external"), BLOCK, num_external_computed_tokens=BLOCK
        )
    assert quota.block_pool.get_num_free_blocks() == free
    assert not quota.resident_blocks


def test_payload_is_a_serializable_dataclass_and_zero_token_steps_advance():
    quota = StateQuotaKVCacheManager(_config(), capacity=4)
    _advance(quota, _request("first"))
    base = SchedulerOutput.make_empty()
    first = quota.snapshot()
    second = quota.snapshot()
    assert second.sequence == first.sequence + 1
    assert second.resident_blocks == first.resident_blocks
    output = StateBudgetSchedulerOutput(
        **{field.name: getattr(base, field.name) for field in fields(base)},
        metal_state_cache=second,
        state_cache_budget_stats=quota.stats(),
    )
    restored = pickle.loads(pickle.dumps(output))
    assert isinstance(restored, SchedulerOutput)
    assert restored.metal_state_cache == second
    assert restored.state_cache_budget_stats["resident_rows"] == 1


def _scheduler_init(
    monkeypatch, capacity, groups, concurrent, policy=SchedulingPolicy.FCFS
):
    delegate = _config(groups=groups)

    def initialize(self):
        self.vllm_config = SimpleNamespace(
            speculative_config=None, max_concurrent_batches=concurrent
        )
        self.kv_cache_config = delegate.kv_cache_config
        self.kv_cache_manager = delegate
        self.parallel_config = SimpleNamespace(
            pipeline_parallel_size=1, data_parallel_size=1
        )
        self.connector = self.ec_connector = None
        self.max_num_running_reqs = 100
        self.max_num_scheduled_tokens = 128
        self.sched_step_seq = 0
        self.processed_step_seq = 0
        self.num_spec_tokens = 0
        self.policy = policy

    monkeypatch.setattr(Scheduler, "__init__", initialize)
    monkeypatch.setattr(
        "vllm_metal.v1.state_budget_scheduler.state_cache_budget_from_kv_config",
        lambda *args: StateCacheBudget(capacity * 64, 64, capacity, capacity * 64),
    )


@pytest.mark.parametrize("groups,concurrent", [(1, 1), (2, 1), (3, 2)])
def test_working_reserve_bounds_concurrency_and_preserves_fences(
    monkeypatch, groups, concurrent
):
    bound = (concurrent + 1) * groups
    _scheduler_init(
        monkeypatch,
        capacity=bound * 2 + bound - 1,
        groups=groups,
        concurrent=concurrent,
    )
    scheduler = StateBudgetScheduler()
    assert scheduler.max_num_running_reqs == 2
    assert scheduler.defer_block_free
    assert scheduler.kv_cache_manager is scheduler._state_quota


def test_budget_below_one_request_progress_bound_is_rejected(monkeypatch):
    _scheduler_init(monkeypatch, capacity=5, groups=2, concurrent=2)
    with pytest.raises(ValueError, match="at least 6 rows"):
        StateBudgetScheduler()


@pytest.mark.parametrize(
    "scheduler_cls", [StateBudgetScheduler, StateBudgetAsyncScheduler]
)
def test_priority_scheduling_is_rejected(monkeypatch, scheduler_cls):
    _scheduler_init(
        monkeypatch,
        capacity=8,
        groups=1,
        concurrent=2,
        policy=SchedulingPolicy.PRIORITY,
    )
    with pytest.raises(ValueError, match="requires FCFS scheduling"):
        scheduler_cls()


def test_async_scheduler_keeps_upstream_async_hooks():
    assert issubclass(StateBudgetAsyncScheduler, AsyncScheduler)
    assert (
        StateBudgetAsyncScheduler._update_after_schedule
        is AsyncScheduler._update_after_schedule
    )
    assert (
        StateBudgetAsyncScheduler._update_request_with_output
        is AsyncScheduler._update_request_with_output
    )


def test_async_scheduler_constructor_runs_upstream_async_initializer(monkeypatch):
    _scheduler_init(monkeypatch, capacity=6, groups=1, concurrent=2)
    scheduler = StateBudgetAsyncScheduler()
    assert scheduler.max_num_running_reqs == 2
    assert scheduler.pp_size == 1
    assert scheduler._spec_token_placeholders == []


def test_wrapped_upstream_signatures_stay_compatible():
    for upstream, wrapper in [
        (Scheduler.schedule, StateBudgetScheduler.schedule),
        (KVCacheManager.allocate_slots, StateQuotaKVCacheManager.allocate_slots),
    ]:
        expected = inspect.signature(upstream).parameters
        actual = inspect.signature(wrapper).parameters
        assert actual.keys() == expected.keys()
        for name in expected:
            assert actual[name].kind == expected[name].kind
            assert actual[name].default == expected[name].default


def test_schedule_attaches_authoritative_snapshot_and_stats(monkeypatch):
    _scheduler_init(monkeypatch, capacity=4, groups=1, concurrent=1)
    throttles = []

    def schedule(_, throttle_prefills=False):
        throttles.append(throttle_prefills)
        return SchedulerOutput.make_empty()

    monkeypatch.setattr(Scheduler, "schedule", schedule)
    scheduler = StateBudgetScheduler()
    first = scheduler.schedule()
    # EngineCore supplies this argument positionally for both regular and
    # batch-queue scheduling; preserve it through the plugin override.
    second = scheduler.schedule(True)
    assert first.metal_state_cache == StateCacheStep(1, ())
    assert second.metal_state_cache == StateCacheStep(2, ())
    assert second.state_cache_budget_stats["capacity_rows"] == 4
    assert throttles == [False, True]


def test_unknown_cow_output_is_never_dispatched(monkeypatch):
    _scheduler_init(monkeypatch, capacity=4, groups=1, concurrent=1)
    output = SchedulerOutput.make_empty()
    output.kv_cache_block_copies = [(1, 2)]
    monkeypatch.setattr(
        Scheduler, "schedule", lambda _, throttle_prefills=False: output
    )
    with pytest.raises(RuntimeError, match="unexpected CoW output"):
        StateBudgetScheduler().schedule()
