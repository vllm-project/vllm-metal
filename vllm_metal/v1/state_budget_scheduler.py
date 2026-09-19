# SPDX-License-Identifier: Apache-2.0
"""Scheduler admission and eviction for a separately bounded state cache.

Logical blocks still come from vLLM's shared BlockPool.  This adapter accounts
for state-bearing block *incarnations*, including idle prefix checkpoints, and
admits their physical rows against a second quota.  It never changes a live
block's contents or rolls back a partially committed upstream allocation.

The supported align layout allocates at most one new state block per group and
scheduled step.  Partial-prefix CoW, internal checkpoints, speculative decoding
and external transfers require additional reservations and are rejected by the
opt-in configuration rather than silently exceeding that bound.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, cast

from vllm.logger import init_logger
from vllm.v1.core.kv_cache_manager import KVCacheBlocks, KVCacheManager
from vllm.v1.core.sched.async_scheduler import AsyncScheduler
from vllm.v1.core.sched.output import SchedulerOutput
from vllm.v1.core.sched.request_queue import RequestQueue, SchedulingPolicy
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.core.single_type_kv_cache_manager import (
    FullAttentionManager,
    MambaManager,
)
from vllm.v1.kv_cache_interface import MambaSpec
from vllm.v1.request import Request, RequestStatus

from vllm_metal.state_budget import (
    StateCacheStep,
    state_cache_budget_from_kv_config,
    state_cache_rows_per_request,
)

logger = init_logger(__name__)


@dataclass
class StateBudgetSchedulerOutput(SchedulerOutput):
    """Explicit dataclass field survives executor serialization."""

    metal_state_cache: StateCacheStep | None = None
    state_cache_budget_stats: dict[str, int] | None = None


class StateQuotaKVCacheManager:
    """Instance-local adapter around the real upstream cache manager.

    A resident entry remains charged until its hashes are invalidated and a
    later worker snapshot retires its generation.  Allocation snapshots are
    consumed in order, before that step writes any new state, so a row can be
    reused by a different block in the same step without transient overgrowth.
    """

    def __init__(self, delegate: KVCacheManager, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError("state cache capacity must be positive")
        managers = delegate.coordinator.single_type_managers
        if any(
            type(manager) not in (FullAttentionManager, MambaManager)
            for manager in managers
        ):
            raise ValueError(
                "state cache budget requires ordinary FullAttentionManager and "
                "align MambaManager groups; other cache manager types are unsupported"
            )
        self._delegate = delegate
        self.capacity = capacity
        self._resident: dict[int, int] = {}
        self._generation = 0
        self._sequence = 0
        self.max_resident_requests: int | None = None
        self._seen_state_ids: set[int] = set()
        self.reused_block_ids = 0
        self.evicted_checkpoints = 0
        self.retired_uncached = 0
        self.quota_stalls = 0
        self.admission_stalls = 0
        self.pressure_wait_steps = 0
        self.pressure_wait_global_steps = 0
        self.pressure_wait_state_steps = 0
        self._state_managers = {
            group: manager
            for group, manager in enumerate(managers)
            if type(manager) is MambaManager
        }
        if not self._state_managers:
            raise ValueError("state cache budget requires align state groups")
        for manager in self._state_managers.values():
            spec = manager.kv_cache_spec
            if (
                not isinstance(spec, MambaSpec)
                or spec.mamba_cache_mode != "align"
                or spec.num_speculative_blocks
                or spec.num_prefill_checkpoint_blocks
                or manager.block_size != delegate.block_pool.hash_block_size
            ):
                raise ValueError(
                    "state cache budget requires aligned full-block prefix hits, "
                    "without speculative or internal checkpoint blocks"
                )

    def __getattr__(self, name: str) -> Any:
        return getattr(self._delegate, name)

    @property
    def num_state_groups(self) -> int:
        return len(self._state_managers)

    @property
    def resident_blocks(self) -> tuple[tuple[int, int], ...]:
        return tuple(sorted(self._resident.items()))

    def request_has_state(self, request_id: str) -> bool:
        """An admitted request owns its reserve until its state table is freed.

        In particular, a resumable request can be WAITING between input chunks
        while still owning a source row.  Its running-list membership is not
        the ownership boundary.  The upstream deferred-free path pops these
        tables immediately; its remaining physical rows stay separately charged
        until their GPU fence completes.
        """
        return any(
            not block.is_null
            for manager in self._state_managers.values()
            for block in manager.req_to_blocks.get(request_id, ())
        )

    def resident_request_ids(self) -> set[str]:
        return {
            request_id
            for manager in self._state_managers.values()
            for request_id, blocks in manager.req_to_blocks.items()
            if any(not block.is_null for block in blocks)
        }

    def snapshot(self) -> StateCacheStep:
        # Includes zero-token steps: worker metadata has its own sequence and
        # does not reuse the scheduler's non-empty GPU-step fence counter.
        self._sequence += 1
        return StateCacheStep(self._sequence, self.resident_blocks)

    def stats(self) -> dict[str, int]:
        return {
            "capacity_rows": self.capacity,
            "resident_rows": len(self._resident),
            "allocation_generations": self._generation,
            "evicted_checkpoints": self.evicted_checkpoints,
            "retired_uncached": self.retired_uncached,
            "reused_block_ids": self.reused_block_ids,
            "quota_stalls": self.quota_stalls,
            "admission_stalls": self.admission_stalls,
            "pressure_wait_steps": self.pressure_wait_steps,
            "pressure_wait_global_steps": self.pressure_wait_global_steps,
            "pressure_wait_state_steps": self.pressure_wait_state_steps,
            "resident_requests": len(self.resident_request_ids()),
        }

    def retire_uncached(self, protected: set[int] | None = None) -> None:
        """Forget unreferenced rows that cannot produce any prefix hit."""
        protected = protected or set()
        pool = self._delegate.block_pool
        for block_id in tuple(self._resident):
            block = pool.blocks[block_id]
            if (
                block_id not in protected
                and block.ref_cnt == 0
                and block.block_hash is None
                and not pool.cached_block_hashes_by_block.get(block_id)
            ):
                del self._resident[block_id]
                self.retired_uncached += 1

    def _make_room(self, needed: int, protected: set[int]) -> bool:
        self.retire_uncached(protected)
        shortage = len(self._resident) + needed - self.capacity
        if shortage <= 0:
            return True
        pool = self._delegate.block_pool
        # Use the same cold-to-hot ordering as BlockPool.  Merely dropping the
        # hash preserves the free-list references and never frees a live block.
        victims = []
        for block in pool.free_block_queue.get_all_free_blocks():
            if (
                block.block_id in self._resident
                and block.block_id not in protected
                and block.ref_cnt == 0
                and not block.is_null
            ):
                victims.append(block.block_id)
                if len(victims) == shortage:
                    break
        if len(victims) < shortage:
            self.quota_stalls += 1
            return False
        pool.evict_blocks(set(victims))
        for block_id in victims:
            del self._resident[block_id]
        self.evicted_checkpoints += len(victims)
        return True

    def _remove_completed_state_rows(
        self, request_id: str, processed_computed_tokens: int
    ) -> None:
        """Release expired align states even when null gaps separate them.

        Upstream's shared range helper stops its reverse scan at the first
        null block. Align prefill can skip several positions per step, so that
        dense-table shortcut can miss an older state behind a gap. With async
        batches, Mamba's last-state shortcut may already refer to a newer row.
        Such missed rows remain referenced and can exhaust the state quota.

        Only positions strictly before the completed source are eligible.
        The completed source and every in-flight destination remain owned;
        freeing a row also preserves any other request's reference/hash.
        """
        if processed_computed_tokens <= 0:
            return
        pool = self._delegate.block_pool
        for manager in self._state_managers.values():
            blocks = manager.req_to_blocks.get(request_id)
            if not blocks:
                continue
            source_index = (processed_computed_tokens - 1) // manager.block_size
            freed = []
            for index in range(min(source_index, len(blocks)) - 1, -1, -1):
                block = blocks[index]
                if block.is_null:
                    continue
                freed.append(block)
                blocks[index] = pool.null_block
            if freed:
                pool.free_blocks(freed)

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,
        num_new_computed_tokens: int = 0,
        new_computed_blocks: KVCacheBlocks | None = None,
        num_lookahead_tokens: int = 0,
        num_external_computed_tokens: int = 0,
        delay_cache_blocks: bool = False,
        num_encoder_tokens: int = 0,
        full_sequence_must_fit: bool = False,
        reserved_blocks: int = 0,
        has_scheduled_reqs: bool = True,
    ) -> KVCacheBlocks | None:
        if (
            num_lookahead_tokens
            or num_external_computed_tokens
            or delay_cache_blocks
            or num_encoder_tokens
        ):
            raise ValueError(
                "state cache budget does not support lookahead or external cache loads"
            )
        delegate = self._delegate
        if (
            self.max_resident_requests is not None
            and not self.request_has_state(request.request_id)
            and len(self.resident_request_ids()) >= self.max_resident_requests
        ):
            # Protect a continuation's reserve even if upstream temporarily
            # removed it from both waiting queues while skipping blocked work.
            # No block tables or prefix references have been changed yet.
            self.admission_stalls += 1
            return None
        groups = (
            new_computed_blocks.blocks
            if new_computed_blocks is not None
            else delegate.empty_kv_cache_blocks.blocks
        )
        protected = {
            block.block_id for group in groups for block in group if not block.is_null
        }
        for group in self._state_managers:
            if any(
                not block.is_null and block.block_id not in self._resident
                for block in groups[group]
            ):
                raise RuntimeError("prefix hit refers to an untracked state block")

        local_computed = request.num_computed_tokens + num_new_computed_tokens
        total_computed = min(local_computed, delegate.max_model_len)
        tokens_to_compute = total_computed + num_new_tokens
        # Upstream permits this release even when admission later fails.  It
        # is idempotent when the delegate repeats it, and protects in-flight
        # source states by using only completed tokens.
        processed_computed = max(0, total_computed - request.num_in_flight_tokens)
        self._remove_completed_state_rows(request.request_id, processed_computed)
        delegate.coordinator.remove_skipped_blocks(
            request.request_id,
            processed_computed,
            num_prompt_tokens=request.num_prompt_tokens,
        )
        needed = 0
        for group, manager in self._state_managers.items():
            if local_computed % manager.block_size and groups[group]:
                raise ValueError(
                    "state cache budget does not support partial prefix hits"
                )
            if request.request_id in manager._partial_hit_reqs:
                raise RuntimeError("unexpected partial state copy under a state budget")
            count = manager.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=min(tokens_to_compute, delegate.max_model_len),
                new_computed_blocks=groups[group],
                total_computed_tokens=local_computed,
                num_local_computed_tokens=local_computed,
                num_tokens_main_model=tokens_to_compute,
                apply_admission_cap=True,
            )
            if count > delegate.block_pool.num_gpu_blocks:
                # Upstream's same-step checkpoint sentinel: the bytes have
                # not yet been computed, so another request cannot restore it.
                return None
            touched = sum(
                block.ref_cnt == 0 and not block.is_null for block in groups[group]
            )
            new_count = count - touched
            if new_count not in (0, 1):
                raise RuntimeError(
                    "unsupported state allocation: expected at most one new row "
                    f"per group, got {new_count}"
                )
            needed += new_count
        if not self._make_room(needed, protected):
            return None

        result = delegate.allocate_slots(
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            num_external_computed_tokens=num_external_computed_tokens,
            delay_cache_blocks=delay_cache_blocks,
            num_encoder_tokens=num_encoder_tokens,
            full_sequence_must_fit=full_sequence_must_fit,
            reserved_blocks=reserved_blocks,
            has_scheduled_reqs=has_scheduled_reqs,
        )
        if result is None:
            return None
        for manager in delegate.coordinator.single_type_managers:
            if manager._pending_cow_copies:
                raise RuntimeError("unexpected CoW allocation under a state budget")
        for group, blocks in enumerate(result.blocks):
            for block in blocks:
                if block.is_null:
                    continue
                self._resident.pop(block.block_id, None)
                if group in self._state_managers:
                    self.reused_block_ids += int(block.block_id in self._seen_state_ids)
                    self._seen_state_ids.add(block.block_id)
                    self._generation += 1
                    self._resident[block.block_id] = self._generation
        if len(self._resident) > self.capacity:
            raise RuntimeError("state allocation exceeded its preflight quota")
        return result


class StateBudgetScheduler(Scheduler):
    """Mixin implementation also used ahead of AsyncScheduler in the MRO."""

    kv_cache_manager: KVCacheManager
    max_num_running_reqs: int
    max_num_scheduled_tokens: int

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        budget = state_cache_budget_from_kv_config(
            self.vllm_config, self.kv_cache_config
        )
        if budget is None:
            raise ValueError("state budget scheduler requires state_cache_budget_mib")
        if self.policy is not SchedulingPolicy.FCFS:
            # Priority scheduling can cancel an earlier allocation in this
            # same step after its prefix hash was published but before any
            # worker computed its state.  This mode cannot lease such entries.
            raise ValueError("state cache budget currently requires FCFS scheduling")
        if (
            self.vllm_config.speculative_config is not None
            or self.connector is not None
            or self.ec_connector is not None
            or self.parallel_config.pipeline_parallel_size != 1
            or self.parallel_config.data_parallel_size != 1
        ):
            raise ValueError(
                "state budget scheduler requires local non-speculative inference"
            )
        self._state_quota = StateQuotaKVCacheManager(
            self.kv_cache_manager, budget.capacity
        )
        # With no partial/internal/speculative states, one group can retain
        # the completed source plus at most one new state per in-flight step.
        # Reserve that bound per admitted request so cold checkpoints cannot
        # prevent the next copy-forward destination from being allocated.
        rows_per_request = state_cache_rows_per_request(
            self.vllm_config, self._state_quota.num_state_groups
        )
        if budget.capacity < rows_per_request:
            raise ValueError(
                f"state cache budget needs at least {rows_per_request} rows "
                "for one request's completed source and in-flight destinations; "
                f"got {budget.capacity}"
            )
        self.max_num_running_reqs = min(
            self.max_num_running_reqs, budget.capacity // rows_per_request
        )
        self._state_quota.max_resident_requests = self.max_num_running_reqs
        # Force the existing fence protocol even without a KV connector:
        # ref_cnt must not reach zero while an asynchronous step still reads
        # a state row that the new quota can actively evict.
        self.defer_block_free = True
        self.kv_cache_manager = cast(KVCacheManager, self._state_quota)
        self._cache_fence_wait_global = False
        self._cache_fence_wait_state = False
        logger.info(
            "Bounded state cache: %d rows, %d rows/request working reserve, "
            "at most %d running requests",
            budget.capacity,
            rows_per_request,
            self.max_num_running_reqs,
        )

    def _select_waiting_queue_for_scheduling(self) -> RequestQueue | None:
        # A queued streaming-input update moves its request directly from
        # RUNNING to WAITING, preserving state but no longer contributing to
        # the upstream running/stream-wait counts.  Resume that existing owner
        # before allocating its reservation to a new request.  Later-arriving
        # updates instead leave their owner in skipped_waiting, so inspect both.
        for queue in (self.skipped_waiting, self.waiting):
            for request in queue:
                if (
                    request.resumable
                    and request.status == RequestStatus.WAITING
                    and self._state_quota.request_has_state(request.request_id)
                ):
                    if queue.peek_request() is not request:
                        queue.remove_request(request)
                        queue.prepend_request(request)
                    return queue
        return super()._select_waiting_queue_for_scheduling()

    def _running_cache_allocation_needs(self) -> tuple[int, int]:
        """Upper-bound new global blocks and state rows for running requests.

        Count the missing table suffix, rather than charging every decode for
        a new page. Each request independently gets the full token/input limit:
        omitting shared-budget subtraction stays conservative when core skips
        earlier requests. Alignment and other core restrictions only reduce it.
        Exact manager types and the no-CoW/spec/checkpoint guards are required
        for these ordinary append and one-state-row allocation rules.
        """
        token_limit = min(
            self.max_num_scheduled_tokens,
            self.scheduler_config.max_num_batched_tokens,
        )
        if token_limit <= 0:
            return 0, 0
        managers = self._state_quota.coordinator.single_type_managers
        global_needed = state_needed = 0
        for request in self.running:
            tokens = min(
                request.num_tokens_with_spec
                + request.num_output_placeholders
                - request.num_computed_tokens,
                token_limit,
                self.max_model_len
                - request.num_computed_tokens
                - self.num_sampled_tokens_per_step,
            )
            threshold = self.scheduler_config.long_prefill_token_threshold
            if threshold > 0:
                tokens = min(tokens, threshold)
            if tokens <= 0:
                continue
            if self.need_mamba_block_aligned_split:
                tokens = self._mamba_block_aligned_split(request, tokens)
            if tokens <= 0:
                continue
            total_tokens = request.num_computed_tokens + tokens
            for manager in managers:
                if request.request_id in manager._partial_hit_reqs:
                    raise RuntimeError(
                        "unexpected partial cache CoW under a state budget"
                    )
                required = (total_tokens + manager.block_size - 1) // manager.block_size
                present = len(manager.req_to_blocks.get(request.request_id, ()))
                missing = max(0, required - present)
                if type(manager) is MambaManager:
                    missing = int(missing > 0)
                    state_needed += missing
                global_needed += missing
        return global_needed, state_needed

    def _should_wait_for_cache_fence(self) -> bool:
        """Drain in-flight work before a shortage causes cascading preemption.

        Core retries allocation immediately after each preemption. Deferred
        frees cannot supply blocks within that call, so retrying can evict all
        running requests. Only wait while a positive-token step is outstanding;
        after it drains, ordinary core preemption can release blocks immediately.
        """
        self._cache_fence_wait_global = False
        self._cache_fence_wait_state = False
        if self.processed_step_seq >= self.sched_step_seq:
            return False
        global_needed, state_needed = self._running_cache_allocation_needs()
        if global_needed == 0 and state_needed == 0:
            return False
        quota = self._state_quota
        pool = quota.block_pool

        def shortages() -> tuple[bool, bool]:
            # Idle resident checkpoints are evictable; only referenced rows
            # reduce the physical quota available to this running batch.
            pinned = sum(
                pool.blocks[block_id].ref_cnt > 0 for block_id in quota._resident
            )
            return (
                global_needed > pool.get_num_free_blocks(),
                state_needed > quota.capacity - pinned,
            )

        global_short, state_short = shortages()
        if not (global_short or state_short):
            return False

        # Allocation normally performs this idempotent release per request.
        # Do it before deciding to wait so already-completed old states do not
        # masquerade as GPU-pinned resources. The completed source and every
        # in-flight destination remain referenced, including shared sources.
        for request in self.running:
            processed = max(
                0,
                min(request.num_computed_tokens, quota.max_model_len)
                - request.num_in_flight_tokens,
            )
            quota._remove_completed_state_rows(request.request_id, processed)
        quota.retire_uncached()
        global_short, state_short = shortages()
        self._cache_fence_wait_global = global_short
        self._cache_fence_wait_state = state_short
        return global_short or state_short

    def schedule(self, throttle_prefills: bool = False) -> SchedulerOutput:
        maximum = self.max_num_scheduled_tokens
        try:
            if self._should_wait_for_cache_fence():
                self.max_num_scheduled_tokens = 0
                self._state_quota.pressure_wait_steps += 1
                self._state_quota.pressure_wait_global_steps += int(
                    self._cache_fence_wait_global
                )
                self._state_quota.pressure_wait_state_steps += int(
                    self._cache_fence_wait_state
                )
            # Let core finalize the empty step normally: finished IDs, previous
            # batch membership and the state catalog must reach the worker even
            # while the engine's bounded FIFO drains earlier positive-token work.
            output = super().schedule(throttle_prefills)
        finally:
            self.max_num_scheduled_tokens = maximum
        if output.kv_cache_block_copies:
            raise RuntimeError("unexpected CoW output under a state budget")
        self._state_quota.retire_uncached()
        values = {field.name: getattr(output, field.name) for field in fields(output)}
        return StateBudgetSchedulerOutput(
            **values,
            metal_state_cache=self._state_quota.snapshot(),
            state_cache_budget_stats=self._state_quota.stats(),
        )


class StateBudgetAsyncScheduler(StateBudgetScheduler, AsyncScheduler):
    """Keep AsyncScheduler's scheduling/output hooks and add the state quota."""
