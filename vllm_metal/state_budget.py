# SPDX-License-Identifier: Apache-2.0
"""Per-engine stable-state budget and scheduler/worker lease protocol.

The user budget covers stable GDN arrays. A separate allowance reserves known
state-row temporaries in addition to the existing forward profile estimate.
Neither is a bound on total process peak memory. Importing this module does not
load MLX or vLLM, so platform discovery can parse the option.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from vllm.config import VllmConfig
    from vllm.v1.kv_cache_interface import KVCacheConfig

STATE_CACHE_BUDGET_KEY = "state_cache_budget_mib"
STATE_BUDGET_SCHEDULER = "vllm_metal.v1.state_budget_scheduler.StateBudgetScheduler"
STATE_BUDGET_ASYNC_SCHEDULER = (
    "vllm_metal.v1.state_budget_scheduler.StateBudgetAsyncScheduler"
)


@dataclass(frozen=True)
class StateCacheStep:
    """Authoritative state leases after scheduling one ordered worker step.

    Each pair is a scheduler block ID and its allocation generation. A missing
    or replaced generation revokes the old mapping before this step can reuse
    its physical slot. Generations never identify token positions.
    """

    sequence: int
    resident_blocks: tuple[tuple[int, int], ...]


@dataclass(frozen=True)
class StateCacheBudget:
    requested_bytes: int
    slot_bytes: int
    capacity: int
    allocated_bytes: int

    @classmethod
    def from_bytes(cls, requested_bytes: int, slot_bytes: int) -> StateCacheBudget:
        if requested_bytes <= 0 or slot_bytes <= 0:
            raise ValueError(
                "state cache budget and physical slot bytes must be positive"
            )
        capacity = requested_bytes // slot_bytes
        if capacity < 1:
            raise ValueError(
                f"{STATE_CACHE_BUDGET_KEY} is smaller than one physical state slot "
                f"({requested_bytes} < {slot_bytes} bytes)"
            )
        return cls(requested_bytes, slot_bytes, capacity, capacity * slot_bytes)


@dataclass(frozen=True)
class StateCacheScratchAllowance:
    """Additional planning allowance for known row-shaped state temporaries.

    This is separate from both stable pools and the existing forward profile.
    It is not a bound on all activations, allocator or Metal driver memory.
    """

    in_flight_batches: int
    active_requests: int
    bytes_per_request: int

    @property
    def payload_copies(self) -> int:
        return 3 * self.in_flight_batches + 1

    @property
    def total_bytes(self) -> int:
        return self.payload_copies * self.active_requests * self.bytes_per_request


def state_cache_budget_bytes(vllm_config: VllmConfig) -> int | None:
    additional = getattr(vllm_config, "additional_config", None)
    if not isinstance(additional, dict) or STATE_CACHE_BUDGET_KEY not in additional:
        return None
    value = additional[STATE_CACHE_BUDGET_KEY]
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{STATE_CACHE_BUDGET_KEY} must be a positive integer in MiB")
    return value * 1024 * 1024


def resolve_state_cache_budget(
    vllm_config: VllmConfig, slot_bytes: int
) -> StateCacheBudget | None:
    requested = state_cache_budget_bytes(vllm_config)
    if requested is None:
        return None
    return StateCacheBudget.from_bytes(requested, slot_bytes)


def state_cache_rows_per_request(vllm_config: VllmConfig, state_groups: int) -> int:
    """Completed source plus one destination per possible in-flight batch."""
    batches = vllm_config.max_concurrent_batches
    if type(batches) is not int or batches < 1 or state_groups < 1:
        raise ValueError(
            "state cache working reserve requires positive batch/group counts"
        )
    return (batches + 1) * state_groups


def state_cache_scratch_allowance(
    vllm_config: VllmConfig, budget: StateCacheBudget, state_groups: int
) -> StateCacheScratchAllowance:
    """Plan ``(3B + 1) * R * F`` bytes in addition to the forward profile.

    Each in-flight batch gets one request-state payload for copy/zero scratch,
    one for new compact updates and one for possible contiguous/dtype staging
    before scatter. One extra payload covers preceding pending updates. This
    deliberately overcounts aliases and non-overlapping operations; the pool
    scatter itself aliases contiguous stable arrays rather than copying them.

    ``R`` follows the scheduler's state working reserve and the one-token-per-
    scheduled-request minimum. ``F`` covers all logical state layers: each
    state group uses one physical row across the canonical pools. The profile
    runs a single sequence before paged wrappers are installed, so do not
    subtract any assumed overlap with this batched-state allowance.
    """
    rows_per_request = state_cache_rows_per_request(vllm_config, state_groups)
    scheduler = vllm_config.scheduler_config
    limits = (scheduler.max_num_seqs, scheduler.max_num_batched_tokens)
    if any(type(limit) is not int or limit < 1 for limit in limits):
        raise ValueError(
            "state scratch allowance requires positive request/token limits"
        )
    active_requests = min(*limits, budget.capacity // rows_per_request)
    if active_requests < 1:
        raise ValueError("state scratch allowance requires one request's working rows")
    return StateCacheScratchAllowance(
        in_flight_batches=vllm_config.max_concurrent_batches,
        active_requests=active_requests,
        bytes_per_request=budget.slot_bytes * state_groups,
    )


def state_cache_budget_from_kv_config(
    vllm_config: VllmConfig, kv_cache_config: KVCacheConfig
) -> StateCacheBudget | None:
    """Derive the same physical row size from the scheduler's adopted layout."""
    if state_cache_budget_bytes(vllm_config) is None:
        return None
    from vllm.v1.kv_cache_interface import MambaSpec

    from vllm_metal.attention.caches.placement import layer_addresses

    state_specs = {
        name: group.kv_cache_spec
        for group in kv_cache_config.kv_cache_groups
        if isinstance(group.kv_cache_spec, MambaSpec)
        for name in group.layer_names
    }
    if not state_specs:
        raise ValueError(f"{STATE_CACHE_BUDGET_KEY} requires GDN state cache groups")
    seen: set[str] = set()
    pools: dict[int, int] = {}
    for tensor in kv_cache_config.kv_cache_tensors:
        for name, address in layer_addresses(tensor):
            if name not in state_specs:
                continue
            if name in seen:
                raise ValueError("state cache layer has multiple physical placements")
            seen.add(name)
            size = state_specs[name].state_content_size_bytes
            if address in pools and pools[address] != size:
                raise ValueError("shared state cache pool has incompatible row sizes")
            pools[address] = size
    if seen != state_specs.keys():
        raise ValueError("state cache layout does not cover every GDN layer")
    return resolve_state_cache_budget(vllm_config, sum(pools.values()))


def configure_state_cache_budget(vllm_config: VllmConfig) -> None:
    """Select an opt-in local scheduler without silently disabling features."""
    if state_cache_budget_bytes(vllm_config) is None:
        return
    cache = vllm_config.cache_config
    parallel = vllm_config.parallel_config
    scheduler = vllm_config.scheduler_config
    if not cache.enable_prefix_caching or cache.mamba_cache_mode != "align":
        raise ValueError(f"{STATE_CACHE_BUDGET_KEY} requires GDN align prefix caching")
    unsupported = []
    if vllm_config.speculative_config is not None:
        unsupported.append("speculative decoding")
    if vllm_config.kv_transfer_config is not None:
        unsupported.append("KV transfer/offloading connectors")
    if getattr(vllm_config, "ec_transfer_config", None) is not None:
        unsupported.append("encoder transfer connectors")
    if parallel.tensor_parallel_size != 1 or parallel.pipeline_parallel_size != 1:
        unsupported.append("tensor/pipeline parallelism")
    if parallel.data_parallel_size != 1:
        unsupported.append("data parallelism")
    if parallel.distributed_executor_backend != "uni":
        unsupported.append("non-local executors")
    if getattr(cache, "prefix_match_unit", None) is not None:
        unsupported.append("custom prefix_match_unit")
    if getattr(cache, "kv_offloading_size", None):
        unsupported.append("KV offloading")
    if getattr(scheduler, "policy", "fcfs") != "fcfs":
        # Priority preemption may cancel an earlier allocation in the same
        # scheduler step after upstream has already published its hashes.
        # Such never-executed states require a separate rollback protocol.
        unsupported.append("priority scheduling")
    if unsupported:
        raise NotImplementedError(
            f"{STATE_CACHE_BUDGET_KEY} does not yet support " + ", ".join(unsupported)
        )
    allowed = (None, STATE_BUDGET_SCHEDULER, STATE_BUDGET_ASYNC_SCHEDULER)
    if scheduler.scheduler_cls not in allowed:
        raise ValueError(f"{STATE_CACHE_BUDGET_KEY} cannot replace a custom scheduler")
    # An unset async option is resolved to enabled by vLLM for this local,
    # non-speculative path. Preserve an explicit false selection.
    scheduler.scheduler_cls = (
        STATE_BUDGET_SCHEDULER
        if scheduler.async_scheduling is False
        else STATE_BUDGET_ASYNC_SCHEDULER
    )
