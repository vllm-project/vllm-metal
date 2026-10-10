# SPDX-License-Identifier: Apache-2.0
"""The KV offload host pool comes out of the Metal KV budget.

On unified memory the pool is the same physical RAM as the wired cache, so
``--gpu-memory-utilization`` must bound both. The budget reported to vLLM
shrinks by the pool, and vLLM sizes a smaller KVCacheConfig from it.
"""

from __future__ import annotations

import itertools
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

pytest.importorskip("vllm", reason="vllm not installed")

from vllm_metal.config import MetalConfig  # noqa: E402
from vllm_metal.utils import CommitProbe  # noqa: E402
from vllm_metal.v1.cache_policy import WorkerCachePlanner  # noqa: E402
from vllm_metal.v1.worker import MetalWorker  # noqa: E402

_GB = 1_000_000_000
_PER_BLOCK = 1_000_000
_POOL = 2 * _GB


def _planner(
    kv_transfer_config: object, max_model_len: int = 4096
) -> WorkerCachePlanner:
    runner = SimpleNamespace(
        scheduler_memory_reporting_mode=lambda: "paged_attention_layout_budget",
        profile_run=lambda: _GB,
    )
    worker = MetalWorker.__new__(MetalWorker)
    worker.model_runner = runner  # type: ignore[assignment]
    worker.cache_config = SimpleNamespace(
        block_size=16, gpu_memory_utilization=0.5, num_gpu_blocks_override=None
    )
    worker.vllm_config = SimpleNamespace(
        cache_config=worker.cache_config,
        kv_transfer_config=kv_transfer_config,
        model_config=SimpleNamespace(max_model_len=max_model_len),
    )
    worker.get_cache_block_size_bytes = MagicMock(return_value=_PER_BLOCK)
    return WorkerCachePlanner(worker)


def _offload(connector: str | None, pool: int = _POOL) -> SimpleNamespace:
    return SimpleNamespace(
        kv_connector=connector,
        kv_connector_extra_config={"cpu_bytes_to_use": pool},
    )


def _paging_probe(monkeypatch, free: int, swap_out=None) -> None:
    """A commit probe that sees ``free`` bytes available. By default it paged
    the whole sample, so there is no headroom."""
    monkeypatch.setenv("VLLM_METAL_KV_COMMIT_PROBE", "1")
    monkeypatch.setattr(
        "vllm_metal.v1.cache_policy.probe_commit",
        lambda nbytes: CommitProbe(
            probed_bytes=nbytes,
            swap_out_before=0,
            swap_out_after=nbytes if swap_out is None else swap_out(nbytes),
            available_before=free,
            available_after=free,
            compressed_before=0,
            compressed_after=0,
            seconds=0.0,
        ),
    )


def _chunk_bytes(blocks_per_chunk: int = 1) -> int:
    """One host chunk, aligned as vLLM's shared offload region aligns it."""
    from vllm.v1.kv_offload.cpu.shared_offload_region import SharedOffloadRegion

    align = SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    return -(-_PER_BLOCK * blocks_per_chunk // align) * align


@pytest.fixture(autouse=True)
def _fixed_device(monkeypatch) -> None:
    monkeypatch.setattr(
        "vllm_metal.v1.cache_policy.get_config",
        lambda: MetalConfig(mlx_device="gpu"),
    )
    monkeypatch.setattr(WorkerCachePlanner, "_metal_limit_bytes", lambda _: 10 * _GB)
    monkeypatch.setattr(WorkerCachePlanner, "get_model_memory_usage", lambda _: _GB)


def test_offload_off_matches_upstream() -> None:
    """Without offload every number and message is upstream's."""
    planner = _planner(None)
    plan = planner._paged_attention_plan(overhead=_GB)

    # 10GB * 0.5 - 1GB weights - 1GB overhead.
    assert plan.kv_budget == 3 * _GB
    assert plan.num_blocks == 3000
    assert plan.kv_offload_pool == 0
    assert plan.format_breakdown() == (
        "metal_limit=10.00GB, fraction=0.5, usable_metal=5.00GB, "
        "model_memory=1.00GB, overhead=1.00GB, kv_budget=3.00GB"
    )
    assert plan.format_mitigations() == (
        "Mitigations: increase --gpu-memory-utilization (currently 0.5); "
        "use a smaller or more quantized model."
    )
    assert planner.determine_available_memory() == 3 * _GB


def test_pool_comes_out_of_the_reported_budget() -> None:
    without = _planner(None)._paged_attention_plan(overhead=_GB)
    planner = _planner(_offload("MetalOffloadingConnector"))
    plan = planner._paged_attention_plan(overhead=_GB)

    assert plan.kv_offload_pool == _POOL
    assert plan.kv_budget == without.kv_budget - _POOL
    assert plan.num_blocks == without.num_blocks - _POOL // _PER_BLOCK
    assert "kv_offload_pool=2.00GB, kv_budget=1.00GB" in plan.format_breakdown()
    assert plan.format_mitigations().startswith(
        "Mitigations: lower --kv-offloading-size; "
    )
    # vLLM sizes its KVCacheConfig from this number.
    assert planner.determine_available_memory() == 1 * _GB


def test_inert_transfer_config_does_not_budget() -> None:
    """A transfer config with no connector is inert upstream."""
    plan = _planner(_offload(None))._paged_attention_plan(overhead=_GB)
    assert plan.kv_offload_pool == 0
    assert plan.kv_budget == 3 * _GB


def test_other_connector_does_not_budget() -> None:
    """Only the Metal offloading connector owns a host pool."""
    plan = _planner(_offload("NixlConnector"))._paged_attention_plan(overhead=_GB)
    assert plan.kv_offload_pool == 0
    assert plan.kv_budget == 3 * _GB


def test_pool_larger_than_budget_fails_with_the_mitigation() -> None:
    planner = _planner(_offload("MetalOffloadingConnector", pool=4 * _GB))
    with pytest.raises(ValueError, match="lower --kv-offloading-size"):
        planner.determine_available_memory()


def _auto_pool(pool: int) -> SimpleNamespace:
    from vllm_metal.v1.kv_offload.config import AUTO_POOL_KEY

    config = _offload("MetalOffloadingConnector", pool=pool)
    config.kv_connector_extra_config[AUTO_POOL_KEY] = True
    return config


def test_automatic_pool_is_capped_to_fit_the_budget() -> None:
    """A long max_model_len must not stop the server from starting."""
    kv_transfer_config = _auto_pool(4 * _GB)
    planner = _planner(kv_transfer_config)

    # A quarter of the 3GB budget left after weights and overhead.
    assert planner.determine_available_memory() == 3 * _GB - 750_000_000
    # The scheduler sizes its pool from the same config after this.
    extra = kv_transfer_config.kv_connector_extra_config
    assert extra["cpu_bytes_to_use"] == 750_000_000


def test_automatic_pool_under_the_cap_is_kept() -> None:
    kv_transfer_config = _auto_pool(500_000_000)
    plan = _planner(kv_transfer_config)._paged_attention_plan(overhead=_GB)
    assert plan.kv_offload_pool == 500_000_000
    assert kv_transfer_config.kv_connector_extra_config["cpu_bytes_to_use"] == (
        500_000_000
    )


def test_automatic_pool_leaves_room_for_one_full_request() -> None:
    """Offloading must not stop a server that fits one request without it."""
    kv_transfer_config = _auto_pool(4 * _GB)
    # 2700 blocks of 1MB plus the null block: one request takes 2.701GB of
    # the 3GB budget.
    planner = _planner(kv_transfer_config, max_model_len=16 * 2700)

    assert planner.determine_available_memory() == 2_701_000_000
    assert kv_transfer_config.kv_connector_extra_config["cpu_bytes_to_use"] == (
        299_000_000
    )


def test_request_that_never_fits_is_left_to_vllm() -> None:
    """One request needs more than the whole budget, with or without
    offloading. The pool takes one chunk and vLLM's max_model_len check
    reports the real problem."""
    kv_transfer_config = _auto_pool(4 * _GB)
    extra = kv_transfer_config.kv_connector_extra_config
    planner = _planner(kv_transfer_config, max_model_len=16 * 3000)

    chunk = _chunk_bytes()
    assert planner.determine_available_memory() == 3 * _GB - chunk
    assert extra["cpu_bytes_to_use"] == chunk


def test_request_that_fits_only_without_a_pool_fails_with_the_fix() -> None:
    """One request fits, but not beside the smallest pool: offloading is
    what fails here, so say so, with the numbers."""
    # 2999 blocks of 1MB (request plus null block) leave 1MB, under one chunk.
    planner = _planner(_auto_pool(4 * _GB), max_model_len=16 * 2998)
    with pytest.raises(ValueError) as error:
        planner.determine_available_memory()
    message = str(error.value)
    assert "cannot hold one --max-model-len request (2860.1 MiB)" in message
    assert "the smallest host pool (1.0 MiB)" in message
    assert "KV budget (2861.0 MiB)" in message
    assert message.endswith("Raise --gpu-memory-utilization or lower --max-model-len.")


def test_offload_pool_is_held_back_from_the_paging_cap(monkeypatch) -> None:
    """A paging machine offers the KV pool free memory less the offload pool.

    The offload host pool is pageable, but it is the same RAM: a cap that spent
    it on KV would put the offload pool back into swap.
    """
    free = 3 * _GB
    pool = _GB
    planner = _planner(_offload("MetalOffloadingConnector", pool=pool))
    _paging_probe(monkeypatch, free=free)

    plan = planner._paged_attention_plan(overhead=_GB)

    # Without the carve-out the cap would be free less the reserve (1.93GB),
    # which is below the 2GB plan but above this.
    assert plan.kv_offload_pool == pool
    assert plan.kv_budget == free - (1 << 30) - pool


def test_automatic_pool_is_capped_again_when_the_probe_shrinks_the_budget(
    monkeypatch,
) -> None:
    """On a paging machine the probe shrinks only the KV cache. The automatic
    pool must give memory back so one request still fits."""
    free = 3 * _GB
    request_blocks = 1500
    kv_transfer_config = _auto_pool(4 * _GB)
    planner = _planner(kv_transfer_config, max_model_len=16 * request_blocks)
    _paging_probe(monkeypatch, free=free)

    plan = planner._paged_attention_plan(overhead=_GB)

    # What the machine holds for KV and pool together, as the probe found it.
    total = free - (1 << 30)
    # One request plus vLLM's null block.
    request = (request_blocks + 1) * _PER_BLOCK
    expected_pool = min(total // 4, total - request)
    assert plan.kv_offload_pool == expected_pool
    assert plan.kv_budget == total - expected_pool
    assert plan.num_blocks == (total - expected_pool) // _PER_BLOCK
    assert plan.kv_budget >= request
    extra = kv_transfer_config.kv_connector_extra_config
    assert extra["cpu_bytes_to_use"] == expected_pool


def test_recap_that_cannot_fit_says_the_machine_is_paging(monkeypatch) -> None:
    """The probe leaves less than one request plus one chunk. The error names
    paging, not a flag the user never set."""
    planner = _planner(_auto_pool(4 * _GB), max_model_len=16 * 1500)
    _paging_probe(monkeypatch, free=2 * _GB)

    with pytest.raises(ValueError, match="machine is paging") as error:
        planner._paged_attention_plan(overhead=_GB)
    assert "--kv-offloading-size" not in str(error.value)


def test_recap_uses_the_room_when_free_memory_is_below_the_pool(monkeypatch) -> None:
    """With less free memory than the pool, the probe's own result floors at
    0. The re-cap uses the room the probe saw, so the server still starts."""
    planner = _planner(_auto_pool(4 * _GB), max_model_len=16 * 50)
    # 100MB free past the reserve, under the 750MB pool.
    _paging_probe(monkeypatch, free=100_000_000 + (1 << 30))

    plan = planner._paged_attention_plan(overhead=_GB)

    # A quarter of the 100MB room; the rest holds the 51MB request.
    assert plan.kv_offload_pool == 25_000_000
    assert plan.kv_budget == 75_000_000


def test_auto_fit_max_model_len_does_not_reserve_a_full_request() -> None:
    """--max-model-len -1: vLLM fits the length after planning."""
    kv_transfer_config = _auto_pool(4 * _GB)
    planner = _planner(kv_transfer_config, max_model_len=16 * 3000)
    planner._worker.vllm_config.model_config.original_max_model_len = -1

    assert planner.determine_available_memory() == 3 * _GB - 750_000_000


def test_block_override_does_not_reserve_a_full_request() -> None:
    """vLLM sizes the cache from num_gpu_blocks_override, not this budget."""
    kv_transfer_config = _auto_pool(4 * _GB)
    planner = _planner(kv_transfer_config, max_model_len=16 * 3000)
    planner._worker.cache_config.num_gpu_blocks_override = 100

    plan = planner._paged_attention_plan(overhead=_GB)

    assert plan.kv_offload_pool == 750_000_000


def test_pool_below_one_host_chunk_fails_with_the_fix() -> None:
    """The host pool holds whole aligned chunks, so a cap below one chunk
    cannot work even when it is above one GPU block."""
    kv_transfer_config = _auto_pool(4 * _GB)
    kv_transfer_config.kv_connector_extra_config["blocks_per_chunk"] = 2
    # Leaves exactly one GPU block (1MB) next to the request: under 2MB.
    planner = _planner(kv_transfer_config, max_model_len=16 * 2998)
    with pytest.raises(ValueError, match="--gpu-memory-utilization"):
        planner.determine_available_memory()


def test_no_budget_at_all_reports_the_breakdown(monkeypatch) -> None:
    """With no room even before the pool, the planner's own error explains it,
    without the automatic pool or advice about a flag the user never set."""
    monkeypatch.setattr(WorkerCachePlanner, "get_model_memory_usage", lambda _: 5 * _GB)
    planner = _planner(_auto_pool(4 * _GB))
    with pytest.raises(ValueError, match="not enough Metal memory") as error:
        planner.determine_available_memory()
    assert "kv_offload_pool" not in str(error.value)
    assert "--kv-offloading-size" not in str(error.value)


def test_recap_shrinks_to_one_chunk_when_a_request_still_fits(monkeypatch) -> None:
    """After a paging probe, a quarter of what is left can be below one chunk.
    The pool shrinks to one chunk rather than keeping its pre-probe size."""
    chunk = 256 * _PER_BLOCK
    kv_transfer_config = _auto_pool(4 * _GB)
    kv_transfer_config.kv_connector_extra_config["blocks_per_chunk"] = 256
    planner = _planner(kv_transfer_config, max_model_len=16 * 50)
    # The probe leaves 900MB for KV cache and pool together.
    _paging_probe(monkeypatch, free=900_000_000 + (1 << 30))

    plan = planner._paged_attention_plan(overhead=_GB)

    assert plan.kv_offload_pool == chunk
    assert plan.kv_budget == 900_000_000 - chunk


def test_pool_below_one_aligned_chunk_fails(monkeypatch) -> None:
    """One GPU block is not enough: the host chunk is page aligned."""
    per_block = 12_000  # one aligned chunk is 16KiB
    planner = _planner(_auto_pool(4 * _GB), max_model_len=16 * 249_998)
    planner._worker.get_cache_block_size_bytes = MagicMock(return_value=per_block)
    # The cap left after one request is exactly one GPU block.
    with pytest.raises(ValueError, match="cannot hold"):
        planner.determine_available_memory()


def test_block_size_key_sets_the_chunk_floor() -> None:
    """--kv-transfer-config block_size in tokens sets blocks per chunk."""
    kv_transfer_config = _auto_pool(4 * _GB)
    kv_transfer_config.kv_connector_extra_config["block_size"] = 64  # 4 blocks
    # Leaves 2MB next to the request: below one 4-block chunk.
    planner = _planner(kv_transfer_config, max_model_len=16 * 2997)
    with pytest.raises(ValueError, match="cannot hold"):
        planner.determine_available_memory()


def test_automatic_pool_below_one_chunk_is_raised_to_one_chunk() -> None:
    """A short max_model_len with large host chunks: two requests are less
    than one chunk, which would give the host pool no chunks at all."""
    kv_transfer_config = _auto_pool(32 * _PER_BLOCK)
    extra = kv_transfer_config.kv_connector_extra_config
    extra["block_size"] = 1024  # 64 blocks per chunk
    _planner(kv_transfer_config, max_model_len=256).determine_available_memory()

    assert extra["cpu_bytes_to_use"] == _chunk_bytes(64)


def test_cap_equal_to_one_chunk_starts() -> None:
    """Budget minus one request is exactly one chunk: start with that chunk."""
    kv_transfer_config = _auto_pool(4 * _GB)
    extra = kv_transfer_config.kv_connector_extra_config
    extra["blocks_per_chunk"] = 256
    # 2744 blocks of 1MB (request plus null block) leave 256MB of 3GB.
    _planner(kv_transfer_config, max_model_len=16 * 2743).determine_available_memory()

    assert extra["cpu_bytes_to_use"] == 256 * _PER_BLOCK


def test_request_reservation_rounds_a_partial_block_up() -> None:
    """One token over a block boundary needs the whole next block."""
    # Rounded up: 2998 + null = 2999 blocks, leaving 1MB, under one chunk.
    # Rounded down it would leave 2MB and start.
    planner = _planner(_auto_pool(4 * _GB), max_model_len=16 * 2997 + 1)
    with pytest.raises(ValueError, match="cannot hold"):
        planner.determine_available_memory()


def test_no_request_reserved_and_no_room_for_one_chunk_fails() -> None:
    """--max-model-len -1 with chunks larger than the budget: the message
    names the pool alone and does not mention --max-model-len."""
    planner = _planner(_auto_pool(4 * _GB))
    planner._worker.vllm_config.model_config.original_max_model_len = -1
    planner._worker.get_cache_block_size_bytes = MagicMock(return_value=4 * _GB)
    with pytest.raises(ValueError) as error:
        planner.determine_available_memory()
    message = str(error.value)
    assert "cannot hold the smallest host pool" in message
    assert "--max-model-len" not in message


def test_request_that_exactly_fills_the_budget_fails_with_the_fix() -> None:
    """A request that uses the whole budget fits without offloading, but not
    beside any pool, so offloading is the cause and says so."""
    # 3000 blocks of 1MB (request plus null block) fill the 3GB budget.
    planner = _planner(_auto_pool(4 * _GB), max_model_len=16 * 2999)
    with pytest.raises(ValueError, match="cannot hold one --max-model-len request"):
        planner.determine_available_memory()


def test_zero_budget_with_auto_fit_reports_the_breakdown(monkeypatch) -> None:
    """A budget of exactly 0, with no request reserved: still the planner's error."""
    # 10GB * 0.5 - 4GB weights - 1GB overhead = 0.
    monkeypatch.setattr(WorkerCachePlanner, "get_model_memory_usage", lambda _: 4 * _GB)
    planner = _planner(_auto_pool(4 * _GB))
    planner._worker.vllm_config.model_config.original_max_model_len = -1
    with pytest.raises(ValueError, match="not enough Metal memory"):
        planner.determine_available_memory()


# Re-capping inside the probe. Budget: 10GB * 0.5 - 1GB weights - 1GB
# overhead = 3GB, so the automatic pool is 750MB. The reserve is 1GiB.
_RESERVE = 1 << 30
_AUTO = 750_000_000
_MML = 16 * 100
_REQUEST = 101 * _PER_BLOCK  # 100 blocks plus the null block


@pytest.mark.parametrize("past_reserve", [_AUTO, 300_000_000, _REQUEST + 2_000_000])
def test_automatic_pool_shrinks_when_free_is_at_or_below_the_pool(
    monkeypatch, past_reserve
) -> None:
    """Room for one request and one chunk: the pool shrinks and startup works."""
    kv_transfer_config = _auto_pool(4 * _GB)
    extra = kv_transfer_config.kv_connector_extra_config
    planner = _planner(kv_transfer_config, max_model_len=_MML)
    _paging_probe(monkeypatch, free=_RESERVE + past_reserve)

    plan = planner._paged_attention_plan(overhead=_GB)

    assert plan.kv_budget >= _REQUEST, plan
    assert _chunk_bytes() <= plan.kv_offload_pool < _AUTO, plan
    assert plan.kv_budget + plan.kv_offload_pool <= past_reserve, plan
    assert extra["cpu_bytes_to_use"] == plan.kv_offload_pool
    assert planner.determine_available_memory() == plan.kv_budget


@pytest.mark.parametrize(
    ("past_reserve", "without_offloading_helps"),
    [(0, False), (50_000_000, False), (_REQUEST, True), (_REQUEST + 500_000, True)],
)
def test_no_room_error_names_paging(
    monkeypatch, past_reserve, without_offloading_helps
) -> None:
    """It never advises --kv-offloading-size. It suggests starting without
    offloading only when one request would then fit."""
    planner = _planner(_auto_pool(4 * _GB), max_model_len=_MML)
    _paging_probe(monkeypatch, free=_RESERVE + past_reserve)

    with pytest.raises(ValueError, match="machine is paging") as error:
        planner._paged_attention_plan(overhead=_GB)

    message = str(error.value)
    assert "--kv-offloading-size" not in message
    assert ("without KV offloading" in message) == without_offloading_helps


@pytest.mark.parametrize("past_reserve", [_AUTO, 300_000_000, _AUTO + 500_000_000])
def test_user_pool_is_never_recapped(monkeypatch, past_reserve) -> None:
    """--kv-offloading-size is the user's choice: the probe shrinks KV only."""
    kv_transfer_config = _offload("MetalOffloadingConnector", pool=_AUTO)
    planner = _planner(kv_transfer_config, max_model_len=_MML)
    _paging_probe(monkeypatch, free=_RESERVE + past_reserve)

    plan = planner._paged_attention_plan(overhead=_GB)

    assert plan.kv_offload_pool == _AUTO
    assert kv_transfer_config.kv_connector_extra_config["cpu_bytes_to_use"] == _AUTO
    assert plan.kv_budget == max(0, past_reserve - _AUTO)


@pytest.mark.parametrize("swap", ["none", "at_tolerance"])
def test_no_paging_keeps_the_pool(monkeypatch, swap) -> None:
    """Low free memory without paging is reported, not acted on."""
    from vllm_metal.v1.cache_policy import kv_swap_tolerance_bytes

    kv_transfer_config = _auto_pool(4 * _GB)
    planner = _planner(kv_transfer_config, max_model_len=_MML)
    _paging_probe(
        monkeypatch,
        free=_RESERVE + 300_000_000,
        swap_out=lambda n: 0 if swap == "none" else kv_swap_tolerance_bytes(n),
    )

    plan = planner._paged_attention_plan(overhead=_GB)

    assert plan.kv_offload_pool == _AUTO
    assert plan.kv_budget == 3 * _GB - _AUTO


def test_recap_at_the_exact_fit_boundary_shrinks_to_one_chunk(monkeypatch) -> None:
    """Room minus one chunk is exactly one request: start with that chunk."""
    kv_transfer_config = _auto_pool(750 * _PER_BLOCK)
    kv_transfer_config.kv_connector_extra_config["blocks_per_chunk"] = 256
    planner = _planner(kv_transfer_config, max_model_len=16 * 49)  # 50 blocks
    _paging_probe(monkeypatch, free=_RESERVE + 306 * _PER_BLOCK)

    plan = planner._paged_attention_plan(overhead=_GB)

    assert plan.kv_offload_pool == _chunk_bytes(256)
    assert plan.kv_budget == 50 * _PER_BLOCK


def test_auto_fit_room_of_one_chunk_recaps_to_it(monkeypatch) -> None:
    """With no request reserved, a room of exactly one chunk holds the pool."""
    planner = _planner(_auto_pool(4 * _GB), max_model_len=_MML)
    planner._worker.vllm_config.model_config.original_max_model_len = -1
    _paging_probe(monkeypatch, free=_RESERVE + _chunk_bytes())

    plan = planner._paged_attention_plan(overhead=_GB)

    assert plan.kv_offload_pool == _chunk_bytes()
    assert plan.kv_budget == 0


@pytest.mark.parametrize("past_reserve", [0, 500_000])
def test_block_override_with_no_room_still_starts(monkeypatch, past_reserve) -> None:
    """With --num-gpu-blocks-override vLLM sizes the cache from the override,
    so a probe that leaves less than one chunk must not fail startup."""
    kv_transfer_config = _auto_pool(4 * _GB)
    planner = _planner(kv_transfer_config, max_model_len=_MML)
    planner._worker.cache_config.num_gpu_blocks_override = 64
    _paging_probe(monkeypatch, free=_RESERVE + past_reserve)

    planner.determine_available_memory()
    plan = planner._paged_attention_plan(overhead=_GB)

    assert plan.kv_offload_pool == _AUTO
    assert plan.kv_budget >= 0


def test_turboquant_workspace_stays_out_of_the_room(monkeypatch) -> None:
    """The prefill workspace is reserved before KV sizing, so the room
    excludes it just as the probe does."""
    workspace = 200_000_000
    monkeypatch.setattr(
        "vllm_metal.v1.cache_policy.get_config",
        lambda: SimpleNamespace(turboquant=True),
    )
    planner = _planner(_auto_pool(4 * _GB), max_model_len=_MML)
    planner._worker.model_runner.tq_prefill_workspace_bytes = workspace
    past_reserve = 300_000_000
    _paging_probe(monkeypatch, free=_RESERVE + workspace + past_reserve)

    plan = planner._paged_attention_plan(overhead=_GB)

    assert plan.kv_budget + plan.kv_offload_pool == past_reserve
    assert plan.kv_offload_pool == past_reserve // 4


_PAST = [0, 1_000_000, 50_000_000, _REQUEST, _REQUEST + 1_000_000, 187_500_000]
_PAST += [300_000_000, _AUTO - 1, _AUTO, _AUTO + 1, _GB, 3 * _GB, 4 * _GB]


def test_recap_holds_for_every_case(monkeypatch) -> None:
    """On a paging machine with an automatic pool: the pool never grows, it
    stays at least one chunk, KV plus pool fits the room, the config matches
    the plan, and startup fails only when one request and one chunk cannot
    fit."""
    failures = []
    for past, requested, mml, bpc in itertools.product(
        _PAST,
        [4 * _GB, _AUTO, 100_000_000, 1_000_000],
        [16 * 50, _MML, 16 * 1500],
        [1, 256],
    ):
        case = (past, requested, mml, bpc)

        def config(requested=requested, bpc=bpc):
            c = _auto_pool(requested)
            c.kv_connector_extra_config["blocks_per_chunk"] = bpc
            return c

        monkeypatch.setenv("VLLM_METAL_KV_COMMIT_PROBE", "0")
        before = _planner(config(), mml)._paged_attention_plan(overhead=_GB)

        kv_transfer_config = config()
        extra = kv_transfer_config.kv_connector_extra_config
        planner = _planner(kv_transfer_config, mml)
        _paging_probe(monkeypatch, free=_RESERVE + past)
        chunk = _chunk_bytes(bpc)
        request = (-(-mml // 16) + 1) * _PER_BLOCK
        try:
            plan = planner._paged_attention_plan(overhead=_GB)
        except ValueError:
            if past >= request + chunk:
                failures.append((case, "raised with room"))
            continue
        ok = (
            plan.kv_offload_pool <= before.kv_offload_pool
            and plan.kv_offload_pool >= min(chunk, before.kv_offload_pool)
            and extra["cpu_bytes_to_use"] == plan.kv_offload_pool
            and max(0, plan.kv_budget) + plan.kv_offload_pool <= max(past, chunk)
        )
        if not ok:
            failures.append((case, plan.kv_budget, plan.kv_offload_pool))

    assert not failures, f"{len(failures)} cases: {failures[:5]}"
