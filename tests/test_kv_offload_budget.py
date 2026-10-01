# SPDX-License-Identifier: Apache-2.0
"""The KV offload host pool comes out of the Metal KV budget.

On unified memory the pool is the same physical RAM as the wired cache, so
``--gpu-memory-utilization`` must bound both. The budget reported to vLLM
shrinks by the pool, and vLLM sizes a smaller KVCacheConfig from it.
"""

from __future__ import annotations

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


def _planner(kv_transfer_config: object) -> WorkerCachePlanner:
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
        cache_config=worker.cache_config, kv_transfer_config=kv_transfer_config
    )
    worker.get_cache_block_size_bytes = MagicMock(return_value=_PER_BLOCK)
    return WorkerCachePlanner(worker)


def _offload(connector: str | None, pool: int = _POOL) -> SimpleNamespace:
    return SimpleNamespace(
        kv_connector=connector,
        kv_connector_extra_config={"cpu_bytes_to_use": pool},
    )


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


def test_offload_pool_is_held_back_from_the_paging_cap(monkeypatch) -> None:
    """A paging machine offers the KV pool free memory less the offload pool.

    The offload host pool is pageable, but it is the same RAM: a cap that spent
    it on KV would put the offload pool back into swap.
    """
    free = 3 * _GB
    pool = _GB
    planner = _planner(_offload("MetalOffloadingConnector", pool=pool))
    monkeypatch.setenv("VLLM_METAL_KV_COMMIT_PROBE", "1")
    monkeypatch.setattr(
        "vllm_metal.v1.cache_policy.probe_commit",
        lambda nbytes: CommitProbe(
            probed_bytes=nbytes,
            swap_out_before=0,
            swap_out_after=nbytes,  # paged the whole sample: no headroom
            available_before=free,
            available_after=free,
            seconds=0.0,
        ),
    )

    plan = planner._paged_attention_plan(overhead=_GB)

    # Without the carve-out the cap would be free less the reserve (1.93GB),
    # which is below the 2GB plan but above this.
    assert plan.kv_offload_pool == pool
    assert plan.kv_budget == free - (1 << 30) - pool
