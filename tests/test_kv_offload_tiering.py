# SPDX-License-Identifier: Apache-2.0
"""Tests for the shared region, and the spec driving upstream's tiering."""

import hashlib
import uuid
from types import SimpleNamespace

import numpy as np
import psutil
import pytest
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.tiering.base import TransferJob

from tests.kv_offload_helpers import (
    gpu_spec,
    make_region,
    make_storage,
    make_turboquant_storage,
    make_worker,
    randomize,
    snapshot_blocks,
    zero_blocks,
)
from vllm_metal.v1.kv_offload.fs_tier import MetalFileSystemTierManager
from vllm_metal.v1.kv_offload.spec import MetalTieringOffloadingSpec

NUM_CHUNKS = 4
BLOCKS_PER_CHUNK = 2


def _attach(storage, rank, engine_id):
    return make_region(
        storage,
        blocks_per_chunk=BLOCKS_PER_CHUNK,
        num_cpu_chunks=NUM_CHUNKS,
        rank=rank,
        engine_id=engine_id,
    )


@pytest.mark.parametrize("turboquant", [False, True], ids=["fp16", "turboquant"])
def test_memoryview_round_trips_chunk_rows(turboquant: bool) -> None:
    """The fs tier reads and writes whole chunk rows through the
    scheduler-side memoryview; a load must restore from those bytes."""
    storage = make_turboquant_storage() if turboquant else make_storage()
    engine_id = f"test-{uuid.uuid4().hex[:8]}"
    worker = make_worker(
        storage,
        blocks_per_chunk=BLOCKS_PER_CHUNK,
        num_cpu_chunks=NUM_CHUNKS,
        region=_attach(storage, 0, engine_id),
    )
    scheduler = _attach(storage, None, engine_id)
    view = scheduler.create_kv_memoryview()
    flat = view.cast("B")
    try:
        randomize(storage)
        original = snapshot_blocks(storage, [3, 4])
        assert worker.submit_store(1, gpu_spec([3, 4]), CPULoadStoreSpec([1]))

        # Address rows as tiering/fs/io.py does. Move chunk 1 to chunk 3, so
        # a carve that disagrees with the view cannot cancel out.
        assert view.strides is not None
        row = view.strides[0]
        stored = bytes(flat[row : 2 * row])
        assert any(stored)
        flat[3 * row : 4 * row] = stored
        flat[row : 2 * row] = b"\x00" * row

        zero_blocks(storage, [3, 4])
        assert worker.submit_load(2, CPULoadStoreSpec([3]), gpu_spec([3, 4]))
        for a, b in zip(original, snapshot_blocks(storage, [3, 4]), strict=True):
            np.testing.assert_array_equal(a, b)
    finally:
        del flat, view
        worker.shutdown()
        scheduler.cleanup()


def test_region_registry_shares_and_releases() -> None:
    storage = make_storage()
    engine_id = f"test-{uuid.uuid4().hex[:8]}"
    a = _attach(storage, 0, engine_id)
    b = _attach(storage, None, engine_id)
    a._base[0] = 42
    assert int(b._base[0]) == 42

    other = _attach(storage, None, f"other-{uuid.uuid4().hex[:8]}")
    assert int(other._base[0]) == 0
    other.cleanup()

    # Releasing one attachment, even twice, keeps the buffer for the other.
    a.cleanup()
    a.cleanup()
    c = _attach(storage, None, engine_id)
    assert int(c._base[0]) == 42
    b.cleanup()
    c.cleanup()

    # After the last release a new attachment gets a fresh buffer.
    d = _attach(storage, None, engine_id)
    assert int(d._base[0]) == 0
    d.cleanup()


def test_released_region_frees_its_pages() -> None:
    """Sleep frees the pool in place. USS is the footprint on macOS."""
    storage = make_storage()
    region = make_region(storage, num_cpu_chunks=4096)
    sample = region.total_size_bytes
    try:
        # Random bytes, so the compressor cannot shrink the pages first.
        rng = np.random.default_rng(0)
        region._base.numpy()[:] = rng.integers(-128, 128, sample, dtype=np.int8)
        before = psutil.Process().memory_full_info().uss
        region.release_pages()
        released = psutil.Process().memory_full_info().uss
        assert before - released > sample // 2

        # Writes before reuse_pages are not counted and the kernel may drop
        # them. The pool is counted again once it is reused.
        region.reuse_pages()
        region._base.numpy()[:] = rng.integers(-128, 128, sample, dtype=np.int8)
        reused = psutil.Process().memory_full_info().uss
        assert reused - released > sample // 2
    finally:
        region.cleanup()


def test_failed_release_keeps_the_pool(caplog) -> None:
    """Freeing the pool is an optimisation, so a failure must not fail sleep."""
    storage = make_storage()
    region = make_region(storage)

    def refuse(advice):
        raise OSError("madvise refused")

    assert region._state is not None
    mapping = region._state.mapping
    region._state.mapping = SimpleNamespace(madvise=refuse)
    try:
        region._base[0] = 7
        region.release_pages()
        assert int(region._base[0]) == 7
        assert "Could not free" in caplog.text
    finally:
        region._state.mapping = mapping
        region.cleanup()


def test_failed_reuse_raises() -> None:
    """Wake must not go on: the kernel can drop writes to pages not taken back."""
    region = make_region(make_storage())

    def refuse(advice):
        raise OSError("madvise refused")

    assert region._state is not None
    mapping = region._state.mapping
    region._state.mapping = SimpleNamespace(madvise=refuse)
    try:
        with pytest.raises(OSError):
            region.reuse_pages()
    finally:
        region._state.mapping = mapping
        region.cleanup()


def test_released_pool_round_trips_after_reuse() -> None:
    """After wake the same views store and load, and the scheduler side
    reads the stored bytes, so the disk tier still sees them."""
    storage = make_storage()
    engine_id = f"test-{uuid.uuid4().hex[:8]}"
    worker = make_worker(
        storage,
        blocks_per_chunk=BLOCKS_PER_CHUNK,
        num_cpu_chunks=NUM_CHUNKS,
        region=_attach(storage, 0, engine_id),
    )
    scheduler = _attach(storage, None, engine_id)
    try:
        randomize(storage)
        assert worker.submit_store(1, gpu_spec([3, 4]), CPULoadStoreSpec([1]))
        worker.release_host_pool()
        worker.restore_host_pool()

        randomize(storage, seed=1)
        original = snapshot_blocks(storage, [3, 4])
        assert worker.submit_store(2, gpu_spec([3, 4]), CPULoadStoreSpec([1]))
        row = scheduler.total_size_bytes // NUM_CHUNKS
        assert bytes(scheduler._base[row : 2 * row].numpy()) == bytes(
            worker.region._base[row : 2 * row].numpy()
        )
        assert any(scheduler._base[row : 2 * row].numpy())

        zero_blocks(storage, [3, 4])
        assert worker.submit_load(3, CPULoadStoreSpec([1]), gpu_spec([3, 4]))
        for a, b in zip(original, snapshot_blocks(storage, [3, 4]), strict=True):
            np.testing.assert_array_equal(a, b)
    finally:
        worker.shutdown()
        scheduler.cleanup()


def test_reuse_without_release_keeps_the_pool() -> None:
    """Wake after a sleep that could not free the pool must not lose it."""
    region = make_region(make_storage())
    try:
        region._base[0] = 7
        region.reuse_pages()
        assert int(region._base[0]) == 7
    finally:
        region.cleanup()


def test_region_geometry_mismatch_rejected() -> None:
    storage = make_storage()
    engine_id = f"test-{uuid.uuid4().hex[:8]}"
    a = _attach(storage, 0, engine_id)
    try:
        with pytest.raises(ValueError, match="mismatched geometry"):
            make_region(
                storage,
                blocks_per_chunk=BLOCKS_PER_CHUNK,
                num_cpu_chunks=NUM_CHUNKS + 1,
                rank=None,
                engine_id=engine_id,
            )
    finally:
        a.cleanup()


def test_empty_pool_rejected_before_registering() -> None:
    storage = make_storage()
    engine_id = f"test-{uuid.uuid4().hex[:8]}"
    with pytest.raises(ValueError, match="holds no chunks"):
        make_region(storage, num_cpu_chunks=0, engine_id=engine_id)
    # Nothing was left in the registry under that id.
    region = make_region(storage, num_cpu_chunks=NUM_CHUNKS, engine_id=engine_id)
    assert int(region._base[0]) == 0
    region.cleanup()


def _offloading_config(
    blocks_per_chunk: int = 1,
    tokens_per_block: int = 16,
    extra_config: dict | None = None,
    enable_kv_cache_events: bool = False,
):
    """A real ``OffloadingConfig``, built without a model download."""
    from vllm.v1.kv_offload.config import (
        OffloadingCacheConfig,
        OffloadingConfig,
        OffloadingGroupConfig,
        OffloadingModelConfig,
        OffloadingParallelConfig,
    )

    return OffloadingConfig(
        groups=(
            OffloadingGroupConfig(
                tokens_per_block=tokens_per_block, layer_names=("l0",), group_id=0
            ),
        ),
        worker_kv_bytes_per_block=1 << 14,
        enable_kv_cache_events=enable_kv_cache_events,
        extra_config=extra_config or {"cpu_bytes_to_use": 64 << 20},
        engine_id="test-engine",
        model=OffloadingModelConfig(name="test-model", dtype="float16"),
        cache=OffloadingCacheConfig(
            tokens_per_hash=tokens_per_block, blocks_per_chunk=blocks_per_chunk
        ),
        parallel=OffloadingParallelConfig(
            rank=0,
            world_size=1,
            tp_size=1,
            pp_size=1,
            pcp_size=1,
            dcp_size=1,
            data_parallel_index=0,
            data_parallel_size=1,
            data_parallel_rank_local=0,
            is_parallelism_agnostic=True,
        ),
    )


def test_tiering_spec_builds_manager_through_upstream(tmp_path) -> None:
    """Drive upstream's get_manager: catches signature drift at the region
    call site, and a silent fall back to upstream's fs tier."""
    spec = MetalTieringOffloadingSpec(
        _offloading_config(
            extra_config={
                "cpu_bytes_to_use": 64 << 20,
                "secondary_tiers": [
                    {"type": "fs", "root_dir": str(tmp_path), "max_size_gib": 1}
                ],
            }
        )
    )
    manager = spec.get_manager()
    assert manager is not None
    (tier,) = manager.secondary_tiers
    assert isinstance(tier, MetalFileSystemTierManager)


def test_reset_empties_the_host_pool_index() -> None:
    """Sleep frees the pool's pages. That is only safe because the reset
    vLLM runs first leaves no hit that would read them."""
    manager = MetalTieringOffloadingSpec(_offloading_config()).get_manager()
    ctx = ReqContext(req_id="r-sleep")
    manager.on_new_request(ctx)
    key = make_offload_key(hashlib.sha256(b"slept").digest(), 0)
    assert manager.prepare_store([key], ctx) is not None
    manager.complete_store([key], ctx)
    assert manager.lookup(key, ctx) is LookupResult.HIT

    manager.reset_cache()
    assert manager.lookup(key, ctx) is LookupResult.MISS


def test_events_reach_the_manager_from_the_disk_tier(tmp_path) -> None:
    """A stored block must surface as an event at the manager, where
    a KV-aware router reads it, not only at the tier."""
    spec = MetalTieringOffloadingSpec(
        _offloading_config(
            extra_config={
                "cpu_bytes_to_use": 64 << 20,
                "secondary_tiers": [
                    {
                        "type": "fs",
                        "root_dir": str(tmp_path),
                        "max_size_gib": 1,
                        "enable_kv_events": True,
                    }
                ],
            },
            enable_kv_cache_events=True,
        )
    )
    manager = spec.get_manager()
    (tier,) = manager.secondary_tiers
    assert tier.events is not None, "events not enabled on the tier"

    key = hashlib.sha256(b"routed").digest() + (0).to_bytes(4, "big")
    tier.submit_store(
        TransferJob(
            job_id=1,
            keys=[key],
            chunk_ids=[0],
            req_context=ReqContext(req_id="r-route"),
            is_promotion=False,
        )
    )
    tier.drain_jobs()
    list(tier.get_finished_jobs())

    events = list(manager.take_events())
    assert events, "no event reached the manager; a router would see nothing"
    assert any(key in e.keys for e in events)
