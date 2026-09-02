# SPDX-License-Identifier: Apache-2.0
"""Tests for the tiering (shared-region) variant of Metal KV offloading.

The tiering worker carves its host pool out of a MetalSharedOffloadRegion —
anonymous RAM shared between the scheduler-side manager and the worker-side
worker within one process — so that secondary tiers (disk) can address whole
CPU blocks as raw byte rows via ``create_kv_memoryview``. These tests
exercise the worker round-trip through the region and the byte-level
contract the fs tier relies on: store via the worker, read/write the same
block through the memoryview, load back via the worker.
"""

import mmap
import uuid

import mlx.core as mx
import numpy as np
import pytest
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from tests.kv_offload_helpers import (
    gpu_block_bytes,
    gpu_spec,
    make_cache,
    randomize,
    snapshot_blocks,
    zero_blocks,
)
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion
from vllm_metal.v1.kv_offload.worker import MetalKVOffloadWorker

NUM_CPU_BLOCKS = 4
FACTOR = 2


def _region_geometry(cache: MetalPagedKVCache) -> tuple[int, int]:
    raw_row = gpu_block_bytes(cache) * FACTOR
    aligned_row = -(-raw_row // mmap.PAGESIZE) * mmap.PAGESIZE
    return aligned_row, raw_row


def _make_regions(
    cache: MetalPagedKVCache,
) -> tuple[MetalSharedOffloadRegion, MetalSharedOffloadRegion]:
    """Worker-rank and scheduler-rank attachments to one shared region."""
    aligned_row, raw_row = _region_geometry(cache)
    instance = f"test-{uuid.uuid4().hex[:8]}"
    worker = MetalSharedOffloadRegion(
        engine_id=instance,
        num_blocks=NUM_CPU_BLOCKS,
        rank=0,
        kv_bytes_per_block=aligned_row,
        cpu_page_size=raw_row,
    )
    scheduler = MetalSharedOffloadRegion(
        engine_id=instance,
        num_blocks=NUM_CPU_BLOCKS,
        rank=None,
        kv_bytes_per_block=aligned_row,
        cpu_page_size=raw_row,
    )
    return worker, scheduler


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
def test_region_backed_roundtrip(dtype: mx.Dtype) -> None:
    cache = make_cache(dtype)
    worker_region, scheduler_region = _make_regions(cache)
    worker = MetalKVOffloadWorker(
        cache,
        block_size_factor=FACTOR,
        num_cpu_blocks=NUM_CPU_BLOCKS,
        region=worker_region,
    )
    try:
        randomize(cache)
        original = snapshot_blocks(cache, [1, 2, 5, 6])

        assert worker.submit_store(1, gpu_spec([1, 2, 5, 6]), CPULoadStoreSpec([0, 2]))
        assert len(worker.get_finished()) == 1

        zero_blocks(cache, [1, 2, 5, 6])
        assert worker.submit_load(2, CPULoadStoreSpec([0, 2]), gpu_spec([1, 2, 5, 6]))
        assert len(worker.get_finished()) == 1

        restored = snapshot_blocks(cache, [1, 2, 5, 6])
        for a, b in zip(original, restored, strict=True):
            np.testing.assert_array_equal(a, b)
    finally:
        worker.shutdown()
        scheduler_region.cleanup()


def test_memoryview_sees_worker_stores_and_feeds_loads() -> None:
    """Byte-level contract used by the fs tier: whole-block rows round-trip
    through the scheduler-side memoryview."""
    cache = make_cache()
    worker_region, scheduler_region = _make_regions(cache)
    worker = MetalKVOffloadWorker(
        cache,
        block_size_factor=FACTOR,
        num_cpu_blocks=NUM_CPU_BLOCKS,
        region=worker_region,
    )
    view = scheduler_region.create_kv_memoryview()
    flat = view.cast("B")
    try:
        randomize(cache)
        original = snapshot_blocks(cache, [3, 4])

        # Handler stores GPU blocks 3,4 into CPU block 1...
        assert worker.submit_store(1, gpu_spec([3, 4]), CPULoadStoreSpec([1]))
        worker.get_finished()

        # ...the scheduler-side view of CPU block 1 must carry those bytes.
        # Address it exactly like the fs tier does (tiering/fs/io.py): flat
        # byte cast, block b at [b * strides[0], (b + 1) * strides[0]).
        assert view.strides is not None
        row_stride = view.strides[0]
        stored_bytes = bytes(flat[row_stride : 2 * row_stride])
        assert any(stored_bytes)  # not all zero

        # Simulate the fs tier: persist the row, scribble over the CPU
        # block, then restore the row through the same memoryview.
        flat[row_stride : 2 * row_stride] = b"\x00" * row_stride
        assert not any(bytes(flat[row_stride : 2 * row_stride]))
        flat[row_stride : 2 * row_stride] = stored_bytes

        # A worker load from that CPU block must reproduce the GPU blocks.
        zero_blocks(cache, [3, 4])
        assert worker.submit_load(2, CPULoadStoreSpec([1]), gpu_spec([3, 4]))
        worker.get_finished()

        restored = snapshot_blocks(cache, [3, 4])
        for a, b in zip(original, restored, strict=True):
            np.testing.assert_array_equal(a, b)
    finally:
        del flat, view
        worker.shutdown()
        scheduler_region.cleanup()


def test_region_registry_shares_and_releases() -> None:
    """Two attachments to one instance id share bytes; distinct ids do not;
    cleanup releases the registry entry only when the last ref drops."""
    cache = make_cache()
    aligned_row, raw_row = _region_geometry(cache)
    instance = f"test-{uuid.uuid4().hex[:8]}"

    def attach(rank):
        return MetalSharedOffloadRegion(
            engine_id=instance,
            num_blocks=NUM_CPU_BLOCKS,
            rank=rank,
            kv_bytes_per_block=aligned_row,
            cpu_page_size=raw_row,
        )

    a = attach(0)
    b = attach(None)
    a._base[0] = 42
    assert int(b._base[0]) == 42, "attachments must share the same buffer"

    other = MetalSharedOffloadRegion(
        engine_id=f"other-{uuid.uuid4().hex[:8]}",
        num_blocks=NUM_CPU_BLOCKS,
        rank=None,
        kv_bytes_per_block=aligned_row,
        cpu_page_size=raw_row,
    )
    assert int(other._base[0]) == 0, "distinct instance ids must not share"
    other.cleanup()

    a.cleanup()
    # Buffer still alive through b.
    assert int(b._base[0]) == 42
    b.cleanup()

    # A fresh attachment after full release gets a new zeroed buffer.
    c = attach(None)
    assert int(c._base[0]) == 0
    c.cleanup()


def test_region_geometry_mismatch_rejected() -> None:
    cache = make_cache()
    aligned_row, raw_row = _region_geometry(cache)
    instance = f"test-{uuid.uuid4().hex[:8]}"
    a = MetalSharedOffloadRegion(
        engine_id=instance,
        num_blocks=NUM_CPU_BLOCKS,
        rank=0,
        kv_bytes_per_block=aligned_row,
        cpu_page_size=raw_row,
    )
    try:
        with pytest.raises(AssertionError, match="mismatched geometry"):
            MetalSharedOffloadRegion(
                engine_id=instance,
                num_blocks=NUM_CPU_BLOCKS + 1,
                rank=None,
                kv_bytes_per_block=aligned_row,
                cpu_page_size=raw_row,
            )
    finally:
        a.cleanup()


def test_region_backed_roundtrip_turboquant() -> None:
    """TurboQuant packed + scale/zero arrays through the shared region and
    the byte-level memoryview contract the disk tier relies on — the region
    carve is most fragile exactly here (mixed dtypes, packed layouts)."""
    cache = make_cache(head_dim=64, turboquant=True, k_quant="q8_0", v_quant="q3_0")
    worker_region, scheduler_region = _make_regions(cache)
    worker = MetalKVOffloadWorker(
        cache,
        block_size_factor=FACTOR,
        num_cpu_blocks=NUM_CPU_BLOCKS,
        region=worker_region,
    )
    randomize(cache)
    gpu_blocks = [1, 2, 3, 4]
    original = snapshot_blocks(cache, gpu_blocks)
    assert worker.submit_store(1, gpu_spec(gpu_blocks), CPULoadStoreSpec([0, 2]))
    worker.get_finished()

    # Byte-level round trip through the scheduler-side memoryview (what the
    # fs tier reads/writes): copy row bytes out and back, then restore.
    # Access exactly as the fs tier does (io.py): flat byte cast + slice.
    mv = scheduler_region.create_kv_memoryview().cast("B")
    row_bytes = (
        scheduler_region.row_stride
        if hasattr(scheduler_region, "row_stride")
        else len(mv) // NUM_CPU_BLOCKS
    )
    row_copy = bytes(mv[0:row_bytes])
    mv[0:row_bytes] = b"\x00" * row_bytes
    mv[0:row_bytes] = row_copy

    zero_blocks(cache, gpu_blocks)
    assert worker.submit_load(2, CPULoadStoreSpec([0, 2]), gpu_spec(gpu_blocks))
    worker.get_finished()
    for a, b in zip(original, snapshot_blocks(cache, gpu_blocks), strict=True):
        np.testing.assert_array_equal(a, b)
    worker.shutdown()
    scheduler_region.cleanup()


def _offloading_config(
    blocks_per_chunk: int = 1,
    tokens_per_block: int = 16,
    extra_config: dict | None = None,
    enable_kv_cache_events: bool = False,
):
    """A real ``OffloadingConfig``, as vLLM hands to an offloading spec.

    Built directly rather than through ``build_offloading_config`` so the test
    needs no model download; the dataclass is upstream's, so a field rename
    still breaks this.
    """
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
                tokens_per_block=tokens_per_block, layer_names=("l0",)
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


@pytest.mark.parametrize("blocks_per_chunk", [1, 2, 4])
def test_gpu_blocks_per_offloaded_block_tracks_blocks_per_chunk(
    blocks_per_chunk: int,
) -> None:
    """The GPU-blocks-per-offloaded-block ratio must follow the config.

    Deriving it from tokens_per_hash pins it to 1 for every single-group
    model, which silently sizes the host pool short and makes a mid-chunk
    restore read the wrong sub-slot."""
    from vllm_metal.v1.kv_offload.spec import (
        MetalTieringOffloadingSpec,
        gpu_blocks_per_offloaded_block,
    )

    spec = MetalTieringOffloadingSpec(_offloading_config(blocks_per_chunk))
    assert gpu_blocks_per_offloaded_block(spec) == blocks_per_chunk


def test_tiering_spec_builds_manager_through_upstream(tmp_path) -> None:
    """get_manager() runs upstream's body with the Metal classes rebound.

    Constructing MetalSharedOffloadRegion by hand cannot catch a signature
    drift against the upstream call site; only driving get_manager() can.

    The isinstance check matters as much as the construction. The fs tier is
    selected by module_path and the region by rebinding the tiering module's
    SharedOffloadRegion. If upstream ever resolves either differently, the
    routing silently no-ops and production quietly gets the upstream tier:
    buffered I/O, 0o644 files, none of the macOS work. Nothing else would
    report it."""
    from vllm_metal.v1.kv_offload.fs_tier import MetalFileSystemTierManager
    from vllm_metal.v1.kv_offload.spec import MetalTieringOffloadingSpec

    spec = MetalTieringOffloadingSpec(
        _offloading_config(
            extra_config={
                "cpu_bytes_to_use": 64 << 20,
                "secondary_tiers": [{"type": "fs", "root_dir": str(tmp_path)}],
            }
        )
    )
    manager = spec.get_manager()
    assert manager is not None
    (tier,) = manager.secondary_tiers
    assert isinstance(tier, MetalFileSystemTierManager)


def test_host_pool_larger_than_ram_is_refused() -> None:
    """A pool the machine cannot hold must fail at construction.

    np.zeros pages in lazily, so an oversized pool starts cleanly and only
    thrashes once blocks are written. On a fixed-RAM Mac that wedges the
    machine instead of failing the request. Upstream guards the equivalent
    with check_shm_free_space before ftruncate."""
    import psutil
    import pytest

    from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion

    total = psutil.virtual_memory().total
    with pytest.raises(ValueError, match="pageable"):
        MetalSharedOffloadRegion(
            engine_id="test-oversized",
            num_blocks=1,
            rank=0,
            kv_bytes_per_block=total,  # the whole machine, in one block
            cpu_page_size=total,
        )


def test_events_reach_the_manager_from_the_disk_tier(tmp_path) -> None:
    """A stored block must surface as an event at the manager, not just at
    the tier.

    This is the path KV-aware routing consumes: the scheduler drains
    OffloadingConnector.take_events(), which delegates to the manager, which
    yields from the primary tier and every secondary tier. The tier-level
    test proves the tier records the event; only this proves it survives the
    hop the router actually reads from.

    Pinned because a Metal-specific submit_store override once dropped the
    bookkeeping this is built from, and nothing failed."""
    import hashlib

    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.tiering.base import TransferJob

    from vllm_metal.v1.kv_offload.spec import MetalTieringOffloadingSpec

    spec = MetalTieringOffloadingSpec(
        _offloading_config(
            extra_config={
                "cpu_bytes_to_use": 64 << 20,
                "secondary_tiers": [
                    {
                        "type": "fs",
                        "root_dir": str(tmp_path),
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
            block_ids=[0],
            req_context=ReqContext(req_id="r-route"),
            is_promotion=False,
        )
    )
    tier.drain_jobs()
    list(tier.get_finished_jobs())

    events = list(manager.take_events())
    assert events, "no event reached the manager; a router would see nothing"
    assert any(key in e.keys for e in events)
