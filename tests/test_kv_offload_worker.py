# SPDX-License-Identifier: Apache-2.0
"""Tests for the Metal KV offloading worker."""

import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import mlx.core as mx
import numpy as np
import pytest
import torch
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.offloading.common import (
    OffloadingConnectorMetadata,
    TransferJob,
)
from vllm.engine.arg_utils import EngineArgs
from vllm.v1.kv_offload.base import LookupResult, ReqContext, make_offload_key
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from tests.kv_offload_helpers import (
    device_ptrs,
    gpu_block_bytes,
    gpu_spec,
    make_region,
    make_storage,
    make_turboquant_storage,
    make_worker,
    randomize,
    snapshot_blocks,
    zero_blocks,
)
from tests.tiny_configs import TINY_QWEN3_CONFIG
from vllm_metal.config import reset_config
from vllm_metal.v1.kv_offload.connector import MetalOffloadingConnector
from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion

ALIGN = MetalSharedOffloadRegion.BLOCK_SIZE_ALIGNMENT

# 1 MiB, 64 host chunks of the test storage's 16 KiB blocks.
_POOL_GIB = 1 / 1024


def _roundtrip(
    storage,
    gpu_blocks: list[int],
    cpu_chunks: list[int],
    *,
    blocks_per_chunk: int = 1,
    block_index: int = 0,
    blocks_per_slice: int | None = None,
) -> None:
    worker = make_worker(storage, blocks_per_chunk=blocks_per_chunk, num_cpu_chunks=16)
    if blocks_per_slice is not None:
        worker._blocks_per_slice = blocks_per_slice
    randomize(storage)
    original = snapshot_blocks(storage, gpu_blocks)
    ptrs = device_ptrs(worker, gpu_blocks, block_index)

    assert worker.submit_store(1, ptrs, CPULoadStoreSpec(cpu_chunks))
    zero_blocks(storage, gpu_blocks)
    assert worker.submit_load(2, CPULoadStoreSpec(cpu_chunks), ptrs)
    assert [r.job_id for r in worker.get_finished()] == [1, 2]

    for a, b in zip(original, snapshot_blocks(storage, gpu_blocks), strict=True):
        np.testing.assert_array_equal(a, b)


def _backing_bytes(storage) -> np.ndarray:
    """A copy of vLLM's allocation, read through torch rather than MLX."""
    tensor = next(iter(storage.tensors.values()))
    raw = torch.empty(0, dtype=torch.uint8).set_(tensor.untyped_storage())
    return raw.numpy().copy()


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16, torch.float32])
def test_roundtrip_dtypes(dtype: torch.dtype) -> None:
    _roundtrip(make_storage(dtype), [5, 2, 7], [0, 1, 3])


def test_roundtrip_turboquant() -> None:
    _roundtrip(make_turboquant_storage(), [1, 6], [2, 0])


@pytest.mark.parametrize("blocks_per_chunk", [1, 2])
def test_roundtrip_mla(blocks_per_chunk: int) -> None:
    _roundtrip(
        make_storage(mla=True, head_dim=36),
        [3, 4, 5, 6],
        [1, 3, 0, 2][: 4 // blocks_per_chunk],
        blocks_per_chunk=blocks_per_chunk,
    )


def test_mla_latent_cache_reads_loaded_rows() -> None:
    """MLA attention reads its own latent views of the storage. A store must
    see rows it wrote, and a load must reach what it reads."""
    from vllm_metal.attention.caches.mla_cache import MLAPagedLatentCache

    storage = make_storage(mla=True, head_dim=36)
    cache = MLAPagedLatentCache.from_upstream(storage, list(storage.tensors))
    worker = make_worker(storage)
    block, block_size = 2, storage.config.kv_cache_groups[0].kv_cache_spec.block_size
    slots = mx.arange(block * block_size, (block + 1) * block_size)
    for layer in range(cache.num_layers):
        rows = mx.random.normal((block_size, cache.latent_dim)) + layer
        cache.write_slots(layer, slots, rows)
    mx.eval(*cache.latent_caches)
    written = [np.array(latent[block]) for latent in cache.latent_caches]

    assert worker.submit_store(1, device_ptrs(worker, [block]), CPULoadStoreSpec([0]))
    zero_blocks(storage, [block])
    mx.eval(*cache.latent_caches)
    assert not np.array_equal(np.array(cache.latent_caches[0][block]), written[0])
    assert worker.submit_load(2, CPULoadStoreSpec([0]), device_ptrs(worker, [block]))

    for latent, expected in zip(cache.latent_caches, written, strict=True):
        np.testing.assert_array_equal(np.array(latent[block]), expected)


def test_roundtrip_chunk_spans_blocks() -> None:
    _roundtrip(make_storage(), [3, 4, 5, 6], [1, 3], blocks_per_chunk=2)


def test_roundtrip_unaligned_first_chunk() -> None:
    # Logical block 3 with 2 blocks per chunk lands in sub-slot 1.
    _roundtrip(make_storage(), [5], [2], blocks_per_chunk=2, block_index=3)


def test_transfers_are_sliced() -> None:
    # Slices of 3 straddle chunks of 4 and start mid-chunk.
    blocks = list(range(1, 31))
    cpu = [9, 2, 14, 0, 7, 11, 5, 3]
    _roundtrip(
        make_storage(num_blocks=64),
        blocks,
        cpu,
        blocks_per_chunk=4,
        block_index=2,
        blocks_per_slice=3,
    )


def test_expedited_load_is_finished_when_waited_on() -> None:
    # vLLM's synchronous loads submit with expedited=True, then wait.
    storage = make_storage()
    worker = make_worker(storage, blocks_per_chunk=1, num_cpu_chunks=16)
    randomize(storage)
    original = snapshot_blocks(storage, [5, 2])
    ptrs = device_ptrs(worker, [5, 2])
    assert worker.submit_store(1, ptrs, CPULoadStoreSpec([0, 1]))
    zero_blocks(storage, [5, 2])

    assert worker.submit_load(2, CPULoadStoreSpec([0, 1]), ptrs, expedited=True)
    worker.wait({2})

    assert [r.job_id for r in worker.get_finished()] == [1, 2]
    for a, b in zip(original, snapshot_blocks(storage, [5, 2]), strict=True):
        np.testing.assert_array_equal(a, b)


def test_load_peak_memory_is_bounded() -> None:
    """A load holds one slice at a time, not the whole job."""
    storage = make_storage(
        num_layers=4, num_kv_heads=8, head_dim=128, num_blocks=256, block_size=16
    )
    worker = make_worker(storage, num_cpu_chunks=256)
    worker._blocks_per_slice = 16
    blocks = list(range(256))
    job = gpu_block_bytes(storage) * len(blocks)
    randomize(storage)
    assert worker.submit_store(1, device_ptrs(worker, blocks), CPULoadStoreSpec(blocks))
    zero_blocks(storage, blocks)

    mx.reset_peak_memory()
    base = mx.get_active_memory()
    assert worker.submit_load(2, CPULoadStoreSpec(blocks), device_ptrs(worker, blocks))
    assert mx.get_peak_memory() - base < job // 4


def test_load_reaches_the_storage_allocation() -> None:
    """Read back through torch: an MLX index-assign on a page view would pass
    an MLX-side read but never reach these bytes."""
    storage = make_storage()
    worker = make_worker(storage)
    randomize(storage)
    before = _backing_bytes(storage)

    blocks = [1, 4, 6]
    assert worker.submit_store(
        1, device_ptrs(worker, blocks), CPULoadStoreSpec([0, 1, 2])
    )
    zero_blocks(storage, blocks)
    assert not np.array_equal(_backing_bytes(storage), before)

    assert worker.submit_load(
        2, CPULoadStoreSpec([0, 1, 2]), device_ptrs(worker, blocks)
    )
    np.testing.assert_array_equal(_backing_bytes(storage), before)


def test_load_is_ordered_after_pending_zeroing() -> None:
    """Zeroing shipped in an earlier step may still be pending in the graph;
    the load must land after it."""
    storage = make_storage()
    worker = make_worker(storage)
    randomize(storage)
    blocks = [2, 5]
    original = snapshot_blocks(storage, blocks)
    assert worker.submit_store(1, device_ptrs(worker, blocks), CPULoadStoreSpec([0, 1]))

    storage.zero_blocks(blocks)  # deliberately not evaluated
    assert worker.submit_load(2, CPULoadStoreSpec([0, 1]), device_ptrs(worker, blocks))
    mx.eval(*storage.buffers)

    for a, b in zip(original, snapshot_blocks(storage, blocks), strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("kind", ["fp16", "turboquant", "mla"])
def test_pool_mirrors_every_cache_byte(kind: str) -> None:
    storage = {
        "fp16": make_storage,
        "turboquant": make_turboquant_storage,
        "mla": lambda: make_storage(mla=True, head_dim=36),
    }[kind]()
    worker = make_worker(storage)
    mirrored = sum(host.shape[2] for host in worker._host)
    assert mirrored * storage.config.num_blocks == storage.nbytes


def test_store_snapshots_the_host_pool() -> None:
    """MLX is lazy: a store must copy the bytes it was called with, or it
    could land the next step's KV instead."""
    storage = make_storage()
    worker = make_worker(storage)
    randomize(storage)
    gpu_blocks, cpu_chunks = [5, 2], [0, 1]
    assert worker.submit_store(
        1, device_ptrs(worker, gpu_blocks), CPULoadStoreSpec(cpu_chunks)
    )
    pooled = [host[cpu_chunks, 0].copy() for host in worker._host]

    zero_blocks(storage, gpu_blocks)

    for host, before in zip(worker._host, pooled, strict=True):
        assert before.any()
        np.testing.assert_array_equal(host[cpu_chunks, 0], before)


def test_transfer_size_counts_every_page_byte() -> None:
    storage = make_storage()
    worker = make_worker(storage, blocks_per_chunk=2)
    blocks = [1, 4, 6]
    expected = len(blocks) * gpu_block_bytes(storage)
    assert worker.submit_store(
        1, device_ptrs(worker, blocks, 1), CPULoadStoreSpec([0, 2])
    )
    assert worker.submit_load(
        2, CPULoadStoreSpec([0, 2]), device_ptrs(worker, blocks, 1)
    )
    assert [r.transfer_size for r in worker.get_finished()] == [expected, expected]


def test_empty_group_is_a_noop() -> None:
    worker = make_worker(make_storage())
    assert worker.submit_store(1, device_ptrs(worker, []), CPULoadStoreSpec([]))
    (result,) = worker.get_finished()
    assert result.success and result.transfer_size == 0


def test_load_rejects_mismatched_block_indices() -> None:
    """Resolving the pointers asserts the length match, but asserts vanish
    under -O, so _slices re-checks before trusting block_indices[0]."""
    storage = make_storage()
    worker = make_worker(storage)
    ptrs = replace(device_ptrs(worker, [1, 2]), block_indices=())

    with pytest.raises(ValueError, match="block_indices"):
        worker.submit_load(1, CPULoadStoreSpec([0, 1]), ptrs)


def test_load_rejects_negative_block_index() -> None:
    storage = make_storage()
    worker = make_worker(storage)
    ptrs = device_ptrs(worker, [1, 2], block_index=-1)

    with pytest.raises(ValueError, match="non-negative"):
        worker.submit_load(1, CPULoadStoreSpec([0, 1]), ptrs)


def test_unknown_spec_types_raise() -> None:
    worker = make_worker(make_storage())
    with pytest.raises(ValueError, match="unexpected load spec types"):
        worker.submit_load(1, CPULoadStoreSpec([0]), CPULoadStoreSpec([1]))
    with pytest.raises(ValueError, match="unexpected store spec types"):
        worker.submit_store(2, CPULoadStoreSpec([0]), CPULoadStoreSpec([1]))
    assert worker.get_finished() == []


def test_block_id_bounds_are_exact() -> None:
    storage = make_storage()
    worker = make_worker(storage)
    n = storage.config.num_blocks
    # A pointer past the last block, and one below the first.
    for bad in ([n], [-1]):
        with pytest.raises(ValueError, match="GPU block ids out of range"):
            worker.submit_store(1, device_ptrs(worker, bad), CPULoadStoreSpec([0]))
        with pytest.raises(ValueError, match="GPU block ids out of range"):
            worker.submit_load(1, CPULoadStoreSpec([0]), device_ptrs(worker, bad))
    for bad in ([4], [-1]):
        with pytest.raises(ValueError, match="CPU chunk ids out of range"):
            worker.submit_store(1, device_ptrs(worker, [0]), CPULoadStoreSpec(bad))
    assert worker.get_finished() == []
    assert worker.submit_store(1, device_ptrs(worker, [n - 1]), CPULoadStoreSpec([3]))


def test_multi_group_device_pointers_raise() -> None:
    worker = make_worker(make_storage())
    ptrs = device_ptrs(worker, [1, 2])
    two_groups = replace(ptrs, group_block_counts=(2, 0), block_indices=(0, 0))
    with pytest.raises(ValueError, match="single KV cache group"):
        worker.submit_store(1, two_groups, CPULoadStoreSpec([0, 1]))


def test_duplicate_load_destinations_raise() -> None:
    worker = make_worker(make_storage())
    with pytest.raises(ValueError, match="duplicate GPU block ids"):
        worker.submit_load(1, CPULoadStoreSpec([0, 1]), device_ptrs(worker, [5, 5]))


@pytest.mark.parametrize(
    ("blocks", "skip"), [([1, 2, 3], 0), ([1, 2], 1)], ids=["count", "skipped-head"]
)
def test_too_few_cpu_chunks_raise(blocks: list[int], skip: int) -> None:
    worker = make_worker(make_storage(), blocks_per_chunk=2)
    ptrs = device_ptrs(worker, blocks, skip)
    with pytest.raises(ValueError, match="CPU chunks too few"):
        worker.submit_store(1, ptrs, CPULoadStoreSpec([0]))


@pytest.mark.parametrize(
    ("delta", "ok"),
    [(0, True), (ALIGN - 1, True), (-1, False), (ALIGN, False)],
    ids=["exact", "padding", "overrun", "layout-disagreement"],
)
def test_chunk_size_must_match_the_carve(delta: int, ok: bool) -> None:
    """The spec pads chunks by less than one alignment unit; anything else is
    a layout disagreement. Two blocks per chunk, so sub-slots are counted."""
    storage = make_storage()
    size = gpu_block_bytes(storage) * 2 + delta
    if ok:
        make_worker(storage, blocks_per_chunk=2, kv_bytes_per_chunk=size)
        return
    with pytest.raises(ValueError, match="host pool carves"):
        make_worker(storage, blocks_per_chunk=2, kv_bytes_per_chunk=size)


def test_region_chunk_count_mismatch_raises() -> None:
    storage = make_storage()
    region = make_region(storage, num_cpu_chunks=4)
    try:
        with pytest.raises(ValueError, match="region has 4 chunks"):
            make_worker(storage, num_cpu_chunks=2, region=region)
    finally:
        region.cleanup()


def test_shut_down_worker_raises() -> None:
    worker = make_worker(make_storage())
    worker.shutdown()
    with pytest.raises(RuntimeError, match="shut down"):
        worker.submit_load(1, CPULoadStoreSpec([0]), device_ptrs(worker, [1]))


class TestConnectorHostPool:
    """Sleep and wake do nothing without a Metal offload worker."""

    def _connector(self, worker) -> MetalOffloadingConnector:
        connector = MetalOffloadingConnector.__new__(MetalOffloadingConnector)
        connector.connector_worker = SimpleNamespace(worker=worker)
        return connector

    def test_ignores_a_worker_of_another_type(self) -> None:
        connector = self._connector(MagicMock(spec=[]))
        connector.release_host_pool()
        connector.restore_host_pool()

    def test_without_a_worker_is_a_no_op(self) -> None:
        connector = MetalOffloadingConnector.__new__(MetalOffloadingConnector)
        connector.connector_worker = None
        connector.release_host_pool()
        connector.restore_host_pool()


def _offloading_engine_config(model_dir):
    """A real engine config with --kv-offloading-size, built offline."""
    (model_dir / "config.json").write_text(json.dumps(TINY_QWEN3_CONFIG))
    reset_config()
    try:
        return EngineArgs(
            model=str(model_dir),
            max_model_len=128,
            skip_tokenizer_init=True,
            block_size=16,
            kv_offloading_size=_POOL_GIB,
            distributed_executor_backend="uni",
        ).create_engine_config()
    finally:
        reset_config()


def _store_then_load(connector, storage, blocks, chunks):
    """Store as job 1 through handle_preemptions, load as job 2 through
    start_load_kv."""
    randomize(storage)
    original = snapshot_blocks(storage, blocks)
    store = TransferJob(
        req_id="r", src_spec=gpu_spec(blocks), dst_spec=CPULoadStoreSpec(chunks)
    )
    load = TransferJob(
        req_id="r", src_spec=CPULoadStoreSpec(chunks), dst_spec=gpu_spec(blocks)
    )
    connector.handle_preemptions(
        OffloadingConnectorMetadata(
            load_jobs={}, store_jobs={1: store}, jobs_to_flush={1}
        )
    )
    zero_blocks(storage, blocks)
    connector.bind_connector_metadata(
        OffloadingConnectorMetadata(load_jobs={2: load}, store_jobs={})
    )
    connector.start_load_kv(None)
    return original, snapshot_blocks(storage, blocks)


class TestConnectorRegistration:
    """Upstream's connector worker drives the Metal worker it registers."""

    @pytest.fixture
    def offloading(self, tmp_path, monkeypatch):
        # Below the MLX_MAX_MB_PER_BUFFER threshold, so the hook leaves the
        # environment alone on any host.
        monkeypatch.setattr(
            "vllm_metal.platform.psutil.virtual_memory",
            lambda: SimpleNamespace(total=64 * (1 << 30), available=48 * (1 << 30)),
        )
        vllm_config = _offloading_engine_config(tmp_path)
        storage = make_storage(num_kv_heads=2, head_dim=64, block_size=16)
        # Both roles share one process, as under the uni executor.
        scheduler = MetalOffloadingConnector(
            vllm_config, KVConnectorRole.SCHEDULER, storage.config
        )
        connector = MetalOffloadingConnector(
            vllm_config, KVConnectorRole.WORKER, storage.config
        )
        connector.register_kv_caches(storage)
        yield connector, storage, scheduler
        connector.shutdown()
        scheduler.shutdown()

    def test_store_and_load_round_trip_through_upstreams_worker(
        self, offloading
    ) -> None:
        connector, storage, _ = offloading

        original, restored = _store_then_load(connector, storage, [5, 2, 7], [0, 1, 2])
        finished = connector.get_finished(set())
        worker_meta = connector.build_connector_worker_meta()

        for a, b in zip(original, restored, strict=True):
            np.testing.assert_array_equal(a, b)
        assert finished == (set(), {"r"})
        assert worker_meta is not None
        assert worker_meta.completed_jobs == {1: 1, 2: 1}

    def test_store_and_load_round_trip_through_upstreams_worker_after_sleep(
        self, offloading
    ) -> None:
        connector, storage, _ = offloading
        registered = connector.connector_worker.worker.kv_caches.tensors[0].tensor

        storage.release()
        connector.release_host_pool()
        asleep_nbytes = registered.untyped_storage().nbytes()
        connector.restore_host_pool()
        storage.reallocate()
        original, restored = _store_then_load(connector, storage, [3, 4], [0, 1])

        assert (asleep_nbytes, registered.untyped_storage().nbytes()) == (
            0,
            storage.nbytes,
        )
        for a, b in zip(original, restored, strict=True):
            np.testing.assert_array_equal(a, b)

    def test_sleep_and_wake_reach_the_worker_region(
        self, offloading, monkeypatch
    ) -> None:
        connector, _, _ = offloading
        region = connector.connector_worker.worker.region
        calls: list[str] = []
        monkeypatch.setattr(region, "release_pages", lambda: calls.append("release"))
        monkeypatch.setattr(region, "reuse_pages", lambda: calls.append("reuse"))

        connector.release_host_pool()
        connector.restore_host_pool()

        assert calls == ["release", "reuse"]

    @pytest.mark.parametrize("reset_connector", [True, False])
    def test_release_host_pool_leaves_no_offload_hit_when_the_pool_is_freed(
        self, offloading, monkeypatch, reset_connector
    ) -> None:
        """vLLM resets the connector first only with reset_connector=True."""
        connector, _, scheduler = offloading
        manager = scheduler.connector_scheduler.manager
        ctx = ReqContext(req_id="r-sleep")
        manager.on_new_request(ctx)
        key = make_offload_key(hashlib.sha256(b"slept").digest(), 0)
        manager.prepare_store([key], ctx)
        manager.complete_store([key], ctx)
        before = manager.lookup(key, ctx)
        region = connector.connector_worker.worker.region
        free_pages = region.release_pages
        at_release = []

        def release_pages() -> None:
            at_release.append(manager.lookup(key, ctx))
            free_pages()

        monkeypatch.setattr(region, "release_pages", release_pages)

        if reset_connector:
            scheduler.reset_cache()
        connector.release_host_pool()

        assert (before, at_release) == (LookupResult.HIT, [LookupResult.MISS])
