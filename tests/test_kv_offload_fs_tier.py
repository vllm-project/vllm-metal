# SPDX-License-Identifier: Apache-2.0
"""Tests for the Metal fs secondary tier (vllm_metal/v1/kv_offload/fs_tier.py)."""

import hashlib
import os
import stat
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import vllm.v1.kv_offload.tiering.spec as tiering_spec_module
from vllm.v1.kv_offload.base import LookupResult, ReqContext
from vllm.v1.kv_offload.file_mapper import FileMapper
from vllm.v1.kv_offload.tiering.base import ScheduleEndContext, TransferJob
from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
from vllm.v1.kv_offload.tiering.fs import io as upstream_io

from vllm_metal.v1.kv_offload import fs_tier
from vllm_metal.v1.kv_offload.fs_tier import (
    EVICTED_BLOCKS_METRIC,
    NOINDEX_DIRNAME,
    STORE_BYTES_METRIC,
    TMP_REAP_AGE_S,
    MetalFileSystemTierManager,
    _make_private_dir,
    prepare_root_dir,
)
from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion
from vllm_metal.v1.kv_offload.spec import (
    MetalTieringOffloadingSpec,
    _metal_shared_region,
    route_fs_tiers_to_metal,
)

BLOCK = 4096


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def _key(label: bytes) -> bytes:
    return hashlib.sha256(label).digest() + (0).to_bytes(4, "big")


def _fake_spec(block_bytes: int) -> SimpleNamespace:
    return SimpleNamespace(
        blocks_per_chunk=1,
        kv_bytes_per_chunk=block_bytes,
        # Read by upstream's __init__ only when enable_kv_events is set.
        kv_events_config=SimpleNamespace(enable_kv_cache_events=True),
        config=SimpleNamespace(
            engine_id="test-engine",
            replicated_layout=False,
            canonical_layout=False,
            extra_config={},
            model=SimpleNamespace(name="test-model", dtype="float16"),
            cache=SimpleNamespace(tokens_per_hash=16, blocks_per_chunk=1),
            parallel=SimpleNamespace(
                tp_size=1,
                pp_size=1,
                pcp_size=1,
                dcp_size=1,
                rank=0,
                is_parallelism_agnostic=True,
            ),
            groups=(SimpleNamespace(tokens_per_block=16, layer_names=("layer0",)),),
        ),
    )


def _tier(tmp_path, block_bytes=BLOCK, num_blocks=4, pool=None, **kwargs):
    if pool is None:
        pool = np.zeros((num_blocks, block_bytes), dtype=np.uint8)
    return MetalFileSystemTierManager(
        offloading_spec=_fake_spec(block_bytes),
        primary_kv_view=memoryview(pool),
        tier_type="fs",
        root_dir=str(tmp_path / "kv-store"),
        n_read_threads=2,
        n_write_threads=2,
        **kwargs,
    )


def _job(job_id, keys, chunk, promotion=False) -> TransferJob:
    return TransferJob(
        job_id=job_id,
        keys=keys,
        chunk_ids=np.array([chunk] * len(keys)),
        is_promotion=promotion,
        req_context=ReqContext(req_id=f"r-{job_id}"),
    )


def _finish(tier) -> list[tuple[int, bool]]:
    tier.drain_jobs()
    return [(r.job_id, r.success) for r in tier.get_finished_jobs()]


def _store_keys(tier, keys, first_job_id: int = 100) -> None:
    """Store each key from pool block 0, one job each."""
    for i, key in enumerate(keys):
        tier.submit_store(_job(first_job_id + i, [key], 0))
        assert _finish(tier) == [(first_job_id + i, True)]


def _lookup(tier, key, ctx) -> LookupResult:
    """Resolve an async lookup: submit, flush, wait for the worker's result."""
    result = tier.lookup(key, ctx)
    if result is not LookupResult.RETRY:
        return result
    tier.on_schedule_end(ScheduleEndContext(set(), set()))
    pending = tier._lookup_manager._pending_results
    pending.put(pending.get(timeout=10))
    return tier.lookup(key, ctx)


def test_prepare_root_dir(tmp_path: Path) -> None:
    root = tmp_path / "kv-store"
    store_dir = prepare_root_dir(str(root))

    assert store_dir == str(root / NOINDEX_DIRNAME)
    assert _mode(root) == 0o700
    assert _mode(Path(store_dir)) == 0o700
    assert prepare_root_dir(str(root)) == store_dir  # idempotent


def test_prepare_root_dir_respects_noindex_name(tmp_path: Path) -> None:
    root = tmp_path / "kv-store.noindex"
    assert prepare_root_dir(str(root)) == str(root)
    assert _mode(root) == 0o700


def test_make_private_dir_does_not_chmod_existing(tmp_path: Path) -> None:
    pre = tmp_path / "shared"
    pre.mkdir()
    os.chmod(pre, 0o755)
    _make_private_dir(str(pre))
    assert _mode(pre) == 0o755

    fresh = tmp_path / "fresh"
    _make_private_dir(str(fresh))
    assert _mode(fresh) == 0o700


def test_root_dir_owned_by_another_user_is_refused(tmp_path: Path, monkeypatch) -> None:
    root = tmp_path / "theirs"
    root.mkdir()
    monkeypatch.setattr("os.geteuid", lambda: os.stat(root).st_uid + 1)
    with pytest.raises(PermissionError, match="owned by uid"):
        _make_private_dir(str(root))


def test_fs_tier_config_routes_to_metal_class() -> None:
    extra = {"secondary_tiers": [{"type": "fs", "root_dir": "/tmp/unused"}]}
    route_fs_tiers_to_metal(extra)
    tier = extra["secondary_tiers"][0]
    assert tier["module_path"] == "vllm_metal.v1.kv_offload.fs_tier"
    assert SecondaryTierFactory.get_tier_class(tier) is MetalFileSystemTierManager
    assert "MetalFileSystemTierManager" not in SecondaryTierFactory._registry


def test_metal_region_swapped_only_within_scope() -> None:
    original = tiering_spec_module.SharedOffloadRegion
    assert original is not MetalSharedOffloadRegion
    with _metal_shared_region():
        assert tiering_spec_module.SharedOffloadRegion is MetalSharedOffloadRegion
    assert tiering_spec_module.SharedOffloadRegion is original


def test_metric_definitions_resolve_metal_tier_classes(monkeypatch) -> None:
    """Runs before any spec instance exists, so it must route the config itself."""
    sentinel = {"metal_fs_metric": object()}
    monkeypatch.setattr(
        MetalFileSystemTierManager,
        "build_metric_definitions",
        classmethod(lambda cls, cfg: sentinel),
    )
    metrics = MetalTieringOffloadingSpec.build_metric_definitions(
        {"secondary_tiers": [{"type": "fs", "root_dir": "/tmp/unused"}]}
    )
    assert "metal_fs_metric" in metrics


def test_layout_signature_discriminates_turboquant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def config(turboquant, k_quant="q8_0", v_quant="q3_0"):
        cfg = SimpleNamespace(turboquant=turboquant, k_quant=k_quant, v_quant=v_quant)
        monkeypatch.setattr(fs_tier, "get_config", lambda: cfg)

    config(False)
    assert fs_tier.layout_signature() == ""

    config(True)
    tq = fs_tier.layout_signature()
    config(True, k_quant="q5_0")
    tq_k = fs_tier.layout_signature()
    config(True, v_quant="q2_0")
    tq_v = fs_tier.layout_signature()
    assert len({"", tq, tq_k, tq_v}) == 4

    config(True)
    assert fs_tier.layout_signature() == tq


def test_blocks_round_trip_through_disk(tmp_path: Path) -> None:
    pool = np.zeros((4, BLOCK), dtype=np.uint8)
    pool[0] = np.random.default_rng(0).integers(0, 256, BLOCK, dtype=np.uint8)
    tier = _tier(tmp_path, pool=pool)
    try:
        key = _key(b"round-trip")
        tier.submit_store(_job(1, [key], 0))
        assert _finish(tier) == [(1, True)]
        assert Path(tier.file_mapper.get_file_name(key)).stat().st_size == BLOCK

        tier.submit_load(_job(2, [key], 1, promotion=True))
        assert _finish(tier) == [(2, True)]
        np.testing.assert_array_equal(pool[1], pool[0])
    finally:
        tier.shutdown()


def test_restart_restores_only_the_loaded_weights(tmp_path: Path) -> None:
    q4, q8 = tmp_path / "q4.gguf", tmp_path / "q8.gguf"
    q4.write_bytes(b"q4")
    q8.write_bytes(b"q8")
    alias = tmp_path / "alias.gguf"
    alias.symlink_to(q4)
    key = _key(b"same prompt")

    def restart(weights: Path | None, expected: LookupResult) -> None:
        tier = _tier(tmp_path, gguf_weights_path=str(weights) if weights else None)
        try:
            assert _lookup(tier, key, ReqContext(req_id="probe")) is expected
            if expected is LookupResult.MISS:
                _store_keys(tier, [key])
        finally:
            tier.shutdown()

    for weights in (q4, q8, None):
        restart(weights, LookupResult.MISS)
    for weights in (alias, q8, None):
        restart(weights, LookupResult.HIT)
    q8.replace(q4)
    restart(alias, LookupResult.MISS)
    before = q4.stat()
    q4.write_bytes(b"v2")
    os.utime(q4, ns=(before.st_atime_ns, before.st_mtime_ns))
    restart(q4, LookupResult.MISS)


def test_legacy_block_is_evicted_without_being_restored(tmp_path: Path) -> None:
    key = _key(b"legacy prefix")
    store = prepare_root_dir(str(tmp_path / "kv-store"))
    legacy_mapper = FileMapper.from_offloading_spec(
        store, _fake_spec(BLOCK), parallel_agnostic=True
    )
    legacy = Path(legacy_mapper.get_file_name(key))
    legacy.parent.mkdir(parents=True)
    legacy.write_bytes(bytes(BLOCK))

    tier = _tier(tmp_path, max_size_gib=BLOCK / (1 << 30))
    try:
        assert _lookup(tier, key, ReqContext(req_id="probe")) is LookupResult.MISS
        _store_keys(tier, [key])
        assert not legacy.exists()
        assert Path(tier.file_mapper.get_file_name(key)).is_file()
    finally:
        tier.shutdown()


def test_failed_load_overrides_cached_hit(tmp_path: Path) -> None:
    """Upstream's failed-load negative cache (vllm#49328) still works here."""
    pool = np.zeros((4, BLOCK), dtype=np.uint8)
    pool[0] = 7
    tier = _tier(tmp_path, pool=pool, max_size_gib=0)
    try:
        key = _key(b"block-A")
        path = Path(tier.file_mapper.get_file_name(key))
        _store_keys(tier, [key])

        ctx = ReqContext(req_id="r-hit")
        assert _lookup(tier, key, ctx) is LookupResult.HIT
        path.write_bytes(b"x" * 10)  # corrupt after the cached HIT
        tier.submit_load(
            TransferJob(
                job_id=2,
                keys=[key],
                chunk_ids=np.array([1]),
                is_promotion=True,
                req_context=ctx,
            )
        )
        assert _finish(tier) == [(2, False)]
        assert tier.lookup(key, ctx) is LookupResult.MISS
        # Upstream deleted the short-read file; the cap stops counting it.
        assert not path.exists()
        assert tier._store_bytes == 0
        tier.on_request_finished(ctx)

        _store_keys(tier, [key], first_job_id=3)
        assert _lookup(tier, key, ReqContext(req_id="r-4")) is LookupResult.HIT
    finally:
        tier.shutdown()


def test_store_emits_block_stored_event(tmp_path: Path) -> None:
    """A KV-aware router learns what this instance holds from these events."""
    tier = _tier(tmp_path, enable_kv_events=True)
    try:
        assert tier.events is not None
        key = _key(b"evented")
        tier.submit_store(_job(1, [key], 0))
        assert _finish(tier) == [(1, True)]
        events = list(tier.take_events())
        assert events and key in events[0].keys
    finally:
        tier.shutdown()


def test_empty_jobs_complete_and_leave_no_bookkeeping(tmp_path: Path) -> None:
    tier = _tier(tmp_path)
    try:
        for job_id in range(4):
            job = _job(job_id, [], 0, promotion=bool(job_id % 2))
            if job.is_promotion:
                tier.submit_load(job)
            else:
                tier.submit_store(job)
        done = threading.Event()
        threading.Thread(
            target=lambda: (tier.drain_jobs(), done.set()), daemon=True
        ).start()
        assert done.wait(timeout=10), "drain_jobs() hung on a keyless job"
        assert len(list(tier.get_finished_jobs())) == 4
        assert tier._store_job_paths == {} and tier._load_job_paths == {}
        assert tier._load_job_keys == {}
    finally:
        tier.shutdown()


def test_store_cap_evicts_oldest_before_a_store(tmp_path: Path) -> None:
    tier = _tier(tmp_path, max_size_gib=2.5 * BLOCK / (1 << 30))
    try:
        a, b, c = _key(b"A"), _key(b"B"), _key(b"C")
        _store_keys(tier, [a, b])
        assert tier._store_bytes == 2 * BLOCK
        _store_keys(tier, [c], first_job_id=200)
        pa, pb, pc = (Path(tier.file_mapper.get_file_name(k)) for k in (a, b, c))
        assert not pa.exists()
        assert pb.exists() and pc.exists()
        assert tier._store_bytes == 2 * BLOCK <= tier._max_store_bytes
        assert _lookup(tier, a, ReqContext(req_id="r-a")) is LookupResult.MISS
    finally:
        tier.shutdown()


def _removed_keys(tier) -> list[bytes]:
    return [k for e in tier.take_events() if e.removed for k in e.keys]


def test_eviction_emits_block_removed_event(tmp_path: Path) -> None:
    """A KV-aware router must stop routing to this instance for evicted blocks."""
    cap = 2.5 * BLOCK / (1 << 30)
    tier = _tier(tmp_path, max_size_gib=cap, enable_kv_events=True)
    try:
        a, b, c = _key(b"A"), _key(b"B"), _key(b"C")
        _store_keys(tier, [a, b, c])
        assert _removed_keys(tier) == [a]
    finally:
        tier.shutdown()


def test_eviction_overrides_cached_hit(tmp_path: Path) -> None:
    """A live request keeps A's HIT cached; eviction must turn it into a miss."""
    tier = _tier(tmp_path, max_size_gib=2.5 * BLOCK / (1 << 30))
    try:
        a, b, c = _key(b"A"), _key(b"B"), _key(b"C")
        _store_keys(tier, [a])
        assert _lookup(tier, a, ReqContext(req_id="x")) is LookupResult.HIT
        _store_keys(tier, [b, c], first_job_id=200)
        assert not Path(tier.file_mapper.get_file_name(a)).exists()
        assert _lookup(tier, a, ReqContext(req_id="y")) is LookupResult.MISS
    finally:
        tier.shutdown()


def test_eviction_skips_files_of_in_flight_jobs(tmp_path: Path) -> None:
    tier = _tier(tmp_path, max_size_gib=1.5 * BLOCK / (1 << 30))
    try:
        a, b = _key(b"A"), _key(b"B")
        _store_keys(tier, [a])
        pa = tier.file_mapper.get_file_name(a)
        tier._load_job_paths[999] = [pa]  # an in-flight load reading A
        _store_keys(tier, [b], first_job_id=200)
        assert os.path.exists(pa)
        assert next(iter(tier._store_index)) == pa  # still the oldest
    finally:
        tier._load_job_paths.pop(999, None)
        tier.shutdown()


def test_eviction_keeps_index_entry_when_remove_fails(
    tmp_path: Path, monkeypatch
) -> None:
    """A failed remove leaves the file on disk, so the index entry stays."""
    tier = _tier(tmp_path, max_size_gib=2.5 * BLOCK / (1 << 30), enable_kv_events=True)
    try:
        a, b, c = _key(b"A"), _key(b"B"), _key(b"C")
        _store_keys(tier, [a, b])
        pa = tier.file_mapper.get_file_name(a)
        pb = tier.file_mapper.get_file_name(b)
        real_remove = os.remove

        def fail_on_a(path, *args, **kwargs):
            if path == pa:
                raise PermissionError("denied")
            return real_remove(path, *args, **kwargs)

        monkeypatch.setattr(os, "remove", fail_on_a)
        _store_keys(tier, [c], first_job_id=200)

        # A could not be deleted: it keeps its index entry (and its bytes in
        # the accounting) while B is the block actually evicted under the cap.
        assert os.path.exists(pa)
        assert pa in tier._store_index
        assert not os.path.exists(pb)
        assert _removed_keys(tier) == [b]
        assert tier._store_bytes == 2 * BLOCK
    finally:
        tier.shutdown()


def test_store_cap_seeds_from_a_previous_run(tmp_path: Path) -> None:
    first = _tier(tmp_path, max_size_gib=0)
    try:
        keys = [_key(b"old"), _key(b"mid"), _key(b"new")]
        _store_keys(first, keys)
        paths = [Path(first.file_mapper.get_file_name(k)) for k in keys]
        for i, path in enumerate(paths):
            os.utime(path, (1_000_000 + i, 1_000_000 + i))
    finally:
        first.shutdown()

    cap = 2 * BLOCK / (1 << 30)
    second = _tier(tmp_path, max_size_gib=cap, enable_kv_events=True)
    try:
        assert second._store_bytes == 2 * BLOCK
        assert not paths[0].exists(), "oldest by mtime should go first"
        assert paths[1].exists() and paths[2].exists()
        # The key comes back from the file name.
        assert _removed_keys(second) == [keys[0]]
    finally:
        second.shutdown()


@pytest.fixture(autouse=True)
def _roomy_disk(monkeypatch: pytest.MonkeyPatch) -> None:
    """Keep the default cap's free-space check off the host's real disk."""
    _disk(monkeypatch, free_gib=800)


def _disk(monkeypatch: pytest.MonkeyPatch, free_gib: int) -> None:
    usage = SimpleNamespace(total=1000 << 30, used=0, free=free_gib << 30)
    monkeypatch.setattr(fs_tier.shutil, "disk_usage", lambda _: usage)


def test_store_cap_default_is_ten_percent_of_the_volume(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disk(monkeypatch, free_gib=800)
    tier = _tier(tmp_path)
    try:
        assert tier._max_store_bytes == 100 << 30
    finally:
        tier.shutdown()


def test_store_cap_default_refused_when_the_disk_cannot_hold_it(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disk(monkeypatch, free_gib=50)
    with pytest.raises(ValueError, match="Set max_size_gib"):
        MetalFileSystemTierManager._resolve_store_cap(str(tmp_path), None, 0)


def test_store_cap_default_counts_the_existing_store(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # On a restart the store's own files are space it can use.
    _disk(monkeypatch, free_gib=50)
    assert MetalFileSystemTierManager._resolve_store_cap(
        str(tmp_path), None, 60 << 30
    ) == (100 << 30)


def test_store_cap_explicit_size_is_trusted(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _disk(monkeypatch, free_gib=50)
    assert MetalFileSystemTierManager._resolve_store_cap(str(tmp_path), 20, 0) == (
        20 << 30
    )


def test_store_cap_zero_is_unbounded(tmp_path: Path) -> None:
    tier = _tier(tmp_path, max_size_gib=0)
    try:
        keys = [_key(bytes([i])) for i in range(5)]
        _store_keys(tier, keys)
        assert all(Path(tier.file_mapper.get_file_name(k)).exists() for k in keys)
        assert tier._store_bytes == 5 * BLOCK
    finally:
        tier.shutdown()


def test_only_stale_temp_files_are_reaped(tmp_path: Path) -> None:
    """A fresh temp file may be another live instance's in-flight write."""
    first = _tier(tmp_path, max_size_gib=0)
    try:
        _store_keys(first, [_key(b"kept")])
        store_dir = Path(first.file_mapper.get_file_name(_key(b"kept"))).parent
    finally:
        first.shutdown()
    stale = store_dir / "dead.bin_1.tmp"
    fresh = store_dir / "live.bin_2.tmp"
    stale.write_bytes(b"x" * 100)
    fresh.write_bytes(b"x" * 100)
    old = time.time() - TMP_REAP_AGE_S - 1
    os.utime(stale, (old, old))

    second = _tier(tmp_path, max_size_gib=0)
    try:
        assert not stale.exists()
        assert fresh.exists()
        assert second._store_bytes == BLOCK
    finally:
        second.shutdown()


def test_partial_store_failure_counts_written_files(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    real_store = upstream_io._store_block
    calls = 0

    def fail_second(*args, **kwargs):
        nonlocal calls
        calls += 1
        if calls == 2:
            raise OSError("disk full")
        real_store(*args, **kwargs)

    monkeypatch.setattr(upstream_io, "_HAS_FSIO_C", False)
    monkeypatch.setattr(upstream_io, "_store_block", fail_second)
    tier = _tier(tmp_path, max_size_gib=0)
    try:
        a, b = _key(b"A"), _key(b"B")
        tier.submit_store(_job(1, [a, b], 0))
        assert _finish(tier) == [(1, False)]
        assert list(tier._store_index) == [tier.file_mapper.get_file_name(a)]
        assert tier._store_bytes == BLOCK
    finally:
        tier.shutdown()


def test_store_metrics_are_defined_and_reported(tmp_path: Path) -> None:
    definitions = MetalFileSystemTierManager.build_metric_definitions({})
    assert STORE_BYTES_METRIC in definitions and EVICTED_BLOCKS_METRIC in definitions

    tier = _tier(tmp_path, max_size_gib=2.5 * BLOCK / (1 << 30))
    try:
        _store_keys(tier, [_key(b"A"), _key(b"B")])
        stats = tier.get_stats()
        assert stats is not None
        assert stats._values[STORE_BYTES_METRIC][()] == 2 * BLOCK
        assert EVICTED_BLOCKS_METRIC not in stats._values

        _store_keys(tier, [_key(b"C")], first_job_id=300)
        stats = tier.get_stats()
        assert stats is not None
        assert stats._values[STORE_BYTES_METRIC][()] == 2 * BLOCK
        assert stats._values[EVICTED_BLOCKS_METRIC][()] == 1
        # Counters are deltas: nothing new since the last report.
        assert EVICTED_BLOCKS_METRIC not in tier.get_stats()._values
    finally:
        tier.shutdown()
