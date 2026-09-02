# SPDX-License-Identifier: Apache-2.0
"""Tests for the macOS-tuned fs secondary tier (vllm_metal/v1/kv_offload/fs_tier.py).

The io functions must keep upstream's semantics (byte-identical on-disk format,
atomic replace, dedup on existing files) while adding the macOS integrations:
F_NOCACHE, 0o600 files, 0o700 directories, and .noindex nesting.
"""

import os
import stat
from pathlib import Path

import numpy as np
import pytest

from vllm_metal.v1.kv_offload.fs_tier import (
    NOINDEX_DIRNAME,
    MetalFileSystemTierManager,
    load_block,
    prepare_root_dir,
    store_block,
)

BLOCK = 4096


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def test_store_load_roundtrip(tmp_path: Path) -> None:
    data = np.random.default_rng(0).integers(0, 256, BLOCK * 2, dtype=np.uint8)
    dest = tmp_path / "ab" / "cd_g0" / "hash.bin"

    store_block(str(dest), memoryview(data), BLOCK, BLOCK)

    assert _mode(dest) == 0o600
    assert [p.name for p in dest.parent.iterdir()] == ["hash.bin"]  # no .tmp left

    out = np.zeros(BLOCK * 2, dtype=np.uint8)
    load_block(str(dest), memoryview(out), BLOCK, BLOCK)
    np.testing.assert_array_equal(out[BLOCK:], data[BLOCK:])
    np.testing.assert_array_equal(out[:BLOCK], 0)


def test_store_skips_existing(tmp_path: Path) -> None:
    """Content-hash dedup parity with upstream: existing files are kept."""
    dest = tmp_path / "hash.bin"
    first = np.full(BLOCK, 1, dtype=np.uint8)
    second = np.full(BLOCK, 2, dtype=np.uint8)

    store_block(str(dest), memoryview(first), 0, BLOCK)
    store_block(str(dest), memoryview(second), 0, BLOCK)

    out = np.zeros(BLOCK, dtype=np.uint8)
    load_block(str(dest), memoryview(out), 0, BLOCK)
    np.testing.assert_array_equal(out, first)


def test_load_removes_unreadable_file(tmp_path: Path) -> None:
    dest = tmp_path / "short.bin"
    dest.write_bytes(b"x" * (BLOCK // 2))
    out = np.zeros(BLOCK, dtype=np.uint8)
    with pytest.raises(OSError, match="Short read"):
        load_block(str(dest), memoryview(out), 0, BLOCK)
    assert not dest.exists()


def test_load_missing_file_raises_without_cleanup(tmp_path: Path) -> None:
    """A transient-style failure (here: file vanished) propagates as-is —
    load_block must not attempt deletion when it never validated the file."""
    out = np.zeros(BLOCK, dtype=np.uint8)
    with pytest.raises(FileNotFoundError):
        load_block(str(tmp_path / "gone.bin"), memoryview(out), 0, BLOCK)


def test_prepare_root_dir(tmp_path: Path) -> None:
    root = tmp_path / "kv-store"
    store_dir = prepare_root_dir(str(root))

    assert store_dir == str(root / NOINDEX_DIRNAME)
    assert _mode(root) == 0o700
    assert _mode(Path(store_dir)) == 0o700
    # Idempotent against an existing store.
    assert prepare_root_dir(str(root)) == store_dir


def test_prepare_root_dir_respects_noindex_name(tmp_path: Path) -> None:
    root = tmp_path / "kv-store.noindex"
    assert prepare_root_dir(str(root)) == str(root)
    assert _mode(root) == 0o700


def test_fs_tier_config_routes_to_metal_class() -> None:
    """An 'fs' tier config is rewritten to load the Metal class through
    upstream's out-of-tree module_path hook, and upstream resolves it.

    Without this the tier silently falls back to upstream's: buffered I/O,
    0o644 block files, and nothing logged."""
    from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory

    from vllm_metal.v1.kv_offload.spec import route_fs_tiers_to_metal

    extra = {"secondary_tiers": [{"type": "fs", "root_dir": "/tmp/unused"}]}
    route_fs_tiers_to_metal(extra)
    tier = extra["secondary_tiers"][0]
    assert tier["module_path"] == "vllm_metal.v1.kv_offload.fs_tier"
    assert SecondaryTierFactory.get_tier_class(tier) is MetalFileSystemTierManager
    # Untouched: the registry is upstream's and stays that way.
    assert "MetalFileSystemTierManager" not in SecondaryTierFactory._registry


def test_metal_region_swapped_only_within_scope() -> None:
    """The shared region has no out-of-tree hook upstream, so it is still
    rebound. Pin that it is restored."""
    import vllm.v1.kv_offload.tiering.spec as tiering_spec_module

    from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion
    from vllm_metal.v1.kv_offload.spec import _metal_tiering_classes

    original = tiering_spec_module.SharedOffloadRegion
    assert original is not MetalSharedOffloadRegion
    with _metal_tiering_classes():
        assert tiering_spec_module.SharedOffloadRegion is MetalSharedOffloadRegion
    assert tiering_spec_module.SharedOffloadRegion is original


def test_metric_definitions_resolve_metal_tier_classes(monkeypatch) -> None:
    """build_metric_definitions runs at stat-logger construction, before any
    spec instance exists, so it has to route the tier config itself or a
    Metal tier's metrics silently never register (then KeyError at first
    observation)."""
    from vllm_metal.v1.kv_offload.spec import MetalTieringOffloadingSpec

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


def test_region_alignment_matches_upstream_snapshot() -> None:
    """TieringOffloadingSpec snapshots SharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
    at import time; a Metal override of the class attr would never be seen by
    that snapshot, so pin the two to be equal."""
    from vllm.v1.kv_offload.tiering.spec import TieringOffloadingSpec

    from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion

    assert (
        MetalSharedOffloadRegion.BLOCK_SIZE_ALIGNMENT
        == TieringOffloadingSpec.BLOCK_SIZE_ALIGNMENT
    )


def test_layout_signature_discriminates_turboquant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TurboQuant settings must produce disjoint store paths.

    Everything else upstream's FileMapper already hashes into the path, so
    the signature is empty unless TurboQuant is on."""
    from types import SimpleNamespace

    from vllm_metal.v1.kv_offload import fs_tier

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

    assert len({"", tq, tq_k, tq_v}) == 4  # all disjoint

    config(True)
    assert fs_tier.layout_signature() == tq  # deterministic


def test_make_private_dir_does_not_chmod_existing(tmp_path: Path) -> None:
    """Pointing root_dir at a pre-existing (e.g. shared) directory must not
    strip other users' permissions."""
    from vllm_metal.v1.kv_offload.fs_tier import _make_private_dir

    pre = tmp_path / "shared"
    pre.mkdir()
    os.chmod(pre, 0o755)
    _make_private_dir(str(pre))
    assert _mode(pre) == 0o755  # untouched (warning logged instead)

    fresh = tmp_path / "fresh"
    _make_private_dir(str(fresh))
    assert _mode(fresh) == 0o700


def _fake_spec(block_bytes: int):
    from types import SimpleNamespace

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


def _await_lookup(tier, key, ctx, timeout: float = 10.0):
    """Drive the async lookup loop (lookup -> flush -> drain) to a verdict."""
    import time as _time

    from vllm.v1.kv_offload.base import LookupResult, ScheduleEndContext

    deadline = _time.monotonic() + timeout
    while _time.monotonic() < deadline:
        result = tier.lookup(key, ctx)
        if result is not LookupResult.RETRY:
            return result
        tier.on_schedule_end(ScheduleEndContext(set(), set()))
        _time.sleep(0.05)
    raise TimeoutError("lookup never resolved")


def test_manager_end_to_end_and_livelock_guard(tmp_path: Path) -> None:
    """Constructs the real MetalFileSystemTierManager and pins: store/load
    round-trip, lookup hit, and upstream's failed-load negative cache
    (vllm#49328) still reaching through this subclass's submit_load."""
    import time as _time

    import numpy as np
    from vllm.v1.kv_offload.base import LookupResult, ReqContext
    from vllm.v1.kv_offload.tiering.base import TransferJob

    num_blocks, block_bytes = 4, 4096
    pool = np.zeros((num_blocks, block_bytes), dtype=np.uint8)
    tier = MetalFileSystemTierManager(
        offloading_spec=_fake_spec(block_bytes),
        primary_kv_view=memoryview(pool),
        tier_type="fs",
        root_dir=str(tmp_path / "kv-store"),
        n_read_threads=2,
        n_write_threads=2,
    )
    try:
        import hashlib

        key = hashlib.sha256(b"block-A").digest() + (0).to_bytes(4, "big")
        path = Path(tier.file_mapper.get_file_name(key))

        # Store block 0.
        pool[0] = 7
        tier.submit_store(
            TransferJob(
                job_id=1,
                keys=[key],
                block_ids=np.array([0]),
                is_promotion=False,
                req_context=ReqContext(req_id="r-store"),
            )
        )
        tier.drain_jobs()
        results = list(tier.get_finished_jobs())
        assert [(r.job_id, r.success) for r in results] == [(1, True)]
        assert path.stat().st_size == block_bytes

        ctx1 = ReqContext(req_id="r-1")
        assert _await_lookup(tier, key, ctx1, timeout=15) is LookupResult.HIT
        tier.on_request_finished(ctx1)

        # Livelock guard: cache a True verdict, THEN corrupt, then fail the
        # load. The failed load must override the stale True, or the same
        # doomed promotion is re-issued every step. vLLM 0.28.0 does this
        # upstream (vllm#49328) off _load_job_keys, which submit_load fills.
        ctx3 = ReqContext(req_id="r-3")
        assert _await_lookup(tier, key, ctx3, timeout=15) is LookupResult.HIT
        path.write_bytes(b"x" * 10)  # corrupt AFTER the cached True
        tier.submit_load(
            TransferJob(
                job_id=2,
                keys=[key],
                block_ids=np.array([1]),
                is_promotion=True,
                req_context=ctx3,
            )
        )
        tier.drain_jobs()
        _time.sleep(0.1)
        results = list(tier.get_finished_jobs())
        assert [(r.job_id, r.success) for r in results] == [(2, False)]
        assert tier.lookup(key, ctx3) is LookupResult.MISS  # stale HIT overridden
        tier.on_request_finished(ctx3)

        # A fresh store restores the block, and the key hits again.
        pool[0] = 7
        tier.submit_store(
            TransferJob(
                job_id=3,
                keys=[key],
                block_ids=np.array([0]),
                is_promotion=False,
                req_context=ReqContext(req_id="r-restore"),
            )
        )
        tier.drain_jobs()
        assert [(r.job_id, r.success) for r in tier.get_finished_jobs()] == [(3, True)]
        ctx4 = ReqContext(req_id="r-4")
        assert _await_lookup(tier, key, ctx4, timeout=15) is LookupResult.HIT
        tier.on_request_finished(ctx4)
    finally:
        tier.shutdown()


def _tier(tmp_path, block_bytes=4096, num_blocks=4, **kwargs):
    """A real MetalFileSystemTierManager over a numpy pool."""
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


def test_store_emits_block_stored_event(tmp_path: Path) -> None:
    """The tier must announce what it stored.

    Upstream builds the BlockStored event from _store_job_keys, populated in
    submit_store. An override that omits it stores blocks and never announces
    them, so a KV-aware router never learns this instance holds them and
    routes prefix matches elsewhere. Silent, and it defeats the disk tier in
    a routed deployment."""
    import hashlib

    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.tiering.base import TransferJob

    tier = _tier(tmp_path, enable_kv_events=True)
    try:
        assert tier.events is not None, "fake spec did not enable events"
        key = hashlib.sha256(b"evented").digest() + (0).to_bytes(4, "big")
        tier.submit_store(
            TransferJob(
                job_id=1,
                keys=[key],
                block_ids=[0],
                req_context=ReqContext(req_id="r-ev"),
                is_promotion=False,
            )
        )
        tier.drain_jobs()
        assert [(r.job_id, r.success) for r in tier.get_finished_jobs()] == [(1, True)]
        events = list(tier.take_events())
        assert events, "no BlockStored event emitted for a successful store"
        assert key in events[0].keys
    finally:
        tier.shutdown()


def test_empty_job_completes_without_hanging(tmp_path: Path) -> None:
    """A keyless job must not wedge drain_jobs().

    Enqueueing zero tasks increments the pool's in-flight count and nothing
    ever decrements it, because completion is driven by task callbacks.
    drain_jobs() waits on that count with no timeout, so one empty job hangs
    reset_cache and sleep/wake for good."""
    import threading

    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.tiering.base import TransferJob

    tier = _tier(tmp_path)
    try:
        tier.submit_store(
            TransferJob(
                job_id=7,
                keys=[],
                block_ids=[],
                req_context=ReqContext(req_id="r-e"),
                is_promotion=False,
            )
        )
        done = threading.Event()
        threading.Thread(
            target=lambda: (tier.drain_jobs(), done.set()), daemon=True
        ).start()
        assert done.wait(timeout=10), "drain_jobs() hung on a keyless job"
        assert (7, True) in [(r.job_id, r.success) for r in tier.get_finished_jobs()]
    finally:
        tier.shutdown()


def test_empty_job_leaves_no_bookkeeping(tmp_path: Path) -> None:
    """A keyless job must complete and leave no bookkeeping behind.

    Upstream enqueues exactly one task per job, so an empty job still has a
    task to complete and the pool's in-flight count comes back down. Pinned
    because an earlier per-block fan-out enqueued zero tasks here and wedged
    drain_jobs() forever."""
    from vllm.v1.kv_offload.base import ReqContext
    from vllm.v1.kv_offload.tiering.base import TransferJob

    tier = _tier(tmp_path)
    try:
        for job_id in range(5):
            for promotion in (False, True):
                job = TransferJob(
                    job_id=job_id * 2 + promotion,
                    keys=[],
                    block_ids=[],
                    req_context=ReqContext(req_id=f"r-{job_id}"),
                    is_promotion=promotion,
                )
                if promotion:
                    tier.submit_load(job)
                else:
                    tier.submit_store(job)
        tier.drain_jobs()
        list(tier.get_finished_jobs())
        assert tier._store_job_keys == {}
        assert tier._load_job_keys == {}
    finally:
        tier.shutdown()
