# SPDX-License-Identifier: Apache-2.0
"""Tests for the macOS-tuned fs secondary tier (vllm_metal/v1/kv_offload/fs_tier.py).

The io functions must keep upstream's semantics (atomic replace, dedup on
existing files, remove-on-unreadable) while adding the macOS integrations:
0o600 files, 0o700 directories, .noindex nesting, thread QoS, and the
CRC32-footer on-disk format (<payload><8-byte LE CRC32>).
"""

import os
import stat
import sys
import threading
import zlib
from pathlib import Path

import numpy as np
import pytest

from vllm_metal.v1.kv_offload.fs_tier import (
    _CRC_FOOTER_BYTES,
    NOINDEX_DIRNAME,
    QOS_CLASS_UTILITY,
    MetalFileSystemTierManager,
    load_block,
    on_disk_block_size,
    prepare_root_dir,
    set_current_thread_qos,
    store_block,
)

BLOCK = 4096


def _mode(path: Path) -> int:
    return stat.S_IMODE(os.stat(path).st_mode)


def _on_disk_bytes(payload: bytes) -> bytes:
    """A block file exactly as store_block would write it."""
    crc = zlib.crc32(payload) & 0xFFFFFFFF
    return payload + crc.to_bytes(_CRC_FOOTER_BYTES, "little")


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


def test_store_overwrites_wrong_size_existing(tmp_path: Path) -> None:
    """Size-aware dedup: a stale/foreign file of the wrong length must not
    mask the honest write (unlike upstream's bare exists() dedup)."""
    dest = tmp_path / "hash.bin"
    dest.write_bytes(b"x" * (BLOCK // 2))  # wrong size — no footer, truncated
    data = np.full(BLOCK, 3, dtype=np.uint8)

    store_block(str(dest), memoryview(data), 0, BLOCK)

    assert dest.stat().st_size == on_disk_block_size(BLOCK)
    out = np.zeros(BLOCK, dtype=np.uint8)
    load_block(str(dest), memoryview(out), 0, BLOCK)
    np.testing.assert_array_equal(out, data)


def test_load_removes_unreadable_file(tmp_path: Path) -> None:
    dest = tmp_path / "short.bin"
    dest.write_bytes(b"x" * (BLOCK // 2))
    out = np.zeros(BLOCK, dtype=np.uint8)
    with pytest.raises(OSError, match="Short/oversized read"):
        load_block(str(dest), memoryview(out), 0, BLOCK)
    assert not dest.exists()


def test_load_rejects_crc_mismatch_and_removes_file(tmp_path: Path) -> None:
    """A right-sized file with corrupt payload fails the CRC and is deleted,
    so the next lookup is a clean miss instead of poisoned KV on the GPU."""
    dest = tmp_path / "rot.bin"
    good = _on_disk_bytes(bytes(range(256)) * (BLOCK // 256))
    flipped = bytearray(good)
    flipped[BLOCK // 2] ^= 0x01  # single bit of rot in the payload
    dest.write_bytes(bytes(flipped))

    out = np.zeros(BLOCK, dtype=np.uint8)
    with pytest.raises(OSError, match="CRC mismatch"):
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


@pytest.mark.slow
@pytest.mark.skipif(sys.platform != "darwin", reason="macOS Time Machine")
def test_time_machine_exclusion_applies(tmp_path: Path) -> None:
    """The fire-and-forget tmutil thread really lands the sticky exclusion
    (tmutil blocks ~10s per call on backupd XPC, hence the slow marker)."""
    import subprocess

    from vllm_metal.v1.kv_offload.fs_tier import _exclude_from_time_machine

    root = tmp_path / "kv-store"
    root.mkdir()
    _exclude_from_time_machine(str(root)).join(timeout=60)
    result = subprocess.run(
        ["tmutil", "isexcluded", str(root)], capture_output=True, text=True, timeout=60
    )
    assert "[Excluded]" in result.stdout


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS QoS")
def test_thread_qos_settable() -> None:
    results: list[bool] = []
    t = threading.Thread(
        target=lambda: results.append(set_current_thread_qos(QOS_CLASS_UTILITY))
    )
    t.start()
    t.join()
    assert results == [True]


def test_tiering_scope_routes_fs_tier() -> None:
    """MetalTieringOffloadingSpec.get_manager resolves 'fs' to the Metal tier
    (and the shared region to the Metal region) only within its scope."""
    import vllm.v1.kv_offload.tiering.spec as tiering_spec_module
    from vllm.v1.kv_offload.tiering.factory import SecondaryTierFactory
    from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

    from vllm_metal.v1.kv_offload.shared_region import MetalSharedOffloadRegion
    from vllm_metal.v1.kv_offload.spec import _metal_tiering_classes

    assert SecondaryTierFactory._registry["fs"]() is FileSystemTierManager
    with _metal_tiering_classes():
        assert SecondaryTierFactory._registry["fs"]() is MetalFileSystemTierManager
        assert tiering_spec_module.SharedOffloadRegion is MetalSharedOffloadRegion
    assert SecondaryTierFactory._registry["fs"]() is FileSystemTierManager
    assert tiering_spec_module.SharedOffloadRegion is not MetalSharedOffloadRegion


def test_metric_definitions_resolve_metal_tier_classes(monkeypatch) -> None:
    """build_metric_definitions runs at stat-logger construction, OUTSIDE
    get_manager's rebinding scope — the override must route the tier-class
    lookup to the Metal classes or a Metal tier's metrics would silently not
    register (then KeyError at first observation)."""
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


def test_layout_signature_disjoint_and_deterministic() -> None:
    """Same model, different KV byte layout => different store paths.
    Upstream FileMapper can't see this on Metal (cache_dtype is 'auto',
    TurboQuant lives in additional_config)."""
    from types import SimpleNamespace

    from vllm_metal.v1.kv_offload.fs_tier import layout_signature

    def spec(block_bytes, dtype, add):
        return SimpleNamespace(
            kv_bytes_per_offloaded_block=block_bytes,
            vllm_config=SimpleNamespace(
                model_config=SimpleNamespace(dtype=dtype),
                additional_config=add,
            ),
        )

    fp16 = layout_signature(spec(458752, "float16", {}))
    bf16 = layout_signature(spec(458752, "bfloat16", {}))
    tq = layout_signature(spec(179200, "float16", {"turboquant": True}))
    tq2 = layout_signature(
        spec(179200, "float16", {"turboquant": True, "k_quant": "q5_0"})
    )
    assert len({fp16, bf16, tq, tq2}) == 4  # all disjoint
    assert fp16 == layout_signature(spec(458752, "float16", {}))  # deterministic


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
        block_size_factor=1,
        kv_bytes_per_offloaded_block=block_bytes,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(model="test-model", dtype="float16"),
            cache_config=SimpleNamespace(block_size=16, cache_dtype="auto"),
            parallel_config=SimpleNamespace(
                tensor_parallel_size=1,
                pipeline_parallel_size=1,
                prefill_context_parallel_size=1,
                decode_context_parallel_size=1,
                rank=0,
            ),
            additional_config={},
            use_v2_model_runner=False,
        ),
        kv_cache_config=SimpleNamespace(
            kv_cache_groups=[
                SimpleNamespace(
                    kv_cache_spec=SimpleNamespace(block_size=16),
                    layer_names=["layer0"],
                )
            ]
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
    round-trip, size-validating lookups (truncated file == miss), and the
    failed-load negative cache that breaks the promotion livelock."""
    import time as _time

    import numpy as np
    from vllm.v1.kv_offload.base import LookupResult, ReqContext
    from vllm.v1.kv_offload.tiering.base import JobMetadata

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
            JobMetadata(
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
        assert path.stat().st_size == on_disk_block_size(block_bytes)

        # Size-validated lookup: healthy (footer-sized) file hits...
        ctx1 = ReqContext(req_id="r-1")
        assert _await_lookup(tier, key, ctx1, timeout=15) is LookupResult.HIT
        tier.on_request_finished(ctx1)

        # ...truncated file is a clean MISS, not a doomed promotion...
        path.write_bytes(b"x" * (block_bytes // 2))
        ctx2 = ReqContext(req_id="r-2")
        assert _await_lookup(tier, key, ctx2, timeout=15) is LookupResult.MISS
        tier.on_request_finished(ctx2)

        # ...and so is a legacy payload-only file (pre-CRC-footer layout):
        # the lookup's expected size must include the footer.
        path.write_bytes(b"y" * block_bytes)
        ctx2b = ReqContext(req_id="r-2b")
        assert _await_lookup(tier, key, ctx2b, timeout=15) is LookupResult.MISS
        tier.on_request_finished(ctx2b)

        # Livelock guard: cache a True verdict, THEN corrupt, then fail the
        # load — the negative cache must answer False despite the stale True.
        store_ok = np.full(block_bytes, 9, dtype=np.uint8)
        path.write_bytes(_on_disk_bytes(store_ok.tobytes()))
        ctx3 = ReqContext(req_id="r-3")
        assert _await_lookup(tier, key, ctx3, timeout=15) is LookupResult.HIT
        path.write_bytes(b"x" * 10)  # corrupt AFTER the cached True
        tier.submit_load(
            JobMetadata(
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
        # A fresh store lifts the negative cache.
        tier.submit_store(
            JobMetadata(
                job_id=3,
                keys=[key],
                block_ids=np.array([0]),
                is_promotion=False,
                req_context=ReqContext(req_id="r-restore"),
            )
        )
        tier.drain_jobs()
        list(tier.get_finished_jobs())
        assert key not in tier._failed_load_keys
    finally:
        tier.shutdown()
