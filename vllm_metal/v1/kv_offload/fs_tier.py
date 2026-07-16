# SPDX-License-Identifier: Apache-2.0
"""macOS-tuned filesystem secondary tier for Metal KV offloading.

Subclasses upstream ``FileSystemTierManager`` to use the operating system
properly on macOS; the tier semantics (file naming, atomic writes, dedup,
lookup) are inherited unchanged:

- **F_NOCACHE** on block file descriptors. Upstream opens with ``O_DIRECT``,
  which silently degrades to ``0`` on macOS, so multi-GB KV churn would flow
  through the Unified Buffer Cache and evict genuinely useful page cache.
  ``fcntl(F_NOCACHE)`` is the macOS idiom for this write-once/read-rarely
  streaming pattern (advisory, no alignment requirements).
- **0o600 block files under a 0o700 root.** KV blocks are conversation-derived
  data (prompt content is recoverable from them), and with ``PYTHONHASHSEED``
  pinned the content-hash filenames let anyone who can list the directory
  test for the presence of known prompts. Upstream creates 0o644 files.
- **Spotlight and Time Machine exclusion.** Blocks are stored under a
  ``blocks.noindex`` subdirectory (the ``.noindex`` name suffix is the
  reliable per-directory Spotlight opt-out) and the root gets a best-effort
  ``tmutil addexclusion`` so cache churn never bloats backups or lingers in
  APFS local snapshots after deletion.
- **Thread QoS.** Load-priority I/O threads run ``QOS_CLASS_USER_INITIATED``
  (restores are on the request critical path); store-priority threads run
  ``QOS_CLASS_UTILITY`` so bulk writes are steered toward E-cores instead of
  competing with inference on P-cores.

Scheduler-side only (no mlx import): ``MetalTieringOffloadingSpec`` routes the
``fs`` tier type here for the duration of manager construction.
"""

from __future__ import annotations

import ctypes
import functools
import os
import stat
import subprocess
import sys
import threading
import zlib
from typing import TYPE_CHECKING

from vllm.logger import init_logger
from vllm.v1.kv_offload.base import LookupResult
from vllm.v1.kv_offload.tiering.fs import manager as _fs_manager_module

# Pinned to vLLM 0.25.1 (like the SharedOffloadRegion rebinding in spec.py):
# the tmp-suffix and mkdir helpers are reused so the atomic-replace protocol
# cannot drift from upstream's.
from vllm.v1.kv_offload.tiering.fs.io import _ensure_dirs, _get_tmp_suffix
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager
from vllm.v1.kv_offload.tiering.fs.thread_pool import DualQueueThreadPool

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.base import JobMetadata

logger = init_logger(__name__)

# macOS pthread/qos.h constants.
QOS_CLASS_USER_INITIATED = 0x19
QOS_CLASS_UTILITY = 0x11

NOINDEX_DIRNAME = "blocks.noindex"

_F_NOCACHE: int | None = None
if sys.platform == "darwin":
    import fcntl

    _F_NOCACHE = getattr(fcntl, "F_NOCACHE", None)


def set_current_thread_qos(qos_class: int) -> bool:
    """Set the calling thread's macOS QoS class; False if unavailable."""
    if sys.platform != "darwin":
        return False
    try:
        libc = ctypes.CDLL(None, use_errno=True)
        fn = libc.pthread_set_qos_class_self_np
    except (OSError, AttributeError):
        return False
    return int(fn(ctypes.c_uint(qos_class), ctypes.c_int(0))) == 0


def _set_nocache(fd: int) -> None:
    """Advise the UBC not to retain pages for this fd (best-effort)."""
    if _F_NOCACHE is None:
        return
    try:
        fcntl.fcntl(fd, _F_NOCACHE, 1)
    except OSError as exc:
        logger.warning_once("F_NOCACHE failed: %s", exc)


# On-disk layout: <block_size KV bytes><8-byte little-endian CRC32 footer>.
# The footer catches silent corruption a size check cannot — a torn write
# (F_NOCACHE makes power-loss torn pages more likely), bit rot, or a
# foreign/stale writer in a shared store — before poisoned KV reaches the GPU
# cache. It is a corruption check, NOT an authentication code: a hostile
# writer who controls the file can forge the CRC. The defense against that is
# store ownership (see _make_private_dir); a per-deployment HMAC is the
# roadmap control for adversarial-writer stores. layout_signature already
# fences this format change, so no migration for existing fp16 stores.
_CRC_FOOTER_BYTES = 8


def on_disk_block_size(block_size: int) -> int:
    """File size for a KV block of ``block_size`` payload bytes (incl footer)."""
    return block_size + _CRC_FOOTER_BYTES


def _footer(crc: int) -> bytes:
    return crc.to_bytes(_CRC_FOOTER_BYTES, "little")


def store_block(
    dest_path: str,
    buffer: memoryview,
    offset: int,
    block_size: int,
) -> None:
    """Upstream ``io.store_block`` with F_NOCACHE, 0o600 files, CRC footer.

    Dedup is size-aware: a pre-existing file of the wrong length (a stale or
    hostile placement) does not mask the honest write."""
    payload_len = block_size + _CRC_FOOTER_BYTES
    try:
        if os.path.getsize(dest_path) == payload_len:
            return  # already stored, correct size — trust the CRC on load
    except OSError:
        pass  # absent (or unstatable) — (re)write it

    tmp_path = dest_path + _get_tmp_suffix()
    _ensure_dirs(dest_path)

    view_slice = buffer.cast("B")[offset : offset + block_size]
    footer = _footer(zlib.crc32(view_slice) & 0xFFFFFFFF)
    try:
        fd = os.open(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            _set_nocache(fd)
            written = os.write(fd, view_slice) + os.write(fd, footer)
            if written < payload_len:
                raise OSError(
                    f"Short write: expected {payload_len} bytes, wrote {written}"
                )
        finally:
            os.close(fd)
        os.replace(tmp_path, dest_path)
    except Exception:
        try:
            os.remove(tmp_path)
        except OSError as cleanup_exc:
            logger.warning("Failed to remove temp file %s: %s", tmp_path, cleanup_exc)
        raise


def load_block(
    source_path: str,
    view: memoryview,
    offset: int,
    block_size: int,
) -> None:
    """Read one KV block and verify its CRC footer.

    A bad file (short/size-mismatched/CRC-mismatched) is deleted so the tier
    turns it into a clean miss; a *transient* error (fd exhaustion, EIO) is
    propagated WITHOUT deleting — mass-unlinking valid blocks under an EMFILE
    burst would be worse than a retry."""
    fd: int | None = None
    view_slice = view.cast("B")[offset : offset + block_size]
    footer = bytearray(_CRC_FOOTER_BYTES)
    bad_file = False
    try:
        fd = os.open(source_path, os.O_RDONLY)
        _set_nocache(fd)
        bytes_read = os.readv(fd, [view_slice, footer])
        if bytes_read != block_size + _CRC_FOOTER_BYTES:
            bad_file = True
            raise OSError(
                f"Short/oversized read: expected {block_size + _CRC_FOOTER_BYTES} "
                f"bytes, read {bytes_read}"
            )
        expected = int.from_bytes(footer, "little")
        actual = zlib.crc32(view_slice) & 0xFFFFFFFF
        if actual != expected:
            bad_file = True
            raise OSError(
                f"KV block CRC mismatch (got {actual:#010x}, expected "
                f"{expected:#010x}); corrupt or tampered store"
            )
    except Exception:
        if bad_file:
            try:
                os.remove(source_path)
            except OSError as cleanup_exc:
                logger.warning(
                    "Failed to remove bad block file %s: %s", source_path, cleanup_exc
                )
        raise
    finally:
        if fd is not None:
            os.close(fd)


def _exclude_from_time_machine(path: str) -> threading.Thread:
    """Sticky-exclude ``path`` from Time Machine, off the startup path.

    ``tmutil addexclusion`` blocks ~10s on an XPC round-trip to backupd
    (measured on macOS 15), so it runs in a fire-and-forget daemon thread:
    the exclusion is best-effort and must not delay engine start. Returns the
    thread (for tests)."""

    def run() -> None:
        try:
            result = subprocess.run(
                ["tmutil", "addexclusion", path],
                capture_output=True,
                text=True,
                check=False,
                timeout=60,
            )
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            logger.warning("Time Machine exclusion unavailable for %s: %s", path, exc)
            return
        if result.returncode != 0:
            logger.warning(
                "tmutil addexclusion %s failed: %s", path, result.stderr.strip()
            )

    thread = threading.Thread(target=run, name="vllm_kv_tmutil", daemon=True)
    thread.start()
    return thread


def _make_private_dir(path: str) -> None:
    """mkdir -p; chmod 0700 only if WE created it. chmod-ing a pre-existing
    user directory (root_dir=/tmp, a shared mount, ...) would strip other
    users' access and e.g. /tmp's sticky bit — warn instead."""
    if os.path.isdir(path):
        mode = stat.S_IMODE(os.stat(path).st_mode)
        if mode & 0o077:
            logger.warning(
                "KV store directory %s is group/world-accessible (%o); KV "
                "blocks encode conversation-derived data — consider "
                "chmod 700.",
                path,
                mode,
            )
        return
    os.makedirs(path, exist_ok=True)
    os.chmod(path, 0o700)


def prepare_root_dir(root_dir: str) -> str:
    """Harden the KV store directory and return the directory to store under.

    Owner-only permissions (on directories this code creates), Time Machine
    exclusion, and a ``.noindex`` nesting level so Spotlight never indexes
    the block churn. Idempotent; safe to call on every startup against an
    existing store.
    """
    _make_private_dir(root_dir)
    _exclude_from_time_machine(root_dir)
    if os.path.basename(os.path.normpath(root_dir)).endswith(".noindex"):
        return root_dir
    nested = os.path.join(root_dir, NOINDEX_DIRNAME)
    _make_private_dir(nested)
    return nested


def layout_signature(offloading_spec) -> str:
    """Directory component binding stored blocks to the KV byte layout.

    Upstream ``FileMapper`` hashes ``cache_config.cache_dtype`` — which on
    Metal is always "auto" (the real cache dtype follows model_config.dtype)
    — and knows nothing of TurboQuant (``additional_config``). Two runs of
    the SAME model with different ``--dtype`` or quant settings would
    therefore share block paths with incompatible byte layouts: loads where
    the on-disk file is larger than the reader's block silently restore
    wrong-layout bytes, and short reads DELETE the other config's valid
    files. Binding the store path to (block bytes, cache dtype, quant
    config) makes the layouts disjoint on disk. Deterministic across
    restarts for a fixed config, so cross-restart reuse is preserved.
    """
    block_bytes = int(getattr(offloading_spec, "kv_bytes_per_offloaded_block", 0))
    dtype = str(offloading_spec.vllm_config.model_config.dtype).replace("torch.", "")
    sig = f"layout-{block_bytes}B-{dtype}"
    add = getattr(offloading_spec.vllm_config, "additional_config", None) or {}
    if isinstance(add, dict) and add.get("turboquant"):
        sig += f"-tq-{add.get('k_quant', 'q8_0')}-{add.get('v_quant', 'q3_0')}"
    return sig


class MetalDualQueueThreadPool(DualQueueThreadPool):
    """DualQueueThreadPool whose workers set a macOS QoS class on entry."""

    def _worker(self, load_priority: bool) -> None:
        qos = QOS_CLASS_USER_INITIATED if load_priority else QOS_CLASS_UTILITY
        if not set_current_thread_qos(qos):
            logger.debug_once("Thread QoS unavailable; using default scheduling")
        super()._worker(load_priority)


class MetalFsAsyncLookupManager(_fs_manager_module.FsAsyncLookupManager):
    """Lookup that validates file SIZE, not just existence.

    A truncated or foreign-layout block file passes a bare exists() check and
    then fails fatally at load time; worse, the failed load DELETES the file
    while the lookup cache keeps answering True for the lifetime of the
    requesting request — re-initiating the same doomed promotion every step
    (request-level livelock, verified against upstream
    tiering/manager.py::complete_store(success=False) + async_lookup.py cache
    semantics). Size validation turns bad files into clean misses up front.
    """

    def batch_lookup(self, keys, req_context):
        expected = on_disk_block_size(self._tier._block_size)
        results = []
        for key in keys:
            path = self._tier.file_mapper.get_file_name(key)
            try:
                ok = os.path.getsize(path) == expected
            except OSError:
                ok = False
            if not ok and os.path.exists(path):
                logger.warning(
                    "fs tier: %s exists but is not %d bytes; treating as miss",
                    path,
                    expected,
                )
            results.append(ok)
        return results


class MetalFileSystemTierManager(FileSystemTierManager):
    """FileSystemTierManager with the macOS integrations described above."""

    def __init__(
        self,
        offloading_spec,
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        **kwargs,
    ) -> None:
        store_dir = os.path.join(
            prepare_root_dir(root_dir), layout_signature(offloading_spec)
        )
        _make_private_dir(store_dir)
        if store_dir != root_dir:
            logger.info("KV store blocks live under %s (Spotlight-excluded)", store_dir)
        # The upstream constructor instantiates its module-global
        # DualQueueThreadPool; rebind it around the super() call so the pool
        # gets QoS-aware workers (single-process on Metal, so this cannot
        # race another constructor).
        original_pool = _fs_manager_module.DualQueueThreadPool
        _fs_manager_module.DualQueueThreadPool = MetalDualQueueThreadPool  # type: ignore[misc]
        try:
            super().__init__(
                offloading_spec, primary_kv_view, tier_type, store_dir, **kwargs
            )
        finally:
            _fs_manager_module.DualQueueThreadPool = original_pool  # type: ignore[misc]
        # Replace the exists()-based lookup manager with the size-validating
        # one (shut down the upstream instance's thread first).
        self._lookup_manager.shutdown()
        self._lookup_manager = MetalFsAsyncLookupManager(
            tier=self, tier_type=self.tier_type
        )
        # Negative cache breaking the failed-promotion livelock: keys whose
        # load failed are answered False at lookup until re-stored, because
        # the async lookup cache would otherwise keep answering a stale True
        # for the lifetime of the requesting request.
        self._failed_load_keys: set = set()
        self._inflight_load_keys: dict = {}

    def lookup(self, key, req_context):
        if key in self._failed_load_keys:
            return LookupResult.MISS
        return super().lookup(key, req_context)

    def get_finished_jobs(self):
        for result in super().get_finished_jobs():
            keys = self._inflight_load_keys.pop(result.job_id, None)
            if keys is not None and not result.success:
                self._failed_load_keys.update(keys)
                logger.warning(
                    "fs tier: load failed for %d block(s); marking them "
                    "absent so the scheduler recomputes instead of "
                    "re-promoting (livelock guard)",
                    len(keys),
                )
            yield result

    # Upstream's submit_store/submit_load bind the plain io callbacks from
    # their module namespace; re-issue them with the F_NOCACHE versions.
    def submit_store(self, job_metadata: JobMetadata) -> None:
        # A fresh store re-creates the file: lift the livelock negative-cache.
        self._failed_load_keys.difference_update(job_metadata.keys)
        tasks = (
            functools.partial(
                store_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids, strict=True)
        )
        self._pool.enqueue_store(job_metadata.job_id, len(job_metadata.keys), tasks)

    def submit_load(self, job_metadata: JobMetadata) -> None:
        # Track keys per job so a failed load can negative-cache them
        # (see get_finished_jobs).
        self._inflight_load_keys[job_metadata.job_id] = list(job_metadata.keys)
        tasks = (
            functools.partial(
                load_block,
                self.file_mapper.get_file_name(key),
                self._primary_kv_view,
                int(bid) * self._block_size,
                self._block_size,
            )
            for key, bid in zip(job_metadata.keys, job_metadata.block_ids, strict=True)
        )
        self._pool.enqueue_load(job_metadata.job_id, len(job_metadata.keys), tasks)
