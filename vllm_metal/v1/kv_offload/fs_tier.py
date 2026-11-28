# SPDX-License-Identifier: Apache-2.0
"""Upstream's fs tier with a disk cap, a private root and Spotlight exclusion.

Block IO, file naming and dedup are upstream's. KV blocks are
conversation-derived, and with the sha256 default (fixed seed) their filenames
are predictable, so the store lives under a 0o700 root that other users cannot
list. Blocks sit under a ``.noindex`` directory so Spotlight skips them.
"""

from __future__ import annotations

import collections
import hashlib
import json
import os
import shutil
import stat
import time
from collections.abc import Iterable
from typing import TYPE_CHECKING

from vllm.distributed.kv_transfer.kv_connector.v1.offloading.metrics import (
    OffloadingConnectorStats,
)
from vllm.logger import init_logger
from vllm.v1.kv_offload.base import (
    OffloadingCounterMetadata,
    OffloadingEvent,
    OffloadingGaugeMetadata,
    OffloadingMetricMetadata,
    OffloadKey,
    make_offload_key,
)
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

from vllm_metal.config import get_config

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.base import JobResult, TransferJob

logger = init_logger(__name__)

NOINDEX_DIRNAME = "blocks.noindex"

# Reported through upstream's tier stats, beside the CPU tier's metrics.
STORE_BYTES_METRIC = "vllm:kv_offload_fs_store_bytes"
EVICTED_BLOCKS_METRIC = "vllm:kv_offload_fs_evicted_blocks"

# A block write takes milliseconds. A younger temp file may belong to another
# live instance sharing the store, so only older ones are reaped.
TMP_REAP_AGE_S = 600


def _make_private_dir(path: str) -> None:
    """mkdir with mode 0o700. Refuse a directory another user owns."""
    if os.path.isdir(path):
        st = os.stat(path)
        # Its owner could pre-place a block file that dedup keeps and a load
        # restores.
        if st.st_uid != os.geteuid():
            raise PermissionError(
                f"KV store directory {path} is owned by uid {st.st_uid}, not "
                f"this process (uid {os.geteuid()}). Use a directory you own."
            )
        # Do not chmod a directory this process did not create (e.g. /tmp).
        mode = stat.S_IMODE(st.st_mode)
        if mode & 0o077:
            logger.warning(
                "KV store directory %s is group/world-accessible (%o); "
                "consider chmod 700.",
                path,
                mode,
            )
        return
    os.makedirs(path, exist_ok=True)
    os.chmod(path, 0o700)


def prepare_root_dir(root_dir: str) -> str:
    """Create the private root and return the ``.noindex`` dir to store under."""
    _make_private_dir(root_dir)
    if os.path.basename(os.path.normpath(root_dir)).endswith(".noindex"):
        return root_dir
    nested = os.path.join(root_dir, NOINDEX_DIRNAME)
    _make_private_dir(nested)
    return nested


def layout_signature() -> str:
    """Path component for TurboQuant settings, which ``FileMapper`` cannot see.

    TurboQuant changes bytes per element without changing anything
    ``FileMapper`` hashes. Without this, two quant settings would share block
    paths, and a short read of the other layout's file deletes it.
    """
    cfg = get_config()
    if not cfg.turboquant:
        return ""
    return f"tq-{cfg.k_quant}-{cfg.v_quant}"


def _weights_namespace(gguf_weights_path: str | None) -> str:
    """Separate weights that upstream's mapper sees as the same companion model."""
    if gguf_weights_path is None:
        return "native"
    path = os.path.realpath(gguf_weights_path, strict=True)
    st = os.stat(path)
    identity = json.dumps(
        [path, st.st_dev, st.st_ino, st.st_size, st.st_ctime_ns]
    ).encode()
    return "gguf-" + hashlib.sha256(identity).hexdigest()


class MetalFileSystemTierManager(FileSystemTierManager):
    """``FileSystemTierManager`` with a store cap and LRU eviction.

    Upstream never deletes a block file, so the store grows with the working
    set. ``max_size_gib`` caps it (default 10% of the volume, 0 disables).
    Oldest blocks go first, by store order with a touch on each load. Files
    an in-flight job reads or writes are never evicted.
    """

    def __init__(
        self,
        offloading_spec,
        primary_kv_view: memoryview,
        tier_type: str,
        root_dir: str,
        max_size_gib: float | None = None,
        *,
        gguf_weights_path: str | None = None,
        **kwargs,
    ) -> None:
        # The factory passes the class name; report "fs" like the CUDA path.
        tier_type = "fs"
        store_dir = prepare_root_dir(root_dir)
        signature = layout_signature()
        if signature:
            store_dir = os.path.join(store_dir, signature)
        _make_private_dir(store_dir)
        if store_dir != root_dir:
            logger.info("KV store blocks live under %s (Spotlight-excluded)", store_dir)
        super().__init__(
            offloading_spec,
            primary_kv_view,
            tier_type,
            os.path.join(store_dir, _weights_namespace(gguf_weights_path)),
            **kwargs,
        )
        # Scheduler-thread bookkeeping. Paths of in-flight jobs pin their files.
        # path -> (size, key); key is None if the path does not map back to one.
        self._store_index: collections.OrderedDict[
            str, tuple[int, OffloadKey | None]
        ] = collections.OrderedDict()
        self._store_bytes = 0
        self._store_job_paths: dict[int, dict[str, OffloadKey]] = {}
        self._load_job_paths: dict[int, list[str]] = {}
        self._cap_announced = False
        self._evicted_since_stats = 0
        self._index_existing_store(store_dir)
        self._max_store_bytes = self._resolve_store_cap(
            store_dir, max_size_gib, self._store_bytes
        )
        self._evict_to(self._max_store_bytes)

    @staticmethod
    def _resolve_store_cap(
        store_dir: str, max_size_gib: float | None, existing_bytes: int
    ) -> int:
        """Cap in bytes; 0 means unbounded.

        The default is 10% of the volume. If the disk cannot hold that, fail at
        startup rather than fill it. The store's own files count as free.
        """
        if max_size_gib is None:
            usage = shutil.disk_usage(store_dir)
            cap = usage.total // 10
            available = usage.free + existing_bytes
            if available < cap:
                raise ValueError(
                    f"The KV store's default cap is {cap / (1 << 30):.1f} GiB "
                    f"(10% of the volume), but only {available / (1 << 30):.1f} "
                    f"GiB is free for {store_dir}. Set max_size_gib on the fs "
                    "tier to a size the disk can hold."
                )
            logger.info(
                "KV store cap defaulting to %.1f GiB, 10%% of the volume; set "
                "max_size_gib on the tier to change it (0 disables the cap)",
                cap / (1 << 30),
            )
            return cap
        if max_size_gib <= 0:
            logger.warning(
                "KV store cap disabled (max_size_gib=%s); the store grows with "
                "the working set",
                max_size_gib,
            )
            return 0
        return int(max_size_gib * (1 << 30))

    def _index_existing_store(self, store_dir: str) -> None:
        """Seed the LRU from files on disk and reap stale temp files."""
        found: list[tuple[float, str, int]] = []
        reaped = 0
        cutoff = time.time() - TMP_REAP_AGE_S
        for dirpath, _dirnames, filenames in os.walk(store_dir):
            for name in filenames:
                path = os.path.join(dirpath, name)
                try:
                    st = os.stat(path)
                    if name.endswith(".tmp"):
                        if st.st_mtime < cutoff:
                            os.remove(path)
                            reaped += 1
                        continue
                except OSError:
                    continue
                if name.endswith(".bin"):
                    found.append((st.st_mtime, path, st.st_size))
        found.sort()
        for _mtime, path, size in found:
            self._store_index[path] = (size, self._key_from_path(path))
            self._store_bytes += size
        if found:
            logger.info(
                "KV store holds %d block files, %.1f GiB",
                len(found),
                self._store_bytes / (1 << 30),
            )
        if reaped:
            logger.info("KV store: removed %d stale temp file(s)", reaped)

    def _key_from_path(self, path: str) -> OffloadKey | None:
        """Invert ``FileMapper.get_file_name``. None for another layout's file."""
        stem = os.path.splitext(os.path.basename(path))[0]
        group = os.path.basename(os.path.dirname(path)).rpartition("_g")[2]
        try:
            key = make_offload_key(bytes.fromhex(stem), int(group))
        except (ValueError, OverflowError):
            return None
        return key if self.file_mapper.get_file_name(key) == path else None

    def _on_removed(self, keys: list[OffloadKey]) -> None:
        """Report removed blocks to KV event consumers and the lookup cache."""
        if not keys:
            return
        if self.events is not None:
            self.events.append(
                OffloadingEvent(
                    keys=keys,
                    medium=self.medium,
                    removed=True,
                    locality=self.locality,
                )
            )
        # Override cached HITs only. Pending probes must keep their phase, and
        # they check the disk after this.
        states = self._lookup_manager._lookup_state
        self._lookup_manager.mark_miss(
            [k for k in keys if (s := states.get(k)) is not None and s.result]
        )

    def _pinned_paths(self) -> set[str]:
        pinned: set[str] = set()
        for paths in self._store_job_paths.values():
            pinned.update(paths)
        for paths in self._load_job_paths.values():
            pinned.update(paths)
        return pinned

    def _evict_to(self, target_bytes: int) -> None:
        """Delete oldest block files until the store is at or under target."""
        if not self._max_store_bytes or self._store_bytes <= target_bytes:
            return
        pinned = self._pinned_paths()
        evicted = 0
        removed: list[OffloadKey] = []
        skipped: list[tuple[str, tuple[int, OffloadKey | None]]] = []
        while self._store_bytes > target_bytes and self._store_index:
            path, entry = self._store_index.popitem(last=False)
            if path in pinned:
                skipped.append((path, entry))
                continue
            try:
                os.remove(path)
            except FileNotFoundError:
                pass
            except OSError as exc:
                logger.warning("KV store eviction of %s failed: %s", path, exc)
                # The file still occupies the store; keep it indexed so the
                # accounting stays honest and a later pass retries it.
                skipped.append((path, entry))
                continue
            size, key = entry
            self._store_bytes -= size
            evicted += 1
            if key is not None:
                removed.append(key)
        self._evicted_since_stats += evicted
        self._on_removed(removed)
        # Pinned files go back at the head, still the oldest.
        for path, entry in reversed(skipped):
            self._store_index[path] = entry
            self._store_index.move_to_end(path, last=False)
        if evicted and not self._cap_announced:
            self._cap_announced = True
            logger.info(
                "KV store reached its %.1f GiB cap; evicting oldest blocks "
                "(%d this step). Raise max_size_gib on the tier for more "
                "persistence.",
                self._max_store_bytes / (1 << 30),
                evicted,
            )

    @classmethod
    def build_metric_definitions(
        cls, extra_config: dict
    ) -> dict[str, OffloadingMetricMetadata]:
        definitions = dict(super().build_metric_definitions(extra_config))
        definitions[STORE_BYTES_METRIC] = OffloadingGaugeMetadata(
            documentation="Bytes of KV block files in the fs tier store."
        )
        definitions[EVICTED_BLOCKS_METRIC] = OffloadingCounterMetadata(
            documentation="KV block files evicted from the fs tier store by its cap."
        )
        return definitions

    def get_stats(self) -> OffloadingConnectorStats | None:
        stats = super().get_stats() or OffloadingConnectorStats()
        stats.set_gauge(STORE_BYTES_METRIC, self._store_bytes)
        if self._evicted_since_stats:
            stats.increase_counter(EVICTED_BLOCKS_METRIC, self._evicted_since_stats)
            self._evicted_since_stats = 0
        return stats

    def _account_store(self, paths: dict[str, OffloadKey]) -> None:
        for path, key in paths.items():
            if path in self._store_index:
                self._store_index.move_to_end(path)
            else:
                self._store_index[path] = (self._block_size, key)
                self._store_bytes += self._block_size

    def _on_loaded(self, paths: list[str]) -> None:
        for path in paths:
            if path in self._store_index:
                self._store_index.move_to_end(path)

    def get_finished_jobs(self) -> Iterable[JobResult]:
        results = list(super().get_finished_jobs())
        for result in results:
            store_paths = self._store_job_paths.pop(result.job_id, None)
            if store_paths is not None:
                if not result.success:
                    # A failed batch stops at the first bad block. Count the
                    # files it did write so they stay under the cap.
                    store_paths = {
                        p: k for p, k in store_paths.items() if os.path.exists(p)
                    }
                self._account_store(store_paths)
                continue
            load_paths = self._load_job_paths.pop(result.job_id, None)
            if load_paths is None:
                continue
            # Upstream keeps the blocks loaded before a failure
            # (successful_keys); the rest are a miss.
            n_loaded = len(load_paths)
            if not result.success:
                n_loaded = len(result.successful_keys or ())
                # Upstream deletes a short-read file; stop counting it.
                removed: list[OffloadKey] = []
                for path in load_paths[n_loaded:]:
                    if path in self._store_index and not os.path.exists(path):
                        size, key = self._store_index.pop(path)
                        self._store_bytes -= size
                        if key is not None:
                            removed.append(key)
                self._on_removed(removed)
            self._on_loaded(load_paths[:n_loaded])
        return results

    def submit_store(self, job_metadata: TransferJob) -> None:
        paths = {self.file_mapper.get_file_name(key): key for key in job_metadata.keys}
        self._store_job_paths[job_metadata.job_id] = paths
        # Make room first so the store never crosses the cap. Deduped files
        # over-reserve by their size for one step.
        if self._max_store_bytes:
            self._evict_to(self._max_store_bytes - len(paths) * self._block_size)
        super().submit_store(job_metadata)

    def submit_load(self, job_metadata: TransferJob) -> None:
        self._load_job_paths[job_metadata.job_id] = [
            self.file_mapper.get_file_name(key) for key in job_metadata.keys
        ]
        super().submit_load(job_metadata)
