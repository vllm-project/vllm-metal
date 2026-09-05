# SPDX-License-Identifier: Apache-2.0
"""macOS-tuned filesystem secondary tier for Metal KV offloading.

Subclasses upstream ``FileSystemTierManager`` to use the operating system
properly on macOS; the tier semantics (file naming, atomic writes, dedup,
lookup) are inherited unchanged:

- **F_NOCACHE** on block file descriptors. macOS has no ``O_DIRECT``, so
  upstream's ``probe_o_direct`` reports it unavailable and falls back to
  buffered I/O; multi-GB KV churn would then flow through the Unified Buffer
  Cache and evict genuinely useful page cache. ``fcntl(F_NOCACHE)`` is the
  macOS idiom for this write-once/read-rarely streaming pattern (advisory, no
  alignment requirements).
- **0o600 block files under a 0o700 root.** KV blocks are conversation-derived
  data (prompt content is recoverable from them), and with ``PYTHONHASHSEED``
  pinned the content-hash filenames let anyone who can list the directory
  test for the presence of known prompts. Upstream creates 0o644 files.
- **Spotlight exclusion.** Blocks are stored under a ``blocks.noindex``
  subdirectory; the ``.noindex`` name suffix is the reliable per-directory
  Spotlight opt-out. Backup exclusion is left to the user (``tmutil
  addexclusion``), since it is a sticky change to their backup config.

Scheduler-side only (no mlx import): ``MetalTieringOffloadingSpec`` routes the
``fs`` tier type here through upstream's ``module_path`` hook.
"""

from __future__ import annotations

import fcntl
import functools
import os
import stat
from typing import TYPE_CHECKING

from vllm.logger import init_logger

# The tmp-suffix and mkdir helpers are reused so the atomic-replace protocol
# cannot drift from upstream's.
from vllm.v1.kv_offload.tiering.fs.io import (
    _ensure_dirs,
    _get_tmp_suffix,
    _validate_offsets,
)
from vllm.v1.kv_offload.tiering.fs.manager import FileSystemTierManager

from vllm_metal.config import get_config

if TYPE_CHECKING:
    from vllm.v1.kv_offload.tiering.base import TransferJob

logger = init_logger(__name__)

NOINDEX_DIRNAME = "blocks.noindex"


# Only the F_NOCACHE constant is Darwin-specific (fcntl itself is POSIX,
# imported unconditionally above so non-darwin mypy runs resolve the name).
_F_NOCACHE: int | None = getattr(fcntl, "F_NOCACHE", None)


def _set_nocache(fd: int) -> None:
    """Advise the UBC not to retain pages for this fd (best-effort).

    Must be called after open and BEFORE the first read or write. Setting it
    afterwards is a measured no-op: page-cache growth is then identical to
    buffered I/O (0.25 of bytes moved, against 0.02 when set first).
    """
    if _F_NOCACHE is None:
        return
    try:
        fcntl.fcntl(fd, _F_NOCACHE, 1)
    except OSError as exc:
        logger.warning_once("F_NOCACHE failed: %s", exc)


# On-disk format is upstream's byte-for-byte: block_size KV bytes, no header
# and no footer, so a store written here is readable by an upstream fs tier and
# vice versa.
#
# That means undetected corruption is possible: a torn write on power loss, bit
# rot, or a stale writer produces a right-sized file whose contents are wrong,
# and it will be restored into the KV cache without complaint. Upstream carries
# the same exposure on Linux (neither side fsyncs before the atomic replace,
# and O_DIRECT has the same torn-page behaviour as F_NOCACHE), and accepts it.
# Metal accepts it on the same terms rather than diverging the format. If this
# ever needs closing, close it upstream so both platforms benefit.


def store_block(
    dest_path: str,
    buffer: memoryview,
    offset: int,
    block_size: int,
) -> None:
    """Upstream ``_store_block`` with F_NOCACHE and 0o600 files."""
    _validate_offsets(buffer, [offset], block_size)
    if os.path.exists(dest_path):
        return

    tmp_path = dest_path + _get_tmp_suffix()
    _ensure_dirs(dest_path)

    view_slice = buffer.cast("B")[offset : offset + block_size]
    try:
        fd = os.open(tmp_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY | os.O_TRUNC, 0o600)
        try:
            _set_nocache(fd)
            written = os.write(fd, view_slice)
            if written < block_size:
                raise OSError(
                    f"Short write: expected {block_size} bytes, wrote {written}"
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
    """Upstream ``_load_block`` with F_NOCACHE.

    Deletion semantics match upstream exactly: a short read is provable
    corruption and the file goes, every other error (fd exhaustion, EIO)
    propagates untouched. F_NOCACHE is the only difference, and it exists
    because upstream hard-codes the open flags around ``use_o_direct``
    rather than taking an opener."""
    _validate_offsets(view, [offset], block_size)
    fd: int | None = None
    view_slice = view.cast("B")[offset : offset + block_size]
    truncated = False
    try:
        fd = os.open(source_path, os.O_RDONLY)
        _set_nocache(fd)
        bytes_read = os.readv(fd, [view_slice])
        if bytes_read < block_size:
            truncated = True
            raise OSError(f"Short read: expected {block_size} bytes, read {bytes_read}")
    except Exception:
        if truncated:
            try:
                os.remove(source_path)
            except OSError as cleanup_exc:
                logger.warning(
                    "Failed to remove truncated block file %s: %s",
                    source_path,
                    cleanup_exc,
                )
        raise
    finally:
        if fd is not None:
            os.close(fd)


def batch_store_block(
    paths: list[str],
    view: memoryview,
    offsets: list[int],
    block_size: int,
) -> None:
    """Upstream ``batch_store_block``, F_NOCACHE variant.

    Upstream's fast path is a C extension whose open flags are fixed at
    build time, so the Python loop is the only place F_NOCACHE can be set.
    """
    _validate_offsets(view, offsets, block_size)
    for path, offset in zip(paths, offsets, strict=True):
        store_block(path, view, offset, block_size)


def batch_load_block(
    paths: list[str],
    view: memoryview,
    offsets: list[int],
    block_size: int,
) -> None:
    """Upstream ``batch_load_block``, F_NOCACHE variant.

    On failure the OSError carries ``num_succeeded``, which upstream's
    get_finished_jobs uses to keep the blocks that did load and mark only
    the rest a miss. Dropping that attribute would silently turn a partial
    failure into a full recompute.
    """
    _validate_offsets(view, offsets, block_size)
    for i, (path, offset) in enumerate(zip(paths, offsets, strict=True)):
        try:
            load_block(path, view, offset, block_size)
        except OSError as exc:
            exc.num_succeeded = i  # type: ignore[attr-defined]
            raise


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

    Owner-only permissions (on directories this code creates) and a
    ``.noindex`` nesting level so Spotlight never indexes the block churn.
    Idempotent; safe to call on every startup against an existing store.
    """
    _make_private_dir(root_dir)
    if os.path.basename(os.path.normpath(root_dir)).endswith(".noindex"):
        return root_dir
    nested = os.path.join(root_dir, NOINDEX_DIRNAME)
    _make_private_dir(nested)
    return nested


def layout_signature() -> str:
    """Directory component for KV layouts upstream's ``FileMapper`` cannot see.

    ``FileMapper`` already hashes model name, model dtype, tokens_per_hash,
    blocks_per_file and the KV cache groups into the store path, so those
    layouts are disjoint on disk without help. TurboQuant is the exception:
    it changes bytes per element while leaving every one of those fields
    unchanged, and upstream knows nothing about it. Two runs of the same
    model with different quant settings would otherwise share block paths
    with incompatible byte layouts, where a load whose on-disk file is
    larger than the reader's block silently restores wrong bytes and a short
    read DELETES the other config's valid files.

    Empty when TurboQuant is off, so the common case adds no nesting.
    Deterministic for a fixed config, so cross-restart reuse is preserved.
    """
    cfg = get_config()
    if not cfg.turboquant:
        return ""
    return f"tq-{cfg.k_quant}-{cfg.v_quant}"


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
        # The factory passes the class name through as tier_type because the
        # tier is selected by module_path. Report "fs" so metric labels and
        # log lines match the CUDA path.
        tier_type = "fs"
        store_dir = prepare_root_dir(root_dir)
        signature = layout_signature()
        if signature:
            store_dir = os.path.join(store_dir, signature)
        _make_private_dir(store_dir)
        if store_dir != root_dir:
            logger.info("KV store blocks live under %s (Spotlight-excluded)", store_dir)
        super().__init__(
            offloading_spec, primary_kv_view, tier_type, store_dir, **kwargs
        )
        # super().__init__ just logged "O_DIRECT is not supported ... falling
        # back to buffered I/O". True of upstream's io path, false of this
        # one: macOS has no O_DIRECT and F_NOCACHE does the same job. Say so,
        # or the first thing a user sees from this tier contradicts it.
        logger.info(
            "Metal fs tier uses fcntl(F_NOCACHE) in place of O_DIRECT; the "
            "buffered-I/O fallback logged above does not apply to it."
        )

    # Upstream binds its batch io callbacks from its own module namespace;
    # re-issue them with the F_NOCACHE variants. One task per job, exactly as
    # upstream does, so job bookkeeping, partial-keep on a failed load and the
    # pool's in-flight accounting all behave identically.
    def submit_store(self, job_metadata: TransferJob) -> None:
        keys = list(job_metadata.keys)
        if self.events is not None:
            self._store_job_keys[job_metadata.job_id] = keys
        task = functools.partial(
            batch_store_block,
            [self.file_mapper.get_file_name(key) for key in keys],
            self._primary_kv_view,
            [int(bid) * self._block_size for bid in job_metadata.block_ids],
            self._block_size,
        )
        self._pool.enqueue_store(job_metadata.job_id, 1, [task])

    def submit_load(self, job_metadata: TransferJob) -> None:
        job_id = job_metadata.job_id
        keys = list(job_metadata.keys)
        self._load_job_keys[job_id] = keys
        paths = [self.file_mapper.get_file_name(key) for key in keys]
        offsets = [int(bid) * self._block_size for bid in job_metadata.block_ids]

        def load_task() -> None:
            try:
                batch_load_block(
                    paths, self._primary_kv_view, offsets, self._block_size
                )
            except OSError as exc:
                # Runs on a pool thread; written before task_done publishes the
                # job, so get_finished_jobs reads it safely under the GIL.
                self._load_progress[job_id] = getattr(exc, "num_succeeded", 0)
                raise

        self._pool.enqueue_load(job_id, 1, [load_task])
