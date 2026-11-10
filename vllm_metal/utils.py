# SPDX-License-Identifier: Apache-2.0
"""Metal utility functions for vLLM Metal plugin."""

import logging
import mmap
import os
import re
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

logger = logging.getLogger(__name__)

# Stride for the probe's touch: the OS page size, which is the granularity a
# fault commits. One write per stride commits every page it covers, and the
# value comes from the runtime rather than a page size assumed here.
_PAGE_STRIDE = mmap.PAGESIZE

# ``vm_stat`` prints both counters; only the second one tracks the swap file.
_PAGE_SIZE_RE = re.compile(r"page size of (\d+) bytes")
_SWAP_OUTS_RE = re.compile(r"^Swapouts:\s*(\d+)\.", re.MULTILINE)
_COMPRESSIONS_RE = re.compile(r"^Compressions:\s*(\d+)\.", re.MULTILINE)
_VM_STAT_TIMEOUT_SECONDS = 5.0


def _parse_page_counter(
    vm_stat_output: str, counter: re.Pattern[str], name: str
) -> int:
    page_size = _PAGE_SIZE_RE.search(vm_stat_output)
    pages = counter.search(vm_stat_output)
    if page_size is None or pages is None:
        raise ValueError(f"vm_stat output is missing its page size or {name} line")
    return int(pages.group(1)) * int(page_size.group(1))


def _parse_swap_out_bytes(vm_stat_output: str) -> int:
    """Cumulative swap-file writes in bytes, from ``vm_stat`` output.

    ``vm_stat`` reports ``Pageouts`` and ``Swapouts`` as separate counters and
    both are in pages; the swap file is the second one.
    """
    return _parse_page_counter(vm_stat_output, _SWAP_OUTS_RE, "Swapouts")


def _parse_compressed_bytes(vm_stat_output: str) -> int:
    """Cumulative bytes the kernel has compressed, from ``vm_stat`` output.

    macOS answers memory pressure with the compressor first and the swap file
    only once the compressor is full, so a sample the compressor absorbed shows
    up here and not in ``Swapouts``.
    """
    return _parse_page_counter(vm_stat_output, _COMPRESSIONS_RE, "Compressions")


class PagingCounters(NamedTuple):
    """The kernel's cumulative displacement counters, in bytes."""

    swap_out_bytes: int
    compressed_bytes: int


def _paging_counters() -> PagingCounters:
    """The kernel's cumulative swap-file and compressor counters.

    Not ``psutil.swap_memory().sout``: on macOS that is the *Pageouts* counter,
    which counts page-outs generally and so also moves for file writeback that
    has nothing to do with memory pressure, while missing swap the kernel does
    under pressure. ``vm_stat`` reports ``Swapouts`` for the swap file itself.
    Swap occupancy is wrong for a different reason: it falls as readily as it
    rises, so a machine that pages 128 MiB out while reclaiming 128 MiB back
    reads as no change at all.

    Raises ``RuntimeError`` when ``vm_stat`` is missing, fails, hangs past its
    timeout, or prints something else. The probe is a default-on safety gate,
    so it must not pass an unverified plan silently; the error names
    ``VLLM_METAL_KV_COMMIT_PROBE=0`` as the explicit opt-out.
    """
    try:
        output = subprocess.run(
            ["/usr/bin/vm_stat"],
            capture_output=True,
            text=True,
            check=True,
            timeout=_VM_STAT_TIMEOUT_SECONDS,
        ).stdout
        return PagingCounters(
            swap_out_bytes=_parse_swap_out_bytes(output),
            compressed_bytes=_parse_compressed_bytes(output),
        )
    except (OSError, subprocess.SubprocessError, ValueError) as exc:
        raise RuntimeError(
            f"KV commit probe could not read vm_stat ({exc}); set "
            "VLLM_METAL_KV_COMMIT_PROBE=0 to skip the probe."
        ) from exc


def get_model_download_path(
    model_repo_name: str, *, revision: str | None = None
) -> str:
    """
    Get the path to the model, downloading from ModelScope if configured, otherwise will pass the model_repo_name.

    When VLLM_USE_MODELSCOPE=True, downloads the model from ModelScope (modelscope.cn)
    instead of HuggingFace. Useful in regions where HuggingFace is slow or blocked.

    Args:
        model_repo_name: Model repo name from HuggingFace or ModelScope
        revision: Requested ModelScope revision; HuggingFace loaders receive it separately.

    Returns:
        Local folder path (string) of repo snapshot

    Example:

    ```bash
    VLLM_USE_MODELSCOPE=True VLLM_METAL_MODELSCOPE_CACHE=/path/to/cache vllm serve Qwen/Qwen2.5-0.5B
    ```
    """
    if Path(model_repo_name).exists():
        return model_repo_name

    # Reuse vLLM core's own env parsing (accepts "1"/"true") so the plugin
    # and core cannot drift apart on which spellings enable ModelScope.
    from vllm.envs import VLLM_USE_MODELSCOPE

    if VLLM_USE_MODELSCOPE:
        try:
            from modelscope.hub.snapshot_download import snapshot_download

            import vllm_metal.envs as envs

            model_cache_dir = envs.VLLM_METAL_MODELSCOPE_CACHE

            logger.info(f"Downloading model {model_repo_name} from ModelScope...")
            model_path = snapshot_download(
                model_repo_name,
                cache_dir=model_cache_dir,
                **({"revision": revision} if revision is not None else {}),
            )
            logger.info(f"Model downloaded to {model_path}")
            return str(model_path)
        except ImportError:
            logger.warning(
                "modelscope not installed, falling back to default loader (HuggingFace)"
            )
        except Exception as e:
            logger.warning(f"Failed to download from ModelScope: {e}")

    # Fallback: Let mlx_lm or mlx_vlm handle the download natively from HuggingFace
    return model_repo_name


def resolve_mlx_lm_model_path(
    model_name: str | Path, *, revision: str | None = None
) -> Path:
    """Resolve a local checkpoint using mlx-lm's file filters and Hub selection."""
    model_path = Path(model_name)
    if model_path.exists():
        return model_path

    from mlx_lm.utils import DEFAULT_ALLOW_PATTERNS

    if os.getenv("MLXLM_USE_MODELSCOPE", "False").lower() == "true":
        from modelscope import snapshot_download
    else:
        from huggingface_hub import snapshot_download

    return Path(
        snapshot_download(
            str(model_name), revision=revision, allow_patterns=DEFAULT_ALLOW_PATTERNS
        )
    )


def set_wired_limit() -> None:
    """
    Set Metal wired memory limit for optimal GPU performance.

    Pins model weights in GPU-accessible memory to prevent memory paging
    and GPU stalls during inference.

    See: https://github.com/ml-explore/mlx-lm/pull/652
    """
    try:
        import mlx.core as mx

        device_info = mx.device_info()
        max_wired = int(device_info.get("max_recommended_working_set_size", 0))
        if max_wired > 0:
            mx.set_wired_limit(max_wired)
            logger.info(f"Set Metal wired_limit to {max_wired / (1024**3):.1f} GB")
    except Exception as e:
        logger.warning(f"Failed to set wired_limit: {e}")


@dataclass(frozen=True)
class CommitProbe:
    """What forcing ``probed_bytes`` resident did to the machine.

    ``available_before`` is the machine before the touch; ``available_after``
    is the machine once the sample has been dropped again.
    """

    probed_bytes: int
    swap_out_before: int
    swap_out_after: int
    compressed_before: int
    compressed_after: int
    available_before: int
    available_after: int
    seconds: float

    @property
    def swap_out_bytes(self) -> int:
        """Bytes the kernel wrote to the swap file while the sample was resident.

        A cumulative count, so it cannot be hidden by reclamation, unlike swap
        occupancy.
        """
        return max(0, self.swap_out_after - self.swap_out_before)

    @property
    def compressed_bytes(self) -> int:
        """Bytes the kernel compressed while the sample was resident."""
        return max(0, self.compressed_after - self.compressed_before)

    @property
    def displaced_bytes(self) -> int:
        """Bytes the kernel moved out of the way to back the sample.

        The compressor and the swap file are the kernel's two answers to
        pressure, in that order, so both counters add up to one signal.
        """
        return self.swap_out_bytes + self.compressed_bytes

    def describe(self) -> str:
        return (
            f"forced {self.probed_bytes / 2**20:.0f} MiB resident in "
            f"{self.seconds:.2f}s: available "
            f"{self.available_before / 2**30:.1f}->{self.available_after / 2**30:.1f} GiB, "
            f"compressed +{self.compressed_bytes / 2**20:.0f} MiB, "
            f"swap out +{self.swap_out_bytes / 2**20:.0f} MiB"
        )


def probe_commit(nbytes: int) -> CommitProbe:
    """Force ``nbytes`` resident, measure what that cost, then drop it again.

    A lazily allocated KV pool is unbacked, so nothing tells the machine -- or
    the user -- whether the pool fits until a request writes a block, and by
    then the answer arrives mid-generation as a swap storm or a jetsam kill.
    Touching one byte per page asks the question at startup, where the answer
    is still cheap to act on: size the pool down, or refuse to start.

    The sample lives in its own anonymous mapping, so closing it drops the pages
    outright: this is a check, not a reservation, and it does not hand the lazy
    pool a resident footprint. (Anonymous pages freed by ``munmap`` are recycled
    rather than written to swap, so nothing of the sample is left for the next
    allocation to inherit.)

    Paging is read from the kernel's cumulative compressor and swap-file
    counters (see :func:`_paging_counters`) rather than swap occupancy, which
    both rises and falls during a probe and so can hide the writes this is
    looking for. The compressor comes first: macOS compresses other pages to
    make room before it writes swap, so a probe that reads only the swap file
    misses a sample the compressor absorbed.
    ``available_after`` is read once the sample has
    been dropped, so the pair describes the machine before the probe and with
    the probe's memory already back.

    Raises whatever mapping ``nbytes`` raises -- a machine that cannot map the
    buffer has already answered.
    """
    import psutil

    if nbytes <= 0:
        raise ValueError("probe_commit needs a positive size")

    available_before = int(psutil.virtual_memory().available)
    before = _paging_counters()
    started = time.perf_counter()

    buf = mmap.mmap(-1, nbytes)
    try:
        # One store per stride, through a memoryview: the write commits the
        # page, and an indexed store costs less than building a slice per page.
        with memoryview(buf) as view:
            for offset in range(0, nbytes, _PAGE_STRIDE):
                view[offset] = 1
    finally:
        buf.close()

    # Read after the mapping is gone, so these describe the machine with the
    # probe's memory already back. Swap-outs are cumulative, so the paging the
    # probe caused is counted whether it happened before or after the release.
    after = _paging_counters()
    available_after = int(psutil.virtual_memory().available)

    return CommitProbe(
        probed_bytes=nbytes,
        swap_out_before=before.swap_out_bytes,
        swap_out_after=after.swap_out_bytes,
        available_before=available_before,
        available_after=available_after,
        seconds=time.perf_counter() - started,
        compressed_before=before.compressed_bytes,
        compressed_after=after.compressed_bytes,
    )
