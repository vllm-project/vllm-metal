# SPDX-License-Identifier: Apache-2.0
"""Metal utility functions for vLLM Metal plugin."""

import logging
import mmap
import time
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# Stride for the probe's touch: the OS page size, which is the granularity a
# fault commits. One write per stride commits every page it covers, and the
# value comes from the runtime rather than a page size assumed here.
_PAGE_STRIDE = mmap.PAGESIZE


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


def set_wired_limit() -> None:
    """
    Set Metal wired memory limit for optimal GPU performance.

    Pins model weights in GPU-accessible memory to prevent memory paging
    and GPU stalls during inference.

    See: https://github.com/ml-explore/mlx-lm/pull/652
    """
    try:
        import mlx.core as mx

        device_info = mx.metal.device_info()
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
    available_before: int
    available_after: int
    seconds: float

    @property
    def swap_out_bytes(self) -> int:
        """Bytes the kernel wrote to swap while the sample was resident.

        Cumulative swap-out counts, not occupancy: occupancy falls as readily
        as it rises, so a machine that pages 128 MiB out while reclaiming
        128 MiB back reads as no change at all.
        """
        return max(0, self.swap_out_after - self.swap_out_before)

    def describe(self) -> str:
        return (
            f"forced {self.probed_bytes / 2**20:.0f} MiB resident in "
            f"{self.seconds:.2f}s: available "
            f"{self.available_before / 2**30:.1f}->{self.available_after / 2**30:.1f} GiB, "
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

    Paging is read from the cumulative swap-out counter rather than swap
    occupancy, which both rises and falls during a probe and so can hide the
    writes this is looking for. ``available_after`` is read once the sample has
    been dropped, so the pair describes the machine before the probe and with
    the probe's memory already back.

    Raises whatever mapping ``nbytes`` raises -- a machine that cannot map the
    buffer has already answered.
    """
    import psutil

    if nbytes <= 0:
        raise ValueError("probe_commit needs a positive size")

    available_before = int(psutil.virtual_memory().available)
    swap_out_before = int(psutil.swap_memory().sout)
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
    swap_out_after = int(psutil.swap_memory().sout)
    available_after = int(psutil.virtual_memory().available)

    return CommitProbe(
        probed_bytes=nbytes,
        swap_out_before=swap_out_before,
        swap_out_after=swap_out_after,
        available_before=available_before,
        available_after=available_after,
        seconds=time.perf_counter() - started,
    )
