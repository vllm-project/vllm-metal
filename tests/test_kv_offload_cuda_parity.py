# SPDX-License-Identifier: Apache-2.0
"""Metal host-pool placement matches upstream's ``compute_sub_block_ptrs``.

The CUDA worker builds its copy descriptors with that function; it is pure
pointer math and runs anywhere. Each page region is checked separately.
"""

import numpy as np
import pytest
import torch
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import compute_sub_block_ptrs

from tests.kv_offload_helpers import (
    device_ptrs,
    make_storage,
    make_turboquant_storage,
    make_worker,
    randomize,
    snapshot_blocks,
    zero_blocks,
)

NUM_CPU_BLOCKS = 4

CASES = [
    pytest.param(1, [5, 2, 7], [0, 1, 3], 0, id="factor1"),
    pytest.param(2, [1, 2, 3, 4], [0, 2], 0, id="factor2-aligned"),
    pytest.param(2, [5], [2], 3, id="factor2-unaligned"),
    pytest.param(4, [1, 2, 3], [0, 3], 6, id="factor4-unaligned-spanning"),
]


def _expected_offsets(
    pool: np.ndarray, factor: int, cpu_blocks: list[int], skip: int, count: int
) -> np.ndarray:
    """Byte offsets into ``pool`` where upstream CUDA would place each block.

    ``pool`` is one page region's host pool, shape
    (num_cpu_chunks, factor, page_bytes). Upstream sees it as the
    (num_cpu_blocks, row_bytes) int8 CPU tensor it computes pointers over;
    the reshape keeps the region's row stride, so the pointers are the real
    ones.
    """
    rows = pool.reshape(NUM_CPU_BLOCKS, -1).view(np.int8)
    cpu_tensor = torch.from_numpy(rows)
    ptrs = np.empty(count, dtype=np.uint64)
    compute_sub_block_ptrs(
        np.asarray(cpu_blocks), factor, ptrs, cpu_tensor, skip_count=skip
    )
    return (ptrs - np.uint64(cpu_tensor.data_ptr())).astype(np.int64)


@pytest.mark.parametrize(("factor", "gpu_blocks", "cpu_blocks", "block_index"), CASES)
@pytest.mark.parametrize("turboquant", [False, True], ids=["fp16", "turboquant"])
def test_store_placement_matches_cuda(
    factor: int,
    gpu_blocks: list[int],
    cpu_blocks: list[int],
    block_index: int,
    turboquant: bool,
) -> None:
    storage = make_turboquant_storage() if turboquant else make_storage()
    worker = make_worker(
        storage, blocks_per_chunk=factor, num_cpu_chunks=NUM_CPU_BLOCKS
    )
    randomize(storage)
    original = snapshot_blocks(storage, gpu_blocks)
    ptrs = device_ptrs(worker, gpu_blocks, block_index)

    assert worker.submit_store(1, ptrs, CPULoadStoreSpec(cpu_blocks))
    assert [r.job_id for r in worker.get_finished()] == [1]

    skip = block_index % factor
    # Read placement off the region itself: host arrays are strided views.
    region_bytes = (
        np.asarray(worker.region.create_kv_memoryview()).reshape(-1).view(np.uint8)
    )
    region_ptr = region_bytes.ctypes.data
    for host, blocks in zip(worker._host, original, strict=True):
        pool_bytes = region_bytes[host.ctypes.data - region_ptr :]
        block_bytes = blocks.reshape(len(gpu_blocks), -1).view(np.uint8)
        page = block_bytes.shape[1]
        offsets = _expected_offsets(host, factor, cpu_blocks, skip, len(gpu_blocks))
        for i, off in enumerate(offsets):
            np.testing.assert_array_equal(
                pool_bytes[off : off + page],
                block_bytes[i],
                err_msg=f"gpu block {gpu_blocks[i]} not at CUDA offset {off}",
            )

    # Load back from the CUDA-verified layout, pinning both directions.
    zero_blocks(storage, gpu_blocks)
    assert worker.submit_load(2, CPULoadStoreSpec(cpu_blocks), ptrs)
    assert [r.job_id for r in worker.get_finished()] == [2]
    for a, b in zip(original, snapshot_blocks(storage, gpu_blocks), strict=True):
        np.testing.assert_array_equal(a, b)


def test_scheduler_side_is_upstream_code() -> None:
    """The scheduler-facing surface is upstream's own functions, not ports."""
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )

    from vllm_metal.v1.kv_offload.connector import MetalOffloadingConnector

    overridden = {
        name for name in vars(MetalOffloadingConnector) if not name.startswith("_")
    }
    # Worker-side registration and sleep/wake hooks only.
    assert overridden == {
        "register_kv_caches",
        "release_host_pool",
        "restore_host_pool",
    }, overridden
    for name in (
        "get_num_new_matched_tokens",
        "update_state_after_alloc",
        "build_connector_meta",
        "request_finished",
        "take_events",
    ):
        assert getattr(MetalOffloadingConnector, name) is getattr(
            OffloadingConnector, name
        ), name
