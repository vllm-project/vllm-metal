# SPDX-License-Identifier: Apache-2.0
"""Functional parity of the Metal offload path with vLLM's CUDA path.

The scheduler side needs no parity testing: it *is* the upstream code
(asserted below by function identity). The worker-side worker is the one
reimplemented piece, so its observable contract — which bytes land in which
host-pool slot for a given TransferSpec — is checked here against upstream's
own placement arithmetic: ``compute_sub_block_ptrs`` from
``vllm/v1/kv_offload/cpu/gpu_worker.py``, the exact function the CUDA DMA
path uses to build its copy descriptors (it is pure numpy/torch pointer math
and runs on any platform).

For every transfer the CUDA worker copies GPU block ``i`` to the byte range
``[ptrs[i], ptrs[i] + page_size)`` of the CPU tensor, where ``ptrs`` comes
from ``compute_sub_block_ptrs`` over the CPU tensor with the transfer's
``skip = block_index % block_size_factor``. The Metal worker must place the
same bytes at the same offsets of its host pool, per cache array.
"""

import numpy as np
import pytest
import torch
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec
from vllm.v1.kv_offload.cpu.gpu_worker import compute_sub_block_ptrs

from tests.kv_offload_helpers import (
    gpu_spec,
    make_cache,
    randomize,
    snapshot_blocks,
    zero_blocks,
)
from vllm_metal.v1.kv_offload.worker import MetalKVOffloadWorker

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

    ``pool`` is one cache array's host pool, shape
    (num_cpu_blocks, factor, *block_shape). Upstream sees it as the flat
    (num_cpu_blocks, row_bytes) int8 CPU tensor it computes pointers over.
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
    if turboquant:
        cache = make_cache(head_dim=64, turboquant=True, k_quant="q8_0", v_quant="q3_0")
    else:
        cache = make_cache()
    worker = MetalKVOffloadWorker(
        cache, block_size_factor=factor, num_cpu_blocks=NUM_CPU_BLOCKS
    )
    randomize(cache)
    original = snapshot_blocks(cache, gpu_blocks)

    assert worker.submit_store(
        1, gpu_spec(gpu_blocks, block_index), CPULoadStoreSpec(cpu_blocks)
    )
    assert [r.job_id for r in worker.get_finished()] == [1]

    skip = block_index % factor
    for ref, blocks in zip(worker._refs, original, strict=True):
        pool_bytes = ref.cpu.reshape(-1).view(np.uint8)
        block_bytes = blocks.reshape(len(gpu_blocks), -1).view(np.uint8)
        page = block_bytes.shape[1]
        offsets = _expected_offsets(ref.cpu, factor, cpu_blocks, skip, len(gpu_blocks))
        for i, off in enumerate(offsets):
            np.testing.assert_array_equal(
                pool_bytes[off : off + page],
                block_bytes[i],
                err_msg=f"gpu block {gpu_blocks[i]} not at CUDA offset {off}",
            )

    # Load back through the Metal worker from the CUDA-verified layout, so
    # both directions are pinned to the same placement.
    zero_blocks(cache, gpu_blocks)
    assert worker.submit_load(
        2, CPULoadStoreSpec(cpu_blocks), gpu_spec(gpu_blocks, block_index)
    )
    assert [r.job_id for r in worker.get_finished()] == [2]
    for a, b in zip(original, snapshot_blocks(cache, gpu_blocks), strict=True):
        np.testing.assert_array_equal(a, b)


def test_scheduler_side_is_upstream_code() -> None:
    """The scheduler-facing surface is upstream's own functions, not ports."""
    from vllm.distributed.kv_transfer.kv_connector.v1.offloading_connector import (
        OffloadingConnector,
    )
    from vllm.v1.kv_offload.cpu.spec import CPUOffloadingSpec

    from vllm_metal.v1.kv_offload.connector import MetalOffloadingConnector
    from vllm_metal.v1.kv_offload.spec import MetalOffloadingSpec

    # The CPU-tier spec inherits the manager (LRU/ARC policies, store
    # thresholds, block tracking) unmodified.
    assert MetalOffloadingSpec.get_manager is CPUOffloadingSpec.get_manager

    # The connector overrides only worker-side KV-cache registration; every
    # scheduler-side hook is inherited.
    overridden = {
        name for name in vars(MetalOffloadingConnector) if not name.startswith("_")
    }
    assert overridden == {"register_kv_caches"}, overridden
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
