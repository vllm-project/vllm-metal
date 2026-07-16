# SPDX-License-Identifier: Apache-2.0
"""Unit tests for the Metal KV offloading worker (plain-numpy host pool).

Round-trips blocks through the host pool and asserts byte-exact restoration,
including bf16 (numpy-dtype-less) caches, TurboQuant packed caches, and
offloaded blocks spanning multiple GPU blocks (block_size_factor > 1) with an
unaligned first block.
"""

import mlx.core as mx
import numpy as np
import pytest
from vllm.v1.kv_offload.base import GPULoadStoreSpec
from vllm.v1.kv_offload.cpu.common import CPULoadStoreSpec

from tests.kv_offload_helpers import (
    gpu_spec,
    make_cache,
    randomize,
    snapshot_blocks,
    zero_blocks,
)
from vllm_metal.v1.kv_offload.worker import MetalKVOffloadWorker


def _run_roundtrip(
    cache,
    *,
    block_size_factor: int = 1,
    gpu_blocks: list[int],
    cpu_blocks: list[int],
    block_index: int = 0,
) -> None:
    worker = MetalKVOffloadWorker(
        cache, block_size_factor=block_size_factor, num_cpu_blocks=4
    )

    randomize(cache)
    original = snapshot_blocks(cache, gpu_blocks)

    # Store GPU -> CPU.
    ok = worker.submit_store(
        1, gpu_spec(gpu_blocks, block_index), CPULoadStoreSpec(cpu_blocks)
    )
    assert ok
    finished = worker.get_finished()
    assert [r.job_id for r in finished] == [1]
    assert finished[0].success
    assert finished[0].transfer_size > 0

    # Clobber the GPU blocks, then load them back CPU -> GPU.
    zero_blocks(cache, gpu_blocks)
    ok = worker.submit_load(
        2, CPULoadStoreSpec(cpu_blocks), gpu_spec(gpu_blocks, block_index)
    )
    assert ok
    finished = worker.get_finished()
    assert [r.job_id for r in finished] == [2]

    restored = snapshot_blocks(cache, gpu_blocks)
    for a, b in zip(original, restored, strict=True):
        np.testing.assert_array_equal(a, b)


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_roundtrip_dtypes(dtype: mx.Dtype) -> None:
    cache = make_cache(dtype)
    _run_roundtrip(cache, gpu_blocks=[5, 2, 7], cpu_blocks=[0, 1, 3])


def test_roundtrip_turboquant() -> None:
    cache = make_cache(
        head_dim=64,
        turboquant=True,
        k_quant="q8_0",
        v_quant="q3_0",
    )
    _run_roundtrip(cache, gpu_blocks=[1, 6], cpu_blocks=[2, 0])


def test_roundtrip_block_size_factor() -> None:
    """One CPU block spans two GPU blocks."""
    cache = make_cache()
    _run_roundtrip(
        cache,
        block_size_factor=2,
        gpu_blocks=[3, 4, 5, 6],
        cpu_blocks=[1, 3],
    )


def test_roundtrip_unaligned_first_block() -> None:
    """Request enters its first offloaded block mid-block (skip one sub-slot)."""
    cache = make_cache()
    worker = MetalKVOffloadWorker(cache, block_size_factor=2, num_cpu_blocks=4)
    randomize(cache)

    # First store gpu blocks [2,3] as the two sub-slots of CPU block 1.
    assert worker.submit_store(
        1, gpu_spec([2, 3], block_index=0), CPULoadStoreSpec([1])
    )

    # Now store one more gpu block at logical block index 3: skip = 3 % 2 = 1,
    # so the worker must place it in the second sub-slot of CPU block 2.
    assert worker.submit_store(2, gpu_spec([5], block_index=3), CPULoadStoreSpec([2]))
    assert len(worker.get_finished()) == 2

    original = snapshot_blocks(cache, [5])
    zero_blocks(cache, [5])
    assert worker.submit_load(3, CPULoadStoreSpec([2]), gpu_spec([5], block_index=3))
    assert len(worker.get_finished()) == 1
    restored = snapshot_blocks(cache, [5])
    for a, b in zip(original, restored, strict=True):
        np.testing.assert_array_equal(a, b)


def test_unknown_spec_types_raise() -> None:
    """Errors must propagate: the connector-side caller asserts a successful
    submission (job failure is unsupported upstream), so raising here surfaces
    the root cause instead of a bare assert — swallowing it would hide it."""
    cache = make_cache()
    worker = MetalKVOffloadWorker(cache, block_size_factor=1, num_cpu_blocks=4)
    with pytest.raises(ValueError, match="unexpected load spec types"):
        worker.submit_load(1, CPULoadStoreSpec([0]), CPULoadStoreSpec([1]))
    with pytest.raises(ValueError, match="unexpected store spec types"):
        worker.submit_store(2, CPULoadStoreSpec([0]), CPULoadStoreSpec([1]))
    assert worker.get_finished() == []


def test_uncovered_per_block_array_rejected() -> None:
    """CACHE_ATTRS completeness guard: a per-block array list the offload
    inventory doesn't know about must fail registration, not silently drop
    state (the 'new model family' corruption mode)."""
    cache = make_cache()
    cache.gdn_conv_state = [
        mx.zeros((cache.num_blocks, 4, 8)) for _ in range(2)
    ]  # simulate a new model family's extra per-block state
    with pytest.raises(NotImplementedError, match="gdn_conv_state"):
        MetalKVOffloadWorker(cache, block_size_factor=1, num_cpu_blocks=4)


def test_out_of_range_block_ids_raise() -> None:
    """OOB ids corrupt silently at the MLX layer (take returns garbage,
    scatter drops writes) — the worker must fail fast instead."""
    cache = make_cache()  # 8 GPU blocks
    worker = MetalKVOffloadWorker(cache, block_size_factor=1, num_cpu_blocks=4)
    with pytest.raises(ValueError, match="GPU block ids out of range"):
        worker.submit_store(1, gpu_spec([5, 99]), CPULoadStoreSpec([0, 1]))
    with pytest.raises(ValueError, match="CPU block ids out of range"):
        worker.submit_store(2, gpu_spec([5, 2]), CPULoadStoreSpec([0, 7]))
    with pytest.raises(ValueError, match="CPU block ids out of range"):
        worker.submit_store(3, gpu_spec([5]), CPULoadStoreSpec([-1]))
    assert worker.get_finished() == []


def test_expected_block_bytes_overrun_raises() -> None:
    """Carving MORE than the spec sized (row overrun) is fatal; carving less
    is upstream's alignment padding and must be accepted (TurboQuant blocks
    are rarely page-aligned — found live on a clean-machine e2e)."""
    cache = make_cache()
    with pytest.raises(ValueError, match="rows would overrun"):
        MetalKVOffloadWorker(
            cache,
            block_size_factor=1,
            num_cpu_blocks=4,
            expected_block_bytes=100,  # far below the real carve
        )


def test_expected_block_bytes_padding_accepted() -> None:
    """Upstream tiering rounds blocks up to SharedOffloadRegion's
    BLOCK_SIZE_ALIGNMENT — mmap.PAGESIZE (16384 on Apple Silicon)."""
    import mmap

    from tests.kv_offload_helpers import gpu_block_bytes

    cache = make_cache()
    page = mmap.PAGESIZE
    aligned = -(-gpu_block_bytes(cache) // page) * page  # upstream round_up
    MetalKVOffloadWorker(
        cache,
        block_size_factor=1,
        num_cpu_blocks=4,
        expected_block_bytes=aligned,
    )


def test_expected_block_bytes_match_accepted() -> None:
    from tests.kv_offload_helpers import gpu_block_bytes

    cache = make_cache()
    MetalKVOffloadWorker(
        cache,
        block_size_factor=1,
        num_cpu_blocks=4,
        expected_block_bytes=gpu_block_bytes(cache),
    )


def test_unsupported_cache_dtype_rejected() -> None:
    cache = make_cache()
    cache.key_caches[0] = mx.zeros(cache.key_caches[0].shape, dtype=mx.int64)
    with pytest.raises(NotImplementedError, match="unsupported cache dtype"):
        MetalKVOffloadWorker(cache, block_size_factor=1, num_cpu_blocks=4)


def test_too_few_cpu_blocks_raise() -> None:
    cache = make_cache()
    worker = MetalKVOffloadWorker(cache, block_size_factor=2, num_cpu_blocks=4)
    with pytest.raises(ValueError, match="CPU blocks too few"):
        worker.submit_store(1, gpu_spec([1, 2, 3]), CPULoadStoreSpec([0]))


def test_empty_group_is_a_successful_noop() -> None:
    cache = make_cache()
    worker = MetalKVOffloadWorker(cache, block_size_factor=1, num_cpu_blocks=4)
    spec = GPULoadStoreSpec([], group_sizes=[0], block_indices=[0])
    assert worker.submit_store(1, spec, CPULoadStoreSpec([]))
    (result,) = worker.get_finished()
    assert result.success and result.transfer_size == 0
