# SPDX-License-Identifier: Apache-2.0
"""Real scheduler role reuse must drain compact state before new KV writes."""

from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import torch
from vllm import SamplingParams
from vllm.config import ObservabilityConfig, ParallelConfig, SchedulerConfig
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.kv_cache_interface import (
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    KVCacheTensor,
    MambaSpec,
)
from vllm.v1.outputs import ModelRunnerOutput
from vllm.v1.request import Request, RequestStatus

from tests.stub_runner import make_cache_config, make_gdn_hybrid_plan
from vllm_metal.attention.runtime.hybrid import HybridPagedAttentionRuntime
from vllm_metal.metal import get_ops


def _scheduler_and_runtime():
    # Two live requests occupy the whole nine-block pool (one null block,
    # one KV block and three state blocks per request). A longer successor
    # must therefore recycle old state IDs as attention IDs.
    blocks, page_bytes = 9, 2048
    attention = FullAttentionSpec(
        block_size=16, num_kv_heads=1, head_size=32, dtype=torch.float16
    )
    state = MambaSpec(
        block_size=512,
        shapes=((1, 4), (1, 4, 32)),
        dtypes=(torch.float16, torch.float32),
        page_size_padded=page_bytes,
        mamba_cache_mode="none",
    )
    groups = [KVCacheGroupSpec(["layers.3.self_attn"], attention)]
    groups.extend(
        KVCacheGroupSpec([f"layers.{i}.linear_attn"], state) for i in range(3)
    )
    kv = KVCacheConfig(
        num_blocks=blocks,
        kv_cache_groups=groups,
        kv_cache_tensors=[
            KVCacheTensor(
                size=blocks * page_bytes,
                layers=group.layer_names,
                layer_stride=blocks * page_bytes,
                block_stride=page_bytes,
            )
            for group in groups
        ],
        kv_cache_layout="LBNHC",
    )
    cache = make_cache_config(
        block_size=16,
        mamba_block_size=512,
        mamba_cache_mode="none",
        enable_prefix_caching=False,
    )
    cache.num_gpu_blocks = blocks
    config = SimpleNamespace(
        scheduler_config=SchedulerConfig(
            max_model_len=512,
            is_encoder_decoder=False,
            max_num_batched_tokens=64,
            max_num_seqs=4,
            async_scheduling=False,
            enable_chunked_prefill=True,
            long_prefill_token_threshold=64,
        ),
        cache_config=cache,
        parallel_config=ParallelConfig(),
        observability_config=ObservabilityConfig(),
        model_config=SimpleNamespace(
            uses_mrope=False,
            uses_xdrope=False,
            is_encoder_decoder=False,
            is_diffusion=False,
            max_model_len=512,
            enable_return_routed_experts=False,
            return_sampling_mask=False,
        ),
        lora_config=None,
        kv_events_config=None,
        is_mm_encoder_only=False,
        kv_transfer_config=None,
        ec_transfer_config=None,
        ec_manager_config=SimpleNamespace(get_encoder_cache_manager_obj=lambda: None),
        speculative_config=None,
        num_speculative_tokens=0,
        num_lookahead_tokens=0,
        num_prefill_lookahead_tokens=0,
        max_in_flight_tokens=64,
        max_concurrent_batches=1,
        use_v2_model_runner=False,
    )
    scheduler = Scheduler(
        config,
        kv,
        structured_output_manager=SimpleNamespace(should_advance=lambda *a, **k: False),
        block_size=512,
        hash_block_size=16,
        mm_registry=SimpleNamespace(supports_multimodal_inputs=lambda _: False),
    )
    runtime = HybridPagedAttentionRuntime(
        hybrid_plan=make_gdn_hybrid_plan(
            4,
            [3],
            conv_kernel_dim=2,
            conv_dim=4,
            num_v_heads=1,
            value_head_dim=4,
            key_head_dim=32,
        ),
        dtype=mx.float16,
        mamba_cache_mode="none",
    )
    runtime.initialize_from_config(kv)
    return scheduler, runtime


def _request(name, tokens, offset):
    return Request(
        request_id=name,
        prompt_token_ids=list(range(offset, offset + tokens)),
        sampling_params=SamplingParams(temperature=0, max_tokens=128, ignore_eos=True),
        pooling_params=None,
    )


def test_none_mode_state_to_kv_reuse_flushes_pending_before_zero_and_write():
    scheduler, runtime = _scheduler_and_runtime()
    for name, offset in (("a", 0), ("b", 100)):
        scheduler.add_request(_request(name, 1, offset))
    for _ in range(2):
        step = scheduler.schedule()
        assert step.num_scheduled_tokens == {"a": 1, "b": 1}
        # Only token production is synthetic; allocation, completion,
        # cancellation, free-list ordering, and re-admission are upstream's.
        scheduler.update_from_output(
            step,
            ModelRunnerOutput(
                req_ids=["a", "b"],
                req_id_to_index={"a": 0, "b": 1},
                sampled_token_ids=[[1001], [1001]],
            ),
        )
    previous = [scheduler.kv_cache_manager.get_block_ids(name) for name in ("a", "b")]
    old_state_ids = {
        block for rows in previous for group in rows[1:] for block in group
    }
    for layer in range(3):
        slots = [rows[layer + 1][0] for rows in previous]
        runtime.state_cache.set_pending_recurrent_state(
            layer, slots, mx.full((2, 1, 4, 32), layer + 7, dtype=mx.float32)
        )
    assert all(runtime.state_cache.has_pending_recurrent_state(i) for i in range(3))

    scheduler.finish_requests(["a", "b"], RequestStatus.FINISHED_ABORTED)
    scheduler.add_request(_request("c", 33, 200))
    step = scheduler.schedule()
    assert step.num_scheduled_tokens == {"c": 33}
    kv_ids = scheduler.kv_cache_manager.get_block_ids("c")[0]
    reused = old_state_ids.intersection(kv_ids)
    assert reused, "the real scheduler must reassign at least one state ID to KV"
    assert reused.issubset(step.new_block_ids_to_zero)

    # Match execute_model's lifecycle and pre-forward zeroing order. Releasing
    # marks pending state for materialization; zero_blocks drains overlapping
    # updates before recording writes into the same shared allocation.
    runtime.release_requests(step.finished_req_ids)
    runtime.zero_blocks(step.new_block_ids_to_zero)
    for slots in runtime.state_cache.pending_recurrent_slot_ids:
        assert not reused.intersection(slots or [])
    mx.eval(runtime.storage.buffer)
    for block in reused:
        assert not runtime.storage.tensors["layers.3.self_attn"][block].any()

    cache = runtime.kv_cache
    ids = sorted(reused)
    new_k, new_v = get_ops().reshape_and_cache(
        mx.full((len(ids), 1, 32), 3, dtype=mx.float16),
        mx.full((len(ids), 1, 32), 5, dtype=mx.float16),
        cache.key_caches[0],
        cache.value_caches[0],
        mx.array([block * 16 for block in ids], dtype=mx.int64),
    )
    cache.replace_layer_cache(0, new_k, new_v)
    runtime.materialize_pending_state()
    mx.eval(runtime.storage.buffer)
    for block in ids:
        np.testing.assert_array_equal(np.array(cache.key_caches[0][block, 0]), 3)
        np.testing.assert_array_equal(np.array(cache.value_caches[0][block, 0]), 5)
