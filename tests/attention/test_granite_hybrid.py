# SPDX-License-Identifier: Apache-2.0
"""Granite's family plan and complete paged forwards against pinned mlx-lm."""

from __future__ import annotations

from dataclasses import asdict
from itertools import accumulate

import mlx.core as mx
import numpy as np
import pytest
import torch
from mlx_lm.models.granitemoehybrid import Model, ModelArgs
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum

from tests.stub_runner import initialize_hybrid_runtime
from vllm_metal.attention.context import (
    OffsetCache,
    PagedAttentionContext,
    clear_context,
    set_context,
)
from vllm_metal.attention.runtime.factory import build_hybrid_runtime_plan
from vllm_metal.attention.runtime.hybrid import HybridPagedAttentionRuntime


def _model_args(**overrides) -> ModelArgs:
    return ModelArgs(
        **{
            "model_type": "granitemoehybrid",
            "vocab_size": 128,
            "hidden_size": 128,
            "intermediate_size": 128,
            "num_hidden_layers": 4,
            "max_position_embeddings": 128,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "attention_bias": False,
            "embedding_multiplier": 2.0,
            "attention_multiplier": 0.125,
            "logits_scaling": 4.0,
            "residual_multiplier": 0.22,
            "layer_types": ["mamba", "attention", "mamba", "attention"],
            "rms_norm_eps": 1e-5,
            "rope_theta": 10000.0,
            "position_embedding_type": "nope",
            "mamba_n_heads": 8,
            "mamba_d_head": 32,
            "mamba_d_state": 32,
            "mamba_d_conv": 4,
            "mamba_n_groups": 2,
            "mamba_proj_bias": False,
            "mamba_conv_bias": True,
            **overrides,
        }
    )


def test_granite_plan_matches_mlx_lm_state_and_upstream_spec() -> None:
    args = _model_args()
    model = Model(args)
    dtypes = (torch.bfloat16, torch.float32)
    plan = build_hybrid_runtime_plan(asdict(args), args.num_hidden_layers, dtypes)
    reference_cache = model.make_cache()
    mx.eval(model(mx.array([[1, 2, 3]]), cache=reference_cache))

    expected_shapes = MambaStateShapeCalculator.mamba2_state_shape(
        intermediate_size=args.mamba_n_heads * args.mamba_d_head,
        tp_world_size=1,
        n_groups=args.mamba_n_groups,
        num_heads=args.mamba_n_heads,
        head_dim=args.mamba_d_head,
        state_size=args.mamba_d_state,
        conv_kernel=args.mamba_d_conv,
    )
    assert plan.layers.attention_indices == (1, 3)
    assert plan.layers.state_indices == (0, 2)
    assert plan.geometry.state_shapes == expected_shapes
    for layer_idx in plan.layers.state_indices:
        assert plan.geometry.state_shapes == tuple(
            state.shape[1:] for state in reference_cache[layer_idx].cache
        )
    spec = plan.state_cache_spec(
        mamba_block_size=16, page_size_padded=None, mamba_cache_mode="none"
    )
    assert spec.shapes == expected_shapes
    assert spec.dtypes == dtypes
    assert spec.mamba_type == MambaAttentionBackendEnum.MAMBA2


@pytest.mark.parametrize("moe", [False, True], ids=["dense", "moe"])
@pytest.mark.parametrize("rope", [False, True], ids=["nope", "rope"])
def test_granite_paged_requests_match_mlx_lm_through_slot_reuse(moe, rope) -> None:
    """Compare all logits while interleaving chunks, decode and replacement requests."""
    mx.random.seed(7)
    args = _model_args(
        num_local_experts=4 if moe else None,
        num_experts_per_tok=2 if moe else None,
        shared_intermediate_size=64 if moe else None,
        position_embedding_type="rope" if rope else "nope",
    )
    reference = Model(args)
    paged = Model(args)
    paged.update(reference.parameters())
    plan = build_hybrid_runtime_plan(asdict(args), 4, (torch.float32, torch.float32))
    runtime = HybridPagedAttentionRuntime(
        hybrid_plan=plan,
        dtype=mx.float32,
    )
    initialize_hybrid_runtime(
        runtime,
        6,
        block_size=16,
        num_kv_heads=args.num_key_value_heads,
        head_dim=args.hidden_size // args.num_attention_heads,
    )
    assert runtime.patch_model(paged) == 4
    caches = {req: reference.make_cache() for req in ("a", "b", "c")}
    offsets = dict.fromkeys(caches, 0)
    blocks = {"a": [0, 1], "b": [2, 3], "c": [0, 1]}
    state_blocks = {"a": 4, "b": 5, "c": 4}
    next_tokens = {}

    def step(requests, segments, num_decode=0):
        expected = []
        for req, tokens in zip(requests, segments, strict=True):
            expected.append(reference(mx.array([tokens]), cache=caches[req]))
        cu_seqlens = list(accumulate(map(len, segments), initial=0))
        ctx = PagedAttentionContext(
            slot_mapping=[
                blocks[req][pos // 16] * 16 + pos % 16
                for req, tokens in zip(requests, segments, strict=True)
                for pos in range(offsets[req], offsets[req] + len(tokens))
            ],
            block_tables=[blocks[req] for req in requests],
            context_lens=[
                offsets[req] + len(tokens)
                for req, tokens in zip(requests, segments, strict=True)
            ],
            offsets=[offsets[req] for req in requests],
            cu_seqlens=cu_seqlens,
            num_decode_requests=num_decode,
        )
        runtime.populate_step_context(
            req_ids=requests,
            ctx=ctx,
            state_block_ids=[[[state_blocks[req]]] for req in requests],
            step_positions=[
                (offsets[req], len(tokens))
                for req, tokens in zip(requests, segments, strict=True)
            ],
        )
        set_context(ctx)
        try:
            actual = paged(
                mx.array([[token for segment in segments for token in segment]]),
                cache=[OffsetCache(max(ctx.offsets)) for _ in range(4)],
            )
            outputs = [actual]
            runtime.extend_forward_eval_outputs(outputs)
            mx.eval(*outputs)
        finally:
            clear_context()
        for i, (req, tokens) in enumerate(zip(requests, segments, strict=True)):
            result = actual[:, cu_seqlens[i] : cu_seqlens[i + 1]]
            np.testing.assert_allclose(
                np.array(result), np.array(expected[i]), atol=2e-5, rtol=2e-5
            )
            assert mx.array_equal(
                mx.argmax(result, axis=-1), mx.argmax(expected[i], axis=-1)
            )
            next_tokens[req] = int(mx.argmax(result[0, -1]))
            offsets[req] += len(tokens)

    step(["a", "b"], [[1, 2, 3], [4, 5]])
    # A resident decode precedes another request's continued prefill.
    step(["b", "a"], [[next_tokens["b"]], list(range(7, 24))], num_decode=1)
    step(["a", "b"], [[next_tokens["a"]], [next_tokens["b"]]], num_decode=2)
    runtime.release_requests({"a"})
    runtime.materialize_pending_state()
    step(["b", "c"], [[next_tokens["b"]], [13, 14, 15, 16]], num_decode=1)
    for _ in range(4):
        step(["c", "b"], [[next_tokens["c"]], [next_tokens["b"]]], num_decode=2)
    runtime.release_requests({"b", "c"})
    runtime.materialize_pending_state()
