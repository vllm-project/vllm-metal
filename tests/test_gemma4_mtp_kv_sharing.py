# SPDX-License-Identifier: Apache-2.0
"""Tests for Gemma4 MTP assistant KV sharing on Metal."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import mlx.core as mx
import pytest

from tests.stub_runner import make_stub_runner
from vllm_metal.metal_kernel_backend.cache import MetalPagedKVCache
from vllm_metal.metal_kernel_backend.paged_attention import (
    MetalKernelPagedAttentionWrapper,
    patch_model_attention_metal_kernel,
)
from vllm_metal.paged_attention_backend.mha import MHAPagedAttentionBackend
from vllm_metal.paged_attention_common import (
    PagedAttentionContext,
    clear_context,
    set_context,
)
from vllm_metal.v1.gemma4_mtp import (
    Gemma4MTPAssistantMetadata,
    Gemma4MTPAssistantRuntime,
    Gemma4MTPKVSharingPlan,
    Gemma4MTPTargetMetadata,
)

_BLOCK_SIZE = 8


def _assistant_metadata(
    layer_types: tuple[str, ...],
) -> Gemma4MTPAssistantMetadata:
    return Gemma4MTPAssistantMetadata(
        model_type="gemma4_assistant",
        architectures=("Gemma4AssistantForCausalLM",),
        vocab_size=262144,
        hidden_size=256,
        backbone_hidden_size=1536,
        tie_word_embeddings=True,
        num_hidden_layers=len(layer_types),
        layer_types=layer_types,
        use_ordered_embeddings=True,
    )


def _target_metadata(
    layer_types: tuple[str, ...],
) -> Gemma4MTPTargetMetadata:
    return Gemma4MTPTargetMetadata(
        vocab_size=262144,
        hidden_size=1536,
        non_shared_layer_types=layer_types,
    )


def _target_args(
    layer_types: list[str] | None = None,
) -> dict[str, Any]:
    if layer_types is None:
        layer_types = [
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        ]
    return {
        "model_type": "gemma4_text",
        "vocab_size": 262144,
        "hidden_size": 1536,
        "num_hidden_layers": len(layer_types),
        "num_kv_shared_layers": 0,
        "layer_types": layer_types,
    }


def _target_cache(num_layers: int) -> MetalPagedKVCache:
    return MetalPagedKVCache(
        num_layers=num_layers,
        num_kv_heads=1,
        head_dim=4,
        num_blocks=1,
        block_size=_BLOCK_SIZE,
        dtype=mx.float16,
    )


def test_kv_sharing_plan_uses_last_non_shared_target_layer_by_type() -> None:
    assistant = _assistant_metadata(
        (
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        )
    )
    target = _target_metadata(
        (
            "sliding_attention",
            "full_attention",
            "sliding_attention",
            "sliding_attention",
            "full_attention",
        )
    )

    plan = Gemma4MTPKVSharingPlan.from_metadata(
        assistant=assistant,
        target=target,
    )

    assert plan.assistant_layer_to_target_cache_idx == (3, 3, 4)
    assert plan.layer_types == assistant.layer_types


def test_kv_sharing_plan_rejects_non_tail_layout() -> None:
    assistant = _assistant_metadata(("full_attention",))
    target = _target_metadata(("full_attention", "sliding_attention"))

    with pytest.raises(ValueError, match="tail-match"):
        Gemma4MTPKVSharingPlan.from_metadata(
            assistant=assistant,
            target=target,
        )


def test_patch_assistant_model_installs_read_only_wrappers() -> None:
    plan = Gemma4MTPKVSharingPlan(
        assistant_layer_to_target_cache_idx=(1, 0),
        layer_types=("full_attention", "sliding_attention"),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=object()),
                SimpleNamespace(self_attn=object()),
            ]
        )
    )
    cache = _target_cache(num_layers=2)

    patched = plan.patch_assistant_model(
        model,
        target_kv_cache=cache,
        block_size=_BLOCK_SIZE,
    )

    assert patched == 2
    wrappers = [layer.self_attn for layer in model.model.layers]
    assert all(
        isinstance(wrapper, MetalKernelPagedAttentionWrapper) for wrapper in wrappers
    )
    assert [wrapper._mk_cache_idx for wrapper in wrappers] == [1, 0]
    assert all(wrapper._mk_force_shared_kv for wrapper in wrappers)


def test_patch_assistant_model_rejects_missing_target_cache_slot() -> None:
    plan = Gemma4MTPKVSharingPlan(
        assistant_layer_to_target_cache_idx=(1,),
        layer_types=("full_attention",),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=object())])
    )
    cache = _target_cache(num_layers=1)

    with pytest.raises(ValueError, match="outside target cache"):
        plan.patch_assistant_model(
            model,
            target_kv_cache=cache,
            block_size=_BLOCK_SIZE,
        )


def test_patch_assistant_model_validates_before_mutating() -> None:
    original_attn = object()
    plan = Gemma4MTPKVSharingPlan(
        assistant_layer_to_target_cache_idx=(0, 1),
        layer_types=("sliding_attention", "full_attention"),
    )
    model = SimpleNamespace(
        model=SimpleNamespace(
            layers=[
                SimpleNamespace(self_attn=original_attn),
                SimpleNamespace(),
            ]
        )
    )
    cache = _target_cache(num_layers=2)

    with pytest.raises(ValueError, match="has no attention module"):
        plan.patch_assistant_model(
            model,
            target_kv_cache=cache,
            block_size=_BLOCK_SIZE,
        )

    assert model.model.layers[0].self_attn is original_attn


def test_runtime_patches_real_assistant_model_layers() -> None:
    from vllm_metal.v1.gemma4_mtp_model import (
        Gemma4MTPAssistantModel,
        Gemma4MTPAssistantModelArgs,
    )

    layer_types = ("sliding_attention", "full_attention")
    text_config = {
        "model_type": "gemma4_text",
        "vocab_size": 16,
        "hidden_size": 8,
        "intermediate_size": 16,
        "num_attention_heads": 2,
        "num_key_value_heads": 1,
        "head_dim": 4,
        "global_head_dim": 4,
        "num_hidden_layers": 2,
        "layer_types": list(layer_types),
        "hidden_size_per_layer_input": 0,
    }
    model = Gemma4MTPAssistantModel(
        Gemma4MTPAssistantModelArgs(
            vocab_size=16,
            backbone_hidden_size=12,
            text_config=text_config,
        )
    )
    runtime = Gemma4MTPAssistantRuntime(
        model_name="/assistant",
        model=model,
        metadata=Gemma4MTPAssistantMetadata(
            model_type="gemma4_assistant",
            architectures=("Gemma4AssistantForCausalLM",),
            vocab_size=16,
            hidden_size=8,
            backbone_hidden_size=12,
            tie_word_embeddings=True,
            num_hidden_layers=2,
            layer_types=layer_types,
            use_ordered_embeddings=False,
        ),
    )
    cache = _target_cache(num_layers=2)

    wired = runtime.with_target_kv_sharing(
        target_metadata=Gemma4MTPTargetMetadata(
            vocab_size=16,
            hidden_size=12,
            non_shared_layer_types=layer_types,
        ),
        target_kv_cache=cache,
        block_size=_BLOCK_SIZE,
    )

    assert runtime.kv_sharing_plan is None
    assert wired.kv_sharing_plan is not None
    assert wired.forward_ready is False
    assert [layer.self_attn._mk_force_shared_kv for layer in model.model.layers] == [
        True,
        True,
    ]


def test_force_shared_kv_wrapper_injects_layer_shaped_sentinel() -> None:
    cache = MetalPagedKVCache(
        num_layers=1,
        num_kv_heads=3,
        head_dim=5,
        num_blocks=1,
        block_size=_BLOCK_SIZE,
        dtype=mx.float16,
        kv_heads_per_layer=[3],
        head_dim_per_layer=[5],
    )
    wrapper = MetalKernelPagedAttentionWrapper(
        SimpleNamespace(n_kv_heads=2, head_dim=4),
        layer_idx=0,
        kv_cache=cache,
        block_size=_BLOCK_SIZE,
        cache_idx=0,
        force_shared_kv=True,
    )
    ctx = PagedAttentionContext(
        slot_mapping=[0, 1],
        block_tables=[[0]],
        context_lens=[2],
        offsets=[0],
        cu_seqlens=[0, 2],
    )
    x = mx.ones((1, 2, 7), dtype=mx.float16)
    captured: dict[str, tuple[int, ...]] = {}

    def fake_sdpa_forward(
        _inner: object,
        _x: mx.array,
        _ctx: PagedAttentionContext,
        _cache: MetalPagedKVCache,
        _layer_idx: int,
        *,
        shared_kv: tuple[mx.array, mx.array] | None = None,
    ) -> tuple[mx.array, tuple[mx.array, mx.array]]:
        assert shared_kv is not None
        captured["key_shape"] = shared_kv[0].shape
        return mx.zeros_like(_x), shared_kv

    set_context(ctx)
    try:
        with patch(
            "vllm_metal.metal_kernel_backend.paged_attention.sdpa_forward",
            fake_sdpa_forward,
        ):
            caller_kv = (
                mx.zeros((1, 1, 2, 1), dtype=mx.float16),
                mx.zeros((1, 1, 2, 1), dtype=mx.float16),
            )
            output, kv_pair, offset = wrapper(x, shared_kv=caller_kv)
    finally:
        clear_context()

    assert captured["key_shape"] == (1, 2, 2, 4)
    assert kv_pair[0].shape == (1, 2, 2, 4)
    assert output.shape == x.shape
    assert offset == 0


def test_normal_patch_resets_reused_wrapper_shared_kv_mode() -> None:
    old_cache = _target_cache(num_layers=2)
    new_cache = _target_cache(num_layers=2)
    wrapper = MetalKernelPagedAttentionWrapper(
        object(),
        layer_idx=0,
        kv_cache=old_cache,
        block_size=_BLOCK_SIZE,
        cache_idx=0,
        force_shared_kv=True,
    )
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=wrapper)])
    )

    patched = patch_model_attention_metal_kernel(
        model,
        new_cache,
        _BLOCK_SIZE,
        cache_idx_map={0: 1},
    )

    assert patched == 1
    assert wrapper._mk_kv_cache is new_cache
    assert wrapper._mk_cache_idx == 1
    assert wrapper._mk_force_shared_kv is False


def test_cache_policy_installs_gemma4_mtp_kv_sharing() -> None:
    backend = MHAPagedAttentionBackend(
        num_layers=3,
        num_kv_heads=1,
        head_dim=4,
        block_size=_BLOCK_SIZE,
        dtype=mx.float16,
    )
    backend.initialize(num_blocks=1)
    installed = object()

    class _AssistantRuntime:
        def with_target_kv_sharing(
            self,
            *,
            target_metadata: Gemma4MTPTargetMetadata,
            target_kv_cache: MetalPagedKVCache,
            block_size: int,
        ) -> object:
            assert target_metadata.non_shared_layer_types == (
                "sliding_attention",
                "sliding_attention",
                "full_attention",
            )
            assert target_kv_cache is backend.kv_cache
            assert block_size == _BLOCK_SIZE
            return installed

    layer_types = [
        "sliding_attention",
        "sliding_attention",
        "full_attention",
    ]
    runner = make_stub_runner(
        model_args={
            "vocab_size": 262144,
            "hidden_size": 1536,
        },
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(
                text_config=SimpleNamespace(
                    model_type="gemma4_text",
                    vocab_size=262144,
                    hidden_size=1536,
                    num_hidden_layers=3,
                    num_kv_shared_layers=0,
                    layer_types=layer_types,
                )
            )
        ),
        _gemma4_mtp_assistant=_AssistantRuntime(),
    )

    runner._cache_policy._install_gemma4_mtp_kv_sharing(
        backend,
        block_size=_BLOCK_SIZE,
    )

    assert runner._gemma4_mtp_assistant is installed


def test_cache_policy_rejects_gemma4_mtp_without_mha_backend() -> None:
    runner = make_stub_runner(
        model_args=_target_args(),
        _gemma4_mtp_assistant=object(),
    )

    with pytest.raises(NotImplementedError, match="requires the MHA paged"):
        runner._cache_policy._install_gemma4_mtp_kv_sharing(
            object(),
            block_size=_BLOCK_SIZE,
        )
