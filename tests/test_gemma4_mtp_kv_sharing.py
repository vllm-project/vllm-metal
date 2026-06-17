# SPDX-License-Identifier: Apache-2.0
"""Tests for Gemma4 MTP assistant KV sharing on Metal."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import mlx.core as mx
import pytest

from tests.stub_runner import make_stub_runner
from vllm_metal.attention.caches.kv_cache import MetalPagedKVCache
from vllm_metal.attention.context import get_context
from vllm_metal.attention.impls.sdpa_wrapper import (
    SDPAPagedAttentionWrapper,
    patch_sdpa_attention,
)
from vllm_metal.attention.runtime.mha import MHAPagedAttentionRuntime
from vllm_metal.v1.gemma4_mtp import (
    Gemma4MTPAssistantMetadata,
    Gemma4MTPAssistantRuntime,
    Gemma4MTPDraftSeed,
    Gemma4MTPKVSharingBinding,
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


def test_kv_sharing_binding_records_runner_local_cache() -> None:
    plan = Gemma4MTPKVSharingPlan(
        assistant_layer_to_target_cache_idx=(1, 0),
        layer_types=("full_attention", "sliding_attention"),
    )
    cache = _target_cache(num_layers=2)

    binding = Gemma4MTPKVSharingBinding.from_plan(
        plan,
        target_kv_cache=cache,
        block_size=_BLOCK_SIZE,
    )

    assert binding.plan is plan
    assert binding.target_kv_cache is cache
    assert binding.block_size == _BLOCK_SIZE


def test_kv_sharing_binding_rejects_missing_target_cache_slot() -> None:
    plan = Gemma4MTPKVSharingPlan(
        assistant_layer_to_target_cache_idx=(1,),
        layer_types=("full_attention",),
    )
    cache = _target_cache(num_layers=1)

    with pytest.raises(ValueError, match="outside target cache"):
        Gemma4MTPKVSharingBinding.from_plan(
            plan,
            target_kv_cache=cache,
            block_size=_BLOCK_SIZE,
        )


def test_runtime_keeps_cached_assistant_model_unpatched() -> None:
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
    original_attn = [layer.self_attn for layer in model.model.layers]

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
    assert wired.kv_sharing is not None
    assert wired.kv_sharing.target_kv_cache is cache
    assert wired.forward_ready is True
    assert all(
        layer.self_attn is original_attn[idx]
        for idx, layer in enumerate(model.model.layers)
    )
    assert not any(
        isinstance(layer.self_attn, SDPAPagedAttentionWrapper)
        for layer in model.model.layers
    )


def test_runtime_kv_binding_is_not_process_global() -> None:
    layer_types = ("full_attention",)
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=object())])
    )
    runtime = Gemma4MTPAssistantRuntime(
        model_name="/assistant",
        model=model,
        metadata=_assistant_metadata(layer_types),
    )
    target_metadata = _target_metadata(layer_types)
    cache1 = _target_cache(num_layers=1)
    cache2 = _target_cache(num_layers=1)

    wired1 = runtime.with_target_kv_sharing(
        target_metadata=target_metadata,
        target_kv_cache=cache1,
        block_size=_BLOCK_SIZE,
    )
    wired2 = runtime.with_target_kv_sharing(
        target_metadata=target_metadata,
        target_kv_cache=cache2,
        block_size=_BLOCK_SIZE,
    )

    assert runtime.kv_sharing is None
    assert wired1.kv_sharing is not None
    assert wired2.kv_sharing is not None
    assert wired1.kv_sharing.target_kv_cache is cache1
    assert wired2.kv_sharing.target_kv_cache is cache2
    assert not isinstance(
        model.model.layers[0].self_attn,
        SDPAPagedAttentionWrapper,
    )


def test_runtime_proposes_drafts_with_runner_local_kv_context() -> None:
    captured: dict[str, object] = {}

    class AssistantModel:
        def draft_token_ids(
            self,
            input_ids,
            *,
            target_hidden_states,
            target_input_embeddings,
            target_kv_cache,
            target_cache_indices,
        ):
            ctx = get_context()
            assert ctx is not None
            captured["input_ids"] = input_ids.tolist()
            captured["hidden_states"] = target_hidden_states.tolist()
            captured["embeddings"] = target_input_embeddings.tolist()
            captured["cache"] = target_kv_cache
            captured["cache_indices"] = target_cache_indices
            captured["context_lens"] = ctx.context_lens
            captured["offsets"] = ctx.offsets
            captured["block_tables"] = ctx.block_tables
            return mx.array([101, 102], dtype=mx.int32)

    layer_types = ("full_attention",)
    runtime = Gemma4MTPAssistantRuntime(
        model_name="/assistant",
        model=AssistantModel(),
        metadata=Gemma4MTPAssistantMetadata(
            model_type="gemma4_assistant",
            architectures=("Gemma4AssistantForCausalLM",),
            vocab_size=16,
            hidden_size=8,
            backbone_hidden_size=4,
            tie_word_embeddings=True,
            num_hidden_layers=1,
            layer_types=layer_types,
            use_ordered_embeddings=False,
        ),
    )
    cache = _target_cache(num_layers=1)
    wired = runtime.with_target_kv_sharing(
        target_metadata=Gemma4MTPTargetMetadata(
            vocab_size=16,
            hidden_size=4,
            non_shared_layer_types=layer_types,
        ),
        target_kv_cache=cache,
        block_size=_BLOCK_SIZE,
    )

    drafts = wired.propose_draft_token_ids(
        seeds=(
            Gemma4MTPDraftSeed(
                req_id="r0",
                token_id=7,
                target_hidden_row=0,
                target_position=1,
                block_ids=(0,),
            ),
            Gemma4MTPDraftSeed(
                req_id="r1",
                token_id=8,
                target_hidden_row=2,
                target_position=3,
                block_ids=(0,),
            ),
        ),
        target_hidden_states=mx.array(
            [[1.0, 0.0, 0.0, 0.0], [2.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]
        ),
        target_input_embeddings=mx.ones((1, 2, 4)),
    )

    assert drafts == [[101], [102]]
    assert captured["input_ids"] == [[7, 8]]
    assert captured["hidden_states"] == [[[1.0, 0.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]]
    assert captured["embeddings"] == [[[1.0, 1.0, 1.0, 1.0], [1.0, 1.0, 1.0, 1.0]]]
    assert captured["cache"] is cache
    assert captured["cache_indices"] == (0,)
    assert captured["context_lens"] == [2, 4]
    assert captured["offsets"] == [1, 3]
    assert captured["block_tables"] == [[0], [0]]
    assert get_context() is None


def test_runtime_runs_tiny_assistant_forward_over_target_kv() -> None:
    from vllm_metal.v1.gemma4_mtp_model import (
        Gemma4MTPAssistantModel,
        Gemma4MTPAssistantModelArgs,
    )

    layer_types = ("full_attention",)
    hidden_size = 64
    model = Gemma4MTPAssistantModel(
        Gemma4MTPAssistantModelArgs(
            vocab_size=8,
            backbone_hidden_size=hidden_size,
            text_config={
                "model_type": "gemma4_text",
                "vocab_size": 8,
                "hidden_size": hidden_size,
                "intermediate_size": hidden_size * 2,
                "num_attention_heads": 1,
                "num_key_value_heads": 1,
                "head_dim": hidden_size,
                "global_head_dim": hidden_size,
                "num_hidden_layers": 1,
                "layer_types": list(layer_types),
                "hidden_size_per_layer_input": 0,
                "num_kv_shared_layers": 1,
                "use_double_wide_mlp": False,
            },
        )
    )
    runtime = Gemma4MTPAssistantRuntime(
        model_name="/assistant",
        model=model,
        metadata=Gemma4MTPAssistantMetadata(
            model_type="gemma4_assistant",
            architectures=("Gemma4AssistantForCausalLM",),
            vocab_size=8,
            hidden_size=hidden_size,
            backbone_hidden_size=hidden_size,
            tie_word_embeddings=True,
            num_hidden_layers=1,
            layer_types=layer_types,
            use_ordered_embeddings=False,
        ),
    )
    wired = runtime.with_target_kv_sharing(
        target_metadata=Gemma4MTPTargetMetadata(
            vocab_size=8,
            hidden_size=hidden_size,
            non_shared_layer_types=layer_types,
        ),
        target_kv_cache=MetalPagedKVCache(
            num_layers=1,
            num_kv_heads=1,
            head_dim=hidden_size,
            num_blocks=1,
            block_size=_BLOCK_SIZE,
            dtype=mx.float16,
        ),
        block_size=_BLOCK_SIZE,
    )

    drafts = wired.propose_draft_token_ids(
        seeds=(
            Gemma4MTPDraftSeed(
                req_id="r0",
                token_id=1,
                target_hidden_row=0,
                target_position=0,
                block_ids=(0,),
            ),
        ),
        target_hidden_states=mx.ones((1, hidden_size)),
        target_input_embeddings=mx.ones((1, 1, hidden_size)),
    )

    assert len(drafts) == 1
    assert len(drafts[0]) == 1
    assert 0 <= drafts[0][0] < 8


def test_reused_wrapper_rebinds_cache_through_owner_method() -> None:
    old_cache = _target_cache(num_layers=2)
    new_cache = _target_cache(num_layers=2)
    wrapper = SDPAPagedAttentionWrapper(
        object(),
        layer_idx=0,
        kv_cache=old_cache,
        block_size=_BLOCK_SIZE,
        cache_idx=0,
    )
    model = SimpleNamespace(
        model=SimpleNamespace(layers=[SimpleNamespace(self_attn=wrapper)])
    )

    patched = patch_sdpa_attention(
        model,
        new_cache,
        _BLOCK_SIZE,
        cache_idx_map={0: 1},
    )

    assert patched == 1
    assert wrapper._mk_kv_cache is new_cache
    assert wrapper._mk_cache_idx == 1


def test_cache_policy_installs_gemma4_mtp_kv_sharing() -> None:
    backend = MHAPagedAttentionRuntime(
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

    runner._cache_policy.install_gemma4_mtp_kv_sharing(
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
        runner._cache_policy.install_gemma4_mtp_kv_sharing(
            object(),
            block_size=_BLOCK_SIZE,
        )
