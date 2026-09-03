# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the hybrid runtime plan and its GDN family owner."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
import torch
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from tests.stub_runner import make_gdn_hybrid_plan
from vllm_metal.attention.impls.linear import GDNPagedAttentionWrapper
from vllm_metal.attention.impls.sdpa_wrapper import SDPAPagedAttentionWrapper
from vllm_metal.attention.runtime.families.gdn import build_gdn_hybrid_plan
from vllm_metal.attention.runtime.hybrid import HybridPagedAttentionRuntime
from vllm_metal.attention.runtime.hybrid_plan import (
    ATTENTION_LAYER,
    STATE_LAYER,
    HybridRuntimePlan,
)

GDN_ARGS = {
    "full_attention_interval": 4,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 4,
    "linear_key_head_dim": 32,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 3,
}


class _FakeSDPA(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.q_proj = nn.Linear(4, 4)
        self.k_proj = nn.Linear(4, 4)
        self.v_proj = nn.Linear(4, 4)
        self.o_proj = nn.Linear(4, 4)


class _FakeGDN(nn.Module):
    num_k_heads = 1
    num_v_heads = 1

    def __init__(self) -> None:
        super().__init__()
        self.conv1d = nn.Conv1d(4, 4, 2)


class _Layer(nn.Module):
    def __init__(self, attn: nn.Module, linear: bool) -> None:
        super().__init__()
        if linear:
            self.linear_attn = attn
        else:
            self.self_attn = attn


class _FakeModel(nn.Module):
    def __init__(self, roles: str) -> None:
        super().__init__()
        self.layers = [
            _Layer(_FakeGDN() if role == "s" else _FakeSDPA(), linear=(role == "s"))
            for role in roles
        ]


def _make_tiny_plan() -> HybridRuntimePlan:
    """Four layers, attention at 1 and 3, geometry sized for the fakes."""
    return make_gdn_hybrid_plan(
        4,
        [1, 3],
        conv_kernel_dim=2,
        conv_dim=4,
        num_v_heads=1,
        value_head_dim=4,
        key_head_dim=32,
    )


def _make_runtime() -> HybridPagedAttentionRuntime:
    return HybridPagedAttentionRuntime(
        hybrid_plan=_make_tiny_plan(),
        max_num_seqs=2,
        num_kv_heads=1,
        head_dim=4,
        block_size=4,
        dtype=mx.float32,
    )


class TestGdnPlanDecision:
    def test_topology_follows_the_interval_rule(self) -> None:
        plan = build_gdn_hybrid_plan(GDN_ARGS, 8)

        assert plan.layers.attention_indices == (3, 7)
        assert plan.layers.state_indices == (0, 1, 2, 4, 5, 6)
        assert plan.layers.num_attention == 2
        assert plan.layers.num_state == 6
        assert plan.layers.layer_roles == (
            STATE_LAYER,
            STATE_LAYER,
            STATE_LAYER,
            ATTENTION_LAYER,
            STATE_LAYER,
            STATE_LAYER,
            STATE_LAYER,
            ATTENTION_LAYER,
        )

    def test_geometry_packs_qk_and_v_into_the_conv_stream(self) -> None:
        plan = build_gdn_hybrid_plan(GDN_ARGS, 8)

        # conv_dim = 2*32*2 + 4*16 = 192, hand-written.
        assert plan.geometry.conv_kernel_dim == 3
        assert plan.geometry.conv_dim == 192
        assert plan.geometry.num_v_heads == 4
        assert plan.geometry.value_head_dim == 16
        assert plan.geometry.key_head_dim == 32


class TestGdnPlanRejection:
    @pytest.mark.parametrize(
        "missing", ["linear_num_key_heads", "linear_conv_kernel_dim"]
    )
    def test_omitted_linear_dim_rejects_with_the_key_name(self, missing: str) -> None:
        args = {k: v for k, v in GDN_ARGS.items() if k != missing}
        expected = f"GDN hybrid model args are missing required {missing!r}."

        with pytest.raises(ValueError) as excinfo:
            build_gdn_hybrid_plan(args, 8)
        assert str(excinfo.value) == expected

    @pytest.mark.parametrize(
        ("interval", "num_layers"),
        [(1, 8), (9, 8)],
        ids=["no_state_layers", "no_attention_layers"],
    )
    def test_interval_dropping_a_layer_role_rejects(
        self, interval: int, num_layers: int
    ) -> None:
        args = {**GDN_ARGS, "full_attention_interval": interval}
        expected = (
            "GDN hybrid requires 2 <= full_attention_interval <= num_layers so "
            "the model keeps both attention and state layers, got "
            f"full_attention_interval={interval} with num_layers={num_layers}."
        )

        with pytest.raises(ValueError) as excinfo:
            build_gdn_hybrid_plan(args, num_layers)
        assert str(excinfo.value) == expected


class TestStateCacheSpec:
    def test_spec_threads_geometry_family_and_mode(self) -> None:
        plan = make_gdn_hybrid_plan(
            4,
            [1, 3],
            conv_kernel_dim=3,
            conv_dim=192,
            num_v_heads=4,
            value_head_dim=16,
            key_head_dim=32,
        )
        expected = MambaSpec(
            shapes=((2, 192), (4, 16, 32)),
            dtypes=(torch.float16, torch.float32),
            block_size=2048,
            page_size_padded=None,
            mamba_type=MambaAttentionBackendEnum.GDN_ATTN,
            mamba_cache_mode="align",
        )

        spec = plan.state_cache_spec(
            conv_dtype=torch.float16,
            mamba_block_size=2048,
            page_size_padded=None,
            mamba_cache_mode="align",
        )

        assert spec == expected


class TestRuntimeUsesThePlan:
    def test_unsupported_cache_mode_rejects_with_the_family_label(self) -> None:
        expected = (
            "hybrid paged attention does not support mamba_cache_mode='all' "
            "for the 'gdn' state family (supported: ('none', 'align'))"
        )

        with pytest.raises(NotImplementedError) as excinfo:
            HybridPagedAttentionRuntime(
                hybrid_plan=_make_tiny_plan(),
                max_num_seqs=2,
                num_kv_heads=1,
                head_dim=4,
                block_size=4,
                dtype=mx.float32,
                mamba_cache_mode="all",
            )
        assert str(excinfo.value) == expected

    def test_state_cache_is_sized_from_the_plan_geometry(self) -> None:
        runtime = _make_runtime()

        runtime.initialize(num_blocks=2)

        state_cache = runtime.state_cache
        assert state_cache.num_layers == 2
        assert state_cache.conv_kernel_dim == 2
        assert state_cache.conv_dim == 4
        assert state_cache.num_v_heads == 1
        assert state_cache.value_head_dim == 4
        assert state_cache.key_head_dim == 32


class TestHybridPatchModel:
    def test_installs_family_wrappers_at_plan_cache_indices(self) -> None:
        runtime = _make_runtime()
        runtime.initialize(num_blocks=2)
        model = _FakeModel("sasa")

        patched = runtime.patch_model(model)

        assert patched == 4
        sdpa_1 = model.layers[1].self_attn
        sdpa_3 = model.layers[3].self_attn
        gdn_0 = model.layers[0].linear_attn
        gdn_2 = model.layers[2].linear_attn
        assert isinstance(sdpa_1, SDPAPagedAttentionWrapper)
        assert isinstance(sdpa_3, SDPAPagedAttentionWrapper)
        assert isinstance(gdn_0, GDNPagedAttentionWrapper)
        assert isinstance(gdn_2, GDNPagedAttentionWrapper)
        assert sdpa_1._mk_cache_idx == 0
        assert sdpa_3._mk_cache_idx == 1
        assert gdn_0._gdn_cache_idx == 0
        assert gdn_2._gdn_cache_idx == 1
        assert gdn_0._gdn_state_cache is runtime.state_cache
        assert gdn_2._gdn_state_cache is runtime.state_cache

    def test_repatch_rebinds_cached_wrappers_through_owner_methods(self) -> None:
        runtime_a = _make_runtime()
        runtime_a.initialize(num_blocks=2)
        model = _FakeModel("sasa")
        runtime_a.patch_model(model)
        wrapper_before = model.layers[0].linear_attn
        runtime_b = _make_runtime()
        runtime_b.initialize(num_blocks=2)

        patched = runtime_b.patch_model(model)

        assert patched == 4
        assert model.layers[0].linear_attn is wrapper_before
        assert wrapper_before._gdn_state_cache is runtime_b.state_cache
        assert wrapper_before._gdn_state_cache is not runtime_a.state_cache
        assert wrapper_before._gdn_cache_idx == 0

    def test_unclassifiable_layer_rejects_with_the_family_label(self) -> None:
        runtime = _make_runtime()
        runtime.initialize(num_blocks=2)

        class _Mystery(nn.Module):
            pass

        model = _FakeModel("sasa")
        model.layers[2] = _Layer(_Mystery(), linear=True)

        with pytest.raises(RuntimeError, match="is not a 'gdn' state module"):
            runtime.patch_model(model)
