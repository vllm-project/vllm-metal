# SPDX-License-Identifier: Apache-2.0
"""Contract tests for the hybrid runtime plan, its GDN family owner and the
state-family factory."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import pytest
import torch
from mlx_lm.models.nemotron_h import (
    Model,
    ModelArgs,
    NemotronHMamba2Mixer,
    NemotronHMLP,
)
from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
from vllm.v1.attention.backends.registry import MambaAttentionBackendEnum
from vllm.v1.kv_cache_interface import MambaSpec

from tests.stub_runner import (
    NEMOTRON_H_TINY_ARGS,
    make_gdn_hybrid_plan,
    make_nemotron_hybrid_plan,
)
from vllm_metal.attention.impls.linear import GDNPagedAttentionWrapper
from vllm_metal.attention.impls.mamba2 import Mamba2PagedStateWrapper
from vllm_metal.attention.impls.sdpa_wrapper import SDPAPagedAttentionWrapper
from vllm_metal.attention.runtime.factory import build_hybrid_runtime_plan
from vllm_metal.attention.runtime.families.gdn import build_gdn_hybrid_plan
from vllm_metal.attention.runtime.families.nemotron_h import (
    build_nemotron_h_hybrid_plan,
)
from vllm_metal.attention.runtime.hybrid import HybridPagedAttentionRuntime
from vllm_metal.attention.runtime.hybrid_plan import (
    ATTENTION_LAYER,
    STATE_LAYER,
    STATELESS_LAYER,
    HybridRuntimePlan,
)

NEMOTRON_H_ARGS = {
    "model_type": "nemotron_h",
    "hybrid_override_pattern": ["M", "E", "M", "*", "-", "M"],
    "mamba_num_heads": 4,
    "mamba_head_dim": 8,
    "ssm_state_size": 32,
    "n_groups": 2,
    "conv_kernel": 4,
}

GDN_ARGS = {
    "model_type": "qwen3_5",
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
        self.in_proj_qkv = nn.Linear(4, 4)
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


def _make_nemotron_runtime() -> HybridPagedAttentionRuntime:
    """Four real Nemotron-H blocks (M-*M) on an fp32 pool sized for the tiny args."""
    return HybridPagedAttentionRuntime(
        hybrid_plan=make_nemotron_hybrid_plan("M-*M"),
        max_num_seqs=2,
        num_kv_heads=2,
        head_dim=8,
        block_size=4,
        dtype=mx.float32,
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


class TestStateFamilyFactory:
    @pytest.mark.parametrize(
        "model_type",
        ["qwen3_5", "qwen3_5_text", "qwen3_5_moe", "qwen3_5_moe_text", "qwen3_next"],
    )
    def test_routes_gdn_model_types_to_the_gdn_family(self, model_type: str) -> None:
        plan = build_hybrid_runtime_plan({**GDN_ARGS, "model_type": model_type}, 8)

        assert plan.family.label == "gdn"
        assert plan.layers.attention_indices == (3, 7)

    def test_overlapping_fields_of_an_unsupported_hybrid_still_reject(self) -> None:
        args = {**GDN_ARGS, "model_type": "falcon_h1", "mamba_num_heads": 8}
        expected = (
            "Metal hybrid runtime has no state family for model_type='falcon_h1'."
        )

        with pytest.raises(NotImplementedError) as excinfo:
            build_hybrid_runtime_plan(args, 8)
        assert str(excinfo.value) == expected

    def test_routes_nemotron_args_to_the_nemotron_family(self) -> None:
        plan = build_hybrid_runtime_plan(NEMOTRON_H_ARGS, 6)

        assert plan.family.label == "nemotron_h"
        assert plan.layers.layer_roles == (
            STATE_LAYER,
            STATELESS_LAYER,
            STATE_LAYER,
            ATTENTION_LAYER,
            STATELESS_LAYER,
            STATE_LAYER,
        )

    def test_malformed_interval_is_a_gdn_error_not_a_miss(self) -> None:
        args = {**GDN_ARGS, "full_attention_interval": "4"}

        with pytest.raises(ValueError, match="must be positive integers"):
            build_hybrid_runtime_plan(args, 8)


class TestNemotronHPlanDecision:
    def test_string_pattern_reads_as_characters(self) -> None:
        plan = build_nemotron_h_hybrid_plan(
            {**NEMOTRON_H_ARGS, "hybrid_override_pattern": "M*E"}, 3
        )

        assert plan.layers.layer_roles == (
            STATE_LAYER,
            ATTENTION_LAYER,
            STATELESS_LAYER,
        )

    def test_spec_is_mamba2_typed_with_fp32_state(self) -> None:
        plan = build_nemotron_h_hybrid_plan(NEMOTRON_H_ARGS, 6)
        expected = MambaSpec(
            shapes=((3, 160), (4, 8, 32)),
            dtypes=(torch.bfloat16, torch.float32),
            block_size=2048,
            page_size_padded=None,
            mamba_type=MambaAttentionBackendEnum.MAMBA2,
            mamba_cache_mode="none",
        )

        spec = plan.state_cache_spec(
            conv_dtype=torch.bfloat16,
            mamba_block_size=2048,
            page_size_padded=None,
            mamba_cache_mode="none",
        )

        assert spec == expected

    def test_geometry_matches_the_built_mixer(self) -> None:
        plan = build_nemotron_h_hybrid_plan(NEMOTRON_H_TINY_ARGS, 2)
        mixer = NemotronHMamba2Mixer(ModelArgs(**NEMOTRON_H_TINY_ARGS))

        geometry = plan.geometry

        assert geometry.conv_kernel_dim == mixer.conv_kernel_size
        assert geometry.conv_dim == mixer.conv_dim
        assert geometry.num_v_heads == mixer.num_heads
        assert geometry.value_head_dim == mixer.head_dim
        assert geometry.key_head_dim == mixer.ssm_state_size

    def test_spec_shapes_match_vllm_mamba2_calculator(self) -> None:
        plan = build_nemotron_h_hybrid_plan(NEMOTRON_H_ARGS, 6)
        expected = MambaStateShapeCalculator.mamba2_state_shape(
            tp_world_size=1,
            intermediate_size=32,
            n_groups=2,
            num_heads=4,
            head_dim=8,
            state_size=32,
            conv_kernel=4,
        )

        spec = plan.state_cache_spec(
            conv_dtype=torch.bfloat16,
            mamba_block_size=2048,
            page_size_padded=None,
            mamba_cache_mode="none",
        )

        assert spec.shapes == expected


class TestNemotronHPlanRejection:
    @pytest.mark.parametrize("missing", ["hybrid_override_pattern", "n_groups"])
    def test_omitted_key_rejects_with_the_key_name(self, missing: str) -> None:
        args = {k: v for k, v in NEMOTRON_H_ARGS.items() if k != missing}
        expected = f"Nemotron-H hybrid model args are missing required {missing!r}."

        with pytest.raises(ValueError) as excinfo:
            build_nemotron_h_hybrid_plan(args, 6)
        assert str(excinfo.value) == expected

    @pytest.mark.parametrize(("value", "shown"), [(0, "0"), ("32", "'32'")])
    def test_non_positive_or_non_int_dim_rejects(self, value, shown: str) -> None:
        args = {**NEMOTRON_H_ARGS, "ssm_state_size": value}
        expected = (
            "Nemotron-H hybrid model args must be positive integers; invalid "
            f"ssm_state_size={shown}."
        )

        with pytest.raises(ValueError) as excinfo:
            build_nemotron_h_hybrid_plan(args, 6)
        assert str(excinfo.value) == expected

    def test_unknown_block_token_rejects(self) -> None:
        args = {**NEMOTRON_H_ARGS, "hybrid_override_pattern": "M*Z"}
        expected = "Nemotron-H hybrid_override_pattern must use M, *, - or E; got 'Z'."

        with pytest.raises(ValueError) as excinfo:
            build_nemotron_h_hybrid_plan(args, 3)
        assert str(excinfo.value) == expected

    @pytest.mark.parametrize(
        "pattern", ["MEME", "*E*"], ids=["no_attention", "no_state"]
    )
    def test_pattern_missing_a_role_rejects(self, pattern: str) -> None:
        args = {**NEMOTRON_H_ARGS, "hybrid_override_pattern": pattern}
        expected = (
            "Nemotron-H hybrid requires both Mamba-2 ('M') and attention ('*') "
            f"blocks, got hybrid_override_pattern={pattern!r}."
        )

        with pytest.raises(ValueError) as excinfo:
            build_nemotron_h_hybrid_plan(args, len(pattern))
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

    def test_nemotron_layout_wraps_only_attention_and_mixers(self) -> None:
        model = Model(
            ModelArgs(**{**NEMOTRON_H_TINY_ARGS, "hybrid_override_pattern": "M-*M"})
        )
        runtime = _make_nemotron_runtime()
        runtime.initialize(num_blocks=2)

        patched = runtime.patch_model(model)

        assert patched == 3
        assert isinstance(model.layers[0].mixer, Mamba2PagedStateWrapper)
        assert isinstance(model.layers[1].mixer, NemotronHMLP)
        assert isinstance(model.layers[2].mixer, SDPAPagedAttentionWrapper)
        assert isinstance(model.layers[3].mixer, Mamba2PagedStateWrapper)
        assert model.layers[0].mixer._mamba2_cache_idx == 0
        assert model.layers[3].mixer._mamba2_cache_idx == 1
        assert model.layers[3].mixer._mamba2_state_cache is runtime.state_cache

    def test_nemotron_repatch_rebinds_the_mixer_wrappers(self) -> None:
        model = Model(
            ModelArgs(**{**NEMOTRON_H_TINY_ARGS, "hybrid_override_pattern": "M-*M"})
        )
        runtime_a = _make_nemotron_runtime()
        runtime_a.initialize(num_blocks=2)
        runtime_a.patch_model(model)
        wrapper_before = model.layers[3].mixer
        runtime_b = _make_nemotron_runtime()
        runtime_b.initialize(num_blocks=2)

        patched = runtime_b.patch_model(model)

        assert patched == 3
        assert model.layers[3].mixer is wrapper_before
        assert wrapper_before._mamba2_state_cache is runtime_b.state_cache
        assert wrapper_before._mamba2_cache_idx == 1

    def test_state_pool_dtype_mismatch_rejects_at_patch_time(self) -> None:
        runtime = HybridPagedAttentionRuntime(
            hybrid_plan=_make_tiny_plan(),
            max_num_seqs=2,
            num_kv_heads=1,
            head_dim=4,
            block_size=4,
            dtype=mx.bfloat16,
        )
        runtime.initialize(num_blocks=2)
        model = _FakeModel("sasa")
        expected = (
            "state pool dtype mlx.core.bfloat16 does not match the layer 0 mixer "
            "dtype mlx.core.float32; pass --dtype matching the checkpoint so "
            "state and activations keep one dtype."
        )

        with pytest.raises(ValueError) as excinfo:
            runtime.patch_model(model)
        assert str(excinfo.value) == expected

    def test_unclassifiable_layer_rejects_with_the_family_label(self) -> None:
        runtime = _make_runtime()
        runtime.initialize(num_blocks=2)

        class _Mystery(nn.Module):
            pass

        model = _FakeModel("sasa")
        model.layers[2] = _Layer(_Mystery(), linear=True)

        with pytest.raises(RuntimeError, match="is not a 'gdn' state module"):
            runtime.patch_model(model)
