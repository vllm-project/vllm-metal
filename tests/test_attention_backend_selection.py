# SPDX-License-Identifier: Apache-2.0
"""Selection and integration tests for paged-attention backends."""

from __future__ import annotations

import re
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace

import mlx.core as mx
import pytest

import vllm_metal.attention.dispatch as dispatch
from tests.stub_runner import make_stub_runner
from vllm_metal.attention.caches.turboquant import FWHT_SUPPORTED_HEAD_DIMS
from vllm_metal.attention.runtime.mla import MLAPagedAttentionRuntime
from vllm_metal.attention.runtime.sdpa import SDPAPagedAttentionRuntime


@pytest.fixture(autouse=True)
def _isolated_dispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[None]:
    """Keep process-global registrations and the override local to each test."""
    monkeypatch.setattr(dispatch, "_REGISTRY", dispatch._REGISTRY.copy())
    monkeypatch.delenv("VLLM_METAL_ATTENTION_BACKEND", raising=False)
    yield


def _caps(
    *,
    is_mla: bool = False,
    is_hybrid: bool = False,
    kv_paged: bool = True,
    dtype: mx.Dtype = mx.bfloat16,
    head_dims: tuple[int, ...] = (128,),
    block_size: int = 16,
    turboquant: bool = False,
) -> dispatch.AttnCaps:
    return dispatch.AttnCaps(
        is_mla=is_mla,
        is_hybrid=is_hybrid,
        kv_paged=kv_paged,
        dtype=dtype,
        head_dims=head_dims,
        block_size=block_size,
        turboquant=turboquant,
    )


def test_default_candidates_follow_attention_family() -> None:
    assert dispatch.candidate_backends(_caps()) == ("paged_metal_mha",)
    assert dispatch.candidate_backends(_caps(is_mla=True)) == ("paged_metal_mla",)
    assert dispatch.candidate_backends(_caps(is_hybrid=True)) == ()

    assert dispatch.select_attention_backend(_caps()) is dispatch.PagedMetalMHA
    assert (
        dispatch.select_attention_backend(_caps(is_mla=True)) is dispatch.PagedMetalMLA
    )


@pytest.mark.parametrize(
    ("backend_name", "caps", "reason"),
    [
        ("paged_metal_mha", _caps(is_mla=True), "MLA"),
        ("paged_metal_mla", _caps(), "MLA"),
        ("paged_metal_mha", _caps(is_hybrid=True), "hybrid"),
        ("paged_metal_mla", _caps(is_mla=True, is_hybrid=True), "hybrid"),
    ],
)
def test_forced_backend_reports_incompatible_model(
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
    caps: dispatch.AttnCaps,
    reason: str,
) -> None:
    monkeypatch.setenv("VLLM_METAL_ATTENTION_BACKEND", backend_name)

    with pytest.raises(ValueError) as excinfo:
        dispatch.select_attention_backend(caps)

    message = str(excinfo.value)
    assert backend_name in message
    assert reason.lower() in message.lower()


@pytest.mark.parametrize("backend_name", ["paged_metal_mha", "paged_metal_mla"])
def test_hybrid_runtime_rejects_standalone_override_at_policy_boundary(
    monkeypatch: pytest.MonkeyPatch,
    backend_name: str,
) -> None:
    monkeypatch.setenv("VLLM_METAL_ATTENTION_BACKEND", backend_name)
    runner = make_stub_runner(
        is_hybrid=True,
        num_layers=2,
        num_kv_cache_layers=1,
        num_kv_heads=1,
        head_dim=128,
        kv_cache_dtype=mx.bfloat16,
        cache_config=SimpleNamespace(block_size=16),
    )

    with pytest.raises(ValueError, match="hybrid"):
        runner.build_paged_attention_runtime(block_size=16)


@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("dtype", [mx.int8, mx.int32])
def test_cache_dtype_must_be_supported(dtype: mx.Dtype, is_mla: bool) -> None:
    with pytest.raises(ValueError) as excinfo:
        dispatch.select_attention_backend(_caps(dtype=dtype, is_mla=is_mla))

    message = str(excinfo.value)
    assert ("paged_metal_mla" if is_mla else "paged_metal_mha") in message
    assert "dtype" in message.lower()


@pytest.mark.parametrize("is_mla", [False, True])
@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16, mx.float32])
def test_supported_cache_dtypes_are_accepted(dtype: mx.Dtype, is_mla: bool) -> None:
    backend = dispatch.PagedMetalMLA if is_mla else dispatch.PagedMetalMHA
    assert (
        dispatch.select_attention_backend(_caps(dtype=dtype, is_mla=is_mla)) is backend
    )


@pytest.mark.parametrize("is_mla", [False, True])
def test_matching_builtin_override_is_accepted(
    monkeypatch: pytest.MonkeyPatch, is_mla: bool
) -> None:
    backend = dispatch.PagedMetalMLA if is_mla else dispatch.PagedMetalMHA
    monkeypatch.setenv("VLLM_METAL_ATTENTION_BACKEND", backend.name)

    assert dispatch.select_attention_backend(_caps(is_mla=is_mla)) is backend


@pytest.mark.parametrize(
    ("backend", "caps"),
    [
        (dispatch.PagedMetalMHA, _caps(kv_paged=False)),
        (dispatch.PagedMetalMLA, _caps(is_mla=True, kv_paged=False)),
    ],
)
def test_builtins_require_paged_kv_cache(
    backend: type, caps: dispatch.AttnCaps
) -> None:
    assert any("paged KV cache" in reason for reason in backend.validate(caps))


def test_mha_accepts_every_compiled_head_dimension() -> None:
    caps = _caps(head_dims=dispatch.PAGED_ATTENTION_HEAD_SIZES)

    assert dispatch.PagedMetalMHA.validate(caps) == []


def test_mha_validates_every_per_layer_head_dimension() -> None:
    caps = _caps(head_dims=(64, 128, 160, 512))

    reasons = dispatch.PagedMetalMHA.validate(caps)

    assert any("160" in reason and "head" in reason for reason in reasons)


@pytest.mark.parametrize("block_size", [8, 16, 24, 32, 48, 544])
def test_mha_accepts_cache_blocks_translatable_to_a_kernel_block(
    block_size: int,
) -> None:
    assert dispatch.PagedMetalMHA.validate(_caps(block_size=block_size)) == []


@pytest.mark.parametrize("block_size", [-8, 0, 7, 12])
def test_mha_rejects_invalid_cache_block_sizes(block_size: int) -> None:
    reasons = dispatch.PagedMetalMHA.validate(_caps(block_size=block_size))

    assert any("block_size" in reason for reason in reasons)


def test_turboquant_requires_fwht_head_dimensions() -> None:
    accepted = _caps(
        head_dims=tuple(FWHT_SUPPORTED_HEAD_DIMS),
        turboquant=True,
    )
    rejected = _caps(head_dims=(80, 128), turboquant=True)

    assert dispatch.PagedMetalMHA.validate(accepted) == []
    assert any(
        "TurboQuant" in reason for reason in dispatch.PagedMetalMHA.validate(rejected)
    )


def test_mla_does_not_apply_mha_specialization_gates() -> None:
    caps = _caps(is_mla=True, head_dims=(7,), block_size=7)

    assert dispatch.PagedMetalMLA.validate(caps) == []


def test_mla_rejects_turboquant_before_building_runtime() -> None:
    with pytest.raises(ValueError, match="paged_metal_mla.*TurboQuant"):
        dispatch.select_attention_backend(_caps(is_mla=True, turboquant=True))


def test_policy_checks_every_cache_head_dimension() -> None:
    runner = make_stub_runner(
        head_dim=128,
        head_dim_per_layer=[128, 160],
        kv_heads_per_layer=[1, 1],
        kv_cache_dtype=mx.bfloat16,
    )

    with pytest.raises(ValueError, match="unsupported head dimensions.*160"):
        runner.build_paged_attention_runtime(block_size=16)


def test_policy_checks_requested_block_size() -> None:
    runner = make_stub_runner(
        head_dim=128,
        kv_cache_dtype=mx.bfloat16,
        cache_config=SimpleNamespace(block_size=16),
    )

    with pytest.raises(ValueError, match="block_size 12"):
        runner.build_paged_attention_runtime(block_size=12)


def test_mha_head_dimensions_match_metal_instantiations() -> None:
    source = (
        Path(__file__).resolve().parent.parent
        / "vllm_metal"
        / "metal"
        / "kernels_v2"
        / "pagedattention.metal"
    ).read_text()
    start = source.index("#define instantiate_paged_attention_heads")
    end = source.index("\n\n#define", start)
    instantiated = {
        int(value)
        for value in re.findall(
            r"instantiate_paged_attention_inner\(\s*type\s*,\s*k_cache_type\s*,"
            r"\s*v_cache_type\s*,\s*(\d+)\s*,",
            source[start:end],
        )
    }

    assert instantiated
    assert instantiated == set(dispatch.PAGED_ATTENTION_HEAD_SIZES)


def test_unknown_override_lists_registered_backends(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("VLLM_METAL_ATTENTION_BACKEND", "missing")

    with pytest.raises(ValueError) as excinfo:
        dispatch.select_attention_backend(_caps())

    message = str(excinfo.value)
    assert "missing" in message
    assert "paged_metal_mha" in message
    assert "paged_metal_mla" in message


def test_registration_rejects_empty_and_duplicate_names() -> None:
    class EmptyName:
        name = ""

    class DuplicateName:
        name = "paged_metal_mha"

    with pytest.raises(ValueError, match="name"):
        dispatch.register_attention_backend(EmptyName)
    with pytest.raises(ValueError, match="paged_metal_mha"):
        dispatch.register_attention_backend(DuplicateName)


def test_registered_backend_can_be_forced(monkeypatch: pytest.MonkeyPatch) -> None:
    class ExternalBackend:
        name = "external"

        @classmethod
        def validate(cls, caps: dispatch.AttnCaps) -> list[str]:
            return []

        @classmethod
        def build_runtime(cls, builder, *, block_size: int):
            return builder._build_sdpa_backend(block_size)

    assert dispatch.register_attention_backend(ExternalBackend) is ExternalBackend
    monkeypatch.setenv("VLLM_METAL_ATTENTION_BACKEND", "external")

    assert dispatch.select_attention_backend(_caps()) is ExternalBackend


def test_auto_selection_uses_first_valid_candidate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Rejected:
        name = "rejected"

        @classmethod
        def validate(cls, caps: dispatch.AttnCaps) -> list[str]:
            return ["deliberate rejection"]

    class Accepted:
        name = "accepted"

        @classmethod
        def validate(cls, caps: dispatch.AttnCaps) -> list[str]:
            return []

    dispatch.register_attention_backend(Rejected)
    dispatch.register_attention_backend(Accepted)
    monkeypatch.setattr(
        dispatch,
        "candidate_backends",
        lambda caps: ("rejected", "accepted", "paged_metal_mha"),
    )

    assert dispatch.select_attention_backend(_caps()) is Accepted


def test_auto_selection_reports_every_candidate_rejection(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FirstRejected:
        name = "first_rejected"

        @classmethod
        def validate(cls, caps: dispatch.AttnCaps) -> list[str]:
            return ["first reason"]

    class SecondRejected:
        name = "second_rejected"

        @classmethod
        def validate(cls, caps: dispatch.AttnCaps) -> list[str]:
            return ["second reason"]

    dispatch.register_attention_backend(FirstRejected)
    dispatch.register_attention_backend(SecondRejected)
    monkeypatch.setattr(
        dispatch,
        "candidate_backends",
        lambda caps: ("first_rejected", "second_rejected"),
    )

    with pytest.raises(ValueError) as excinfo:
        dispatch.select_attention_backend(_caps())

    message = str(excinfo.value)
    assert "first_rejected" in message and "first reason" in message
    assert "second_rejected" in message and "second reason" in message


@pytest.mark.parametrize(
    ("model_args", "expected_type"),
    [({}, SDPAPagedAttentionRuntime), ({"kv_lora_rank": 64}, MLAPagedAttentionRuntime)],
)
def test_capacity_runtime_build_routes_through_selector(
    monkeypatch: pytest.MonkeyPatch,
    model_args: dict[str, int],
    expected_type: type,
) -> None:
    class CapacityBackend:
        name = "capacity_test"

        @classmethod
        def validate(cls, caps: dispatch.AttnCaps) -> list[str]:
            return []

        @classmethod
        def build_runtime(cls, builder, *, block_size: int):
            if model_args:
                return builder._build_mla_backend(block_size)
            return builder._build_sdpa_backend(block_size)

    dispatch.register_attention_backend(CapacityBackend)
    monkeypatch.setenv("VLLM_METAL_ATTENTION_BACKEND", "capacity_test")
    runner = make_stub_runner(
        model_args=model_args,
        num_layers=2,
        num_kv_cache_layers=2,
        num_kv_heads=1,
        head_dim=64,
        kv_cache_dtype=mx.float16,
        cache_config=SimpleNamespace(block_size=16),
    )

    runtime = runner.build_paged_attention_runtime(block_size=16)

    assert isinstance(runtime, expected_type)


def test_layout_initialization_routes_through_selector(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from tests.test_per_layer_kv_cache import merged_full_attention_config

    calls: list[int] = []

    class LayoutBackend:
        name = "layout_test"

        @classmethod
        def validate(cls, caps: dispatch.AttnCaps) -> list[str]:
            return []

        @classmethod
        def build_runtime(cls, builder, *, block_size: int):
            calls.append(block_size)
            return builder._build_sdpa_backend(block_size)

    dispatch.register_attention_backend(LayoutBackend)
    monkeypatch.setenv("VLLM_METAL_ATTENTION_BACKEND", "layout_test")
    config, _ = merged_full_attention_config()
    runner = make_stub_runner(
        num_layers=4,
        num_kv_cache_layers=4,
        num_kv_heads=16,
        head_dim=256,
        kv_cache_dtype=mx.bfloat16,
        cache_config=SimpleNamespace(block_size=16),
        kv_heads_per_layer=[16, 16, 4, 4],
        head_dim_per_layer=[256, 256, 512, 512],
    )
    runtime = SDPAPagedAttentionRuntime(
        num_layers=4,
        num_kv_heads=16,
        head_dim=256,
        block_size=16,
        dtype=mx.bfloat16,
        kv_heads_per_layer=[16, 16, 4, 4],
        head_dim_per_layer=[256, 256, 512, 512],
    )
    monkeypatch.setattr(
        runner._cache_policy,
        "_build_sdpa_backend",
        lambda block_size: runtime,
    )
    monkeypatch.setattr(runtime, "patch_model", lambda model: 4)

    runner.initialize_kv_cache(config)

    assert calls == [16]
    assert runner.paged_attention_runtime is runtime
    assert runtime.kv_cache.key_caches[2].shape == (3, 16, 4, 512)
