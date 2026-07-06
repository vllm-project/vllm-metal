# SPDX-License-Identifier: Apache-2.0
"""Tests for Metal LoRA adapter loading helpers."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from vllm.lora.layers import LoRAMapping
from vllm.sampling_params import SamplingParams

import vllm_metal.v1.model_runner as model_runner_mod
from tests.stub_runner import make_stub_runner
from vllm_metal.v1.lora import layers as layers_mod
from vllm_metal.v1.lora import model_manager as model_manager_mod
from vllm_metal.v1.lora import peft_loader as peft_loader_mod
from vllm_metal.v1.lora import punica_wrapper as punica_mod
from vllm_metal.v1.lora import runtime as runtime_mod
from vllm_metal.v1.lora import worker_manager as worker_manager_mod

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("vllm.lora.peft_helper")
pytest.importorskip("vllm.lora.utils")
pytest.importorskip("safetensors")


# PunicaWrapperMLX.add_lora_linear
def test_punica_add_lora_linear_no_lora_is_a_passthrough() -> None:
    wrapper = punica_mod.PunicaWrapperMLX(
        max_num_batched_tokens=4, max_batches=2, max_loras=1
    )
    mapping = LoRAMapping(index_mapping=(0, 0), prompt_mapping=(0,))
    wrapper.update_metadata(mapping, lora_index_to_id=[None])

    x = mx.array(np.ones((2, 3), dtype=np.float32))
    y = mx.array(np.full((2, 4), 5.0, dtype=np.float32))
    a_stacked = mx.array(np.ones((2, 1, 3), dtype=np.float32))
    b_stacked = mx.array(np.ones((2, 4, 1), dtype=np.float32))

    out = wrapper.add_lora_linear(y, x, a_stacked, b_stacked, scale=1.0)
    np.testing.assert_array_equal(np.array(out), np.array(y))


# MLXLinearWithLoRA wrapper


def test_linear_wrapper_set_lora_writes_into_correct_slot() -> None:
    base = nn.Linear(input_dims=3, output_dims=4, bias=False)
    wrapper = layers_mod.MLXLinearWithLoRA(
        base_layer=base, max_loras=2, max_lora_rank=4, dtype=mx.float32
    )
    assert wrapper.weight is base.weight
    assert not hasattr(wrapper, "bias")

    lora_a = mx.array(np.ones((2, 3), dtype=np.float32))
    lora_b = mx.array(np.ones((4, 2), dtype=np.float32))

    wrapper.set_lora(slot=1, lora_a=lora_a, lora_b=lora_b)

    a_stacked = np.array(wrapper.lora_a_stacked)
    b_stacked = np.array(wrapper.lora_b_stacked)
    assert not a_stacked[0].any()
    np.testing.assert_array_equal(a_stacked[1, :2, :], np.ones((2, 3)))
    np.testing.assert_array_equal(a_stacked[1, 2:, :], np.zeros((2, 3)))
    np.testing.assert_array_equal(b_stacked[1, :, :2], np.ones((4, 2)))
    np.testing.assert_array_equal(b_stacked[1, :, 2:], np.zeros((4, 2)))


@pytest.mark.parametrize(
    "lora_a_shape,lora_b_shape,err_match",
    [
        ((2, 7), (4, 2), "LoRA weight shape mismatch"),  # in_dim mismatch
        ((4, 3), (4, 4), "exceeds max_lora_rank"),  # rank > max_lora_rank
        ((2, 3, 1), (4, 2), "must be 2-D"),  # A not 2-D
        ((2, 3), (4, 3), "does not match B rank"),  # A rank != B rank
    ],
)
def test_linear_wrapper_rejects_bad_weights(
    lora_a_shape, lora_b_shape, err_match
) -> None:
    base = nn.Linear(input_dims=3, output_dims=4, bias=False)
    wrapper = layers_mod.MLXLinearWithLoRA(
        base_layer=base, max_loras=1, max_lora_rank=2, dtype=mx.float32
    )
    a = mx.array(np.ones(lora_a_shape, dtype=np.float32))
    b = mx.array(np.ones(lora_b_shape, dtype=np.float32))
    with pytest.raises(ValueError, match=err_match):
        wrapper.set_lora(0, a, b)


def test_linear_wrapper_call_with_active_lora_changes_output() -> None:
    base = nn.Linear(input_dims=2, output_dims=2, bias=False)
    base.weight = mx.zeros((2, 2), dtype=mx.float32)  # base output is 0

    wrapper = layers_mod.MLXLinearWithLoRA(
        base_layer=base, max_loras=1, max_lora_rank=1, dtype=mx.float32
    )
    punica = punica_mod.PunicaWrapperMLX(
        max_num_batched_tokens=2, max_batches=1, max_loras=1
    )
    wrapper.set_mapping(punica)
    wrapper.set_lora(
        slot=0,
        lora_a=mx.array(np.array([[1.0, 0.0]], dtype=np.float32)),
        lora_b=mx.array(np.array([[1.0], [0.0]], dtype=np.float32)),
    )

    mapping = LoRAMapping(index_mapping=(42,), prompt_mapping=(42,))
    punica.update_metadata(mapping, lora_index_to_id=[42])

    x = mx.array(np.array([[1.0, 0.0]], dtype=np.float32))
    out = np.array(wrapper(x))
    # base output = [0, 0]; delta = B @ A @ x = [[1],[0]] @ ([1,0]@[1,0]=1) = [[1],[0]]
    np.testing.assert_allclose(out, np.array([[1.0, 0.0]]), rtol=1e-5, atol=1e-6)


# PEFT loader (round-trip through a tmp safetensors file)


def _write_peft_adapter(tmp_path: Path) -> Path:
    from safetensors.numpy import save_file

    config = {
        "peft_type": "LORA",
        "r": 2,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj"],
        "use_rslora": False,
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    a = np.arange(6, dtype=np.float32).reshape(2, 3)
    b = np.arange(8, dtype=np.float32).reshape(4, 2)
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": b,
        },
        str(tmp_path / "adapter_model.safetensors"),
    )
    return tmp_path


def test_peft_loader_normalizes_module_name_and_keeps_orientation(
    tmp_path: Path,
) -> None:
    pytest.importorskip("safetensors.numpy")

    adapter_dir = _write_peft_adapter(tmp_path)
    loaded = peft_loader_mod.load_peft_adapter(adapter_dir, lora_id=1)

    assert loaded.lora_id == 1
    assert loaded.rank == 2
    assert "layers.0.self_attn.q_proj" in loaded.weights

    weights = loaded.weights["layers.0.self_attn.q_proj"]
    assert weights.lora_a.shape == (2, 3)
    assert weights.lora_b.shape == (4, 2)
    assert weights.scaling == pytest.approx(4.0)


@pytest.mark.parametrize(
    "config_override,err_match",
    [
        ({"r": 1024}, "is greater than max_lora_rank"),
        ({"use_dora": True}, "does not yet support DoRA"),
        ({"modules_to_save": ["lm_head"]}, "modules_to_save being None"),
    ],
)
def test_peft_loader_rejects_unsupported_configs(
    tmp_path: Path,
    config_override: dict,
    err_match: str,
) -> None:
    """When called with a LoRAConfig, the loader must surface PEFTHelper's validation."""
    pytest.importorskip("safetensors.numpy")
    adapter_dir = _write_peft_adapter(tmp_path)
    # Patch the adapter_config.json with the unsupported feature.
    cfg_path = adapter_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg.update(config_override)
    cfg_path.write_text(json.dumps(cfg))

    lora_config = SimpleNamespace(
        max_lora_rank=16, max_cpu_loras=3, max_loras=2, bias_enabled=False
    )
    with pytest.raises(ValueError, match=err_match):
        peft_loader_mod.load_peft_adapter(
            adapter_dir, lora_id=1, lora_config=lora_config
        )


def test_peft_loader_rejects_partial_adapter(tmp_path: Path) -> None:
    """Adapter file with only lora_A (or only lora_B) for a module must fail Explicitly."""
    from safetensors.numpy import save_file

    pytest.importorskip("safetensors.numpy")

    config = {
        "peft_type": "LORA",
        "r": 2,
        "lora_alpha": 8,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj"],
        "use_rslora": False,
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    a_only = np.arange(6, dtype=np.float32).reshape(2, 3)
    save_file(
        {"base_model.model.layers.0.self_attn.q_proj.lora_A.weight": a_only},
        str(tmp_path / "adapter_model.safetensors"),
    )

    with pytest.raises(ValueError, match=r"has lora_a but no matching lora_b"):
        peft_loader_mod.load_peft_adapter(tmp_path, lora_id=1)


def test_peft_loader_without_lora_config_skips_validation(tmp_path: Path) -> None:
    """Backwards-compat: loader without a config still loads even oversized ranks."""
    pytest.importorskip("safetensors.numpy")
    adapter_dir = _write_peft_adapter(tmp_path)
    cfg_path = adapter_dir / "adapter_config.json"
    cfg = json.loads(cfg_path.read_text())
    cfg["use_dora"] = True
    cfg_path.write_text(json.dumps(cfg))

    # No lora_config -> loader does not run validate_legal, so this succeeds.
    loaded = peft_loader_mod.load_peft_adapter(adapter_dir, lora_id=1)
    assert loaded.lora_id == 1


# Module-name resolver used by the model manager
@pytest.mark.parametrize(
    "stored_key,lookup,expected_hit",
    [
        ("layers.0.self_attn.q_proj", "layers.0.self_attn.q_proj", True),  # exact
        (
            "language_model.model.layers.0.self_attn.q_proj",
            "layers.0.self_attn.q_proj",
            True,
        ),  # suffix
        (None, "layers.0.x", False),  # missing
    ],
)
def test_lookup_weights_for_module(stored_key, lookup, expected_hit) -> None:
    weights = peft_loader_mod.LoRALayerWeightsMLX(
        module_name=stored_key or "",
        rank=2,
        lora_a=mx.zeros((2, 3)),
        lora_b=mx.zeros((4, 2)),
        scaling=1.0,
    )
    stored = {stored_key: weights} if stored_key is not None else {}
    adapter = peft_loader_mod.LoadedLoRA(lora_id=1, rank=2, weights=stored)
    found = model_manager_mod._lookup_weights_for_module(adapter, lookup)
    assert (found is weights) if expected_hit else (found is None)


# Multi-adapter batching


def _stack_with_null(*per_slot_a: np.ndarray) -> tuple[mx.array, int, int]:
    """Stack rank-1 LoRA A weights into ``(slots+1, rank, in)`` with null tail."""
    null = np.zeros_like(per_slot_a[0])
    stacked = np.stack([*per_slot_a, null])
    return mx.array(stacked), int(stacked.shape[1]), int(stacked.shape[2])


def test_punica_routes_two_adapters_in_one_batch() -> None:
    """Token i gets adapter[idx[i]]'s delta — the whole point of punica."""
    wrapper = punica_mod.PunicaWrapperMLX(
        max_num_batched_tokens=4, max_batches=2, max_loras=2
    )
    # Tokens: [adapter 11, adapter 22, adapter 11, adapter 22].
    mapping = LoRAMapping(index_mapping=(11, 22, 11, 22), prompt_mapping=(11, 22))
    wrapper.update_metadata(mapping, lora_index_to_id=[11, 22])

    a0 = np.array([[1.0, 0.0]], dtype=np.float32)  # adapter 11 picks dim 0
    a1 = np.array([[0.0, 1.0]], dtype=np.float32)  # adapter 22 picks dim 1
    a_stacked, _, _ = _stack_with_null(a0, a1)

    b0 = np.array([[1.0]], dtype=np.float32)  # adapter 11: out scale 1
    b1 = np.array([[10.0]], dtype=np.float32)  # adapter 22: out scale 10
    b_null = np.zeros_like(b0)
    b_stacked = mx.array(np.stack([b0, b1, b_null]))

    x = mx.array(
        np.array([[2.0, 3.0], [2.0, 3.0], [4.0, 5.0], [4.0, 5.0]], dtype=np.float32)
    )
    y = mx.zeros((4, 1), dtype=mx.float32)

    out = np.array(wrapper.add_lora_linear(y, x, a_stacked, b_stacked, scale=1.0))

    # Token 0 (adapter 11): A·x = 2,  B·2 = 2
    # Token 1 (adapter 22): A·x = 3,  B·3 = 30
    # Token 2 (adapter 11): A·x = 4,  B·4 = 4
    # Token 3 (adapter 22): A·x = 5,  B·5 = 50
    np.testing.assert_allclose(out.flatten(), [2.0, 30.0, 4.0, 50.0], rtol=1e-5)


def test_punica_three_adapters_with_no_lora_token() -> None:
    """Mixed batch: 3 adapters + a base-model token routed to the null slot."""
    wrapper = punica_mod.PunicaWrapperMLX(
        max_num_batched_tokens=4, max_batches=4, max_loras=3
    )
    mapping = LoRAMapping(index_mapping=(7, 8, 0, 9), prompt_mapping=(7, 8, 0, 9))
    wrapper.update_metadata(mapping, lora_index_to_id=[7, 8, 9])

    # Three rank-1 adapters that each return a scalar = adapter index + 1.
    a_stacked, _, _ = _stack_with_null(
        np.array([[1.0]], dtype=np.float32),
        np.array([[2.0]], dtype=np.float32),
        np.array([[3.0]], dtype=np.float32),
    )
    b_stacked = mx.array(
        np.stack(
            [
                np.array([[1.0]], dtype=np.float32),
                np.array([[1.0]], dtype=np.float32),
                np.array([[1.0]], dtype=np.float32),
                np.array([[0.0]], dtype=np.float32),  # null slot
            ]
        )
    )

    x = mx.array(np.ones((4, 1), dtype=np.float32))
    y = mx.full((4, 1), 100.0, dtype=mx.float32)

    out = np.array(wrapper.add_lora_linear(y, x, a_stacked, b_stacked, scale=1.0))

    # Adapters add 1, 2, 0 (null), 3 to the base of 100.
    np.testing.assert_allclose(out.flatten(), [101.0, 102.0, 100.0, 103.0], rtol=1e-5)


def test_punica_batched_matches_per_token_single_adapter_runs() -> None:
    """Cross-check: batched multi-adapter == running each adapter alone per token."""
    rng = np.random.default_rng(0)
    in_dim, out_dim, rank = 4, 5, 2

    # Two random adapters.
    a0, a1 = (
        rng.standard_normal((rank, in_dim)).astype(np.float32),
        rng.standard_normal((rank, in_dim)).astype(np.float32),
    )
    b0, b1 = (
        rng.standard_normal((out_dim, rank)).astype(np.float32),
        rng.standard_normal((out_dim, rank)).astype(np.float32),
    )
    a_stacked = mx.array(np.stack([a0, a1, np.zeros_like(a0)]))
    b_stacked = mx.array(np.stack([b0, b1, np.zeros_like(b0)]))

    x_np = rng.standard_normal((4, in_dim)).astype(np.float32)
    x = mx.array(x_np)
    y_base_np = rng.standard_normal((4, out_dim)).astype(np.float32)

    # Reference: hand-compute per token using whichever adapter is assigned.
    assigned = [33, 44, 33, 44]
    a_ref = {33: a0, 44: a1}
    b_ref = {33: b0, 44: b1}
    expected = y_base_np.copy()
    for i, aid in enumerate(assigned):
        expected[i] += b_ref[aid] @ a_ref[aid] @ x_np[i]

    wrapper = punica_mod.PunicaWrapperMLX(
        max_num_batched_tokens=4, max_batches=4, max_loras=2
    )
    wrapper.update_metadata(
        LoRAMapping(index_mapping=tuple(assigned), prompt_mapping=(33, 44)),
        lora_index_to_id=[33, 44],
    )
    out = np.array(
        wrapper.add_lora_linear(mx.array(y_base_np), x, a_stacked, b_stacked, scale=1.0)
    )
    np.testing.assert_allclose(out, expected, rtol=1e-5, atol=1e-5)


def test_punica_update_metadata_reroutes_after_slot_churn() -> None:
    """If the manager moves an active adapter to a different slot between steps,
    the next add_lora_linear must use the new slot — no stale token→slot map."""
    wrapper = punica_mod.PunicaWrapperMLX(
        max_num_batched_tokens=2, max_batches=2, max_loras=2
    )

    # Adapter 11 picks dim 0 (=> output 1), adapter 22 picks dim 1 (=> output 10).
    a0 = np.array([[1.0, 0.0]], dtype=np.float32)
    a1 = np.array([[0.0, 1.0]], dtype=np.float32)
    a_stacked = mx.array(np.stack([a0, a1, np.zeros_like(a0)]))
    b_stacked = mx.array(
        np.stack(
            [
                np.array([[1.0]], dtype=np.float32),
                np.array([[10.0]], dtype=np.float32),
                np.array([[0.0]], dtype=np.float32),
            ]
        )
    )
    x = mx.array(np.array([[1.0, 1.0], [1.0, 1.0]], dtype=np.float32))
    y = mx.zeros((2, 1), dtype=mx.float32)

    # Step 1: adapter 11 is in slot 0.
    wrapper.update_metadata(
        LoRAMapping(index_mapping=(11, 11), prompt_mapping=(11,)),
        lora_index_to_id=[11, None],
    )
    assert wrapper.token_slot_indices.tolist() == [0, 0]
    out1 = np.array(wrapper.add_lora_linear(y, x, a_stacked, b_stacked, scale=1.0))
    np.testing.assert_allclose(out1.flatten(), [1.0, 1.0], rtol=1e-5)

    # Step 2: adapter 11 moved to slot 1 (because slot 0 now holds 22). The
    # weight stack the manager passes also gets reordered: slot 0 = adapter 22's
    # weights, slot 1 = adapter 11's. Token 0 still requests adapter 11 -> slot 1.
    a_stacked_swapped = mx.array(np.stack([a1, a0, np.zeros_like(a0)]))
    b_stacked_swapped = mx.array(
        np.stack(
            [
                np.array([[10.0]], dtype=np.float32),
                np.array([[1.0]], dtype=np.float32),
                np.array([[0.0]], dtype=np.float32),
            ]
        )
    )
    wrapper.update_metadata(
        LoRAMapping(index_mapping=(11, 22), prompt_mapping=(11, 22)),
        lora_index_to_id=[22, 11],
    )
    assert wrapper.token_slot_indices.tolist() == [1, 0]
    out2 = np.array(
        wrapper.add_lora_linear(y, x, a_stacked_swapped, b_stacked_swapped, scale=1.0)
    )
    # Token 0 (adapter 11, slot 1): a1·[1,1]=1, b·1=1.  No, wait — slot 1 holds adapter 11.
    # a_swapped[1] = a0 = [1,0]; a0·[1,1] = 1; b_swapped[1] = [[1.0]]; -> 1.0.
    # Token 1 (adapter 22, slot 0): a_swapped[0] = a1 = [0,1]; a1·[1,1] = 1; b_swapped[0] = [[10.0]]; -> 10.0.
    np.testing.assert_allclose(out2.flatten(), [1.0, 10.0], rtol=1e-5)


# MLXLoRAModelManager — full slot-table + module-wrapping flow


def _lora_config_stub(
    *,
    max_loras: int,
    max_lora_rank: int,
    max_cpu_loras: int | None = None,
    target_modules: list[str] | None = None,
) -> SimpleNamespace:
    """Stand-in for ``vllm.config.lora.LoRAConfig`` — only the fields the manager reads."""
    return SimpleNamespace(
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        max_cpu_loras=max_cpu_loras,
        target_modules=target_modules,
    )


class _TwoLinearModel(nn.Module):
    """Tiny stand-in mlx model with two ``nn.Linear`` layers, both zero-weighted."""

    def __init__(self, in_dim: int = 2, out_dim: int = 2) -> None:
        super().__init__()
        self.fc1 = nn.Linear(input_dims=in_dim, output_dims=out_dim, bias=False)
        self.fc2 = nn.Linear(input_dims=out_dim, output_dims=out_dim, bias=False)
        self.fc1.weight = mx.zeros((out_dim, in_dim), dtype=mx.float32)
        self.fc2.weight = mx.zeros((out_dim, out_dim), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return self.fc2(self.fc1(x))


def _make_adapter(
    lora_id: int, *, fc1_a, fc1_b, scaling: float = 1.0
) -> peft_loader_mod.LoadedLoRA:
    """Build a LoadedLoRA targeting only fc1 (fc2 is a no-op base + no-lora pass)."""
    return peft_loader_mod.LoadedLoRA(
        lora_id=lora_id,
        rank=int(fc1_a.shape[0]),
        weights={
            "fc1": peft_loader_mod.LoRALayerWeightsMLX(
                module_name="fc1",
                rank=int(fc1_a.shape[0]),
                lora_a=mx.array(fc1_a),
                lora_b=mx.array(fc1_b),
                scaling=scaling,
            )
        },
    )


def test_manager_wraps_linears_then_activate_applies_delta() -> None:
    """End-to-end: build manager, register + activate adapter, forward, check delta."""
    model = _TwoLinearModel()
    manager = model_manager_mod.MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=2, max_lora_rank=2),
        max_num_seqs=2,
        max_num_batched_tokens=4,
        dtype=mx.float32,
    )

    # Both linears must have been wrapped.
    assert set(manager.modules) == {"fc1", "fc2"}
    assert isinstance(model.fc1, layers_mod.MLXLinearWithLoRA)

    # Adapter that adds [3, 0] to fc1's output for any input [1, 0].
    adapter = _make_adapter(
        lora_id=1,
        fc1_a=np.array([[1.0, 0.0]], dtype=np.float32),
        fc1_b=np.array([[3.0], [0.0]], dtype=np.float32),
    )
    manager.add_adapter(adapter)
    manager.activate_adapter(adapter.lora_id)
    assert manager.lora_index_to_id[0] == 1

    # Push the per-step mapping so the punica wrapper knows which slot to use.
    mapping = LoRAMapping(index_mapping=(1,), prompt_mapping=(1,))
    manager.set_adapter_mapping(mapping)

    # fc1 base = 0, so output is 0 + LoRA delta.  fc2 base = 0 too.
    out = np.array(model(mx.array(np.array([[1.0, 0.0]], dtype=np.float32))))
    # fc1 output: B @ A @ [1,0] = [[3],[0]] @ ([1,0]·[1,0]=1) = [3,0]
    # fc2 output: weight=0, no LoRA active for fc2 since adapter doesn't target it = [0,0]
    np.testing.assert_allclose(out, np.array([[0.0, 0.0]]), rtol=1e-5, atol=1e-6)


def test_manager_two_adapters_mixed_batch_through_full_forward() -> None:
    """End-to-end: register+activate two adapters, run a mixed batch through the
    wrapped model, verify each token gets its own adapter's delta."""
    model = _TwoLinearModel()
    # Make fc2 a pass-through-on-dim-0 so the test reads fc1's delta directly.
    model.fc2.weight = mx.array(np.eye(2, dtype=np.float32))

    manager = model_manager_mod.MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=2, max_lora_rank=1, max_cpu_loras=2),
        max_num_seqs=2,
        max_num_batched_tokens=4,
        dtype=mx.float32,
    )

    # Adapter 1: fc1 += [5, 0] for input [1, 0].   Adapter 2: fc1 += [0, 7] for input [0, 1].
    a1 = _make_adapter(
        lora_id=1,
        fc1_a=np.array([[1.0, 0.0]], dtype=np.float32),
        fc1_b=np.array([[5.0], [0.0]], dtype=np.float32),
    )
    a2 = _make_adapter(
        lora_id=2,
        fc1_a=np.array([[0.0, 1.0]], dtype=np.float32),
        fc1_b=np.array([[0.0], [7.0]], dtype=np.float32),
    )
    manager.add_adapter(a1)
    manager.add_adapter(a2)
    manager.activate_adapter(1)
    manager.activate_adapter(2)
    assert sorted(manager.list_adapters()) == [1, 2]

    # Mixed batch: token 0 -> adapter 1 with [1,0]; token 1 -> adapter 2 with [0,1].
    mapping = LoRAMapping(index_mapping=(1, 2), prompt_mapping=(1, 2))
    manager.set_adapter_mapping(mapping)

    x = mx.array(np.array([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32))
    out = np.array(model(x))
    # fc1 base = 0; fc1 delta -> [5,0] for token 0, [0,7] for token 1.
    # fc2 = identity -> output equals fc1 delta.
    np.testing.assert_allclose(out, np.array([[5.0, 0.0], [0.0, 7.0]]), rtol=1e-5)


def test_manager_slot_is_reused_after_swap() -> None:
    """Activate A in slot 0, deactivate, activate B — slot 0 must hold B's weights."""
    model = _TwoLinearModel()
    manager = model_manager_mod.MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=1, max_cpu_loras=2),
        max_num_seqs=1,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )
    a = _make_adapter(
        lora_id=1,
        fc1_a=np.array([[1.0, 0.0]], dtype=np.float32),
        fc1_b=np.array([[7.0], [0.0]], dtype=np.float32),
    )
    b = _make_adapter(
        lora_id=2,
        fc1_a=np.array([[0.0, 1.0]], dtype=np.float32),
        fc1_b=np.array([[0.0], [9.0]], dtype=np.float32),
    )
    manager.add_adapter(a)
    manager.activate_adapter(1)
    manager.deactivate_adapter(1)
    manager.add_adapter(b)
    manager.activate_adapter(2)

    fc1 = manager.modules["fc1"]
    # A's weights had B[0,0] = 7; B's adapter has B[1,0] = 9.  Slot 0 must be B's.
    np.testing.assert_array_equal(np.array(fc1.lora_b_stacked[0])[:, 0], [0.0, 9.0])


def test_manager_set_adapter_mapping_caches_identical_mapping(monkeypatch) -> None:
    """Repeated identical mappings must not re-invoke ``update_metadata``."""
    model = _TwoLinearModel()
    manager = model_manager_mod.MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=1),
        max_num_seqs=1,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )

    calls = 0
    real_update = manager.punica_wrapper.update_metadata

    def counting_update(mapping, lora_index_to_id):
        nonlocal calls
        calls += 1
        return real_update(mapping, lora_index_to_id)

    monkeypatch.setattr(manager.punica_wrapper, "update_metadata", counting_update)

    mapping = LoRAMapping(index_mapping=(0,), prompt_mapping=(0,))
    manager.set_adapter_mapping(mapping)
    manager.set_adapter_mapping(mapping)
    manager.set_adapter_mapping(mapping)
    assert calls == 1


def test_manager_activate_invalidates_mapping_cache(monkeypatch) -> None:
    """Activating a new adapter forces the next set_adapter_mapping to re-run."""
    model = _TwoLinearModel()
    manager = model_manager_mod.MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=2, max_lora_rank=1),
        max_num_seqs=1,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )
    adapter = _make_adapter(
        lora_id=5,
        fc1_a=np.array([[1.0, 0.0]], dtype=np.float32),
        fc1_b=np.array([[1.0], [0.0]], dtype=np.float32),
    )
    manager.add_adapter(adapter)
    manager.activate_adapter(5)

    mapping = LoRAMapping(index_mapping=(5,), prompt_mapping=(5,))
    manager.set_adapter_mapping(mapping)
    assert manager._last_mapping == mapping

    manager.deactivate_adapter(5)
    assert manager._last_mapping is None  # invalidated


def test_manager_no_free_slot_raises() -> None:
    """activate_adapter must raise once max_loras slots are full."""
    model = _TwoLinearModel()
    manager = model_manager_mod.MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=1, max_cpu_loras=4),
        max_num_seqs=1,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )
    a = _make_adapter(
        1,
        fc1_a=np.array([[1.0, 0.0]], dtype=np.float32),
        fc1_b=np.array([[1.0], [0.0]], dtype=np.float32),
    )
    b = _make_adapter(
        2,
        fc1_a=np.array([[0.0, 1.0]], dtype=np.float32),
        fc1_b=np.array([[0.0], [1.0]], dtype=np.float32),
    )
    manager.add_adapter(a)
    manager.add_adapter(b)
    manager.activate_adapter(1)
    with pytest.raises(ValueError, match="No free LoRA slots"):
        manager.activate_adapter(2)


def test_manager_add_adapter_over_capacity_raises() -> None:
    """add_adapter must raise once max_cpu_loras adapters are registered."""
    model = _TwoLinearModel()
    manager = model_manager_mod.MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=1, max_cpu_loras=1),
        max_num_seqs=1,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )
    a = _make_adapter(
        1,
        fc1_a=np.array([[1.0, 0.0]], dtype=np.float32),
        fc1_b=np.array([[1.0], [0.0]], dtype=np.float32),
    )
    b = _make_adapter(
        2,
        fc1_a=np.array([[0.0, 1.0]], dtype=np.float32),
        fc1_b=np.array([[0.0], [1.0]], dtype=np.float32),
    )
    manager.add_adapter(a)
    with pytest.raises(RuntimeError, match="capacity"):
        manager.add_adapter(b)


def test_manager_pin_tracks_existing_adapter_only() -> None:
    model = _TwoLinearModel()
    manager = model_manager_mod.MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=1, max_cpu_loras=2),
        max_num_seqs=1,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )
    adapter = _make_adapter(
        7,
        fc1_a=np.array([[1.0, 0.0]], dtype=np.float32),
        fc1_b=np.array([[1.0], [0.0]], dtype=np.float32),
    )
    manager.add_adapter(adapter)
    assert manager.pin_adapter(7) is True
    assert manager.pin_adapter(8) is False
    assert manager.remove_adapter(7) is True
    assert 7 not in manager.list_adapters()


def test_manager_remove_all_adapters_clears_slots_and_registry() -> None:
    model = _TwoLinearModel()
    manager = model_manager_mod.MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=2, max_lora_rank=1, max_cpu_loras=2),
        max_num_seqs=2,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )
    for lora_id in (1, 2):
        adapter = _make_adapter(
            lora_id,
            fc1_a=np.array([[1.0, 0.0]], dtype=np.float32),
            fc1_b=np.array([[float(lora_id)], [0.0]], dtype=np.float32),
        )
        manager.add_adapter(adapter)
        manager.activate_adapter(lora_id)

    manager.remove_all_adapters()

    assert manager.list_adapters() == set()
    assert manager.lora_index_to_id == [None, None]
    np.testing.assert_array_equal(np.array(manager.modules["fc1"].lora_b_stacked), 0.0)


def test_manager_target_modules_filter_excludes_unmatched() -> None:
    """``target_modules=['fc1']`` must wrap fc1 only — fc2 stays a plain Linear."""
    model = _TwoLinearModel()
    manager = model_manager_mod.MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(
            max_loras=1, max_lora_rank=1, target_modules=["fc1"]
        ),
        max_num_seqs=1,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )
    assert set(manager.modules) == {"fc1"}
    assert isinstance(model.fc1, layers_mod.MLXLinearWithLoRA)
    assert not isinstance(model.fc2, layers_mod.MLXLinearWithLoRA)


def test_manager_rejects_zero_wrapped_modules() -> None:
    model = _TwoLinearModel()
    with pytest.raises(RuntimeError, match="no LoRA target modules") as exc_info:
        model_manager_mod.MLXLoRAModelManager(
            model=model,
            lora_config=_lora_config_stub(
                max_loras=1, max_lora_rank=1, target_modules=["missing"]
            ),
            max_num_seqs=1,
            max_num_batched_tokens=2,
            dtype=mx.float32,
        )
    assert "quantized layers" not in str(exc_info.value)
    assert "target_modules" in str(exc_info.value)


@pytest.mark.parametrize(
    ("registered_ids", "active_id", "requested_id", "slot_match"),
    [
        ((1, 2), 2, 1, r"\[2\]"),
        ((), None, 99, r"\[None\]"),
    ],
)
def test_manager_set_adapter_mapping_rejects_missing_active_lora_id(
    registered_ids: tuple[int, ...],
    active_id: int | None,
    requested_id: int,
    slot_match: str,
) -> None:
    model = _TwoLinearModel()
    manager = model_manager_mod.MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=1, max_cpu_loras=2),
        max_num_seqs=1,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )
    for lora_id in registered_ids:
        manager.add_adapter(
            _make_adapter(
                lora_id,
                fc1_a=np.array([[1.0, 0.0]], dtype=np.float32),
                fc1_b=np.array([[1.0], [0.0]], dtype=np.float32),
            )
        )
    if active_id is not None:
        manager.activate_adapter(active_id)

    with pytest.raises(
        ValueError, match=rf"not active.*\[{requested_id}\].*{slot_match}"
    ):
        manager.set_adapter_mapping(
            LoRAMapping(index_mapping=(requested_id,), prompt_mapping=(requested_id,))
        )


# Runner routing: paged forward processes decode tokens before prefill tokens,


def test_paged_lora_routing_orders_decode_before_prefill() -> None:
    """Mixed decode+prefill batch with different lora_ids routes by execution order."""
    decode_state = SimpleNamespace(lora_id=11)
    decode_reqs = [("decode-req", decode_state)]
    prefill_pack = [
        SimpleNamespace(lora_id=22, token_ids=[101, 102, 103, 104]),
    ]

    runner = make_stub_runner()
    entries = runner._paged_lora_routing(decode_reqs, prefill_pack)

    assert entries == [(11, 1), (22, 4)]


def test_handle_new_request_registers_lora_before_paged_prefill() -> None:
    """Paged requests must load their LoRA before prepare_step routes by id."""

    class SpyLoRA:
        def __init__(self) -> None:
            self.added: list[SimpleNamespace] = []

        def add_adapter(self, lora_request: SimpleNamespace) -> bool:
            self.added.append(lora_request)
            return True

    spy_lora = SpyLoRA()
    paged_attention = object()
    runner = make_stub_runner(
        _paged_attention_backend=paged_attention,
        _paged_attention_runtime=paged_attention,
        _lora=spy_lora,
    )
    lora_request = SimpleNamespace(lora_int_id=17)
    new_req = SimpleNamespace(
        req_id="req-lora",
        pooling_params=None,
        prompt_token_ids=[1, 2, 3],
        sampling_params=SamplingParams(temperature=0, max_tokens=1),
        block_ids=([4],),
        num_computed_tokens=0,
        lora_request=lora_request,
    )
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-lora": 3})
    batch = model_runner_mod._ExecutionBatch()

    runner._handle_new_requests(batch, [new_req], scheduler_output)

    assert spy_lora.added == [lora_request]
    assert batch.paged_prefill_entries[0].prefill.lora_id == 17


# MetalLoRARuntime guards


def _runtime_setup_kwargs(**overrides: object) -> dict:
    kwargs: dict = {
        "model": object(),
        "lora_config": _lora_config_stub(max_loras=1, max_lora_rank=8),
        "is_stt": False,
        "paged_attention_enabled": True,
        "speculative_decode_enabled": False,
        "max_num_seqs": 1,
        "max_num_batched_tokens": 8,
        "dtype": mx.float16,
        "max_position_embeddings": None,
    }
    kwargs.update(overrides)
    return kwargs


def test_runtime_rejects_lora_without_paged_attention() -> None:
    rt = runtime_mod.MetalLoRARuntime()
    with pytest.raises(NotImplementedError, match="requires paged attention"):
        rt.setup(**_runtime_setup_kwargs(paged_attention_enabled=False))
    assert rt.enabled is False


def test_runtime_rejects_lora_with_speculative_decode() -> None:
    rt = runtime_mod.MetalLoRARuntime()
    with pytest.raises(NotImplementedError, match="speculative decode"):
        rt.setup(**_runtime_setup_kwargs(speculative_decode_enabled=True))
    assert rt.enabled is False


def test_runtime_stt_disables_lora_without_raising() -> None:
    rt = runtime_mod.MetalLoRARuntime()
    rt.setup(**_runtime_setup_kwargs(is_stt=True))
    assert rt.enabled is False


def test_prepare_step_raises_for_unloaded_lora_id() -> None:
    rt = runtime_mod.MetalLoRARuntime()
    # Manager presence is all prepare_step checks before routing; the raise
    # fires in the routing loop before the manager is ever touched.
    rt._manager = SimpleNamespace(set_active_adapters=lambda *a, **k: None)
    with pytest.raises(ValueError, match="LoRA id 7 was routed .* not "):
        rt.prepare_step([(7, 1)])


def test_prepare_step_marks_prefill_mapping() -> None:
    class CapturingManager:
        def __init__(self) -> None:
            self.mapping = None

        def set_active_adapters(self, lora_requests, mapping) -> None:
            self.mapping = mapping

    class Request:
        pass

    manager = CapturingManager()
    rt = runtime_mod.MetalLoRARuntime()
    rt._manager = manager
    rt._loaded[7] = Request()

    rt.prepare_step([(7, 3)])

    assert manager.mapping.index_mapping == (7, 7, 7)
    assert manager.mapping.prompt_mapping == (7,)
    assert manager.mapping.is_prefill is True


def test_worker_manager_rejects_cpu_loras_gt_max_loras() -> None:
    with pytest.raises(NotImplementedError, match="max_cpu_loras > max_loras"):
        worker_manager_mod.MetalWorkerLoRAManager(
            model=_TwoLinearModel(),
            lora_config=_lora_config_stub(
                max_loras=2, max_lora_rank=8, max_cpu_loras=4
            ),
            max_num_seqs=1,
            max_num_batched_tokens=8,
            dtype=mx.float32,
        )


def test_activate_adapter_rejects_zero_module_match() -> None:
    model = _TwoLinearModel()
    manager = model_manager_mod.MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=2),
        max_num_seqs=1,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )
    # Adapter targets a module the model does not expose under LoRA.
    bogus = peft_loader_mod.LoadedLoRA(
        lora_id=1,
        rank=1,
        weights={
            "does.not.exist": peft_loader_mod.LoRALayerWeightsMLX(
                module_name="does.not.exist",
                rank=1,
                lora_a=mx.array(np.zeros((1, 2), dtype=np.float32)),
                lora_b=mx.array(np.zeros((2, 1), dtype=np.float32)),
                scaling=1.0,
            )
        },
    )
    manager.add_adapter(bogus)
    with pytest.raises(ValueError, match="matched 0 wrapped modules"):
        manager.activate_adapter(1)
    # Slot must be rolled back so a later valid activation can use it.
    assert all(sid is None for sid in manager.lora_index_to_id)


def test_activate_adapter_rejects_ambiguous_suffix_match() -> None:
    """If two adapter keys both suffix-match a wrapped module, fail loudly."""
    model = _TwoLinearModel()
    manager = model_manager_mod.MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=1),
        max_num_seqs=1,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )

    def _w(name: str) -> peft_loader_mod.LoRALayerWeightsMLX:
        return peft_loader_mod.LoRALayerWeightsMLX(
            module_name=name,
            rank=1,
            lora_a=mx.array(np.zeros((1, 2), dtype=np.float32)),
            lora_b=mx.array(np.zeros((2, 1), dtype=np.float32)),
            scaling=1.0,
        )

    # Both keys end with ".fc1" so they both suffix-match the wrapped "fc1".
    ambiguous = peft_loader_mod.LoadedLoRA(
        lora_id=42,
        rank=1,
        weights={
            "language_model.fc1": _w("language_model.fc1"),
            "vision_model.fc1": _w("vision_model.fc1"),
            "fc2": _w("fc2"),
        },
    )
    manager.add_adapter(ambiguous)
    with pytest.raises(ValueError, match="ambiguous suffix matches"):
        manager.activate_adapter(42)
    assert all(sid is None for sid in manager.lora_index_to_id)


def _stub_lora_request(lora_id: int, *, load_inplace: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        lora_int_id=lora_id,
        lora_path=f"/fake/adapter-{lora_id}",
        load_inplace=load_inplace,
    )


def _make_worker_manager() -> worker_manager_mod.MetalWorkerLoRAManager:
    return worker_manager_mod.MetalWorkerLoRAManager(
        model=_TwoLinearModel(),
        lora_config=_lora_config_stub(max_loras=2, max_lora_rank=1),
        max_num_seqs=2,
        max_num_batched_tokens=4,
        dtype=mx.float32,
    )


def _patch_loader(monkeypatch, adapters: dict[int, peft_loader_mod.LoadedLoRA]) -> None:
    monkeypatch.setattr("vllm.lora.utils.get_adapter_absolute_path", lambda p: p)

    def _fake_load(path, *, lora_id, max_position_embeddings, lora_config):
        return adapters[lora_id]

    monkeypatch.setattr(worker_manager_mod, "load_peft_adapter", _fake_load)


def _active_fc1_lora_b(
    manager: worker_manager_mod.MetalWorkerLoRAManager, lora_id: int
) -> np.ndarray:
    slot = manager._mm.lora_index_to_id.index(lora_id)
    return np.array(manager._mm.modules["fc1"].lora_b_stacked[slot])[:, 0]


def test_worker_manager_empty_batch_deactivates_stale_adapters(monkeypatch) -> None:
    """An empty lora_requests set must clear previously active slots so a
    subsequent load is not blocked by stale state."""
    manager = _make_worker_manager()
    a = _make_adapter(
        1,
        fc1_a=np.array([[1.0, 0.0]], dtype=np.float32),
        fc1_b=np.array([[1.0], [0.0]], dtype=np.float32),
    )
    _patch_loader(monkeypatch, {1: a})

    assert manager.add_adapter(_stub_lora_request(1)) is True
    assert 1 in {sid for sid in manager._mm.lora_index_to_id if sid is not None}

    # Empty batch — must deactivate adapter 1 instead of silently skipping.
    manager.set_active_adapters(set(), None)
    assert all(sid is None for sid in manager._mm.lora_index_to_id)


def test_worker_manager_add_adapter_load_inplace_replaces_weights(monkeypatch) -> None:
    """Re-adding the same lora_int_id with load_inplace=True must swap weights."""
    manager = _make_worker_manager()
    v1 = _make_adapter(
        7,
        fc1_a=np.array([[1.0, 0.0]], dtype=np.float32),
        fc1_b=np.array([[3.0], [0.0]], dtype=np.float32),
    )
    v2 = _make_adapter(
        7,
        fc1_a=np.array([[1.0, 0.0]], dtype=np.float32),
        fc1_b=np.array([[5.0], [0.0]], dtype=np.float32),
    )
    state: dict[int, peft_loader_mod.LoadedLoRA] = {7: v1}
    _patch_loader(monkeypatch, state)

    assert manager.add_adapter(_stub_lora_request(7)) is True
    np.testing.assert_array_equal(_active_fc1_lora_b(manager, 7), [3.0, 0.0])

    # Without load_inplace, the duplicate add is a no-op.
    assert manager.add_adapter(_stub_lora_request(7)) is False

    # With load_inplace=True, the new weights must replace the old in the slot.
    state[7] = v2
    assert manager.add_adapter(_stub_lora_request(7, load_inplace=True)) is True
    np.testing.assert_array_equal(_active_fc1_lora_b(manager, 7), [5.0, 0.0])


def test_worker_manager_load_inplace_failure_restores_previous_adapter(
    monkeypatch,
) -> None:
    """A failed replacement must not drop the previous working adapter."""
    manager = _make_worker_manager()
    v1 = _make_adapter(
        7,
        fc1_a=np.array([[1.0, 0.0]], dtype=np.float32),
        fc1_b=np.array([[3.0], [0.0]], dtype=np.float32),
    )
    bad = peft_loader_mod.LoadedLoRA(
        lora_id=7,
        rank=1,
        weights={
            "does.not.exist": peft_loader_mod.LoRALayerWeightsMLX(
                module_name="does.not.exist",
                rank=1,
                lora_a=mx.array(np.zeros((1, 2), dtype=np.float32)),
                lora_b=mx.array(np.zeros((2, 1), dtype=np.float32)),
                scaling=1.0,
            )
        },
    )
    state: dict[int, peft_loader_mod.LoadedLoRA] = {7: v1}
    _patch_loader(monkeypatch, state)

    assert manager.add_adapter(_stub_lora_request(7)) is True
    np.testing.assert_array_equal(_active_fc1_lora_b(manager, 7), [3.0, 0.0])

    state[7] = bad
    with pytest.raises(ValueError, match="matched 0 wrapped modules"):
        manager.add_adapter(_stub_lora_request(7, load_inplace=True))

    assert manager.list_adapters() == {7}
    np.testing.assert_array_equal(_active_fc1_lora_b(manager, 7), [3.0, 0.0])


def test_worker_manager_load_inplace_shape_failure_restores_previous_adapter(
    monkeypatch,
) -> None:
    """Validation must finish before any replacement weights are installed."""
    manager = _make_worker_manager()
    v1 = _make_adapter(
        7,
        fc1_a=np.array([[1.0, 0.0]], dtype=np.float32),
        fc1_b=np.array([[3.0], [0.0]], dtype=np.float32),
    )
    bad = peft_loader_mod.LoadedLoRA(
        lora_id=7,
        rank=1,
        weights={
            "fc1": peft_loader_mod.LoRALayerWeightsMLX(
                module_name="fc1",
                rank=1,
                lora_a=mx.array(np.array([[1.0, 0.0]], dtype=np.float32)),
                lora_b=mx.array(np.array([[5.0], [0.0]], dtype=np.float32)),
                scaling=1.0,
            ),
            "fc2": peft_loader_mod.LoRALayerWeightsMLX(
                module_name="fc2",
                rank=1,
                lora_a=mx.array(np.array([[1.0, 0.0, 0.0]], dtype=np.float32)),
                lora_b=mx.array(np.array([[1.0], [0.0]], dtype=np.float32)),
                scaling=1.0,
            ),
        },
    )
    state: dict[int, peft_loader_mod.LoadedLoRA] = {7: v1}
    _patch_loader(monkeypatch, state)

    assert manager.add_adapter(_stub_lora_request(7)) is True
    np.testing.assert_array_equal(_active_fc1_lora_b(manager, 7), [3.0, 0.0])

    state[7] = bad
    with pytest.raises(ValueError, match="LoRA weight shape mismatch"):
        manager.add_adapter(_stub_lora_request(7, load_inplace=True))

    assert manager.list_adapters() == {7}
    np.testing.assert_array_equal(_active_fc1_lora_b(manager, 7), [3.0, 0.0])


def test_worker_manager_load_inplace_replaces_pinned_adapter(monkeypatch) -> None:
    """Pinning prevents cache eviction upstream, not explicit replacement."""
    manager = _make_worker_manager()
    v1 = _make_adapter(
        7,
        fc1_a=np.array([[1.0, 0.0]], dtype=np.float32),
        fc1_b=np.array([[3.0], [0.0]], dtype=np.float32),
    )
    v2 = _make_adapter(
        7,
        fc1_a=np.array([[1.0, 0.0]], dtype=np.float32),
        fc1_b=np.array([[5.0], [0.0]], dtype=np.float32),
    )
    state: dict[int, peft_loader_mod.LoadedLoRA] = {7: v1}
    _patch_loader(monkeypatch, state)

    assert manager.add_adapter(_stub_lora_request(7)) is True
    assert manager.pin_adapter(7) is True

    state[7] = v2
    assert manager.add_adapter(_stub_lora_request(7, load_inplace=True)) is True
    np.testing.assert_array_equal(_active_fc1_lora_b(manager, 7), [5.0, 0.0])
