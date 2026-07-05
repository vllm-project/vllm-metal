# SPDX-License-Identifier: Apache-2.0
"""Tests for QLoRA support (LoRA adapters on AWQ-quantized base models)."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from vllm_metal.v1.lora import (
    LoRAMapping,
    MLXLoRAModelManager,
    MLXQuantizedLinearWithLoRA,
    PunicaWrapperMLX,
    can_wrap,
    can_wrap_qlora,
)
from vllm_metal.v1.lora import peft_loader as peft_loader_mod
from vllm_metal.v1.lora import worker_manager as worker_manager_mod
from vllm_metal.v1.lora.layers import MLXLinearWithLoRA

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")
pytest.importorskip("vllm.lora.peft_helper")
pytest.importorskip("vllm.lora.utils")
pytest.importorskip("safetensors")


def _make_quantized_linear(
    in_dim: int,
    out_dim: int,
    *,
    group_size: int = 64,
    bits: int = 4,
    bias: bool = False,
) -> nn.QuantizedLinear:
    return nn.QuantizedLinear(
        in_dim, out_dim, bias=bias, group_size=group_size, bits=bits
    )


def _lora_config_stub(
    *,
    max_loras: int,
    max_lora_rank: int,
    max_cpu_loras: int | None = None,
    target_modules: list[str] | None = None,
) -> SimpleNamespace:
    return SimpleNamespace(
        max_loras=max_loras,
        max_lora_rank=max_lora_rank,
        max_cpu_loras=max_cpu_loras,
        target_modules=target_modules,
    )


def _make_adapter(
    lora_id: int,
    *,
    module_name: str = "fc1",
    in_dim: int = 128,
    out_dim: int = 64,
    rank: int = 2,
    scaling: float = 1.0,
) -> peft_loader_mod.LoadedLoRA:
    a = (
        np.random.default_rng(lora_id)
        .standard_normal((rank, in_dim))
        .astype(np.float32)
    )
    b = (
        np.random.default_rng(lora_id + 1000)
        .standard_normal((out_dim, rank))
        .astype(np.float32)
    )
    return peft_loader_mod.LoadedLoRA(
        lora_id=lora_id,
        rank=rank,
        weights={
            module_name: peft_loader_mod.LoRALayerWeightsMLX(
                module_name=module_name,
                rank=rank,
                lora_a=mx.array(a),
                lora_b=mx.array(b),
                scaling=scaling,
            )
        },
    )


# can_wrap / can_wrap_qlora detection


def test_can_wrap_rejects_quantized_linear() -> None:
    ql = _make_quantized_linear(128, 64)
    assert not can_wrap(ql)


def test_can_wrap_qlora_accepts_quantized_linear() -> None:
    ql = _make_quantized_linear(128, 64)
    assert can_wrap_qlora(ql)


def test_can_wrap_qlora_rejects_plain_linear() -> None:
    plain = nn.Linear(4, 8, bias=False)
    assert not can_wrap_qlora(plain)


def test_can_wrap_qlora_rejects_non_quantized_module() -> None:
    class _Dummy(nn.Module):
        pass

    assert not can_wrap_qlora(_Dummy())


def test_can_wrap_qlora_accepts_duck_typed_quantized_module() -> None:
    """A module with ``bits`` and ``group_size`` attrs must be accepted."""

    class _FakeQuantLinear(nn.Module):
        bits = 4
        group_size = 128
        scales = mx.zeros((64, 2))  # (out, groups)

        def __call__(self, x: mx.array) -> mx.array:
            return x

    assert can_wrap_qlora(_FakeQuantLinear())


# MLXQuantizedLinearWithLoRA wrapper


def test_qlora_wrapper_infers_dims_from_quantized_linear() -> None:
    ql = _make_quantized_linear(128, 64)
    w = MLXQuantizedLinearWithLoRA(ql, max_loras=2, max_lora_rank=4, dtype=mx.float32)
    assert w.input_size == 128
    assert w.output_size == 64


def test_qlora_wrapper_set_lora_writes_into_correct_slot() -> None:
    ql = _make_quantized_linear(128, 64)
    w = MLXQuantizedLinearWithLoRA(ql, max_loras=2, max_lora_rank=4, dtype=mx.float32)

    lora_a = mx.array(np.ones((2, 128), dtype=np.float32))
    lora_b = mx.array(np.ones((64, 2), dtype=np.float32))

    w.set_lora(slot=1, lora_a=lora_a, lora_b=lora_b)

    a_stacked = np.array(w.lora_a_stacked)
    b_stacked = np.array(w.lora_b_stacked)

    # Slot 0 must still be zero.
    assert not a_stacked[0].any()
    assert not b_stacked[0].any()
    # Slot 1: first 2 rows of A and first 2 cols of B are ones.
    np.testing.assert_array_equal(a_stacked[1, :2, :], np.ones((2, 128)))
    np.testing.assert_array_equal(a_stacked[1, 2:, :], np.zeros((2, 128)))
    np.testing.assert_array_equal(b_stacked[1, :, :2], np.ones((64, 2)))
    np.testing.assert_array_equal(b_stacked[1, :, 2:], np.zeros((64, 2)))


@pytest.mark.parametrize(
    "lora_a_shape,lora_b_shape,err_match",
    [
        ((2, 99), (64, 2), "QLoRA weight shape mismatch"),  # in_dim mismatch
        ((5, 128), (64, 5), "exceeds max_lora_rank"),  # rank > max
        ((2, 3, 1), (64, 2), "must be 2-D"),  # A not 2-D
        ((2, 128), (64, 3), "does not match B rank"),  # A rank != B rank
    ],
)
def test_qlora_wrapper_rejects_bad_weights(
    lora_a_shape, lora_b_shape, err_match
) -> None:
    ql = _make_quantized_linear(128, 64)
    w = MLXQuantizedLinearWithLoRA(ql, max_loras=1, max_lora_rank=4, dtype=mx.float32)
    a = mx.array(np.ones(lora_a_shape, dtype=np.float32))
    b = mx.array(np.ones(lora_b_shape, dtype=np.float32))
    with pytest.raises(ValueError, match=err_match):
        w.set_lora(0, a, b)


def test_qlora_wrapper_reset_lora_zeroes_slot() -> None:
    ql = _make_quantized_linear(128, 64)
    w = MLXQuantizedLinearWithLoRA(ql, max_loras=1, max_lora_rank=2, dtype=mx.float32)

    w.set_lora(
        slot=0,
        lora_a=mx.array(np.ones((2, 128), dtype=np.float32)),
        lora_b=mx.array(np.ones((64, 2), dtype=np.float32)),
    )
    w.reset_lora(0)
    assert not np.array(w.lora_a_stacked[0]).any()
    assert not np.array(w.lora_b_stacked[0]).any()


def test_qlora_wrapper_passthrough_when_no_punica() -> None:
    """Without a Punica wrapper the base quantized output is returned as-is."""
    ql = _make_quantized_linear(128, 64)
    w = MLXQuantizedLinearWithLoRA(ql, max_loras=1, max_lora_rank=2, dtype=mx.float32)

    x = mx.array(np.random.default_rng(0).standard_normal((4, 128)).astype(np.float32))
    out = np.array(w(x))
    ref = np.array(ql(x))
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


def test_qlora_wrapper_call_with_active_adapter_adds_delta() -> None:
    """Forward with an active adapter must equal base_output + LoRA delta.

    The reference delta is computed in float16 (the adapter dtype) to match
    the precision that ``MLXQuantizedLinearWithLoRA`` uses internally.
    """
    ql = _make_quantized_linear(128, 64, group_size=64)
    w = MLXQuantizedLinearWithLoRA(ql, max_loras=1, max_lora_rank=2, dtype=mx.float16)

    punica = PunicaWrapperMLX(max_num_batched_tokens=4, max_batches=1, max_loras=1)
    w.set_mapping(punica)

    # Rank-2 adapter — stored in float16 by the wrapper
    rng = np.random.default_rng(42)
    lora_a_np = rng.standard_normal((2, 128)).astype(np.float16)
    lora_b_np = rng.standard_normal((64, 2)).astype(np.float16)
    w.set_lora(
        slot=0,
        lora_a=mx.array(lora_a_np),
        lora_b=mx.array(lora_b_np),
    )

    mapping = LoRAMapping(index_mapping=(42,), prompt_mapping=(42,))
    punica.update_metadata(mapping, lora_index_to_id=[42])

    x_np = rng.standard_normal((1, 128)).astype(np.float16)
    x = mx.array(x_np.astype(np.float32))

    # Reference: base + manual delta in float16 then cast to float32
    base_out = np.array(ql(x).astype(mx.float32))
    delta = (
        lora_b_np.astype(np.float32)
        @ lora_a_np.astype(np.float32)
        @ x_np.astype(np.float32).T
    ).T
    expected = base_out + delta

    out = np.array(w(x).astype(mx.float32))
    # Use looser tolerance to account for float16 rounding in Punica kernel vs
    # float32 reference path.
    np.testing.assert_allclose(out, expected, rtol=5e-3, atol=5e-3)


def test_qlora_wrapper_no_lora_passthrough_with_punica() -> None:
    """With a Punica wrapper but ``no_lora=True`` the output must equal base."""
    ql = _make_quantized_linear(128, 64)
    w = MLXQuantizedLinearWithLoRA(ql, max_loras=1, max_lora_rank=2, dtype=mx.float32)

    punica = PunicaWrapperMLX(max_num_batched_tokens=2, max_batches=1, max_loras=1)
    w.set_mapping(punica)

    # no_lora=True (null mapping, all slots empty)
    mapping = LoRAMapping(index_mapping=(0, 0), prompt_mapping=(0,))
    punica.update_metadata(mapping, lora_index_to_id=[None])

    x = mx.array(np.random.default_rng(7).standard_normal((2, 128)).astype(np.float32))
    out = np.array(w(x))
    ref = np.array(ql(x))
    np.testing.assert_allclose(out, ref, rtol=1e-4, atol=1e-4)


# QLoRA in a mixed batch via PunicaWrapperMLX


def test_qlora_punica_two_adapters_in_one_batch() -> None:
    """Each token gets its own adapter's delta on top of the quantized base."""
    ql = _make_quantized_linear(64, 32, group_size=64)
    w = MLXQuantizedLinearWithLoRA(ql, max_loras=2, max_lora_rank=1, dtype=mx.float32)

    punica = PunicaWrapperMLX(max_num_batched_tokens=4, max_batches=2, max_loras=2)
    w.set_mapping(punica)

    rng = np.random.default_rng(99)
    a0 = rng.standard_normal((1, 64)).astype(np.float32)
    a1 = rng.standard_normal((1, 64)).astype(np.float32)
    b0 = rng.standard_normal((32, 1)).astype(np.float32)
    b1 = rng.standard_normal((32, 1)).astype(np.float32)

    w.set_lora(0, mx.array(a0), mx.array(b0))
    w.set_lora(1, mx.array(a1), mx.array(b1))

    # 4-token batch: adapter 11, 22, 11, 22
    mapping = LoRAMapping(index_mapping=(11, 22, 11, 22), prompt_mapping=(11, 22))
    punica.update_metadata(mapping, lora_index_to_id=[11, 22])

    x_np = rng.standard_normal((4, 64)).astype(np.float32)
    x = mx.array(x_np)

    # Reference: base + per-token delta
    base_out = np.array(ql(x).astype(mx.float32))
    assigned_a = [a0, a1, a0, a1]
    assigned_b = [b0, b1, b0, b1]
    expected = base_out.copy()
    for i in range(4):
        expected[i] += (assigned_b[i] @ (assigned_a[i] @ x_np[i])).squeeze()

    out = np.array(w(x).astype(mx.float32))
    np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-3)


def test_qlora_mixed_batch_base_model_tokens_unaffected() -> None:
    """Tokens routed to slot 0 (no LoRA) must equal the quantized base output."""
    ql = _make_quantized_linear(64, 32, group_size=64)
    w = MLXQuantizedLinearWithLoRA(ql, max_loras=1, max_lora_rank=1, dtype=mx.float32)

    punica = PunicaWrapperMLX(max_num_batched_tokens=3, max_batches=3, max_loras=1)
    w.set_mapping(punica)

    rng = np.random.default_rng(7)
    a0 = rng.standard_normal((1, 64)).astype(np.float32)
    b0 = rng.standard_normal((32, 1)).astype(np.float32)
    w.set_lora(0, mx.array(a0), mx.array(b0))

    # Token 0 -> no-LoRA (null slot), tokens 1/2 -> adapter 55
    mapping = LoRAMapping(index_mapping=(0, 55, 55), prompt_mapping=(0, 55))
    punica.update_metadata(mapping, lora_index_to_id=[55])

    x_np = rng.standard_normal((3, 64)).astype(np.float32)
    x = mx.array(x_np)
    out = np.array(w(x).astype(mx.float32))
    base_out = np.array(ql(x).astype(mx.float32))

    # Token 0 must exactly match the base (no delta added).
    np.testing.assert_allclose(out[0], base_out[0], rtol=1e-4, atol=1e-4)
    # Tokens 1 and 2 must differ from the base.
    assert not np.allclose(out[1], base_out[1], atol=1e-6)
    assert not np.allclose(out[2], base_out[2], atol=1e-6)


# Model manager wraps quantized modules


class _QuantizedModel(nn.Module):
    """Tiny model with one quantized and one plain linear (like AWQ LLM)."""

    def __init__(self) -> None:
        super().__init__()
        self.q_proj = _make_quantized_linear(128, 64, group_size=64)
        self.out_proj = nn.Linear(64, 32, bias=False)
        self.out_proj.weight = mx.zeros((32, 64), dtype=mx.float32)

    def __call__(self, x: mx.array) -> mx.array:
        return self.out_proj(self.q_proj(x))


def test_manager_wraps_quantized_modules_as_qlora_wrapper() -> None:
    model = _QuantizedModel()
    manager = MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=4),
        max_num_seqs=2,
        max_num_batched_tokens=4,
        dtype=mx.float32,
    )
    # q_proj is quantized -> MLXQuantizedLinearWithLoRA
    assert "q_proj" in manager.modules
    assert isinstance(manager.modules["q_proj"], MLXQuantizedLinearWithLoRA)
    # out_proj is plain -> MLXLinearWithLoRA
    assert "out_proj" in manager.modules
    assert isinstance(manager.modules["out_proj"], MLXLinearWithLoRA)


def test_manager_only_quantized_model_wraps_all_as_qlora() -> None:
    """A fully-quantized model should wrap everything with the QLoRA wrapper."""

    class _FullyQuantized(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = _make_quantized_linear(128, 64, group_size=64)
            self.v_proj = _make_quantized_linear(128, 64, group_size=64)

        def __call__(self, x: mx.array) -> mx.array:
            return self.q_proj(x) + self.v_proj(x)

    model = _FullyQuantized()
    manager = MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=4),
        max_num_seqs=1,
        max_num_batched_tokens=4,
        dtype=mx.float16,
    )
    assert len(manager.modules) == 2
    for w in manager.modules.values():
        assert isinstance(w, MLXQuantizedLinearWithLoRA)


def test_manager_target_modules_filter_on_quantized_model() -> None:
    """``target_modules=['q_proj']`` wraps only q_proj, leaves out_proj alone."""
    model = _QuantizedModel()
    manager = MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(
            max_loras=1, max_lora_rank=4, target_modules=["q_proj"]
        ),
        max_num_seqs=1,
        max_num_batched_tokens=4,
        dtype=mx.float16,
    )
    assert set(manager.modules) == {"q_proj"}
    assert isinstance(manager.modules["q_proj"], MLXQuantizedLinearWithLoRA)
    assert not isinstance(model.out_proj, MLXQuantizedLinearWithLoRA)


# Full forward through a QLoRA-wrapped quantized model


def test_qlora_end_to_end_forward_applies_delta_to_quantized_output() -> None:
    """Register + activate a QLoRA adapter, run forward, verify delta applies."""

    class _SmallAWQ(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.q_proj = _make_quantized_linear(128, 128, group_size=64)
            # lm_head stays plain so we can read the output easily
            self.lm_head = nn.Linear(128, 128, bias=False)
            self.lm_head.weight = mx.eye(128, dtype=mx.float32)

        def __call__(self, x: mx.array) -> mx.array:
            return self.lm_head(self.q_proj(x))

    model = _SmallAWQ()
    manager = MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=2),
        max_num_seqs=1,
        max_num_batched_tokens=4,
        dtype=mx.float32,
    )

    rng = np.random.default_rng(13)
    a_np = rng.standard_normal((2, 128)).astype(np.float32)
    b_np = rng.standard_normal((128, 2)).astype(np.float32)

    adapter = peft_loader_mod.LoadedLoRA(
        lora_id=1,
        rank=2,
        weights={
            "q_proj": peft_loader_mod.LoRALayerWeightsMLX(
                module_name="q_proj",
                rank=2,
                lora_a=mx.array(a_np),
                lora_b=mx.array(b_np),
                scaling=1.0,
            )
        },
    )
    manager.add_adapter(adapter)
    manager.activate_adapter(1)

    mapping = LoRAMapping(index_mapping=(1,), prompt_mapping=(1,))
    manager.set_adapter_mapping(mapping)

    x_np = rng.standard_normal((1, 128)).astype(np.float32)
    x = mx.array(x_np)

    # Base-only reference (no adapter): temporarily disable punica routing.
    punica = manager.punica_wrapper
    punica._no_lora = True
    base_out = np.array(model(x).astype(mx.float32))
    punica._no_lora = False

    # With adapter active
    out = np.array(model(x).astype(mx.float32))

    # The lm_head is identity, so output = q_proj_out (+ delta).
    delta = (b_np @ a_np @ x_np.T).T  # (1, 128)
    expected = base_out + delta
    np.testing.assert_allclose(out, expected, rtol=1e-3, atol=1e-3)


def test_qlora_mixed_model_forward_plain_module_unaffected_by_qlora_adapter() -> None:
    """A QLoRA adapter targeting only q_proj must leave out_proj output unchanged.

    The adapter modifies q_proj's output; out_proj (plain linear) is also
    wrapped but the adapter's weights have no out_proj entry, so its slot is
    zeroed out and the plain-linear forward is unmodified.
    """
    model = _QuantizedModel()
    manager = MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=2),
        max_num_seqs=1,
        max_num_batched_tokens=4,
        dtype=mx.float32,
    )

    rng = np.random.default_rng(77)
    a_np = rng.standard_normal((2, 128)).astype(np.float32)
    b_np = rng.standard_normal((64, 2)).astype(np.float32)

    adapter = peft_loader_mod.LoadedLoRA(
        lora_id=5,
        rank=2,
        weights={
            "q_proj": peft_loader_mod.LoRALayerWeightsMLX(
                module_name="q_proj",
                rank=2,
                lora_a=mx.array(a_np),
                lora_b=mx.array(b_np),
                scaling=1.0,
            )
        },
    )
    manager.add_adapter(adapter)
    manager.activate_adapter(5)

    mapping = LoRAMapping(index_mapping=(5,), prompt_mapping=(5,))
    manager.set_adapter_mapping(mapping)

    # out_proj slot 0 must be zero (reset_lora was called during activate).
    out_proj_wrapper = manager.modules["out_proj"]
    assert isinstance(out_proj_wrapper, MLXLinearWithLoRA)
    assert not np.array(out_proj_wrapper.lora_a_stacked[0]).any()
    assert not np.array(out_proj_wrapper.lora_b_stacked[0]).any()


# Slot reuse, capacity, pin/remove (quantized variant)


def test_qlora_manager_slot_reused_after_swap() -> None:
    """Slot freed by deactivating one QLoRA adapter is reused by the next."""
    model = _QuantizedModel()
    manager = MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=2, max_cpu_loras=2),
        max_num_seqs=1,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )
    a1 = _make_adapter(1, module_name="q_proj", in_dim=128, out_dim=64)
    a2 = _make_adapter(2, module_name="q_proj", in_dim=128, out_dim=64)

    manager.add_adapter(a1)
    manager.activate_adapter(1)
    manager.deactivate_adapter(1)
    manager.add_adapter(a2)
    manager.activate_adapter(2)

    # Slot 0 must hold adapter 2's B weights.
    q_proj_w = manager.modules["q_proj"]
    assert isinstance(q_proj_w, MLXQuantizedLinearWithLoRA)
    b2_col0 = np.array(q_proj_w.lora_b_stacked[0])[:, 0]
    # b2_col0 corresponds to rank-dim 0 of adapter 2's B; it should not be all
    # zeros (adapter 2 has random weights).
    assert b2_col0.any(), "Slot 0 did not receive adapter 2's weights"


def test_qlora_manager_no_free_slot_raises() -> None:
    model = _QuantizedModel()
    manager = MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=2, max_cpu_loras=4),
        max_num_seqs=1,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )
    a1 = _make_adapter(1, module_name="q_proj", in_dim=128, out_dim=64)
    a2 = _make_adapter(2, module_name="q_proj", in_dim=128, out_dim=64)
    manager.add_adapter(a1)
    manager.add_adapter(a2)
    manager.activate_adapter(1)
    with pytest.raises(ValueError, match="No free LoRA slots"):
        manager.activate_adapter(2)


def test_qlora_manager_add_over_capacity_raises() -> None:
    model = _QuantizedModel()
    manager = MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=2, max_cpu_loras=1),
        max_num_seqs=1,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )
    a1 = _make_adapter(1, module_name="q_proj", in_dim=128, out_dim=64)
    a2 = _make_adapter(2, module_name="q_proj", in_dim=128, out_dim=64)
    manager.add_adapter(a1)
    with pytest.raises(RuntimeError, match="capacity"):
        manager.add_adapter(a2)


def test_qlora_manager_pin_tracks_existing_adapter_only() -> None:
    model = _QuantizedModel()
    manager = MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=2, max_cpu_loras=2),
        max_num_seqs=1,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )
    a = _make_adapter(7, module_name="q_proj", in_dim=128, out_dim=64)
    manager.add_adapter(a)
    # pin acknowledges a registered adapter; unknown ids return False.
    assert manager.pin_adapter(7) is True
    assert manager.pin_adapter(8) is False
    assert manager.remove_adapter(7) is True
    assert 7 not in manager.list_adapters()


def test_qlora_manager_activate_rejects_zero_module_match() -> None:
    model = _QuantizedModel()
    manager = MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=2),
        max_num_seqs=1,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )
    bogus = peft_loader_mod.LoadedLoRA(
        lora_id=1,
        rank=1,
        weights={
            "does.not.exist": peft_loader_mod.LoRALayerWeightsMLX(
                module_name="does.not.exist",
                rank=1,
                lora_a=mx.array(np.zeros((1, 128), dtype=np.float32)),
                lora_b=mx.array(np.zeros((64, 1), dtype=np.float32)),
                scaling=1.0,
            )
        },
    )
    manager.add_adapter(bogus)
    with pytest.raises(ValueError, match="matched 0 wrapped modules"):
        manager.activate_adapter(1)
    assert all(sid is None for sid in manager.lora_index_to_id)
    assert 1 not in manager.lora_index_to_id


def test_qlora_manager_activate_rejects_ambiguous_suffix_match() -> None:
    model = _QuantizedModel()
    manager = MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=2),
        max_num_seqs=1,
        max_num_batched_tokens=2,
        dtype=mx.float32,
    )

    def _w(name: str, in_dim: int, out_dim: int) -> peft_loader_mod.LoRALayerWeightsMLX:
        return peft_loader_mod.LoRALayerWeightsMLX(
            module_name=name,
            rank=1,
            lora_a=mx.array(np.zeros((1, in_dim), dtype=np.float32)),
            lora_b=mx.array(np.zeros((out_dim, 1), dtype=np.float32)),
            scaling=1.0,
        )

    ambiguous = peft_loader_mod.LoadedLoRA(
        lora_id=42,
        rank=1,
        weights={
            "language_model.q_proj": _w("language_model.q_proj", 128, 64),
            "vision_model.q_proj": _w("vision_model.q_proj", 128, 64),
            "out_proj": _w("out_proj", 64, 32),
        },
    )
    manager.add_adapter(ambiguous)
    with pytest.raises(ValueError, match="ambiguous suffix matches"):
        manager.activate_adapter(42)
    assert all(sid is None for sid in manager.lora_index_to_id)
    assert 42 not in manager.lora_index_to_id


# Worker-manager constraints carry over to QLoRA


def test_qlora_worker_manager_rejects_cpu_loras_gt_max_loras() -> None:
    with pytest.raises(NotImplementedError, match="max_cpu_loras > max_loras"):
        worker_manager_mod.MetalWorkerLoRAManager(
            model=_QuantizedModel(),
            lora_config=_lora_config_stub(
                max_loras=2, max_lora_rank=8, max_cpu_loras=4
            ),
            max_num_seqs=1,
            max_num_batched_tokens=8,
            dtype=mx.float32,
        )


def _make_worker_manager_quantized() -> worker_manager_mod.MetalWorkerLoRAManager:
    return worker_manager_mod.MetalWorkerLoRAManager(
        model=_QuantizedModel(),
        lora_config=_lora_config_stub(max_loras=2, max_lora_rank=2),
        max_num_seqs=2,
        max_num_batched_tokens=4,
        dtype=mx.float32,
    )


def _patch_loader(monkeypatch, adapters: dict[int, peft_loader_mod.LoadedLoRA]) -> None:
    monkeypatch.setattr("vllm.lora.utils.get_adapter_absolute_path", lambda p: p)

    def _fake_load(path, *, lora_id, max_position_embeddings, lora_config):
        return adapters[lora_id]

    monkeypatch.setattr(worker_manager_mod, "load_peft_adapter", _fake_load)


def _stub_lora_request(lora_id: int, *, load_inplace: bool = False) -> SimpleNamespace:
    return SimpleNamespace(
        lora_int_id=lora_id,
        lora_path=f"/fake/adapter-{lora_id}",
        load_inplace=load_inplace,
    )


def test_qlora_worker_manager_empty_batch_deactivates_stale_adapters(
    monkeypatch,
) -> None:
    manager = _make_worker_manager_quantized()
    a = _make_adapter(1, module_name="q_proj", in_dim=128, out_dim=64)
    _patch_loader(monkeypatch, {1: a})

    assert manager.add_adapter(_stub_lora_request(1)) is True
    assert 1 in {sid for sid in manager._mm.lora_index_to_id if sid is not None}

    manager.set_active_adapters(set(), None)
    assert all(sid is None for sid in manager._mm.lora_index_to_id)


def test_qlora_worker_manager_load_inplace_replaces_weights(monkeypatch) -> None:
    manager = _make_worker_manager_quantized()
    rng = np.random.default_rng(0)
    v1 = peft_loader_mod.LoadedLoRA(
        lora_id=7,
        rank=1,
        weights={
            "q_proj": peft_loader_mod.LoRALayerWeightsMLX(
                module_name="q_proj",
                rank=1,
                lora_a=mx.array(rng.standard_normal((1, 128)).astype(np.float32)),
                lora_b=mx.array(np.ones((64, 1), dtype=np.float32) * 3.0),
                scaling=1.0,
            )
        },
    )
    v2 = peft_loader_mod.LoadedLoRA(
        lora_id=7,
        rank=1,
        weights={
            "q_proj": peft_loader_mod.LoRALayerWeightsMLX(
                module_name="q_proj",
                rank=1,
                lora_a=mx.array(rng.standard_normal((1, 128)).astype(np.float32)),
                lora_b=mx.array(np.ones((64, 1), dtype=np.float32) * 9.0),
                scaling=1.0,
            )
        },
    )
    state: dict[int, peft_loader_mod.LoadedLoRA] = {7: v1}
    _patch_loader(monkeypatch, state)

    assert manager.add_adapter(_stub_lora_request(7)) is True
    slot = manager._mm.lora_index_to_id.index(7)
    b_col = np.array(manager._mm.modules["q_proj"].lora_b_stacked[slot])[:, 0]
    assert np.allclose(b_col, 3.0, atol=1e-5)

    # Without load_inplace, duplicate is a no-op.
    assert manager.add_adapter(_stub_lora_request(7)) is False

    # With load_inplace, weights swap.
    state[7] = v2
    assert manager.add_adapter(_stub_lora_request(7, load_inplace=True)) is True
    slot = manager._mm.lora_index_to_id.index(7)
    b_col = np.array(manager._mm.modules["q_proj"].lora_b_stacked[slot])[:, 0]
    assert np.allclose(b_col, 9.0, atol=1e-5)


# PEFT loader with quantized model adapter_config.json


def _write_peft_adapter(tmp_path: Path, *, use_rslora: bool = False) -> Path:
    """Write a minimal PEFT adapter directory for q_proj."""
    from safetensors.numpy import save_file

    tmp_path.mkdir(parents=True, exist_ok=True)
    config = {
        "peft_type": "LORA",
        "r": 2,
        "lora_alpha": 4,
        "lora_dropout": 0.0,
        "target_modules": ["q_proj"],
        "use_rslora": use_rslora,
    }
    (tmp_path / "adapter_config.json").write_text(json.dumps(config))
    rng = np.random.default_rng(0)
    a = rng.standard_normal((2, 128)).astype(np.float32)
    b = rng.standard_normal((64, 2)).astype(np.float32)
    save_file(
        {
            "base_model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
            "base_model.model.layers.0.self_attn.q_proj.lora_B.weight": b,
        },
        str(tmp_path / "adapter_model.safetensors"),
    )
    return tmp_path


def test_peft_loader_loads_qlora_adapter(tmp_path: Path) -> None:
    pytest.importorskip("safetensors.numpy")
    adapter_dir = _write_peft_adapter(tmp_path)
    loaded = peft_loader_mod.load_peft_adapter(adapter_dir, lora_id=1)

    assert loaded.lora_id == 1
    assert loaded.rank == 2
    module_key = "layers.0.self_attn.q_proj"
    assert module_key in loaded.weights

    weights = loaded.weights[module_key]
    assert weights.lora_a.shape == (2, 128)
    assert weights.lora_b.shape == (64, 2)
    # lora_alpha=4, r=2 → scaling = lora_alpha / r = 2.0
    assert weights.scaling == pytest.approx(2.0)


def test_peft_loader_qlora_and_manager_round_trip(tmp_path: Path) -> None:
    """Load a PEFT adapter from disk, activate it on a quantized model."""
    pytest.importorskip("safetensors.numpy")
    adapter_dir = _write_peft_adapter(tmp_path)
    loaded = peft_loader_mod.load_peft_adapter(adapter_dir, lora_id=1)

    model = _QuantizedModel()
    manager = MLXLoRAModelManager(
        model=model,
        lora_config=_lora_config_stub(max_loras=1, max_lora_rank=4),
        max_num_seqs=1,
        max_num_batched_tokens=4,
        dtype=mx.float32,
    )
    manager.add_adapter(loaded)
    manager.activate_adapter(1)

    assert manager.lora_index_to_id[0] == 1
    q_proj_w = manager.modules["q_proj"]
    assert isinstance(q_proj_w, MLXQuantizedLinearWithLoRA)
    # Slot 0 must have non-zero weights.
    assert np.array(q_proj_w.lora_a_stacked[0]).any()
    assert np.array(q_proj_w.lora_b_stacked[0]).any()


def test_peft_loader_qlora_rslora_scaling(tmp_path: Path) -> None:
    """RSLoRA scaling = lora_alpha / sqrt(r): check it differs from regular."""
    pytest.importorskip("safetensors.numpy")
    regular_dir = _write_peft_adapter(tmp_path / "regular", use_rslora=False)
    rslora_dir = _write_peft_adapter(tmp_path / "rslora", use_rslora=True)

    regular = peft_loader_mod.load_peft_adapter(regular_dir, lora_id=1)
    rslora = peft_loader_mod.load_peft_adapter(rslora_dir, lora_id=2)

    key = "layers.0.self_attn.q_proj"
    # lora_alpha=4, r=2 → regular: 4/2=2.0, rslora: 4/sqrt(2)≈2.828
    assert regular.weights[key].scaling == pytest.approx(2.0)
    assert rslora.weights[key].scaling == pytest.approx(4.0 / np.sqrt(2.0), rel=1e-4)


# dtype propagation: adapter runs in adapter dtype on quant base


def test_qlora_adapter_dtype_float16_on_quantized_base() -> None:
    """Adapter running in float16 on a quantized base must produce float16 output
    that is correctly cast back to the base output dtype."""
    ql = _make_quantized_linear(128, 64, group_size=64)

    w = MLXQuantizedLinearWithLoRA(ql, max_loras=1, max_lora_rank=2, dtype=mx.float16)
    punica = PunicaWrapperMLX(max_num_batched_tokens=1, max_batches=1, max_loras=1)
    w.set_mapping(punica)

    rng = np.random.default_rng(5)
    w.set_lora(
        0,
        mx.array(rng.standard_normal((2, 128)).astype(np.float16)),
        mx.array(rng.standard_normal((64, 2)).astype(np.float16)),
    )

    mapping = LoRAMapping(index_mapping=(99,), prompt_mapping=(99,))
    punica.update_metadata(mapping, lora_index_to_id=[99])

    x = mx.array(rng.standard_normal((1, 128)).astype(np.float32))
    out = w(x)
    # Output dtype must match the base layer's output dtype.
    assert out.dtype == ql(x).dtype, (
        f"QLoRA output dtype {out.dtype} does not match base output dtype {ql(x).dtype}"
    )


def test_qlora_adapter_dtype_bfloat16_on_quantized_base() -> None:
    """Same as above but adapter dtype is bfloat16."""
    ql = _make_quantized_linear(128, 64, group_size=64)
    w = MLXQuantizedLinearWithLoRA(ql, max_loras=1, max_lora_rank=2, dtype=mx.bfloat16)
    punica = PunicaWrapperMLX(max_num_batched_tokens=1, max_batches=1, max_loras=1)
    w.set_mapping(punica)

    rng = np.random.default_rng(6)
    w.set_lora(
        0,
        mx.array(rng.standard_normal((2, 128)).astype(np.float32)),
        mx.array(rng.standard_normal((64, 2)).astype(np.float32)),
    )
    mapping = LoRAMapping(index_mapping=(77,), prompt_mapping=(77,))
    punica.update_metadata(mapping, lora_index_to_id=[77])

    x = mx.array(rng.standard_normal((1, 128)).astype(np.float32))
    out = w(x)
    assert out.dtype == ql(x).dtype
