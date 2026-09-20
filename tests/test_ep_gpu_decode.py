"""GPU-only expert decode skips remote slots without changing local math."""

import mlx.core as mx
import pytest
from mlx_lm.models.switch_layers import QuantizedSwitchLinear

from vllm_metal.distributed.expert_decode import masked_expert_projection


@pytest.mark.parametrize("dtype", [mx.float16, mx.bfloat16])
@pytest.mark.parametrize("per_slot", [False, True])
@pytest.mark.parametrize("dims", [(128, 64), (2880, 128), (128, 67)])
def test_masked_mxfp4_projection(dtype, per_slot, dims):
    k, n = dims
    mx.random.seed(7)
    proj = QuantizedSwitchLinear(
        k, n, 3, bias=True, group_size=32, bits=4, mode="mxfp4"
    )
    proj.bias = mx.random.normal((3, n)).astype(dtype)
    x = mx.random.normal((1, 1, 4, k) if per_slot else (1, 1, k)).astype(dtype)
    indices = mx.array([[[1, 2, 4, 7]]], dtype=mx.uint32)
    local = mx.array([[[0, 0, 2, 0]]], dtype=mx.uint32)
    expected = proj(
        x[..., None, :] if per_slot else x[..., None, None, :], local
    ).squeeze(-2)
    got = masked_expert_projection(proj, x, indices, 2, 5, per_slot)
    mask = ((indices >= 2) & (indices < 5))[..., None]
    expected = mx.where(mask, expected, mx.zeros_like(expected))
    got = mx.where(mask, got, mx.zeros_like(got))
    mx.eval(got, expected)
    tolerance = 0.04 if dtype == mx.bfloat16 else 0.006
    assert mx.allclose(got, expected, rtol=tolerance, atol=tolerance).item()
    assert mx.all(mx.isfinite(got)).item()


def test_unowned_slots_skip_nan_expert_weights():
    proj = QuantizedSwitchLinear(
        128, 64, 2, bias=False, group_size=32, bits=4, mode="mxfp4"
    )
    proj.scales = mx.full(proj.scales.shape, 255, dtype=mx.uint8)
    x = mx.ones((1, 128), dtype=mx.float16)
    indices = mx.array([[0, 1, 4, 5]], dtype=mx.uint32)
    got = masked_expert_projection(proj, x, indices, 2, 4)
    mx.eval(got)
    assert mx.all(got == 0).item()


def test_other_quantizers_use_native_path():
    from types import SimpleNamespace

    from vllm_metal.distributed.expert_decode import supports_experts

    proj = QuantizedSwitchLinear(128, 64, 2, group_size=32, bits=4, mode="affine")
    assert not supports_experts(
        SimpleNamespace(gate_proj=proj, up_proj=proj, down_proj=proj)
    )
