# SPDX-License-Identifier: Apache-2.0
"""Packed MPS GDN recurrence and the upstream Qwen3.5 core integration."""

from itertools import accumulate
from math import prod
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="Requires Apple Silicon MPS"
)


@pytest.fixture(autouse=True)
def _seed_random_generators():
    torch.manual_seed(1241)


def _reference(q, k, v, decay, beta, state):
    # CPU oracle, using the mathematical [Hv, Dk, Dv] orientation.
    group = v.shape[1] // q.shape[1]
    q, k = (t.float().repeat_interleave(group, dim=1) for t in (q, k))
    state = state.transpose(-1, -2).clone()
    outputs = []
    for qt, kt, vt, gt, bt in zip(
        q, k, v.float(), decay.float(), beta.float(), strict=True
    ):
        state *= gt[:, None, None]
        delta = (vt - torch.einsum("hk,hkv->hv", kt, state)) * bt[:, None]
        state += kt[:, :, None] * delta[:, None, :]
        outputs.append(torch.einsum("hk,hkv->hv", qt, state))
    return torch.stack(outputs), state.transpose(-1, -2)


# Qwen/Qwen3.5-{0.8B,9B} config.json text_config, verified 2026-11-03:
# 0.8B local checkpoint; 9B https://modelscope.cn/models/Qwen/Qwen3.5-9B
# (linear_num_key_heads, linear_num_value_heads, key_head_dim, value_head_dim).
@pytest.mark.parametrize("heads", [(16, 16), (16, 32)], ids=["0.8B", "9B-GQA"])
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@pytest.mark.parametrize("prefill", [False, True], ids=["decode", "prefill-boundaries"])
@requires_mps
@torch.inference_mode()
def test_packed_gdn_matches_recurrence(heads, dtype, prefill):
    from vllm_metal.pytorch_backend.mps_ops import gdn_linear_attention

    hk, hv = heads
    dk = dv = 128
    lengths = (1, 63, 64, 65, 528, 3, 0) if prefill else (1,) * 6 + (0,)
    slots = (5, 2, 1, 0, 4, -1, 3)
    initial = (True, False, True, False, True, False, False)
    bounds = [0, *accumulate(lengths)]
    tokens = bounds[-1]
    q, k = (
        torch.nn.functional.normalize(torch.randn(tokens, hk, dk), dim=-1)
        for _ in range(2)
    )
    q = q * dk**-0.5
    v = torch.randn(tokens, hv, dv) * 0.25
    decay = (-0.01 - torch.rand(tokens, hv) * 0.2).exp()
    beta = torch.rand(tokens, hv)
    # Cover both the original same-dtype ABI and fp32 inputs with narrow Y.
    input_dtype = dtype if prefill else torch.float32
    q, k, v, decay, beta = (t.to(input_dtype) for t in (q, k, v, decay, beta))
    row_size = hv * dv * dk
    # Prefix, unused slot and page gaps detect negative indices/stride errors.
    storage = torch.randn(8, row_size + 17) * 0.1
    state = storage[1:, :row_size].view(7, hv, dv, dk)
    state[2].fill_(float("nan"))  # Fresh requests must not read stale state.
    expected_storage = storage.clone()
    expected_state = expected_storage[1:, :row_size].view_as(state)
    expected = torch.full_like(v, 13.0, dtype=dtype)
    for i, (slot, length) in enumerate(zip(slots, lengths, strict=True)):
        if slot < 0 or not length:
            continue
        span = slice(bounds[i], bounds[i + 1])
        start = state[slot] if initial[i] else torch.zeros_like(state[slot])
        result, expected_state[slot] = _reference(
            q[span], k[span], v[span], decay[span], beta[span], start
        )
        expected[span] = result.to(dtype)

    device_storage = storage.to("mps")
    device_state = device_storage[1:, :row_size].view_as(state)
    output = torch.full_like(v, 13.0, dtype=dtype, device="mps")
    gdn_linear_attention(
        *(t.to("mps") for t in (q, k, v, decay, beta)),
        device_state,
        torch.tensor(bounds, dtype=torch.int32, device="mps"),
        torch.tensor(slots, dtype=torch.int32, device="mps"),
        output,
        torch.tensor(initial, dtype=torch.bool, device="mps"),
    )
    # Consume through Torch on the same stream before copying back.
    atol, rtol = (2e-5, 1e-3) if dtype == torch.float16 else (2e-4, 8e-3)
    torch.testing.assert_close(
        (output.float() + 0).cpu(), expected.float(), atol=atol, rtol=rtol
    )
    torch.testing.assert_close(
        (device_storage + 0).cpu(), expected_storage, atol=1e-5, rtol=1e-4
    )
    active = torch.zeros_like(storage, dtype=torch.bool)
    for slot, length in zip(slots, lengths, strict=True):
        if slot >= 0 and length:
            active[slot + 1, :row_size] = True
    torch.testing.assert_close(
        device_storage.cpu()[~active], storage[~active], atol=0, rtol=0
    )


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16], ids=["fp16", "bf16"])
@requires_mps
@torch.inference_mode()
def test_qwen_core_carries_state_in_one_launch(monkeypatch, dtype):
    from vllm.model_executor.layers.mamba.abstract import MambaBase
    from vllm.model_executor.layers.mamba.mamba_utils import MambaStateShapeCalculator
    from vllm.v1.kv_cache_interface import (
        KVCacheTensor,
        MambaSpec,
        create_kv_cache_views,
    )
    from vllm.v1.kv_cache_layout import KVCacheLayout

    from vllm_metal.pytorch_backend.models import qwen3_5

    hk, hv, dk, dv, width = 2, 4, 32, 8, 4
    channels = 2 * hk * dk + hv * dv
    conv_shape, state_shape = MambaStateShapeCalculator.gated_delta_net_state_shape(
        1, hk, hv, dk, dv, width
    )
    assert state_shape == (hv, dv, dk)
    layer = SimpleNamespace(
        key_dim=hk * dk,
        value_dim=hv * dv,
        tp_size=1,
        head_k_dim=dk,
        head_v_dim=dv,
        activation="silu",
        conv1d=SimpleNamespace(
            weight=torch.randn(channels, 1, width).to("mps", dtype), bias=None
        ),
        A_log=torch.randn(hv, device="mps"),
        dt_bias=torch.randn(hv, device="mps"),
        get_state_shape=lambda: (conv_shape, state_shape),
        get_state_dtype=lambda: (dtype, torch.float32),
    )
    # Exercise the real cache-spec view and layer binding, with a page gap.
    spec = MambaSpec(
        block_size=16, shapes=layer.get_state_shape(), dtypes=layer.get_state_dtype()
    )
    page_stride = spec.page_size_bytes + 64
    raw = torch.zeros(64 + 5 * page_stride, dtype=torch.int8, device="mps")
    placement = KVCacheTensor(
        raw.numel(), ["gdn"], 5 * page_stride, page_stride, offset=64
    )
    pages = create_kv_cache_views(raw, spec, 5, KVCacheLayout.LBHNC, placement)[0]
    MambaBase.bind_kv_cache(layer, pages)
    ssm = layer.kv_cache[1]
    assert ssm.shape == (5, hv, dv, dk)
    assert ssm.stride() == (page_stride // 4, dv * dk, dk, 1)
    assert ssm.storage_offset() == (64 + prod(conv_shape) * 2) // 4
    assert ssm.untyped_storage().data_ptr() == raw.untyped_storage().data_ptr()
    for state in layer.kv_cache:
        state.normal_()
    untouched = [t[4].clone() for t in layer.kv_cache]
    qkv = [torch.randn(n, channels).to(dtype) for n in (76, 5)]
    a = [torch.randn(len(x), hv).to(dtype) for x in qkv]
    b = [torch.randn(len(x), hv).to(dtype) for x in qkv]
    received = [[], []]
    launch = qwen3_5.gdn_linear_attention
    launches = []

    def record(*args):
        assert args[0].shape[1] == hk  # No production head expansion.
        assert all(t.device.type == "mps" and t.dtype == torch.int32 for t in args[6:8])
        launches.append(1)
        launch(*args)

    monkeypatch.setattr(qwen3_5, "gdn_linear_attention", record)
    # A PAD decode accompanies fresh prefill, mixed resumed prefill, and decode.
    for nd, spans in [
        (1, [(-1, 0, 1), (0, 0, 65), (1, 0, 3)]),
        (2, [(1, 3, 4), (-1, 0, 1), (0, 65, 75)]),
        (3, [(0, 75, 76), (1, 4, 5), (-1, 0, 1)]),
    ]:
        lengths = [hi - lo for _, lo, hi in spans]
        bounds = [0, *accumulate(lengths)]
        md = SimpleNamespace(
            num_decodes=nd,
            num_decode_tokens=nd,
            num_prefills=3 - nd,
            num_prefill_tokens=bounds[-1] - nd,
            non_spec_state_indices_tensor=torch.tensor(
                [-1 if r < 0 else r + 1 for r, _, _ in spans], device="mps"
            ),
            non_spec_query_start_loc=torch.tensor(bounds, device="mps"),
            has_initial_state=(
                torch.tensor([lo > 0 for _, lo, _ in spans], device="mps")
                if nd < len(spans)
                else None  # Upstream omits this metadata in decode-only batches.
            ),
        )
        packed = [
            torch.cat([xs[max(r, 0)][lo:hi] for r, lo, hi in spans]).to("mps")
            for xs in (qkv, b, a)
        ]
        out = torch.full((bounds[-1] + 1, hv, dv), 13.0, dtype=dtype, device="mps")
        qwen3_5._gdn_core(layer, *packed, out, md)
        assert torch.all(out[-1] == 13)  # Unscheduled output padding.
        for i, (r, _, _) in enumerate(spans):
            result = out[bounds[i] : bounds[i + 1]].cpu()
            if r < 0:
                assert torch.all(result == 13)
            else:
                received[r].append(result)
    assert len(launches) == 3

    weight = layer.conv1d.weight[:, 0].cpu().float()
    for r in range(2):
        tail = torch.zeros(width - 1, channels)
        convolved = []
        for token in qkv[r].float():
            history = torch.cat([tail, token[None]])
            # Conv rounds before SiLU and again after the activation.
            convolved.append(
                torch.nn.functional.silu((history.T * weight).sum(-1).to(dtype))
                .to(dtype)
                .float()
            )
            tail = history[1:]
        x = torch.stack(convolved)
        q, k, v = x.split([hk * dk, hk * dk, hv * dv], dim=-1)
        q, k = (t.reshape(-1, hk, dk) for t in (q, k))
        q, k = (t * torch.rsqrt((t * t).sum(-1, keepdim=True) + 1e-6) for t in (q, k))
        log_g = -layer.A_log.cpu().exp() * torch.nn.functional.softplus(
            a[r].float() + layer.dt_bias.cpu()
        )
        expected, state = _reference(
            q * dk**-0.5,
            k,
            v.reshape(-1, hv, dv),
            log_g.exp(),
            b[r].float().sigmoid(),
            torch.zeros(hv, dv, dk),
        )
        torch.testing.assert_close(
            torch.cat(received[r]).float(),
            expected.to(dtype).float(),
            atol=1e-2,
            rtol=1e-2,
        )
        torch.testing.assert_close(
            layer.kv_cache[1][r + 1].cpu(), state, atol=1e-3, rtol=1e-3
        )
    for t, saved in zip(layer.kv_cache, untouched, strict=True):
        torch.testing.assert_close(t[4], saved, atol=0, rtol=0)


@pytest.fixture
def gdn_inputs():
    # Metadata-only MPS tensors let Linux exercise the real validation/launch ABI.
    from torch._subclasses.fake_tensor import FakeTensor, FakeTensorMode

    mode = FakeTensorMode()
    # Preserve a nonzero offset and a physical gap between state slots.
    state = FakeTensor(
        mode,
        torch.empty(5, 1041, device="meta")[1:, :1024].view(4, 4, 8, 32),
        torch.device("mps"),
    )
    with mode:

        def make(dtype=torch.float32, output_dtype=torch.float32, tokens=3, requests=2):
            q, k = (
                torch.empty(tokens, 2, 32, dtype=dtype, device="mps") for _ in range(2)
            )
            v = torch.empty(tokens, 4, 8, dtype=dtype, device="mps")
            g, beta = (
                torch.empty(tokens, 4, dtype=dtype, device="mps") for _ in range(2)
            )
            cu = torch.empty(requests + 1, dtype=torch.int32, device="mps")
            slots = torch.empty(requests, dtype=torch.int32, device="mps")
            y = torch.empty(tokens, 4, 8, dtype=output_dtype, device="mps")
            initial = torch.empty(requests, dtype=torch.bool, device="mps")
            return [q, k, v, g, beta, state, cu, slots, y, initial]

        yield make


@pytest.fixture
def gdn_library(monkeypatch):
    from vllm_metal.metal import build
    from vllm_metal.pytorch_backend import mps_ops

    kernel = Mock(thread_execution_width=32, max_threads_per_threadgroup=1024)
    load = Mock()
    monkeypatch.setattr(torch.mps, "load_metallib", load)
    monkeypatch.setattr(
        build, "prepare_metallib", Mock(return_value="gdn_kern.metallib")
    )
    monkeypatch.setattr(
        mps_ops,
        "_load_mps_module",
        Mock(side_effect=AssertionError("GDN must not load the custom extension")),
    )
    mps_ops._gdn_kernel.cache_clear()
    yield kernel, load
    mps_ops._gdn_kernel.cache_clear()


@pytest.mark.parametrize(
    "dtype,output_dtype,name",
    [
        (torch.float32, torch.float32, "float"),
        (torch.float16, torch.float16, "half_state_float"),
        (torch.bfloat16, torch.bfloat16, "bfloat16_t_state_float"),
        (torch.float32, torch.float16, "float_out_half"),
        (torch.float32, torch.bfloat16, "float_out_bfloat16_t"),
    ],
)
def test_gdn_torch_launcher_preserves_buffers_and_shader_abi(
    gdn_inputs, gdn_library, dtype, output_dtype, name
):
    from vllm_metal.pytorch_backend import mps_ops

    kernel, load = gdn_library
    library = SimpleNamespace(**{f"gdn_linear_attention_{name}_initial": kernel})
    load.return_value = library
    args = gdn_inputs(dtype, output_dtype)
    mps_ops.gdn_linear_attention(*args)
    call_args, kwargs = kernel.call_args
    assert all(
        actual is expected
        for actual, expected in zip(call_args[:9], args[:9], strict=True)
    )
    assert call_args[9:15] == (2, 2, 4, 32, 8, 1041)
    assert call_args[15] is args[9]
    assert kwargs == {
        "threads": (256, 1, 8),
        "group_size": (32, 1, 1),
        "arg_casts": dict.fromkeys(range(9, 14), "int32"),
    }
    assert mps_ops._gdn_kernel(dtype, output_dtype) == (library, kernel)
    load.assert_called_once_with("gdn_kern.metallib")
    kernel.assert_called_once()


@pytest.mark.parametrize("width,max_threads", [(16, 1024), (32, 16)])
def test_gdn_torch_launcher_rejects_incompatible_simd(gdn_library, width, max_threads):
    from vllm_metal.pytorch_backend import mps_ops

    kernel, load = gdn_library
    kernel.thread_execution_width, kernel.max_threads_per_threadgroup = (
        width,
        max_threads,
    )
    load.return_value = SimpleNamespace(gdn_linear_attention_float_initial=kernel)
    with pytest.raises(RuntimeError, match="32-lane SIMD"):
        mps_ops._gdn_kernel(torch.float32, torch.float32)
    kernel.assert_not_called()


@pytest.mark.parametrize("tokens,requests", [(0, 2), (3, 0)])
def test_gdn_empty_launch_does_not_load_a_library(
    gdn_inputs, gdn_library, tokens, requests
):
    from vllm_metal.pytorch_backend import mps_ops

    _, load = gdn_library
    mps_ops.gdn_linear_attention(*gdn_inputs(tokens=tokens, requests=requests))
    load.assert_not_called()


@pytest.mark.parametrize(
    "index,change,error",
    [
        (0, lambda t: t.to("cpu"), "must be on MPS"),
        (1, lambda t: t.to(torch.float16), "same dtype"),
        (1, lambda t: t.transpose(0, 1), "packed"),
        (2, lambda t: t.narrow(0, 0, 2), r"Q/K \[T,Hk,Dk\]"),
        (
            3,
            lambda t: torch.empty(t.shape[0], 2, dtype=t.dtype, device=t.device),
            r"g/beta must be",
        ),
        (5, lambda t: t.to(torch.float16), "state must be fp32"),
        (5, lambda t: t.transpose(2, 3), "dense inner rows"),
        (5, lambda t: t.as_strided(t.shape, (1, 256, 32, 1)), "dense inner rows"),
        (6, lambda t: t.to(torch.int64), "metadata must be packed int32"),
        (7, lambda t: t.narrow(0, 0, 1), "metadata must be packed int32"),
        (8, lambda t: t.to(torch.bfloat16).transpose(1, 2), "Y must be packed"),
        (9, lambda t: t.to(torch.int32), "has_initial_state must be packed bool"),
    ],
)
def test_gdn_invalid_buffers_fail_before_loading(
    gdn_inputs, gdn_library, index, change, error
):
    from vllm_metal.pytorch_backend import mps_ops

    _, load = gdn_library
    args = gdn_inputs()
    args[index] = change(args[index])
    with pytest.raises(RuntimeError, match=error):
        mps_ops.gdn_linear_attention(*args)
    load.assert_not_called()


@pytest.mark.parametrize(
    "hk,dk,error",
    [
        (2, 31, "positive multiple of 32"),
        (2, 288, "positive multiple of 32"),
        (3, 32, "Invalid GDN heads"),
        (0, 32, "Invalid GDN heads"),
    ],
)
def test_gdn_invalid_head_geometry_fails_before_loading(
    gdn_inputs, gdn_library, hk, dk, error
):
    from vllm_metal.pytorch_backend import mps_ops

    _, load = gdn_library
    args = gdn_inputs()
    args[:2] = [torch.empty(3, hk, dk, device="mps") for _ in range(2)]
    with pytest.raises(RuntimeError, match=error):
        mps_ops.gdn_linear_attention(*args)
    load.assert_not_called()


@pytest.mark.parametrize("tokens,requests", [(2**25, 2), (3, 2**29)])
def test_gdn_rejects_int32_index_overflow(gdn_inputs, gdn_library, tokens, requests):
    from vllm_metal.pytorch_backend import mps_ops

    _, load = gdn_library
    with pytest.raises(RuntimeError, match="exceeds int32 indexing"):
        mps_ops.gdn_linear_attention(*gdn_inputs(tokens=tokens, requests=requests))
    load.assert_not_called()


@pytest.mark.parametrize(
    "dtype,output_dtype",
    [
        (torch.float16, torch.bfloat16),
        (torch.bfloat16, torch.float16),
        (torch.float16, torch.float32),
    ],
)
def test_gdn_rejects_unsupported_output_casts(
    gdn_inputs, gdn_library, dtype, output_dtype
):
    from vllm_metal.pytorch_backend import mps_ops

    _, load = gdn_library
    with pytest.raises(RuntimeError, match="Y must be packed"):
        mps_ops.gdn_linear_attention(*gdn_inputs(dtype, output_dtype))
    load.assert_not_called()
