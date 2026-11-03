# SPDX-License-Identifier: Apache-2.0

from functools import cache

import torch
from vllm.forward_context import get_forward_context
from vllm.model_executor.layers.mamba.mamba_utils import is_conv_state_dim_first
from vllm.model_executor.layers.mamba.ops.cpu.causal_conv1d import (
    causal_conv1d_fn_cpu,
    causal_conv1d_update_torch,
)
from vllm.utils.torch_utils import _resolve_layer_name
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from vllm_metal.pytorch_backend.mps_ops import gdn_linear_attention


def _gating(a_log, dt_bias, a, b):
    g = -torch.exp(a_log.float()) * torch.nn.functional.softplus(
        a.float() + dt_bias.float()
    )
    return g, torch.sigmoid(b.float())


def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def _split_qkv(layer, qkv):
    kd, vd = layer.key_dim // layer.tp_size, layer.value_dim // layer.tp_size
    q, k, v = qkv.split([kd, kd, vd], dim=-1)
    tokens, hk, hv = qkv.shape[0], kd // layer.head_k_dim, vd // layer.head_v_dim
    q, k = (_l2norm(t.float().view(tokens, hk, layer.head_k_dim)) for t in (q, k))
    v = v.float().reshape(tokens, hv, layer.head_v_dim).contiguous()
    return q * layer.head_k_dim**-0.5, k, v


def _gdn_core(layer, mixed_qkv, b, a, out, md):
    conv = layer.kv_cache[0]
    if not is_conv_state_dim_first():
        conv = conv.transpose(-1, -2)
    # Upstream's MambaSpec already binds [slots, Hv, Dv, Dk], including page gaps.
    ssm = layer.kv_cache[1]
    weight = layer.conv1d.weight.view(layer.conv1d.weight.size(0), -1)
    slots, loc = md.non_spec_state_indices_tensor, md.non_spec_query_start_loc
    nd, ndt = md.num_decodes, md.num_decode_tokens
    npf, npt = md.num_prefills, md.num_prefill_tokens
    num_tokens, num_requests = ndt + npt, nd + npf
    # PAD decode rows stay uninitialized; the shader returns before reading
    # their Q/K/V or touching output/state, so they cannot affect live rows.
    x = torch.empty_like(mixed_qkv[:num_tokens])

    if nd > 0:
        idx = slots[:nd].long()
        live = idx != PAD_SLOT_ID
        idx = idx[live]
        conv_state = conv[idx]
        x[:ndt][live] = causal_conv1d_update_torch(
            mixed_qkv[:ndt][live][..., None],
            conv_state,
            weight,
            layer.conv1d.bias,
            layer.activation,
        )[..., 0]
        conv[idx] = conv_state

    if npf > 0:
        lo, hi = ndt, ndt + npt
        p_slots = slots[nd : nd + npf]
        p_loc = loc[nd : nd + npf + 1] - ndt
        p_init = md.has_initial_state[nd : nd + npf]
        x[lo:hi] = causal_conv1d_fn_cpu(
            mixed_qkv[lo:hi].transpose(0, 1),
            weight,
            layer.conv1d.bias,
            conv,
            p_loc,
            cache_indices=p_slots,
            has_initial_state=p_init,
            activation=layer.activation,
        ).transpose(0, 1)

    q, k, v = _split_qkv(layer, x)
    g, beta = _gating(layer.A_log, layer.dt_bias, a[:num_tokens], b[:num_tokens])
    # Decode-only metadata has no has_initial_state tensor.
    initial = torch.ones(num_requests, dtype=torch.bool, device=x.device)
    if npf > 0:
        initial[nd:] = p_init
    # The shader uses one dtype for Q/K/V and Y. Keeping fp32 Q/K/V requires
    # this output copy and cast back for fp16/bf16; it also preserves PAD rows.
    # A contiguous fp32 output already shares storage with result.
    result = out[:num_tokens].float().contiguous()
    # The kernel copies resumed slots into private FP32 state before recurrence;
    # no pooled state view reaches an MPSGraph matmul (#1244).
    gdn_linear_attention(
        q,
        k,
        v,
        g.exp(),
        beta,
        ssm,
        loc[: num_requests + 1].to(device=x.device, dtype=torch.int32),
        slots[:num_requests].to(device=x.device, dtype=torch.int32),
        result,
        initial,
    )
    out[:num_tokens].copy_(result)


@cache
def install():
    from vllm import ModelRegistry
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: F401
    from vllm.model_executor.models.qwen3_5 import Qwen3_5ForCausalLM
    from vllm.model_executor.models.utils import WeightsMapper

    class MPSQwen35ForCausalLM(Qwen3_5ForCausalLM):
        # vLLM 0.32 keeps the published vision-language checkpoints' vision
        # tower in the text-only mapper, so loading them as text fails.
        hf_to_vllm_mapper = WeightsMapper(
            orig_to_new_prefix={
                "model.language_model.": "model.",
                "model.visual.": None,
                "mtp.": None,
            },
        )

    ModelRegistry.register_model("Qwen3_5ForCausalLM", MPSQwen35ForCausalLM)

    # Upstream's forward keeps the projections; only its core op needs MPS.
    @torch.library.impl("vllm::qwen_gdn_attention_core", "MPS")
    def mps_qwen_gdn_attention_core(
        mixed_qkv, b, a, core_attn_out, layer_name, use_aiter=False
    ):
        if use_aiter:
            raise NotImplementedError("The AITER GDN path requires ROCm")
        context = get_forward_context()
        if context.attn_metadata is None:
            return
        name = _resolve_layer_name(layer_name)
        md = context.attn_metadata[name]
        if md.num_actual_tokens:
            layer = context.no_compile_layers[name]
            _gdn_core(layer, mixed_qkv, b, a, core_attn_out, md)
