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

_CHUNK_SIZE = 64


def _gating(a_log, dt_bias, a, b):
    g = -torch.exp(a_log.float()) * torch.nn.functional.softplus(
        a.float() + dt_bias.float()
    )
    return g, torch.sigmoid(b.float())


def _l2norm(x, eps=1e-6):
    return x * torch.rsqrt((x * x).sum(-1, keepdim=True) + eps)


def delta_rule_step(q, k, v, g, beta, state):
    # state [B, H, K, V] is fp32 and updated in place.
    state.mul_(torch.exp(g)[..., None, None])
    delta = (v - torch.einsum("bhk,bhkv->bhv", k, state)) * beta[..., None]
    state.add_(k[..., None] * delta[..., None, :])
    return torch.einsum("bhk,bhkv->bhv", q, state)


def delta_rule_chunked(q, k, v, g, beta, state, chunk=_CHUNK_SIZE):
    length, heads = q.shape[:2]
    v_dim = v.shape[-1]
    # TODO: Use gdn_linear_attention.metal to avoid padding short prefills to
    # 64 tokens and running 63 forward-substitution steps.
    pad = (-length) % chunk
    if pad:  # k = beta = g = 0 rows leave the state untouched
        q, k, v, g, beta = (
            torch.cat([t, t.new_zeros(pad, *t.shape[1:])]) for t in (q, k, v, g, beta)
        )
    n = q.shape[0] // chunk
    q, k, v, g, beta = (
        t.transpose(0, 1).reshape(heads, n, chunk, *t.shape[2:])
        for t in (q, k, v, g, beta)
    )
    vb, kb = v * beta[..., None], k * beta[..., None]
    decay_log = g.cumsum(-1)
    ones = torch.ones(chunk, chunk, dtype=torch.bool, device=q.device)
    decay = decay_log[..., :, None] - decay_log[..., None, :]
    decay = decay.masked_fill(~ones.tril(), float("-inf")).exp()
    inv = -((kb @ k.transpose(-1, -2)) * decay).masked_fill(~ones.tril(-1), 0.0)
    for i in range(1, chunk):  # (I + A)^-1 by forward substitution
        inv[..., i, :i] += (inv[..., i, :, None] * inv[..., :, :i]).sum(-2)
    inv += torch.eye(chunk, device=q.device)
    u, w = inv @ vb, inv @ (kb * decay_log.exp()[..., None])
    out = torch.empty(heads, n, chunk, v_dim, dtype=torch.float32, device=q.device)
    for c in range(n):
        qc, kc, gc = q[:, c], k[:, c], decay_log[:, c]
        scores = (qc @ kc.transpose(-1, -2)) * decay[:, c]
        new = u[:, c] - w[:, c] @ state
        out[:, c] = (qc * gc.exp()[..., None]) @ state + scores @ new
        kept = (kc * (gc[:, -1:] - gc).exp()[..., None]).transpose(-1, -2) @ new
        state = state * gc[:, -1].exp()[:, None, None] + kept
    return out.reshape(heads, n * chunk, v_dim).transpose(0, 1)[:length], state


def _split_qkv(layer, qkv):
    kd, vd = layer.key_dim // layer.tp_size, layer.value_dim // layer.tp_size
    q, k, v = qkv.split([kd, kd, vd], dim=-1)
    tokens, hk, hv = qkv.shape[0], kd // layer.head_k_dim, vd // layer.head_v_dim
    q, k = (_l2norm(t.float().view(tokens, hk, layer.head_k_dim)) for t in (q, k))
    if hv != hk:
        q, k = (t.repeat_interleave(hv // hk, dim=1) for t in (q, k))
    v = v.float().view(tokens, hv, layer.head_v_dim)
    return q * layer.head_k_dim**-0.5, k, v


def _gdn_core(layer, mixed_qkv, b, a, out, md):
    conv = layer.kv_cache[0]
    if not is_conv_state_dim_first():
        conv = conv.transpose(-1, -2)
    ssm = layer.kv_cache[1]
    ssm = ssm.view(*ssm.shape[:2], layer.head_k_dim, layer.head_v_dim)
    weight = layer.conv1d.weight.view(layer.conv1d.weight.size(0), -1)
    slots, loc = md.non_spec_state_indices_tensor, md.non_spec_query_start_loc
    nd, ndt = md.num_decodes, md.num_decode_tokens
    npf, npt = md.num_prefills, md.num_prefill_tokens

    if nd > 0:
        idx = slots[:nd].long()
        live = idx != PAD_SLOT_ID
        idx = idx[live]
        conv_state = conv[idx]
        x = causal_conv1d_update_torch(
            mixed_qkv[:ndt][live][..., None],
            conv_state,
            weight,
            layer.conv1d.bias,
            layer.activation,
        )[..., 0]
        conv[idx] = conv_state
        q, k, v = _split_qkv(layer, x)
        g, beta = _gating(layer.A_log, layer.dt_bias, a[:ndt][live], b[:ndt][live])
        # TODO: Pass slot IDs and unexpanded Q/K heads to the shared Metal
        # kernel so decode updates state in place without copying active slots.
        state = ssm[idx]
        o = delta_rule_step(q, k, v, g, beta, state)
        ssm[idx] = state
        out[:ndt][live] = o.to(out.dtype)

    if npf > 0:
        lo, hi = ndt, ndt + npt
        p_slots = slots[nd : nd + npf]
        p_loc = loc[nd : nd + npf + 1] - ndt
        p_init = md.has_initial_state[nd : nd + npf]
        x = causal_conv1d_fn_cpu(
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
        g, beta = _gating(layer.A_log, layer.dt_bias, a[lo:hi], b[lo:hi])
        # TODO: Pass the packed sequence offsets and slots to the shared Metal
        # kernel to avoid this per-request Python loop and its host reads.
        bounds, s_list, init = p_loc.tolist(), p_slots.tolist(), p_init.tolist()
        for i, slot in enumerate(s_list):
            s, e = bounds[i], bounds[i + 1]
            if slot == PAD_SLOT_ID or s == e:
                continue
            # Copy the slot: MPSGraph matmuls reject a view at a non-zero offset
            # with more than INT_MAX elements after it, as the shared KV buffer has.
            state = ssm[slot].clone() if init[i] else torch.zeros_like(ssm[slot])
            o, ssm[slot] = delta_rule_chunked(
                q[s:e], k[s:e], v[s:e], g[s:e], beta[s:e], state
            )
            out[lo + s : lo + e] = o.to(out.dtype)


@cache
def install():
    from vllm.model_executor.layers.mamba.gdn import qwen_gdn_linear_attn  # noqa: F401

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
