# SPDX-License-Identifier: Apache-2.0
"""Numerics tests for the MLX DFlash draft model vs a torch reference.

The reference implements the exact math of vLLM 0.24.0's
``DFlashQwen3ForCausalLM`` (qwen3_dflash.py): fc combine, hidden_norm,
per-layer context K/V projection with k_norm + neox RoPE, the standard Qwen3
query path attending non-causally over [context, queries], and the d2t
draft->target readout. A convention mismatch (RoPE style, norm epsilon,
GQA layout, mask) shows up as a hard failure here.
"""

from __future__ import annotations

import math

import mlx.core as mx
import numpy as np
import pytest
import torch

from vllm_metal.v1.dflash_model import (
    DFlashDraftModel,
    DFlashModelArgs,
)

ARGS = DFlashModelArgs(
    hidden_size=64,
    num_hidden_layers=2,
    intermediate_size=96,
    num_attention_heads=4,
    num_key_value_heads=2,
    head_dim=16,
    rms_norm_eps=1e-6,
    rope_theta=10000.0,
    max_position_embeddings=2048,
    vocab_size=120,
    draft_vocab_size=48,
    mask_token_id=110,
    num_aux_layers=3,
    target_hidden_size=64,
)


def _rand(rng: np.random.Generator, *shape: int) -> np.ndarray:
    return (rng.standard_normal(shape) * 0.05).astype(np.float32)


def _build_weights(args: DFlashModelArgs, seed: int = 0) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    h, hd = args.hidden_size, args.head_dim
    nq, nkv = args.num_attention_heads, args.num_key_value_heads
    weights: dict[str, np.ndarray] = {
        "embed_tokens.weight": _rand(rng, args.vocab_size, h),
        "fc.weight": _rand(rng, h, args.target_hidden_size * args.num_aux_layers),
        "hidden_norm.weight": 1.0 + _rand(rng, h),
        "norm.weight": 1.0 + _rand(rng, h),
        "lm_head.weight": _rand(rng, args.draft_vocab_size, h),
        # Injective draft->target map like real checkpoints: distinct target
        # ids, stored as offsets (target_id = draft_id + d2t[draft_id]).
        "d2t": (
            np.sort(
                rng.choice(args.vocab_size, size=args.draft_vocab_size, replace=False)
            )
            - np.arange(args.draft_vocab_size)
        ).astype(np.int64),
    }
    for i in range(args.num_hidden_layers):
        p = f"layers.{i}."
        weights[p + "self_attn.q_proj.weight"] = _rand(rng, nq * hd, h)
        weights[p + "self_attn.k_proj.weight"] = _rand(rng, nkv * hd, h)
        weights[p + "self_attn.v_proj.weight"] = _rand(rng, nkv * hd, h)
        weights[p + "self_attn.o_proj.weight"] = _rand(rng, h, nq * hd)
        weights[p + "self_attn.q_norm.weight"] = 1.0 + _rand(rng, hd)
        weights[p + "self_attn.k_norm.weight"] = 1.0 + _rand(rng, hd)
        weights[p + "input_layernorm.weight"] = 1.0 + _rand(rng, h)
        weights[p + "post_attention_layernorm.weight"] = 1.0 + _rand(rng, h)
        weights[p + "mlp.gate_proj.weight"] = _rand(rng, args.intermediate_size, h)
        weights[p + "mlp.up_proj.weight"] = _rand(rng, args.intermediate_size, h)
        weights[p + "mlp.down_proj.weight"] = _rand(rng, h, args.intermediate_size)
    return weights


# ---------------------------------------------------------------------------
# Torch reference (mirrors vllm 0.24.0 qwen3_dflash.py math)
# ---------------------------------------------------------------------------


def _t_rms(x: torch.Tensor, w: torch.Tensor, eps: float) -> torch.Tensor:
    return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + eps) * w


def _t_rope(x: torch.Tensor, positions: torch.Tensor, theta: float) -> torch.Tensor:
    """Neox-style (half-split) RoPE. x: [..., T, head_dim]."""
    hd = x.shape[-1]
    inv_freq = 1.0 / (theta ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
    freqs = positions.to(torch.float32)[:, None] * inv_freq[None, :]  # [T, hd/2]
    cos, sin = freqs.cos(), freqs.sin()
    x1, x2 = x[..., : hd // 2], x[..., hd // 2 :]
    return torch.cat([x1 * cos - x2 * sin, x2 * cos + x1 * sin], dim=-1)


class _TorchRef:
    def __init__(self, args: DFlashModelArgs, w: dict[str, np.ndarray]) -> None:
        self.args = args
        self.w = {k: torch.from_numpy(v) for k, v in w.items()}

    def combine(self, aux: torch.Tensor) -> torch.Tensor:
        return aux @ self.w["fc.weight"].T

    def context_kv(
        self, combined: torch.Tensor, positions: torch.Tensor
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        a = self.args
        normed = _t_rms(combined, self.w["hidden_norm.weight"], a.rms_norm_eps)
        out = []
        for i in range(a.num_hidden_layers):
            p = f"layers.{i}.self_attn."
            k = (normed @ self.w[p + "k_proj.weight"].T).view(
                -1, a.num_key_value_heads, a.head_dim
            )
            k = _t_rms(k, self.w[p + "k_norm.weight"], a.rms_norm_eps)
            k = _t_rope(k.transpose(0, 1), positions, a.rope_theta)  # [nkv, T, hd]
            v = (
                (normed @ self.w[p + "v_proj.weight"].T)
                .view(-1, a.num_key_value_heads, a.head_dim)
                .transpose(0, 1)
            )
            out.append((k, v))
        return out

    def forward(
        self,
        q_ids: torch.Tensor,
        ctx_kv: list[tuple[torch.Tensor, torch.Tensor]],
        q_positions: torch.Tensor,
    ) -> torch.Tensor:
        a = self.args
        group = a.num_attention_heads // a.num_key_value_heads
        h = self.w["embed_tokens.weight"][q_ids]
        for i in range(a.num_hidden_layers):
            p = f"layers.{i}."
            x = _t_rms(h, self.w[p + "input_layernorm.weight"], a.rms_norm_eps)
            q = (x @ self.w[p + "self_attn.q_proj.weight"].T).view(
                -1, a.num_attention_heads, a.head_dim
            )
            q = _t_rms(q, self.w[p + "self_attn.q_norm.weight"], a.rms_norm_eps)
            q = _t_rope(q.transpose(0, 1), q_positions, a.rope_theta)  # [nq, Q, hd]
            k_new = (x @ self.w[p + "self_attn.k_proj.weight"].T).view(
                -1, a.num_key_value_heads, a.head_dim
            )
            k_new = _t_rms(k_new, self.w[p + "self_attn.k_norm.weight"], a.rms_norm_eps)
            k_new = _t_rope(k_new.transpose(0, 1), q_positions, a.rope_theta)
            v_new = (
                (x @ self.w[p + "self_attn.v_proj.weight"].T)
                .view(-1, a.num_key_value_heads, a.head_dim)
                .transpose(0, 1)
            )

            ck, cv = ctx_kv[i]
            k = torch.cat([ck, k_new], dim=1)  # [nkv, T+Q, hd]
            v = torch.cat([cv, v_new], dim=1)
            k = k.repeat_interleave(group, dim=0)  # [nq, T+Q, hd]
            v = v.repeat_interleave(group, dim=0)
            scores = (q @ k.transpose(1, 2)) / math.sqrt(a.head_dim)
            attn = torch.softmax(scores.to(torch.float32), dim=-1).to(q.dtype) @ v
            attn = attn.transpose(0, 1).reshape(len(q_ids), -1)
            h = h + attn @ self.w[p + "self_attn.o_proj.weight"].T

            x = _t_rms(h, self.w[p + "post_attention_layernorm.weight"], a.rms_norm_eps)
            gate = torch.nn.functional.silu(x @ self.w[p + "mlp.gate_proj.weight"].T)
            up = x @ self.w[p + "mlp.up_proj.weight"].T
            h = h + (gate * up) @ self.w[p + "mlp.down_proj.weight"].T
        return _t_rms(h, self.w["norm.weight"], a.rms_norm_eps)

    def full_vocab_logits(self, hidden: torch.Tensor) -> torch.Tensor:
        """Upstream compute_logits: scatter draft logits into -inf full vocab."""
        a = self.args
        logits = hidden @ self.w["lm_head.weight"].T
        full = torch.full((hidden.shape[0], a.vocab_size), float("-inf"))
        targets = torch.arange(a.draft_vocab_size) + self.w["d2t"]
        full[:, targets] = logits
        return full


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def setup():
    """Build model + torch reference, running MLX on CPU.

    On M5-class GPUs MLX routes fp32 GEMMs through reduced-precision tensor
    cores (~1e-3 relative error), which would drown out real convention bugs.
    The CPU device gives true fp32 so tolerances can stay tight.
    """
    prev_device = mx.default_device()
    mx.set_default_device(mx.cpu)
    try:
        weights = _build_weights(ARGS)
        model = DFlashDraftModel(ARGS)
        model.load_weights([(k, mx.array(v)) for k, v in weights.items()], strict=True)
        mx.eval(model.parameters())
        yield model, _TorchRef(ARGS, weights), weights
    finally:
        mx.set_default_device(prev_device)


def test_draft_forward_matches_torch_reference(setup) -> None:
    """Full draft step vs the torch reference, with context rows starting at
    a nonzero absolute position (exercises RoPE offsets on both the context
    projection and the query path). The dense reference softmax also pins the
    non-causal mask semantics: a causal regression would diverge here."""
    model, ref, _ = setup
    rng = np.random.default_rng(7)
    num_ctx, num_query, ctx_offset = 9, 5, 23
    aux = (
        rng.standard_normal((num_ctx, ARGS.target_hidden_size * ARGS.num_aux_layers))
        * 0.1
    ).astype(np.float32)
    q_ids = np.array(
        [
            3,
            ARGS.mask_token_id,
            ARGS.mask_token_id,
            ARGS.mask_token_id,
            ARGS.mask_token_id,
        ],
        dtype=np.int32,
    )
    query_offset = ctx_offset + num_ctx

    combined = model.combine_hidden_states(mx.array(aux))
    ck, cv = model.project_context_kv(combined, ctx_offset)
    hidden = model(mx.array(q_ids), ck, cv, position_offset=query_offset)
    logits = model.compute_draft_logits(hidden)
    mx.eval(logits)

    t_combined = ref.combine(torch.from_numpy(aux))
    t_ctx = ref.context_kv(t_combined, torch.arange(ctx_offset, ctx_offset + num_ctx))
    t_hidden = ref.forward(
        torch.from_numpy(q_ids.astype(np.int64)),
        t_ctx,
        torch.arange(query_offset, query_offset + num_query),
    )
    t_logits = t_hidden @ ref.w["lm_head.weight"].T

    np.testing.assert_allclose(
        np.array(logits, dtype=np.float32),
        t_logits.numpy(),
        rtol=2e-4,
        atol=2e-4,
    )


def test_d2t_readout_matches_full_vocab_scatter(setup) -> None:
    """argmax(draft) + d2t offset == argmax over upstream's -inf scatter."""
    model, ref, _ = setup
    rng = np.random.default_rng(13)
    hidden = (rng.standard_normal((6, ARGS.hidden_size)) * 0.3).astype(np.float32)

    draft_ids = mx.argmax(model.compute_draft_logits(mx.array(hidden)), axis=-1)
    mapped = model.map_draft_to_target(draft_ids)
    mx.eval(mapped)

    full = ref.full_vocab_logits(torch.from_numpy(hidden))
    expected = torch.argmax(full, dim=-1).numpy()
    np.testing.assert_array_equal(np.array(mapped, dtype=np.int64), expected)
