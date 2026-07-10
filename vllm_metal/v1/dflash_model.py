# SPDX-License-Identifier: Apache-2.0
"""MLX draft model for DFlash speculative decoding.

Ports vLLM 0.24.0's ``DFlashQwen3ForCausalLM``
(``vllm/model_executor/models/qwen3_dflash.py``) to MLX. The draft is a small
stack of standard Qwen3 decoder layers with two DFlash-specific pieces:

- ``combine_hidden_states``: one ``fc`` projection over the concatenation of N
  target-layer hidden states (``aux_hidden_state_layer_ids``), producing the
  context representation.
- ``project_context_kv``: per-layer K/V projections of the fc-combined context
  (``hidden_norm`` -> ``k_proj``/``v_proj`` -> ``k_norm`` -> RoPE). Context K/V
  never flows through the decoder layers; it is attended to directly by the
  query tokens (bonus + K mask tokens) in a single **non-causal** forward.

Weight names match the checkpoints (``layers.{i}.self_attn.q_proj`` etc.), so
loading is a direct tree match with no renames beyond upstream's
``midlayer. -> layers.0.`` compatibility rule.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models import qwen3 as _mlx_qwen3
from mlx_lm.models.rope_utils import initialize_rope
from vllm.logger import init_logger

logger = init_logger(__name__)

_SKIPPED_WEIGHT_PREFIXES = ("t2d",)


@dataclass(frozen=True, slots=True)
class DFlashModelArgs:
    """Draft-model hyperparameters parsed from a speculators checkpoint."""

    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    rope_theta: float
    max_position_embeddings: int
    vocab_size: int
    draft_vocab_size: int
    mask_token_id: int
    num_aux_layers: int
    target_hidden_size: int
    attention_bias: bool = False

    @classmethod
    def from_checkpoint_config(cls, config: dict) -> DFlashModelArgs:
        tl = config["transformer_layer_config"]
        # transformers >=5 serializes rope under ``rope_parameters``; older
        # checkpoints keep a top-level ``rope_theta``. Upstream defaults to
        # 1e6 only when the checkpoint carries neither
        # (set_default_rope_theta in qwen3_dflash.py).
        rope_parameters = tl.get("rope_parameters") or {}
        rope_theta = rope_parameters.get("rope_theta")
        if rope_theta is None:
            rope_theta = tl.get("rope_theta", 1_000_000.0)

        aux_layer_ids = config["aux_hidden_state_layer_ids"]
        hidden_size = int(tl["hidden_size"])
        target_hidden_size = config.get("target_hidden_size") or hidden_size

        num_heads = int(tl["num_attention_heads"])
        head_dim = int(tl.get("head_dim") or hidden_size // num_heads)

        vocab_size = int(tl["vocab_size"])
        draft_vocab_size = int(config.get("draft_vocab_size") or vocab_size)

        return cls(
            hidden_size=hidden_size,
            num_hidden_layers=int(tl["num_hidden_layers"]),
            intermediate_size=int(tl["intermediate_size"]),
            num_attention_heads=num_heads,
            num_key_value_heads=int(tl["num_key_value_heads"]),
            head_dim=head_dim,
            rms_norm_eps=float(tl["rms_norm_eps"]),
            rope_theta=float(rope_theta),
            max_position_embeddings=int(tl.get("max_position_embeddings", 40960)),
            vocab_size=vocab_size,
            draft_vocab_size=draft_vocab_size,
            mask_token_id=int(config["mask_token_id"]),
            num_aux_layers=len(aux_layer_ids),
            target_hidden_size=int(target_hidden_size),
            attention_bias=bool(tl.get("attention_bias", False)),
        )


class DFlashAttention(nn.Module):
    """Qwen3-style attention whose K/V context is supplied externally.

    The forward only projects the query tokens; context K/V comes from
    ``DFlashDraftModel.project_context_kv`` (target hidden states), so this
    layer concatenates [context K/V, query K/V] and runs one non-causal SDPA.
    """

    def __init__(self, args: DFlashModelArgs) -> None:
        super().__init__()
        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        bias = args.attention_bias
        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=bias)
        self.q_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(self.head_dim, eps=args.rms_norm_eps)
        self.rope = initialize_rope(
            self.head_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=None,
            max_position_embeddings=args.max_position_embeddings,
        )

    def project_context_kv(
        self, normed_context: mx.array, position_offset: int
    ) -> tuple[mx.array, mx.array]:
        """Project normed context states to this layer's rotated K and V.

        Args:
            normed_context: ``[num_ctx, hidden]`` — ``hidden_norm`` output.
            position_offset: absolute position of the first context row; the
                rows must be positionally contiguous.

        Returns ``(k, v)`` of shape ``[n_kv_heads, num_ctx, head_dim]``.
        """
        num_ctx = normed_context.shape[0]
        k = self.k_proj(normed_context).reshape(num_ctx, self.n_kv_heads, -1)
        k = self.k_norm(k).transpose(1, 0, 2)
        k = self.rope(k, offset=position_offset)
        v = self.v_proj(normed_context).reshape(num_ctx, self.n_kv_heads, -1)
        v = v.transpose(1, 0, 2)
        return k, v

    def __call__(
        self,
        x: mx.array,
        context_k: mx.array,
        context_v: mx.array,
        position_offset: int,
    ) -> mx.array:
        """Non-causal attention of query tokens over [context, queries].

        Args:
            x: ``[num_query, hidden]`` query hidden states.
            context_k / context_v: ``[n_kv_heads, num_ctx, head_dim]``.
            position_offset: absolute position of the first query token
                (= number of context tokens for DFlash).
        """
        num_query = x.shape[0]
        q = self.q_proj(x).reshape(num_query, self.n_heads, -1)
        q = self.q_norm(q).transpose(1, 0, 2)
        q = self.rope(q, offset=position_offset)

        k = self.k_proj(x).reshape(num_query, self.n_kv_heads, -1)
        k = self.k_norm(k).transpose(1, 0, 2)
        k = self.rope(k, offset=position_offset)
        v = self.v_proj(x).reshape(num_query, self.n_kv_heads, -1).transpose(1, 0, 2)

        keys = mx.concatenate([context_k, k], axis=1)
        values = mx.concatenate([context_v, v], axis=1)

        # Every query attends to all context and every query (block-diffusion
        # parallel drafting): a dense SDPA with no mask.
        out = mx.fast.scaled_dot_product_attention(
            q[None],
            keys[None],
            values[None],
            scale=self.scale,
            mask=None,
        )
        out = out[0].transpose(1, 0, 2).reshape(num_query, -1)
        return self.o_proj(out)


class DFlashBlock(nn.Module):
    """Standard Qwen3 decoder block over externally supplied context K/V."""

    def __init__(self, args: DFlashModelArgs) -> None:
        super().__init__()
        self.self_attn = DFlashAttention(args)
        self.mlp = _mlx_qwen3.MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    def __call__(
        self,
        x: mx.array,
        context_k: mx.array,
        context_v: mx.array,
        position_offset: int,
    ) -> mx.array:
        h = x + self.self_attn(
            self.input_layernorm(x), context_k, context_v, position_offset
        )
        return h + self.mlp(self.post_attention_layernorm(h))


class DFlashDraftModel(nn.Module):
    """DFlash draft model.

    One proposal step:
    1. ``combine_hidden_states`` + ``project_context_kv`` append the newly
       committed tokens' context K/V (owned by the proposer, not this module).
    2. ``__call__`` runs the query tokens ``[bonus, mask * K]`` through the
       decoder stack, attending non-causally over context + queries.
    3. ``compute_draft_logits``/``map_draft_to_target`` read out the drafts.
    """

    def __init__(self, args: DFlashModelArgs) -> None:
        super().__init__()
        self.args = args
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [DFlashBlock(args) for _ in range(args.num_hidden_layers)]
        self.fc = nn.Linear(
            args.target_hidden_size * args.num_aux_layers,
            args.hidden_size,
            bias=False,
        )
        self.hidden_norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.lm_head = nn.Linear(args.hidden_size, args.draft_vocab_size, bias=False)
        if args.draft_vocab_size != args.vocab_size:
            # Offset table: target_id = draft_id + d2t[draft_id]
            # (matches upstream compute_logits: targets = arange + d2t).
            self.d2t = mx.zeros((args.draft_vocab_size,), dtype=mx.int64)
        else:
            self.d2t = None

    # -- context side ---------------------------------------------------------

    def combine_hidden_states(self, aux_hidden_states: mx.array) -> mx.array:
        """Project ``[num_tokens, num_aux * target_hidden]`` aux concat via fc."""
        return self.fc(aux_hidden_states)

    def project_context_kv(
        self, context_states: mx.array, position_offset: int
    ) -> tuple[mx.array, mx.array]:
        """Compute per-layer context K/V from fc-combined target states.

        Mirrors upstream ``precompute_and_store_context_kv``:
        ``hidden_norm`` once, then per-layer K/V projection + k_norm + RoPE.

        Args:
            context_states: ``[num_ctx, hidden]`` fc-combined states for
                positionally contiguous rows starting at ``position_offset``.

        Returns ``(k, v)`` of shape
        ``[num_layers, n_kv_heads, num_ctx, head_dim]``.
        """
        normed = self.hidden_norm(context_states)
        ks, vs = [], []
        for layer in self.layers:
            k, v = layer.self_attn.project_context_kv(normed, position_offset)
            ks.append(k)
            vs.append(v)
        return mx.stack(ks, axis=0), mx.stack(vs, axis=0)

    # -- query side -----------------------------------------------------------

    def __call__(
        self,
        input_ids: mx.array,
        context_k: mx.array,
        context_v: mx.array,
        position_offset: int,
    ) -> mx.array:
        """Run query tokens through the stack; return normed hidden states.

        Args:
            input_ids: ``[num_query]`` — bonus token followed by mask tokens.
            context_k / context_v: ``[num_layers, n_kv_heads, num_ctx, hd]``.
            position_offset: absolute position of the bonus token.
        """
        h = self.embed_tokens(input_ids)
        for i, layer in enumerate(self.layers):
            h = layer(h, context_k[i], context_v[i], position_offset)
        return self.norm(h)

    def compute_draft_logits(self, hidden_states: mx.array) -> mx.array:
        """Draft-vocab logits (no d2t scatter; greedy readout maps ids)."""
        return self.lm_head(hidden_states)

    def map_draft_to_target(self, draft_token_ids: mx.array) -> mx.array:
        """Map draft-vocab argmax ids to target-vocab ids via the d2t offsets."""
        if self.d2t is None:
            return draft_token_ids
        return draft_token_ids + self.d2t[draft_token_ids]


def load_dflash_draft_model(
    model_path: str | Path, *, dtype: mx.Dtype = mx.bfloat16
) -> tuple[DFlashDraftModel, DFlashModelArgs, dict]:
    """Load a speculators-format DFlash checkpoint into MLX.

    Returns ``(model, args, raw_config)``.
    """
    path = Path(model_path)
    config = json.loads((path / "config.json").read_text())
    args = DFlashModelArgs.from_checkpoint_config(config)
    model = DFlashDraftModel(args)

    weight_files = sorted(path.glob("*.safetensors"))
    if not weight_files:
        raise FileNotFoundError(f"No safetensors found under {path}")
    weights: dict[str, mx.array] = {}
    for wf in weight_files:
        weights.update(mx.load(str(wf)))

    sanitized: dict[str, mx.array] = {}
    for name, value in weights.items():
        # Upstream compatibility rename for single-layer checkpoints.
        if "midlayer." in name:
            name = name.replace("midlayer.", "layers.0.")
        if name.startswith(_SKIPPED_WEIGHT_PREFIXES):
            continue
        if name == "d2t":
            sanitized["d2t"] = value.astype(mx.int64)
            continue
        if mx.issubdtype(value.dtype, mx.floating):
            value = value.astype(dtype)
        sanitized[name] = value

    model.load_weights(list(sanitized.items()), strict=True)
    mx.eval(model.parameters())
    model.eval()
    logger.info(
        "Loaded DFlash draft model from %s: layers=%d hidden=%d heads=%d/%d "
        "head_dim=%d rope_theta=%.0f draft_vocab=%d aux_layers=%d",
        path,
        args.num_hidden_layers,
        args.hidden_size,
        args.num_attention_heads,
        args.num_key_value_heads,
        args.head_dim,
        args.rope_theta,
        args.draft_vocab_size,
        args.num_aux_layers,
    )
    return model, args, config
