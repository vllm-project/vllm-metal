# SPDX-License-Identifier: Apache-2.0
"""MLX model classes and weight transform for the GLM-4.7-Flash nextn head.

The ``glm4_moe_lite`` (GLM-4.7-Flash) family ships a single trained nextn layer
(``model.layers.<num_hidden_layers>.*``) that predicts one extra token. This
module runs that layer as a tiny standalone MLX model over the "slot stream":

    slot_p = eh_proj([enorm(embed(t_{p+1})), hnorm(h_p)])

where ``h_p`` is the target's post-final-norm hidden state. One stock mlx_lm
``Glm4MoeLiteDecoderLayer`` forward over the appended slots, then the shared
head norm + untied lm_head, yields one greedy draft token.

The checkpoint this model loads is the flat, post-``sanitize`` layout emitted by
the mlx-vlm ``glm4_moe_lite_mtp`` drafter split; that split and
``convert_nextn_weights`` share the transform so there is one source of truth.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, fields
from typing import Any

import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.glm4_moe_lite import Glm4MoeLiteDecoderLayer
from mlx_lm.models.glm4_moe_lite import ModelArgs as Glm4MoeLiteBackboneArgs

from vllm_metal.v1.mtp_heads.glm4_moe_lite_mtp import (
    GLM4_MOE_LITE_MTP_MODEL_TYPE,
    GLM4_MOE_LITE_MTP_NUM_NEXTN,
    GLM4_MOE_LITE_TARGET_MODEL_TYPE,
    HEAD_SOURCE_HINT,
)

# Flat-layout prefixes shared by the model tree and the weight transform below.
_MTP_ATTN_PREFIX = "model.mtp_block.self_attn"
_MTP_MLP_PREFIX = "model.mtp_block.mlp"


@dataclass
class Glm4MoeLiteMTPArgs(BaseModelArgs):
    """MLX model args for the extracted GLM-4.7-Flash nextn head checkpoint.

    Mirrors the field set that ``mlx_lm.models.glm4_moe_lite.ModelArgs`` needs to
    build one decoder layer, plus the MTP-specific ``model_type`` /
    ``num_nextn_predict_layers`` contract. ``num_hidden_layers`` is ``0`` for the
    standalone head (the backbone is stripped); the head builds exactly one
    ``mtp_block`` regardless.
    """

    model_type: str = GLM4_MOE_LITE_MTP_MODEL_TYPE
    vocab_size: int = 154880
    hidden_size: int = 2048
    intermediate_size: int = 10240
    moe_intermediate_size: int = 1536
    num_hidden_layers: int = 0
    num_attention_heads: int = 20
    num_key_value_heads: int = 20
    n_shared_experts: int | None = 1
    n_routed_experts: int = 64
    routed_scaling_factor: float = 1.8
    kv_lora_rank: int = 512
    q_lora_rank: int | None = 768
    qk_rope_head_dim: int = 64
    qk_nope_head_dim: int = 192
    v_head_dim: int = 256
    topk_method: str = "noaux_tc"
    scoring_func: str = "sigmoid"
    norm_topk_prob: bool = True
    n_group: int = 1
    topk_group: int = 1
    num_experts_per_tok: int = 4
    moe_layer_freq: int = 1
    first_k_dense_replace: int = 1
    max_position_embeddings: int = 202752
    rms_norm_eps: float = 1e-5
    rope_theta: float = 1_000_000.0
    rope_scaling: dict[str, Any] | None = None
    attention_bias: bool = False
    attention_dropout: float = 0.0
    partial_rotary_factor: float = 1.0
    tie_word_embeddings: bool = False
    num_nextn_predict_layers: int = GLM4_MOE_LITE_MTP_NUM_NEXTN
    quantization: dict[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.model_type != GLM4_MOE_LITE_MTP_MODEL_TYPE:
            raise ValueError(
                "Glm4MoeLiteMTP head requires "
                f"model_type={GLM4_MOE_LITE_MTP_MODEL_TYPE!r}, got "
                f"{self.model_type!r}"
            )
        if self.num_nextn_predict_layers != GLM4_MOE_LITE_MTP_NUM_NEXTN:
            raise ValueError(
                "Glm4MoeLiteMTP head only supports "
                f"num_nextn_predict_layers={GLM4_MOE_LITE_MTP_NUM_NEXTN}, got "
                f"{self.num_nextn_predict_layers!r}"
            )
        for name in ("hidden_size", "vocab_size", "num_attention_heads"):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(
                    f"Glm4MoeLiteMTP head {name} must be a positive int, got {value!r}"
                )
        if not self.n_routed_experts or self.n_routed_experts <= 0:
            raise ValueError(
                "Glm4MoeLiteMTP head requires n_routed_experts > 0 to build the "
                f"MoE mtp_block, got {self.n_routed_experts!r}"
            )
        # The head builds exactly one decoder layer at
        # layer_idx=first_k_dense_replace and needs it to be the MoE variant
        # (the trained nextn layer is MoE). Stock Glm4MoeLiteDecoderLayer selects
        # MoE only when layer_idx >= first_k_dense_replace AND
        # layer_idx % moe_layer_freq == 0; the first is trivially true here, so
        # validate the second (and guard the modulo against a zero freq).
        if (
            not isinstance(self.moe_layer_freq, int)
            or isinstance(self.moe_layer_freq, bool)
            or self.moe_layer_freq <= 0
        ):
            raise ValueError(
                "Glm4MoeLiteMTP head requires moe_layer_freq to be a positive "
                f"int, got {self.moe_layer_freq!r}"
            )
        if self.first_k_dense_replace % self.moe_layer_freq != 0:
            raise ValueError(
                "Glm4MoeLiteMTP head builds its mtp_block at "
                f"layer_idx=first_k_dense_replace={self.first_k_dense_replace}, "
                "but stock Glm4MoeLiteDecoderLayer only makes that an MoE layer "
                "when layer_idx % moe_layer_freq == 0 "
                f"(moe_layer_freq={self.moe_layer_freq}). The extracted nextn "
                "layer is MoE, so first_k_dense_replace must be divisible by "
                "moe_layer_freq."
            )
        if self.quantization is not None and not isinstance(self.quantization, Mapping):
            raise ValueError(
                "Glm4MoeLiteMTP head quantization must be a mapping, got "
                f"{type(self.quantization).__name__}"
            )

    @classmethod
    def from_dict(cls, params: Mapping[str, Any]) -> Glm4MoeLiteMTPArgs:
        """Build args from the drafter-split config, reading the nested shape.

        The head's shape lives in the nested ``text_config``; the top level
        carries only the head identity (``model_type``), ``quantization``, and
        ``block_size``. Merge the nested config in with the top level winning for
        the fields it owns, so the shape is read from the checkpoint rather than
        silently falling back to this class's GLM-4.7-Flash defaults.
        """
        merged = dict(params)
        text_config = merged.pop("text_config", None)
        if text_config is not None:
            if not isinstance(text_config, Mapping):
                raise ValueError(
                    "Glm4MoeLiteMTP head text_config must be a mapping, got "
                    f"{type(text_config).__name__}"
                )
            merged = {**text_config, **merged}
        return super().from_dict(merged)

    def backbone_args(self) -> Glm4MoeLiteBackboneArgs:
        """Build the stock mlx_lm backbone args used to construct ``mtp_block``.

        The nextn layer is architecturally a ``glm4_moe_lite`` decoder layer, so
        the stock ``ModelArgs`` drives its construction. ``model_type`` is reset
        to the backbone type (``from_dict`` drops MTP-only fields).
        """
        data = {f.name: getattr(self, f.name) for f in fields(self)}
        data["model_type"] = GLM4_MOE_LITE_TARGET_MODEL_TYPE
        return Glm4MoeLiteBackboneArgs.from_dict(data)


def _split_kv_b_proj(weights: dict[str, Any], args: Glm4MoeLiteMTPArgs) -> None:
    """Split the fused ``kv_b_proj`` into absorbed ``embed_q`` / ``unembed_out``.

    Adapts ``mlx_lm.models.glm4_moe_lite.Model.sanitize`` to the single
    ``mtp_block`` prefix. No-op when the weights are already in the post-split
    (absorbed) layout.
    """
    prefix = _MTP_ATTN_PREFIX
    weight_key = f"{prefix}.kv_b_proj.weight"
    if weight_key not in weights:
        return

    quantized = f"{prefix}.kv_b_proj.scales" in weights
    v = weights.pop(weight_key)
    head_dim = args.qk_nope_head_dim + args.v_head_dim

    bits = group_size = 0
    if quantized:
        dims = args.kv_lora_rank
        scales = weights.pop(f"{prefix}.kv_b_proj.scales")
        biases = weights.pop(f"{prefix}.kv_b_proj.biases")
        # Infer bits/group_size from the packed weight and scale shapes, exactly
        # as the stock sanitize does.
        bits = (v.shape[-1] * 32) // dims
        group_size = dims // scales.shape[-1]
        v = mx.dequantize(v, scales, biases, bits=bits, group_size=group_size)

    num_heads = args.num_attention_heads
    v = v.reshape(num_heads, head_dim, -1)
    wk = mx.contiguous(v[:, : args.qk_nope_head_dim, :].swapaxes(-1, -2))
    wv = mx.contiguous(v[:, args.qk_nope_head_dim :, :])
    if quantized:
        wk, wk_scales, wk_biases = mx.quantize(wk, bits=bits, group_size=group_size)
        wv, wv_scales, wv_biases = mx.quantize(wv, bits=bits, group_size=group_size)
        weights[f"{prefix}.embed_q.scales"] = wk_scales
        weights[f"{prefix}.unembed_out.scales"] = wv_scales
        weights[f"{prefix}.embed_q.biases"] = wk_biases
        weights[f"{prefix}.unembed_out.biases"] = wv_biases
    weights[f"{prefix}.embed_q.weight"] = wk
    weights[f"{prefix}.unembed_out.weight"] = wv


def _stack_experts(weights: dict[str, Any], args: Glm4MoeLiteMTPArgs) -> None:
    """Stack per-expert MoE tensors into the ``switch_mlp`` layout.

    Adapts the expert-stacking half of the stock sanitize to the single
    ``mtp_block`` prefix. No-op when already stacked.
    """
    prefix = _MTP_MLP_PREFIX
    n_experts = args.n_routed_experts
    for proj in ("gate_proj", "down_proj", "up_proj"):
        for suffix in ("weight", "scales", "biases"):
            first = f"{prefix}.experts.0.{proj}.{suffix}"
            if first not in weights:
                continue
            expert_keys = [
                f"{prefix}.experts.{e}.{proj}.{suffix}" for e in range(n_experts)
            ]
            # Pre-validate so a checkpoint with fewer experts than the config's
            # ``n_routed_experts`` fails with a descriptive error rather than a
            # bare KeyError leaking from the pop below.
            for e, key in enumerate(expert_keys):
                if key not in weights:
                    raise ValueError(
                        f"Glm4MoeLiteMTP head checkpoint is missing expert {e} "
                        f"tensor {key!r}: config n_routed_experts={n_experts} "
                        f"expects experts 0..{n_experts - 1} for "
                        f"{prefix}.experts.<e>.{proj}.{suffix} to stack into "
                        f"switch_mlp. {HEAD_SOURCE_HINT}"
                    )
            to_join = [weights.pop(key) for key in expert_keys]
            weights[f"{prefix}.switch_mlp.{proj}.{suffix}"] = mx.stack(to_join)


def convert_nextn_weights(
    weights: Mapping[str, Any],
    args: Glm4MoeLiteMTPArgs,
) -> dict[str, Any]:
    """Rewrite flat nextn weights into the module tree the head loads.

    Idempotent: applies the absorbed-MLA ``kv_b_proj`` split and MoE expert
    stacking only when the pre-split / unstacked tensors are present. Shared by
    the mlx-vlm drafter split (which pre-applies it so the shipped checkpoint is
    post-``sanitize``) and by ``Glm4MoeLiteMTPModel.sanitize`` (which tolerates
    either layout).
    """
    out = dict(weights)
    _split_kv_b_proj(out, args)
    _stack_experts(out, args)
    return out


class _Glm4MoeLiteMTPInner(nn.Module):
    """The ``model.*`` subtree matching the flat extracted checkpoint layout."""

    def __init__(
        self,
        args: Glm4MoeLiteMTPArgs,
        backbone: Glm4MoeLiteBackboneArgs,
    ) -> None:
        super().__init__()
        hidden = args.hidden_size
        eps = args.rms_norm_eps
        self.embed_tokens = nn.Embedding(args.vocab_size, hidden)
        self.enorm = nn.RMSNorm(hidden, eps=eps)
        self.hnorm = nn.RMSNorm(hidden, eps=eps)
        self.eh_proj = nn.Linear(2 * hidden, hidden, bias=False)
        # layer_idx == first_k_dense_replace forces the MoE MLP (use_moe=True),
        # matching the trained nextn layer.
        self.mtp_block = Glm4MoeLiteDecoderLayer(
            backbone,
            layer_idx=backbone.first_k_dense_replace,
        )
        self.shared_head_norm = nn.RMSNorm(hidden, eps=eps)


class Glm4MoeLiteMTPModel(nn.Module):
    """MLX module matching the extracted GLM-4.7-Flash nextn head checkpoint."""

    def __init__(self, args: Glm4MoeLiteMTPArgs) -> None:
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.hidden_size = args.hidden_size
        backbone = args.backbone_args()
        self.model = _Glm4MoeLiteMTPInner(args, backbone)
        # Head is NOT tied; the extracted checkpoint ships a dedicated lm_head
        # (from the trained ``shared_head.head``).
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def build_slot_inputs(
        self,
        token_ids: mx.array,
        hidden_rows: mx.array,
        first_position: int,
    ) -> mx.array:
        """Project ``[embed(t_{p+1}), h_p]`` into one input row per slot.

        ``token_ids`` are the tokens whose embeddings feed each slot (target
        tokens shifted left by one). ``hidden_rows`` are the matching target
        post-final-norm hidden states. Embedding rows at absolute position 0 are
        zeroed (upstream ``inputs_embeds[positions == 0] = 0``), which is why
        ``first_position`` (the absolute position of row 0) is required.
        """
        token_ids = mx.array(token_ids)
        if token_ids.ndim != 1:
            raise ValueError(
                "Glm4MoeLiteMTP build_slot_inputs expects 1-D token_ids "
                f"[num_slots], got shape {tuple(token_ids.shape)}"
            )
        if hidden_rows.ndim != 2:
            raise ValueError(
                "Glm4MoeLiteMTP build_slot_inputs expects 2-D hidden_rows "
                f"[num_slots, hidden], got shape {tuple(hidden_rows.shape)}"
            )
        num_slots = token_ids.shape[0]
        if hidden_rows.shape[0] != num_slots:
            raise ValueError(
                "Glm4MoeLiteMTP build_slot_inputs token/hidden row count "
                f"mismatch: token_ids={num_slots}, hidden_rows={hidden_rows.shape[0]}"
            )
        if hidden_rows.shape[1] != self.hidden_size:
            raise ValueError(
                "Glm4MoeLiteMTP build_slot_inputs hidden_rows last dim must be "
                f"{self.hidden_size}, got {hidden_rows.shape[1]}"
            )

        emb = self.model.embed_tokens(token_ids)
        positions = first_position + mx.arange(num_slots)
        keep = (positions != 0)[:, None].astype(emb.dtype)
        emb = emb * keep
        combined = mx.concatenate(
            [self.model.enorm(emb), self.model.hnorm(hidden_rows)],
            axis=-1,
        )
        return self.model.eh_proj(combined)

    def forward_slots(
        self,
        x: mx.array,
        cache: Any,
        *,
        expected_offset: int | None = None,
    ) -> mx.array:
        """Run one stock decoder-layer forward over ``x`` with an appending cache.

        ``x`` is ``[num_slots, hidden]``; the slab index == sequence position ==
        RoPE offset, so stock mlx_lm attention is correct. Returns the
        post-shared-head-norm hidden states ``[num_slots, hidden]``.

        The slot stream carries two independent position anchors: the caller's
        ``first_position`` passed to :meth:`build_slot_inputs` (which zeroes the
        absolute-position-0 embedding), and the ``cache.offset`` that drives
        RoPE / causal masking here. They must agree for the row at ``x[0]``.
        Pass ``expected_offset`` (typically the same ``first_position``) to
        assert the cache is positioned where the caller thinks it is; when it
        differs from the cache's ``offset``, this fails loud instead of silently
        drafting at the wrong position.
        """
        if x.ndim != 2:
            raise ValueError(
                "Glm4MoeLiteMTP forward_slots expects 2-D x [num_slots, hidden], "
                f"got shape {tuple(x.shape)}"
            )
        if expected_offset is not None and cache.offset != expected_offset:
            raise ValueError(
                "Glm4MoeLiteMTP forward_slots cache offset "
                f"{cache.offset} does not match expected_offset "
                f"{expected_offset}: the slot position anchor "
                "(first_position in build_slot_inputs) and the cache's RoPE "
                "offset have diverged."
            )
        h = x[None]
        mask = create_attention_mask(h, cache, return_array=True)
        h = self.model.mtp_block(h, mask, cache)
        h = self.model.shared_head_norm(h)
        return h[0]

    def compute_logits(self, hidden: mx.array) -> mx.array:
        """Project post-norm hidden states through the untied lm_head."""
        return self.lm_head(hidden)

    def sanitize(self, weights: Mapping[str, Any]) -> dict[str, Any]:
        """Accept pre- or post-split attention and (un)stacked expert layouts."""
        return convert_nextn_weights(weights, self.args)

    @property
    def cast_predicate(self):  # noqa: ANN201
        """Keep ``e_score_correction_bias`` at its source dtype during casts.

        Mirrors ``mlx_lm.models.glm4_moe_lite.Model.cast_predicate``: the
        noaux_tc router correction bias is fp32 in the source shard and must
        stay fp32 (vLLM keeps the noaux_tc bias fp32). An mlx_lm convert that
        honors this predicate will not down-cast that tensor with the rest of
        the weights.
        """

        def predicate(weight_key: str) -> bool:
            return "e_score_correction_bias" not in weight_key

        return predicate
