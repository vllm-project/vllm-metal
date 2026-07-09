# SPDX-License-Identifier: Apache-2.0
"""GLM-4.7-Flash native MTP head: MLX model, weight transform, loader, metadata.

The ``glm4_moe_lite`` (GLM-4.7-Flash) family ships a single trained nextn layer
(``model.layers.<num_hidden_layers>.*``) that predicts one extra token. This
module runs that layer as a tiny standalone MLX model over the "slot stream":

    slot_p = eh_proj([enorm(embed(t_{p+1})), hnorm(h_p)])

where ``h_p`` is the target's post-final-norm hidden state. One stock mlx_lm
``Glm4MoeLiteDecoderLayer`` forward over the appended slots, then the shared
head norm + untied lm_head, yields one greedy draft token.

The extraction tool ``tools/extract_glm47_mtp_head.py`` produces the flat,
post-``sanitize`` checkpoint this model loads; both share the
``convert_nextn_weights`` transform so there is one source of truth.

This module holds the head's model, weight transform, loader, and metadata. The
:class:`~vllm_metal.v1.mtp_heads.NativeMTPHead` that registers it and constructs
its proposer imports these lazily, so ``import vllm_metal.v1.mtp_heads`` stays
free of mlx / mlx_lm.

The loader / metadata / source dataclasses intentionally mirror the shapes of
the Gemma4 MTP path in ``vllm_metal.v1.gemma4_mtp``; the duplication is
deliberate until a second native head exists to factor a shared base against.
"""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, fields
from json import JSONDecodeError, loads
from numbers import Integral, Real
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.models.base import BaseModelArgs, create_attention_mask
from mlx_lm.models.glm4_moe_lite import Glm4MoeLiteDecoderLayer
from mlx_lm.models.glm4_moe_lite import ModelArgs as Glm4MoeLiteBackboneArgs
from vllm.logger import init_logger

from vllm_metal.v1.mlx_lm_paths import mlx_lm_compatible_model_path

logger = init_logger(__name__)

GLM4_MOE_LITE_MTP_MODEL_TYPE = "glm4_moe_lite_mtp"
GLM4_MOE_LITE_MTP_ARCHITECTURE = "Glm4MoeLiteMTPModel"
GLM4_MOE_LITE_MTP_N_PREDICT = 1
GLM4_MOE_LITE_MTP_NUM_NEXTN = 1
GLM4_MOE_LITE_TARGET_MODEL_TYPE = "glm4_moe_lite"

# The extraction tool produces the flat checkpoint this loader consumes; error
# messages point users at it rather than at the raw multi-shard target repo.
EXTRACTION_TOOL = "tools/extract_glm47_mtp_head.py"

_HEAD_DOWNLOAD_ALLOW_PATTERNS = [
    "config.json",
    "*.safetensors",
    "*.safetensors.index.json",
]

# Flat-layout prefixes shared by the model tree, the extraction tool, and the
# weight transform below.
_MTP_ATTN_PREFIX = "model.mtp_block.self_attn"
_MTP_MLP_PREFIX = "model.mtp_block.mlp"

# Compat / target hyperparameters that must match between the extracted head and
# the target model for the shared MLA + embedding math to be valid.
_COMPAT_FIELDS = (
    "hidden_size",
    "vocab_size",
    "kv_lora_rank",
    "qk_rope_head_dim",
    "qk_nope_head_dim",
    "v_head_dim",
    "num_attention_heads",
    "rope_theta",
    "rms_norm_eps",
)


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
    n_routed_experts: int | None = 64
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
                        f"switch_mlp. Re-extract the head with {EXTRACTION_TOOL}."
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
    the extraction tool (which pre-applies it so the shipped checkpoint is
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
        assert the cache is positioned where the caller thinks it is; when the
        cache exposes ``offset`` and it differs, this fails loud instead of
        silently drafting at the wrong position. Omitted by default so existing
        callers stay unaffected.
        """
        if x.ndim != 2:
            raise ValueError(
                "Glm4MoeLiteMTP forward_slots expects 2-D x [num_slots, hidden], "
                f"got shape {tuple(x.shape)}"
            )
        if expected_offset is not None:
            cache_offset = getattr(cache, "offset", None)
            if cache_offset is not None and cache_offset != expected_offset:
                raise ValueError(
                    "Glm4MoeLiteMTP forward_slots cache offset "
                    f"{cache_offset} does not match expected_offset "
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


@dataclass(frozen=True, slots=True)
class Glm4MoeLiteMTPHeadMetadata:
    """Validated shape of an extracted GLM-4.7-Flash nextn head checkpoint."""

    model_type: str
    architectures: tuple[str, ...]
    hidden_size: int
    vocab_size: int
    kv_lora_rank: int
    qk_rope_head_dim: int
    qk_nope_head_dim: int
    v_head_dim: int
    num_attention_heads: int
    rope_theta: float
    rms_norm_eps: float
    num_hidden_layers: int
    num_nextn_predict_layers: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Glm4MoeLiteMTPHeadMetadata:
        model_type = _config_value(config, "model_type")
        if model_type != GLM4_MOE_LITE_MTP_MODEL_TYPE:
            raise ValueError(
                "Glm4MoeLiteMTP head requires "
                f"model_type={GLM4_MOE_LITE_MTP_MODEL_TYPE!r}, got "
                f"{model_type!r}. Produce the head with {EXTRACTION_TOOL}."
            )

        num_hidden_layers = _required_int(config, "num_hidden_layers")
        if num_hidden_layers != 0:
            raise ValueError(
                "Glm4MoeLiteMTP head config must have num_hidden_layers=0 "
                f"(standalone nextn head), got {num_hidden_layers}. This looks "
                "like the full GLM-4.7-Flash target repo; extract the head with "
                f"{EXTRACTION_TOOL}."
            )

        num_nextn = _optional_int(config, "num_nextn_predict_layers")
        if num_nextn is None:
            num_nextn = GLM4_MOE_LITE_MTP_NUM_NEXTN
        if num_nextn != GLM4_MOE_LITE_MTP_NUM_NEXTN:
            raise ValueError(
                "Glm4MoeLiteMTP head only supports "
                f"num_nextn_predict_layers={GLM4_MOE_LITE_MTP_NUM_NEXTN}, got "
                f"{num_nextn}"
            )

        n_predict = _optional_int(config, "n_predict")
        if n_predict is not None and n_predict != GLM4_MOE_LITE_MTP_N_PREDICT:
            raise ValueError(
                "Glm4MoeLiteMTP head config must use "
                f"n_predict={GLM4_MOE_LITE_MTP_N_PREDICT}, got {n_predict}"
            )

        return cls(
            model_type=str(model_type),
            architectures=_architectures(config),
            hidden_size=_required_positive_int(config, "hidden_size"),
            vocab_size=_required_positive_int(config, "vocab_size"),
            kv_lora_rank=_required_positive_int(config, "kv_lora_rank"),
            qk_rope_head_dim=_required_positive_int(config, "qk_rope_head_dim"),
            qk_nope_head_dim=_required_positive_int(config, "qk_nope_head_dim"),
            v_head_dim=_required_positive_int(config, "v_head_dim"),
            num_attention_heads=_required_positive_int(config, "num_attention_heads"),
            rope_theta=_required_positive_float(config, "rope_theta"),
            rms_norm_eps=_required_positive_float(config, "rms_norm_eps"),
            num_hidden_layers=num_hidden_layers,
            num_nextn_predict_layers=num_nextn,
        )

    def validate_compatible_with(self, target_config: Mapping[str, Any]) -> None:
        """Fail loud when the head and target disagree on shared hyperparameters."""
        target = _text_config(target_config)
        self._validate_target_model_type(target_config, target)
        for name in _COMPAT_FIELDS:
            head_value = getattr(self, name)
            target_value = _config_value(target, name)
            if target_value is None:
                raise ValueError(
                    "Glm4MoeLiteMTP target model is missing "
                    f"{name!r} required to validate head compatibility"
                )
            if not _values_match(head_value, target_value):
                raise ValueError(
                    f"Glm4MoeLiteMTP head {name} must match target {name}: "
                    f"head={head_value!r}, target={target_value!r}"
                )

    @staticmethod
    def _validate_target_model_type(
        target_config: Any,
        target_text_config: Any,
    ) -> None:
        """Reject a target whose ``model_type`` is not the GLM MoE-lite backbone.

        The head's absorbed-MLA + embedding math is only valid over a
        ``glm4_moe_lite`` target; accept the ``model_type`` at the top level of
        the target config or inside its resolved text config.
        """
        model_types = {
            model_type
            for model_type in (
                _config_value(target_config, "model_type"),
                _config_value(target_text_config, "model_type"),
            )
            if model_type is not None
        }
        if not model_types:
            raise ValueError(
                "Glm4MoeLiteMTP head requires a "
                f"{GLM4_MOE_LITE_TARGET_MODEL_TYPE!r} target model, got "
                "model_type=None"
            )
        unknown = sorted(
            str(model_type)
            for model_type in model_types
            if model_type != GLM4_MOE_LITE_TARGET_MODEL_TYPE
        )
        if unknown:
            raise ValueError(
                "Glm4MoeLiteMTP head requires a "
                f"{GLM4_MOE_LITE_TARGET_MODEL_TYPE!r} target model, got "
                f"model_type={unknown[0]!r}"
            )


@dataclass(frozen=True, slots=True)
class Glm4MoeLiteMTPHeadSource:
    """Resolved head checkpoint source from a vLLM speculative config."""

    model_name: str
    revision: str | None

    @classmethod
    def from_speculative_config(
        cls,
        speculative_config: Any,
    ) -> Glm4MoeLiteMTPHeadSource:
        draft_model_config = speculative_config.draft_model_config
        return cls(
            model_name=draft_model_config.model,
            revision=draft_model_config.revision,
        )

    def resolve(
        self,
        model_path_resolver: Callable[[str], str],
    ) -> Glm4MoeLiteMTPHeadSource:
        return Glm4MoeLiteMTPHeadSource(
            model_name=model_path_resolver(self.model_name),
            revision=self.revision,
        )

    @property
    def cache_key(self) -> tuple[str, str | None]:
        return (self.model_name, self.revision)


@dataclass(frozen=True, slots=True)
class Glm4MoeLiteMTPHeadRuntime:
    """A loaded head model plus its validated metadata."""

    model_name: str
    model: Any
    metadata: Glm4MoeLiteMTPHeadMetadata


class Glm4MoeLiteMTPHeadLoader:
    """Loads and validates an extracted GLM-4.7-Flash nextn head."""

    _CACHE: ClassVar[dict[tuple[str, str | None], Glm4MoeLiteMTPHeadRuntime]] = {}
    _CACHE_LOCK: ClassVar[Lock] = Lock()

    def __init__(
        self,
        *,
        load_model_fn: Callable[..., tuple[Any, dict[str, Any]]] | None = None,
        download_fn: Callable[[str, str | None], Path] | None = None,
        model_path_resolver: Callable[[str], str] | None = None,
    ) -> None:
        self._load_model = load_model_fn
        self._download = download_fn
        self._model_path_resolver = model_path_resolver

    def load_if_needed(
        self,
        *,
        speculative_config: Any,
        target_config: Mapping[str, Any],
    ) -> Glm4MoeLiteMTPHeadRuntime:
        """Load the head for this speculative config, reusing the cross-instance cache."""
        source = Glm4MoeLiteMTPHeadSource.from_speculative_config(speculative_config)
        if self._model_path_resolver is not None:
            source = source.resolve(self._model_path_resolver)

        cached = self._cached_runtime(source)
        if cached is not None:
            cached.metadata.validate_compatible_with(target_config)
            logger.info("GLM4 MTP head loaded from cache: %s", source.model_name)
            return cached

        return self._load_uncached(source, target_config)

    def _load_uncached(
        self,
        source: Glm4MoeLiteMTPHeadSource,
        target_config: Mapping[str, Any],
    ) -> Glm4MoeLiteMTPHeadRuntime:
        logger.info("Loading GLM4 MTP head: %s", source.model_name)
        start_time = time.time()
        model_path = self._download_model(source)
        self._preflight_config(model_path, target_config)
        model, head_config = self._load_head_model(model_path)
        metadata = self._metadata_from_config(head_config, target_config)
        self._assert_head_tensors(model)
        runtime = Glm4MoeLiteMTPHeadRuntime(
            model_name=source.model_name,
            model=model,
            metadata=metadata,
        )
        with self._CACHE_LOCK:
            self._CACHE[source.cache_key] = runtime
        logger.info(
            "GLM4 MTP head loaded in %.2fs: %s",
            time.time() - start_time,
            source.model_name,
        )
        return runtime

    def _cached_runtime(
        self,
        source: Glm4MoeLiteMTPHeadSource,
    ) -> Glm4MoeLiteMTPHeadRuntime | None:
        with self._CACHE_LOCK:
            return self._CACHE.get(source.cache_key)

    @classmethod
    def clear_cache(cls) -> None:
        """Clear the process-level head cache."""
        with cls._CACHE_LOCK:
            cls._CACHE.clear()

    def _preflight_config(
        self,
        model_path: Path,
        target_config: Mapping[str, Any],
    ) -> None:
        head_config = self._read_config_file(model_path)
        if head_config is None and self._load_model is None:
            raise ValueError(
                "GLM4 MTP head model path must contain config.json: "
                f"{model_path}. Produce the head with {EXTRACTION_TOOL}."
            )
        if head_config is not None:
            self._metadata_from_config(head_config, target_config)

    def _metadata_from_config(
        self,
        head_config: Mapping[str, Any],
        target_config: Mapping[str, Any],
    ) -> Glm4MoeLiteMTPHeadMetadata:
        self._reject_custom_model_file(head_config)
        metadata = Glm4MoeLiteMTPHeadMetadata.from_config(head_config)
        metadata.validate_compatible_with(target_config)
        return metadata

    def _load_head_model(self, model_path: Path) -> tuple[Any, dict[str, Any]]:
        if self._load_model is None:
            from mlx_lm.utils import load_model as load_model_fn
        else:
            load_model_fn = self._load_model

        with mlx_lm_compatible_model_path(model_path) as compatible_model_path:
            return load_model_fn(
                compatible_model_path,
                lazy=False,
                strict=True,
                get_model_classes=self._get_model_classes,
            )

    def _download_model(self, source: Glm4MoeLiteMTPHeadSource) -> Path:
        if self._download is not None:
            return Path(self._download(source.model_name, source.revision))

        model_path = Path(source.model_name)
        if model_path.exists():
            return model_path

        from huggingface_hub import snapshot_download

        return Path(
            snapshot_download(
                source.model_name,
                revision=source.revision,
                allow_patterns=_HEAD_DOWNLOAD_ALLOW_PATTERNS,
            )
        )

    @staticmethod
    def _assert_head_tensors(model: Any) -> None:
        params = dict(tree_flatten(model.parameters()))
        for key, source in (
            ("lm_head.weight", "shared_head.head"),
            ("model.embed_tokens.weight", "the dedicated nextn embedding"),
        ):
            if key not in params:
                raise ValueError(
                    f"GLM4 MTP head checkpoint is missing {key!r} (from {source}). "
                    f"Re-extract the head with {EXTRACTION_TOOL}."
                )

    @staticmethod
    def _read_config_file(model_path: Path) -> dict[str, Any] | None:
        config_path = model_path / "config.json"
        if not config_path.exists():
            return None
        try:
            config = loads(config_path.read_text(encoding="utf-8"))
        except JSONDecodeError as exc:
            raise ValueError(
                f"GLM4 MTP head config.json is not valid JSON: {config_path}"
            ) from exc
        if not isinstance(config, dict):
            raise ValueError("GLM4 MTP head config.json must contain an object")
        return config

    @staticmethod
    def _reject_custom_model_file(config: Mapping[str, Any]) -> None:
        if "model_file" in config:
            model_file = config["model_file"]
            raise ValueError(
                "GLM4 MTP head loader uses built-in Metal model classes and does "
                f"not support custom model_file={model_file!r}"
            )

    @staticmethod
    def _get_model_classes(
        config: dict[str, Any],
    ) -> tuple[type[Any], type[Any]]:
        if config.get("model_type") != GLM4_MOE_LITE_MTP_MODEL_TYPE:
            model_type = config.get("model_type")
            architectures = config.get("architectures")
            raise ValueError(
                "GLM4 MTP head loader only supports "
                f"{GLM4_MOE_LITE_MTP_MODEL_TYPE!r} configs, got "
                f"model_type={model_type!r}, architectures={architectures!r}. "
                f"Produce the head with {EXTRACTION_TOOL}."
            )
        return Glm4MoeLiteMTPModel, Glm4MoeLiteMTPArgs


def _text_config(config: Any | None) -> Any | None:
    if config is None:
        return None
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        return get_text_config()
    return _config_value(config, "text_config", config)


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if config is None:
        return default
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _values_match(head_value: Any, target_value: Any) -> bool:
    if isinstance(head_value, Real) and isinstance(target_value, Real):
        return bool(
            abs(float(head_value) - float(target_value))
            <= 1e-9 * max(1.0, abs(float(head_value)))
        )
    return head_value == target_value


def _required_int(config: Mapping[str, Any], key: str) -> int:
    value = _config_value(config, key)
    if value is None:
        raise ValueError(f"GLM4 MTP head config is missing {key}")
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"GLM4 MTP head {key} must be an integer, got {value!r}")
    return int(value)


def _optional_int(config: Mapping[str, Any], key: str) -> int | None:
    value = _config_value(config, key)
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"GLM4 MTP head {key} must be an integer, got {value!r}")
    return int(value)


def _required_positive_int(config: Mapping[str, Any], key: str) -> int:
    value = _required_int(config, key)
    if value <= 0:
        raise ValueError(f"GLM4 MTP head {key} must be positive, got {value}")
    return value


def _required_positive_float(config: Mapping[str, Any], key: str) -> float:
    value = _config_value(config, key)
    if value is None:
        raise ValueError(f"GLM4 MTP head config is missing {key}")
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"GLM4 MTP head {key} must be a number, got {value!r}")
    if float(value) <= 0.0:
        raise ValueError(f"GLM4 MTP head {key} must be positive, got {value}")
    return float(value)


def _architectures(config: Mapping[str, Any]) -> tuple[str, ...]:
    value = _config_value(config, "architectures", ()) or ()
    if isinstance(value, str):
        raise ValueError("GLM4 MTP head architectures must be a non-string sequence")
    try:
        names = tuple(value)
    except TypeError as exc:
        raise ValueError("GLM4 MTP head architectures must be a sequence") from exc
    if any(not isinstance(name, str) for name in names):
        raise ValueError("GLM4 MTP head architectures entries must be strings")
    return names
