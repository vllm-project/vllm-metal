# SPDX-License-Identifier: MIT
# Copyright (c) 2026 erahim3
"""DSpark drafter config — loaded from the HF checkpoint's config.json.

Supports two drafter families with a shared inference path:
  - gemma4  (gemma4_text): k_eq_v attention, v_norm, partial/proportional rope,
            sandwich norms + layer_scalar, gelu-tanh MLP, logit softcap.
  - qwen3   (qwen3):       standard GQA (separate v_proj, no v_norm), default rope,
            Llama-style 2-norm layer, silu MLP, no softcap.
Only the fields the MLX inference path needs are pulled out.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class DSparkConfig:
    family: str = "gemma4"  # "gemma4" | "qwen3"

    # core dims
    hidden_size: int = 3840
    vocab_size: int = 262144
    num_hidden_layers: int = 5
    intermediate_size: int = 15360
    rms_norm_eps: float = 1e-6

    # attention
    num_attention_heads: int = 16
    num_key_value_heads: int = 8
    num_global_key_value_heads: int = 1
    head_dim: int = 256
    global_head_dim: int = 512
    attention_k_eq_v: bool = True
    attention_bias: bool = False

    # rope
    rope_theta: float = 1_000_000.0
    partial_rotary_factor: float = 0.25
    rope_type: str = "proportional"
    # qwen3_5 drafters rope only the first rope_dims of head_dim (partial rotary);
    # None = full head_dim (all other families).
    rope_dims: int | None = None

    # qwen3_5 gated attention: q_proj emits [q ‖ gate] per head (2× out-features),
    # attention output is multiplied by sigmoid(gate) before o_proj.
    gated_q_proj: bool = False

    # qwen3_5 stores every RMSNorm weight as an additive offset from one (Gemma-style
    # (1+w)·x̂ — the reference vLLM patches call this offset_rms_norm). load_drafter adds
    # 1.0 to all RMSNorm weights at load so plain nn.RMSNorm modules compute the right
    # thing. These inherited fields are not a qualified Qwen3.5 serving path;
    # from_dict rejects that checkpoint family.
    offset_rms_norm: bool = False

    # dspark specifics
    block_size: int = 7
    mask_token_id: int = 4
    target_layer_ids: list[int] = field(default_factory=lambda: [5, 17, 29, 41, 46])
    num_target_layers: int = 48

    # markov + confidence
    markov_rank: int = 256
    markov_head_type: str = "vanilla"
    enable_confidence_head: bool = True
    confidence_head_with_markov: bool = True

    # GIDD log-SNR conditioning (PrismML Bonsai drafters; absent from DeepSpec's).
    # At inference the per-position pattern is fixed — anchor (block pos 0) at
    # max_log_snr, every masked position at min_log_snr — so the resulting additive
    # embedding is a constant per block position (see model.LogSnrEmbed).
    log_snr_conditioning: bool = False
    min_log_snr: float = -9.0
    max_log_snr: float = 9.0

    # logits
    final_logit_softcapping: float | None = 30.0
    pad_token_id: int = 0

    # ---- family-derived knobs (set in from_json) ----
    mlp_activation: str = "gelu_tanh"  # "gelu_tanh" | "silu"
    norm_style: str = "gemma"  # "gemma" (sandwich+scalar) | "qwen" (llama 2-norm)
    use_v_norm: bool = True  # gemma: RMSNormNoScale v_norm; qwen: none
    attention_scaling: float | None = None  # None -> 1/sqrt(attn_head_dim)

    @property
    def attn_head_dim(self) -> int:
        """Head dim used by the drafter's own attention."""
        return self.global_head_dim if self.family == "gemma4" else self.head_dim

    @property
    def n_kv_heads(self) -> int:
        if self.family == "gemma4" and self.attention_k_eq_v:
            return self.num_global_key_value_heads
        return self.num_key_value_heads

    @property
    def scaling(self) -> float:
        if self.attention_scaling is not None:
            return self.attention_scaling
        return self.attn_head_dim**-0.5 if self.family == "qwen3" else 1.0

    @property
    def rope_parameters(self) -> dict:
        return {
            "rope_type": self.rope_type,
            "partial_rotary_factor": self.partial_rotary_factor,
        }

    @classmethod
    def from_json(cls, path: str | Path) -> DSparkConfig:
        with open(path) as f:
            c = json.load(f)
        return cls.from_dict(c, source=str(path))

    @classmethod
    def from_dict(cls, c: dict, *, source: str = "DSpark config") -> DSparkConfig:
        path = source
        mt = c.get("model_type", "")

        # Reject the known-incompatible checkpoint packagings with a specific message before
        # family detection — a vLLM-speculators drafter says model_type "qwen3" too, and would
        # otherwise die a few lines down with an unhelpful KeyError.
        if "speculators_config" in c or "speculators_model_type" in c:
            raise ValueError(
                f"{path}: this checkpoint is in the vLLM 'speculators' format "
                f"(speculators_config present), which vllm-metal cannot load yet — it uses a "
                f"different config schema (aux_hidden_state_layer_ids, transformer_layer_config, "
                f"draft_vocab_size). Use a DeepSpec-native standalone drafter "
                f"(e.g. deepseek-ai/dspark_*_block7)."
            )
        if "block_size" not in c and any(k.startswith("dspark_") for k in c):
            raise ValueError(
                f"{path}: this looks like a full target model with an embedded DSpark drafter "
                f"(dspark_* fields in the target config, e.g. DeepSeek-V4-*-DSpark), not a "
                f"standalone drafter checkpoint. Metal currently loads standalone DeepSpec drafters "
                f"(e.g. deepseek-ai/dspark_*_block7)."
            )

        if mt == "qwen3":
            family = "qwen3"
        elif mt == "gemma4_text":
            family = "gemma4"
        else:
            raise ValueError(
                f"{path}: unsupported drafter family (model_type={mt!r}). Supported drafter "
                f"backbones: qwen3 and gemma4_text. Serving support additionally requires "
                f"a qualified target adapter."
            )

        required = (
            "hidden_size",
            "vocab_size",
            "num_hidden_layers",
            "intermediate_size",
            "num_attention_heads",
            "block_size",
            "mask_token_id",
            "target_layer_ids",
        )
        missing = [k for k in required if k not in c]
        if missing:
            raise ValueError(
                f"{path}: config is missing required DeepSpec drafter fields {missing} — this "
                f"does not look like a DeepSpec-format DSpark drafter checkpoint."
            )

        for name in (*required[:6], "num_target_layers", "markov_rank"):
            value = c.get(name)
            if type(value) is not int or value <= 0:
                raise ValueError(f"{path}: {name} must be a positive integer")
        taps = c["target_layer_ids"]
        if (
            not isinstance(taps, list)
            or not taps
            or any(
                type(i) is not int or not 0 <= i < c["num_target_layers"] - 1
                for i in taps
            )
            or any(a >= b for a, b in zip(taps, taps[1:], strict=False))
        ):
            raise ValueError(
                f"{path}: target_layer_ids must be strictly increasing intermediate "
                "decoder layer indices; embedding and final-layer taps are unsupported"
            )
        if (
            type(c["mask_token_id"]) is not int
            or not 0 <= c["mask_token_id"] < c["vocab_size"]
        ):
            raise ValueError(f"{path}: mask_token_id must be in the target vocabulary")
        if c.get("markov_head_type", "vanilla") != "vanilla":
            raise NotImplementedError(
                f"{path}: only the vanilla DSpark Markov head is implemented"
            )
        if c.get("log_snr_conditioning") or c.get("enable_qwen35_gated_q_proj"):
            raise NotImplementedError(
                f"{path}: GIDD and Qwen3.5 draft variants are unsupported"
            )
        if c.get("layer_types") not in (
            None,
            ["full_attention"] * c["num_hidden_layers"],
        ):
            raise NotImplementedError(
                f"{path}: standalone DSpark drafts require full attention"
            )
        if c.get("use_sliding_window") or c.get("dflash_query_causal"):
            raise NotImplementedError(
                f"{path}: sliding or causal draft blocks are unsupported"
            )
        if family == "qwen3":
            rope = c.get("rope_parameters") or {}
            if (
                rope.get("rope_type", "default") != "default"
                or rope.get("partial_rotary_factor", 1.0) != 1.0
            ):
                raise NotImplementedError(
                    f"{path}: Qwen3 DSpark requires full default RoPE"
                )

        kv_heads = c.get("num_key_value_heads", 8)
        head_dim = c.get("head_dim", c["hidden_size"] // c["num_attention_heads"])
        if family == "gemma4":
            head_dim = c.get("global_head_dim", 512)
            if c.get("attention_k_eq_v", True):
                kv_heads = c.get("num_global_key_value_heads", 1)
        if (
            type(kv_heads) is not int
            or kv_heads <= 0
            or c["num_attention_heads"] % kv_heads
        ):
            raise ValueError(
                f"{path}: KV heads must be positive and divide attention heads"
            )
        if type(head_dim) is not int or head_dim <= 0 or head_dim % 2:
            raise ValueError(
                f"{path}: attention head dimension must be positive and even"
            )
        eps = c.get("rms_norm_eps", 1e-6)
        if type(eps) not in (float, int) or not math.isfinite(eps) or eps <= 0:
            raise ValueError(f"{path}: rms_norm_eps must be finite and positive")

        if family == "qwen3":
            rp = c.get("rope_parameters") or {}
            head_dim = c.get("head_dim", c["hidden_size"] // c["num_attention_heads"])
            # qwen3_5-flavored backbones (e.g. Ornith drafters) declare gated q_proj and
            # partial rotary in the same DeepSpec layout; plain qwen3 configs carry neither,
            # so both knobs default to the classic behavior. The config's mrope fields are a
            # text-only no-op (equal position ids collapse mrope to standard rope — mlx-lm's
            # own qwen3_5 text module does the same).
            gated_q = bool(c.get("enable_qwen35_gated_q_proj", False))
            rope_dims = int(head_dim * float(rp.get("partial_rotary_factor", 1.0)))
            # qwen3_5 house style: norm weights stored offset-from-one (see field docs).
            offset_norms = mt == "qwen3_5"
            if c.get("log_snr_conditioning"):
                lo, hi = c.get("min_log_snr"), c.get("max_log_snr")
                if lo is None or hi is None or not (float(hi) > float(lo)):
                    raise ValueError(
                        f"{path}: log_snr_conditioning is enabled but min/max_log_snr are "
                        f"missing or not ordered (min={lo!r}, max={hi!r}) — the featurization "
                        f"divides by (max - min), so a drafter converted without them would "
                        f"draft from silently-wrong embeddings."
                    )
            return cls(
                family="qwen3",
                hidden_size=c["hidden_size"],
                vocab_size=c["vocab_size"],
                num_hidden_layers=c["num_hidden_layers"],
                intermediate_size=c["intermediate_size"],
                rms_norm_eps=c.get("rms_norm_eps", 1e-6),
                num_attention_heads=c["num_attention_heads"],
                num_key_value_heads=c.get("num_key_value_heads", 8),
                head_dim=head_dim,
                attention_k_eq_v=False,
                attention_bias=c.get("attention_bias", False),
                rope_theta=rp.get("rope_theta", c.get("rope_theta", 1_000_000.0)),
                rope_type="default",
                rope_dims=(rope_dims if rope_dims != head_dim else None),
                gated_q_proj=gated_q,
                offset_rms_norm=offset_norms,
                block_size=c["block_size"],
                mask_token_id=c["mask_token_id"],
                target_layer_ids=list(c["target_layer_ids"]),
                num_target_layers=c.get("num_target_layers", 36),
                markov_rank=c.get("markov_rank", 256),
                markov_head_type=c.get("markov_head_type", "vanilla"),
                enable_confidence_head=c.get("enable_confidence_head", True),
                confidence_head_with_markov=c.get("confidence_head_with_markov", True),
                final_logit_softcapping=c.get("final_logit_softcapping", None),
                pad_token_id=c.get("pad_token_id") or 0,
                mlp_activation="silu",
                norm_style="qwen",
                use_v_norm=False,
                log_snr_conditioning=bool(c.get("log_snr_conditioning", False)),
                min_log_snr=float(c.get("min_log_snr", -9.0)),
                max_log_snr=float(c.get("max_log_snr", 9.0)),
            )

        rope = (c.get("rope_parameters") or {}).get("full_attention", {}) or {}
        return cls(
            family="gemma4",
            hidden_size=c["hidden_size"],
            vocab_size=c["vocab_size"],
            num_hidden_layers=c["num_hidden_layers"],
            intermediate_size=c["intermediate_size"],
            rms_norm_eps=c.get("rms_norm_eps", 1e-6),
            num_attention_heads=c["num_attention_heads"],
            num_key_value_heads=c.get("num_key_value_heads", 8),
            num_global_key_value_heads=c.get("num_global_key_value_heads", 1),
            head_dim=c.get("head_dim", 256),
            global_head_dim=c.get("global_head_dim", 512),
            attention_k_eq_v=c.get("attention_k_eq_v", True),
            attention_bias=c.get("attention_bias", False),
            rope_theta=rope.get("rope_theta", 1_000_000.0),
            partial_rotary_factor=rope.get("partial_rotary_factor", 0.25),
            rope_type=rope.get("rope_type", "proportional"),
            block_size=c["block_size"],
            mask_token_id=c["mask_token_id"],
            target_layer_ids=list(c["target_layer_ids"]),
            num_target_layers=c.get("num_target_layers", 48),
            markov_rank=c.get("markov_rank", 256),
            markov_head_type=c.get("markov_head_type", "vanilla"),
            enable_confidence_head=c.get("enable_confidence_head", True),
            confidence_head_with_markov=c.get("confidence_head_with_markov", True),
            final_logit_softcapping=c.get("final_logit_softcapping", 30.0),
            pad_token_id=c.get("pad_token_id", 0),
            mlp_activation="gelu_tanh",
            norm_style="gemma",
            use_v_norm=True,
        )
