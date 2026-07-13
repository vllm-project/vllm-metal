# SPDX-License-Identifier: Apache-2.0
"""GGUF model-family adapter: the single owner of GGUF name and scope policy.

``GGUFModelAdapter`` owns everything the loader must not inline: the dense-arch
allowlist, the out-of-scope tensor markers, the global-tensor name overrides, the
skip set, architecture normalization and enum resolution, the per-tensor
translation from a GGUF name to a live MLX-LM module-parameter path, and the
per-arch RoPE weight-layout policy (which q/k tensors need llama.cpp's layout
undone, and the row-permutation index to undo it).

The name map combines two sources (see ``codex/GGUF_PR3_DESIGN.md`` §6): a small
static override for the global tensors whose MLX target may be absent from a tied
skeleton (``output`` -> ``lm_head``), and a live-derived map built by inverting
``gguf.get_tensor_name_map`` over the freshly built model's own parameter names.
"""

from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class GGUFLoadError(RuntimeError):
    """Raised when a local GGUF checkpoint cannot be loaded through the MLX path."""


class GGUFModelAdapter:
    """Own GGUF name/scope policy: translate tensor names to live MLX-LM
    parameter paths, and the per-arch RoPE weight-layout policy (which q/k
    tensors need llama.cpp's layout undone, and the row-permutation index)."""

    # Dense decoder architectures this loader is verified against. The scope guard
    # is an allowlist (default-deny): any other arch (linear-attention/SSM hybrid,
    # fused-QKV, MoE, vision-language) is rejected, so a new non-dense arch can
    # never be silently admitted. Grow this set as archs are tested.
    SUPPORTED_DENSE_ARCHS = frozenset({"qwen2", "qwen3", "llama"})

    # Substrings flagging an out-of-scope tensor regardless of the declared arch:
    # fused QKV, SSM/Mamba, MoE experts + router, vision/mmproj. Defense in depth
    # behind the arch allowlist; none appear in a dense decoder's tensor names.
    # ``ffn_gate_inp`` is the MoE router: llama-family MoE (e.g. Mixtral) declares
    # general.architecture "llama", so its expert tensors (caught by ``_exps``)
    # and router (this) are reachable once "llama" is admitted.
    OUT_OF_SCOPE_TENSOR_SUBSTRINGS = (
        "attn_qkv",
        "ssm_",
        "_exps",
        "ffn_gate_inp",
        "mmproj",
        "mm.",
        "v.blk",
        "v.patch",
    )

    # GGUF globals whose MLX target has no live parameter to invert from on a tied
    # model (``output`` -> ``lm_head``), so they translate independently of the
    # live parameter tree. The only load-bearing override for the dense set.
    _STATIC_GLOBAL_OVERRIDE = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }

    # GGUF tensors with no MLX-LM counterpart (precomputed buffers MLX recreates).
    _KNOWN_SKIP = frozenset({"rope_freqs.weight"})

    # Archs whose q/k projection weights are stored in llama.cpp's RoPE layout and
    # must be row-un-permuted at load (llama family). qwen2/qwen3 use the HF layout
    # and are absent here. mistral (a llama-arch family) maps to "llama".
    _ROPE_PERMUTE_ARCHS = frozenset({"llama"})

    # config model_type -> the GGUF arch llama.cpp converts that family under
    # (MistralForCausalLM converts under MODEL_ARCH.LLAMA, and mlx-lm builds
    # "mistral" configs through its llama module). Applied to the CONFIG side
    # only: a .gguf declaring a raw "mistral" arch is not a llama.cpp product
    # and stays allowlist-rejected, as do the multimodal "mistral3"/"mistral4".
    _CONFIG_MODEL_TYPE_TO_GGUF_ARCH = {"mistral": "llama"}

    def __init__(
        self,
        *,
        translation_map: dict[str, str | None],
        permute_index: dict[str, mx.array] | None = None,
    ) -> None:
        self._translation_map = translation_map
        self._permute_index = permute_index or {}

    def translate(self, gguf_name: str) -> str | None:
        """Return the live MLX-LM parameter path for a GGUF tensor name."""
        if gguf_name not in self._translation_map:
            raise GGUFLoadError(f"Unmapped GGUF tensor {gguf_name!r}.")
        return self._translation_map[gguf_name]

    def rope_permute_index(self, gguf_name: str) -> mx.array | None:
        """Row-permutation index that undoes llama.cpp's RoPE q/k layout.

        Returns the precomputed index for a q/k weight that needs it, or ``None``
        for every other tensor / arch (a plain table lookup, like
        :meth:`translate`). The loader applies it to the weight, quantized or
        plain, before install.
        """
        return self._permute_index.get(gguf_name)

    @classmethod
    def resolve_arch(cls, *, gguf_arch: str, config_model_type: str) -> str:
        """Normalize, allowlist-gate, and cross-check the GGUF vs config arch.

        The .gguf is the source of truth for its own architecture; reject anything
        outside the dense allowlist, then require the companion config to describe
        the same model. Returns the canonical arch for ``from_model``.
        """
        arch = cls._normalize_arch(gguf_arch)
        if arch not in cls.SUPPORTED_DENSE_ARCHS:
            raise GGUFLoadError(
                f"Architecture {arch!r} is not a supported dense decoder; the GGUF "
                f"loader supports {sorted(cls.SUPPORTED_DENSE_ARCHS)}."
            )
        config_arch = cls._config_model_type_to_gguf_arch(config_model_type)
        if arch != config_arch:
            raise GGUFLoadError(
                f"GGUF architecture {arch!r} does not match config model_type "
                f"{config_model_type!r} (maps to GGUF arch {config_arch!r}); "
                "the .gguf and config_dir describe different models."
            )
        return arch

    @classmethod
    def from_model(
        cls,
        model: nn.Module,
        *,
        gguf: Any,
        arch: str,
        num_hidden_layers: int,
    ) -> GGUFModelAdapter:
        """Build the reverse map by inverting ``get_tensor_name_map`` over ``model``.

        Enumerates the live model's parameter names so every translated path is a
        real attribute path on the freshly built skeleton (including any family
        prefix), then maps each to its GGUF name via the ``gguf`` package.
        """
        arch_enum = cls._resolve_arch_enum(gguf, arch)
        if arch_enum is None:  # pragma: no cover - allowlisted archs always resolve
            raise GGUFLoadError(f"Unknown GGUF architecture {arch!r}.")
        name_map = gguf.get_tensor_name_map(arch_enum, num_hidden_layers)
        reverse: dict[str, str] = {}
        leaves = cast(
            "list[tuple[str, nn.Module]]",
            tree_flatten(model.leaf_modules(), is_leaf=nn.Module.is_module),
        )
        for module_path, module in leaves:
            params = cast("list[tuple[str, Any]]", tree_flatten(module.parameters()))
            for param_name, _ in params:
                mlx_name = f"{module_path}.{param_name}" if module_path else param_name
                gguf_name = name_map.get_name(
                    mlx_name, try_suffixes=(".weight", ".bias")
                )
                if gguf_name is not None:
                    reverse.setdefault(gguf_name, mlx_name)
        if not reverse:
            raise GGUFLoadError(
                f"Empty GGUF->MLX name map for arch {arch!r} "
                f"({num_hidden_layers} layers); cannot map any weight."
            )
        translation_map: dict[str, str | None] = dict(reverse)
        translation_map.update(cls._STATIC_GLOBAL_OVERRIDE)
        for name in cls._KNOWN_SKIP:
            translation_map[name] = None
        permute_index = cls._build_permute_index(model, arch, num_hidden_layers)
        return cls(translation_map=translation_map, permute_index=permute_index)

    @classmethod
    def _build_permute_index(
        cls, model: nn.Module, arch: str, num_hidden_layers: int
    ) -> dict[str, mx.array]:
        """Precompute the RoPE row-permutation index per q/k GGUF tensor name.

        Empty for archs in the HF RoPE layout (qwen). For llama-family archs the
        index reorders each head's interleaved pairs back to the HF half-split.
        ``out_features`` is read off the BUILT q/k projections (whose head_dim is
        already resolved), not ``args.head_dim`` which stays ``None`` when the
        config omits it (the #449 omitted-default trap). llama.cpp permutes both
        the q/k weight AND its bias, so the same index covers ``.bias``.
        """
        if arch not in cls._ROPE_PERMUTE_ARCHS:
            return {}
        args = model.args
        attn = model.model.layers[0].self_attn
        idx_q = cls._rope_permute_index(
            attn.q_proj.weight.shape[0], args.num_attention_heads
        )
        idx_k = cls._rope_permute_index(
            attn.k_proj.weight.shape[0], args.num_key_value_heads
        )
        permute_index: dict[str, mx.array] = {}
        for i in range(num_hidden_layers):
            for suffix, idx in ((".weight", idx_q), (".bias", idx_q)):
                permute_index[f"blk.{i}.attn_q{suffix}"] = idx
            for suffix, idx in ((".weight", idx_k), (".bias", idx_k)):
                permute_index[f"blk.{i}.attn_k{suffix}"] = idx
        return permute_index

    @staticmethod
    def _rope_permute_index(out_features: int, n_head: int) -> mx.array:
        """Inverse of llama.cpp's HF->GGUF q/k RoPE permutation, as an out-axis
        row index. Reorders each head's interleaved (2, head_dim//2) pairs back
        to the (head_dim//2, 2) half-split HF layout."""
        return (
            mx.arange(out_features)
            .reshape(n_head, out_features // n_head // 2, 2)
            .swapaxes(1, 2)
            .reshape(out_features)
        )

    @staticmethod
    def _normalize_arch(name: str) -> str:
        return name.strip().lower().replace("-", "_")

    @classmethod
    def _config_model_type_to_gguf_arch(cls, model_type: str) -> str:
        """Map a config ``model_type`` to the GGUF arch its family converts under."""
        arch = cls._normalize_arch(model_type)
        return cls._CONFIG_MODEL_TYPE_TO_GGUF_ARCH.get(arch, arch)

    @classmethod
    def _resolve_arch_enum(cls, gguf: Any, arch: str) -> Any | None:
        for arch_enum, name in gguf.MODEL_ARCH_NAMES.items():
            if cls._normalize_arch(str(name)) == arch:
                return arch_enum
        return None
