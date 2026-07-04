# SPDX-License-Identifier: Apache-2.0
"""GGUF model-family adapter: the single owner of GGUF name and scope policy.

``GGUFModelAdapter`` owns everything the loader must not inline: the dense-arch
allowlist, the out-of-scope tensor markers, the global-tensor name overrides, the
skip set, architecture normalization and enum resolution, and the per-tensor
translation from a GGUF name to a live MLX-LM module-parameter path.

The name map combines two sources (see ``codex/GGUF_PR3_DESIGN.md`` §6): a small
static override for the global tensors whose MLX target may be absent from a tied
skeleton (``output`` -> ``lm_head``), and a live-derived map built by inverting
``gguf.get_tensor_name_map`` over the freshly built model's own parameter names.
"""

from __future__ import annotations

from typing import Any, cast

import mlx.nn as nn
from mlx.utils import tree_flatten


class GGUFLoadError(RuntimeError):
    """Raised when a local GGUF checkpoint cannot be loaded through the MLX path."""


class GGUFModelAdapter:
    """Translate GGUF tensor names into live MLX-LM module-parameter paths."""

    # Dense decoder architectures this loader is verified against. The scope guard
    # is an allowlist (default-deny): any other arch (linear-attention/SSM hybrid,
    # fused-QKV, MoE, vision-language) is rejected, so a new non-dense arch can
    # never be silently admitted. Grow this set as archs are tested.
    SUPPORTED_DENSE_ARCHS = frozenset({"qwen2", "qwen3"})

    # Substrings flagging an out-of-scope tensor regardless of the declared arch:
    # fused QKV, SSM/Mamba, MoE experts, vision/mmproj. Defense in depth behind the
    # arch allowlist; none of these appear in a dense decoder's tensor names.
    OUT_OF_SCOPE_TENSOR_SUBSTRINGS = (
        "attn_qkv",
        "ssm_",
        "_exps",
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

    def __init__(self, *, translation_map: dict[str, str | None]) -> None:
        self._translation_map = translation_map

    def translate(self, gguf_name: str) -> str | None:
        """Return the live MLX-LM parameter path for a GGUF tensor name."""
        if gguf_name not in self._translation_map:
            raise GGUFLoadError(f"Unmapped GGUF tensor {gguf_name!r}.")
        return self._translation_map[gguf_name]

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
        if arch != cls._normalize_arch(config_model_type):
            raise GGUFLoadError(
                f"GGUF architecture {arch!r} does not match config model_type "
                f"{cls._normalize_arch(config_model_type)!r}; the .gguf and "
                "config_dir describe different models."
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
        return cls(translation_map=translation_map)

    @staticmethod
    def _normalize_arch(name: str) -> str:
        return name.strip().lower().replace("-", "_")

    @classmethod
    def _resolve_arch_enum(cls, gguf: Any, arch: str) -> Any | None:
        for arch_enum, name in gguf.MODEL_ARCH_NAMES.items():
            if cls._normalize_arch(str(name)) == arch:
                return arch_enum
        return None
