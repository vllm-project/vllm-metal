# SPDX-License-Identifier: Apache-2.0
"""GGUF model-family adapter: the single owner of GGUF name and scope policy.

``GGUFModelAdapter`` owns everything the loader must not inline: the arch
allowlist, the out-of-scope tensor markers, the global-tensor name overrides, the
skip set, architecture normalization and enum resolution, the per-tensor
translation from a GGUF name to a live MLX-LM module-parameter path, and the
per-arch weight-layout policy (which tensors llama.cpp stores in another row,
column or value layout, and how to undo it).

The name map combines two sources (see ``codex/GGUF_PR3_DESIGN.md`` §6): a small
static override for the global tensors whose MLX target may be absent from a tied
skeleton (``output`` -> ``lm_head``), and a live-derived map built by inverting
``gguf.get_tensor_name_map`` over the freshly built model's own parameter names.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any, Protocol, cast

import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten


class GGUFLoadError(RuntimeError):
    """Raised when a GGUF checkpoint cannot be loaded through the MLX path."""


class GGUFPlainTransform(Protocol):
    """Undo how llama.cpp stores a plain tensor relative to its live parameter."""

    def source_shape(self, model_shape: tuple[int, ...]) -> tuple[int, ...]:
        """The shape the file stores a parameter of ``model_shape`` in."""
        ...

    def apply(self, source: mx.array) -> mx.array:
        """Map a shape-checked source tensor into the live parameter's layout."""
        ...


class _SqueezedUnitAxis:
    """The file squeezes a conv kernel ``(C, K, 1)`` to ``(C, K)``."""

    def source_shape(self, model_shape: tuple[int, ...]) -> tuple[int, ...]:
        return model_shape[:-1]

    def apply(self, source: mx.array) -> mx.array:
        return mx.expand_dims(source, -1)


class _NegatedExp:
    """The file stores ``-exp(x)`` (llama.cpp's ``ssm_a = -exp(A_log)``)."""

    def source_shape(self, model_shape: tuple[int, ...]) -> tuple[int, ...]:
        return model_shape

    def apply(self, source: mx.array) -> mx.array:
        return mx.log(-source)


@dataclass(frozen=True)
class GGUFArchSpec:
    """How llama.cpp stores one supported GGUF arch relative to mlx-lm."""

    # Config model_types (besides the arch name) that llama.cpp converts here.
    config_model_types: frozenset[str] = frozenset()
    # Out-of-scope substrings this arch's own tensors carry.
    admitted_substrings: frozenset[str] = frozenset()
    # Live parameter name -> the name llama.cpp converts it under.
    param_aliases: Mapping[str, str] = field(default_factory=dict)
    # Per-layer GGUF tensor -> how llama.cpp changed its layout or values.
    layer_transforms: Mapping[str, GGUFPlainTransform] = field(default_factory=dict)
    # q/k rows stored in llama.cpp's interleaved RoPE layout.
    rope_permuted_qk: bool = False
    # Gated DeltaNet V heads tiled across the K heads instead of grouped.
    tiled_v_heads: bool = False


class GGUFModelAdapter:
    """Own GGUF name/scope policy: translate tensor names to live MLX-LM
    parameter paths, and the per-arch weight-layout policy (which tensors need
    llama.cpp's row, column or value layout undone, and the index or transform)."""

    # Archs this loader is verified against, each with every rule it needs. The
    # scope guard is an allowlist (default-deny), so an untested arch is never
    # silently admitted. MistralForCausalLM converts under llama; a .gguf
    # declaring a raw "mistral" or "qwen3_5" arch is not a llama.cpp product.
    _ARCH_SPECS: Mapping[str, GGUFArchSpec] = {
        "qwen2": GGUFArchSpec(),
        "qwen3": GGUFArchSpec(),
        "llama": GGUFArchSpec(
            config_model_types=frozenset({"mistral"}), rope_permuted_qk=True
        ),
        "qwen35": GGUFArchSpec(
            config_model_types=frozenset({"qwen3_5"}),
            admitted_substrings=frozenset({"attn_qkv", "ssm_"}),
            param_aliases={"dt_bias": "dt_proj.bias"},
            layer_transforms={
                "ssm_a": _NegatedExp(),
                "ssm_conv1d.weight": _SqueezedUnitAxis(),
            },
            tiled_v_heads=True,
        ),
    }

    SUPPORTED_ARCHS = frozenset(_ARCH_SPECS)

    # Substrings flagging an out-of-scope tensor: fused QKV, SSM/Mamba, MoE
    # experts + router, vision/mmproj. Defense in depth behind the arch
    # allowlist; an arch admits only the markers its own decoder tensors carry.
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
    # live parameter tree. The only load-bearing override for every supported arch.
    _STATIC_GLOBAL_OVERRIDE = {
        "token_embd.weight": "model.embed_tokens.weight",
        "output_norm.weight": "model.norm.weight",
        "output.weight": "lm_head.weight",
    }

    # GGUF tensors with no MLX-LM counterpart (precomputed buffers MLX recreates).
    _KNOWN_SKIP = frozenset({"rope_freqs.weight"})

    def __init__(
        self,
        *,
        translation_map: dict[str, str | None],
        permute_index: dict[str, mx.array] | None = None,
        column_index: dict[str, mx.array] | None = None,
        plain_transforms: dict[str, GGUFPlainTransform] | None = None,
    ) -> None:
        self._translation_map = translation_map
        self._permute_index = permute_index or {}
        self._column_index = column_index or {}
        self._plain_transforms = plain_transforms or {}

    def translate(self, gguf_name: str) -> str | None:
        """Return the live MLX-LM parameter path for a GGUF tensor name."""
        if gguf_name not in self._translation_map:
            raise GGUFLoadError(f"Unmapped GGUF tensor {gguf_name!r}.")
        return self._translation_map[gguf_name]

    def row_permute_index(self, gguf_name: str) -> mx.array | None:
        """Row index that undoes llama.cpp's stored row order, or ``None``.

        A plain table lookup, like :meth:`translate`; the loader applies it to
        the tensor, quantized or plain, and to its bias.
        """
        return self._permute_index.get(gguf_name)

    def column_permute_index(self, gguf_name: str) -> mx.array | None:
        """Input-column index ``c`` with live weight ``stored[:, c]``, or ``None``."""
        return self._column_index.get(gguf_name)

    def plain_transform(self, gguf_name: str) -> GGUFPlainTransform | None:
        """How llama.cpp changed a plain tensor's layout or values, or ``None``."""
        return self._plain_transforms.get(gguf_name)

    @classmethod
    def resolve_arch(cls, *, gguf_arch: str, config_model_type: str) -> str:
        """Normalize, allowlist-gate, and cross-check the GGUF vs config arch.

        The .gguf is the source of truth for its own architecture; reject anything
        outside the allowlist, then require the companion config to describe the
        same model. Returns the canonical arch for ``from_model``.
        """
        arch = cls._normalize_arch(gguf_arch)
        if arch not in cls.SUPPORTED_ARCHS:
            raise GGUFLoadError(
                f"Architecture {arch!r} is not supported; the GGUF loader "
                f"supports {sorted(cls.SUPPORTED_ARCHS)}."
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
    def validate_config_model_type(cls, config_model_type: str) -> None:
        """Reject a config family that :meth:`resolve_arch` rejects for every .gguf."""
        config_arch = cls._config_model_type_to_gguf_arch(config_model_type)
        if config_arch not in cls.SUPPORTED_ARCHS:
            raise GGUFLoadError(
                f"Config model_type {config_model_type!r} is not supported; the "
                f"GGUF loader supports {sorted(cls.SUPPORTED_ARCHS)}."
            )

    @classmethod
    def is_out_of_scope_tensor(cls, gguf_name: str, arch: str) -> bool:
        """Whether ``gguf_name`` has an out-of-scope substring ``arch`` does not admit."""
        admitted = cls._ARCH_SPECS[arch].admitted_substrings
        return any(
            substring in gguf_name
            for substring in cls.OUT_OF_SCOPE_TENSOR_SUBSTRINGS
            if substring not in admitted
        )

    @staticmethod
    def text_model(model: nn.Module) -> nn.Module:
        """Return the text model a GGUF fills (``language_model`` when wrapped)."""
        return model.language_model if hasattr(model, "language_model") else model

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
        spec = cls._ARCH_SPECS[arch]
        arch_enum = cls._resolve_arch_enum(gguf, arch)
        if arch_enum is None:  # pragma: no cover - allowlisted archs always resolve
            raise GGUFLoadError(f"Unknown GGUF architecture {arch!r}.")
        name_map = gguf.get_tensor_name_map(arch_enum, num_hidden_layers)
        reverse: dict[str, str] = {}
        params = cast("list[tuple[str, Any]]", tree_flatten(model.parameters()))
        for mlx_name, _ in params:
            gguf_name = name_map.get_name(
                cls._converted_name(mlx_name, spec), try_suffixes=(".weight", ".bias")
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
        v_head_rows, v_head_columns = (
            cls._build_v_head_index(model) if spec.tiled_v_heads else ({}, {})
        )
        rope_rows = (
            cls._build_permute_index(model, num_hidden_layers)
            if spec.rope_permuted_qk
            else {}
        )
        return cls(
            translation_map=translation_map,
            permute_index={**rope_rows, **v_head_rows},
            column_index=v_head_columns,
            plain_transforms={
                f"blk.{i}.{tensor}": transform
                for i in range(num_hidden_layers)
                for tensor, transform in spec.layer_transforms.items()
            },
        )

    @classmethod
    def _build_permute_index(
        cls, model: nn.Module, num_hidden_layers: int
    ) -> dict[str, mx.array]:
        """Precompute the RoPE row-permutation index per q/k GGUF tensor name.

        The index reorders each head's interleaved pairs back to the HF half-split.
        ``out_features`` is read off the BUILT q/k projections (whose head_dim is
        already resolved), not ``args.head_dim`` which stays ``None`` when the
        config omits it (the #449 omitted-default trap). llama.cpp permutes both
        the q/k weight AND its bias, so the same index covers ``.bias``.
        """
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

    @classmethod
    def _build_v_head_index(
        cls, model: nn.Module
    ) -> tuple[dict[str, mx.array], dict[str, mx.array]]:
        """Precompute the rows and ``ssm_out`` columns that regroup tiled V heads.

        Head counts and dims are read off each BUILT Gated DeltaNet. llama.cpp
        reorders only when ``nv != nk``, and only the V part (after the
        ``2 * nk * dk`` q/k rows) of the fused qkv and conv tensors.
        """
        rows: dict[str, mx.array] = {}
        columns: dict[str, mx.array] = {}
        for i, layer in enumerate(model.model.layers):
            if "linear_attn" not in layer:
                continue
            gdn = layer.linear_attn
            nk, nv = gdn.num_k_heads, gdn.num_v_heads
            if nv == nk:
                continue
            heads = cls._v_head_untile_index(nk, nv, 1)
            values = cls._v_head_untile_index(nk, nv, gdn.head_v_dim)
            qk_rows = 2 * gdn.key_dim
            qkv = mx.concatenate([mx.arange(qk_rows), qk_rows + values])
            for tensor, index in (
                ("attn_qkv.weight", qkv),
                ("ssm_conv1d.weight", qkv),
                ("attn_gate.weight", values),
                ("ssm_alpha.weight", heads),
                ("ssm_beta.weight", heads),
                ("ssm_a", heads),
                ("ssm_dt.bias", heads),
            ):
                rows[f"blk.{i}.{tensor}"] = index
            columns[f"blk.{i}.ssm_out.weight"] = values
        return rows, columns

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
    def _v_head_untile_index(
        num_k_heads: int, num_v_heads: int, head_dim: int
    ) -> mx.array:
        """Inverse of llama.cpp's grouped->tiled V-head reorder, as a row index.

        The file holds V heads as ``(nv // nk, nk, head_dim)``; mlx-lm groups
        them as ``(nk, nv // nk, head_dim)``.
        """
        return (
            mx.arange(num_v_heads * head_dim)
            .reshape(num_v_heads // num_k_heads, num_k_heads, head_dim)
            .swapaxes(0, 1)
            .reshape(-1)
        )

    @staticmethod
    def _converted_name(mlx_name: str, spec: GGUFArchSpec) -> str:
        """Return the name llama.cpp converts a live parameter under."""
        module_path, _, param_name = mlx_name.rpartition(".")
        converted = spec.param_aliases.get(param_name)
        return mlx_name if converted is None else f"{module_path}.{converted}"

    @staticmethod
    def _normalize_arch(name: str) -> str:
        return name.strip().lower().replace("-", "_")

    @classmethod
    def _config_model_type_to_gguf_arch(cls, model_type: str) -> str:
        """Map a config ``model_type`` to the GGUF arch its family converts under."""
        model_type = cls._normalize_arch(model_type)
        for arch, spec in cls._ARCH_SPECS.items():
            if model_type in spec.config_model_types:
                return arch
        return model_type

    @classmethod
    def _resolve_arch_enum(cls, gguf: Any, arch: str) -> Any | None:
        for arch_enum, name in gguf.MODEL_ARCH_NAMES.items():
            if cls._normalize_arch(str(name)) == arch:
                return arch_enum
        return None
