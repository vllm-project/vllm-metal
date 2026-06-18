# SPDX-License-Identifier: Apache-2.0
"""MLX-native local GGUF loader (dense decoder-only, Q8_0/Q4_0).

Joins PR 1's ``GGUFMLXQuantizedTensor`` (MLX-native repack) and PR 2's
``GGUFLinear``/``GGUFEmbedding`` wrappers into a working mlx-lm model. The model
skeleton is built through the public ``mlx_lm.utils.load_model(..., strict=False)``
path; this loader layers the GGUF-specific orchestration on top: preflight, read,
partition, install the quantized wrappers, load the plain tensors, and verify
every parameter ended up populated.

``GGUFModelLoader`` is the owner; ``load_gguf_model`` is a thin convenience
wrapper. Model-family name/scope policy lives in
:class:`vllm_metal.gguf.adapter.GGUFModelAdapter`. Design rationale is in
``codex/GGUF_PR3_DESIGN.md`` §5.

Scope: dense decoder-only, Q8_0/Q4_0 per tensor. Linear-attention/SSM hybrids,
fused-QKV, MoE, K-quants, Q4_1, vision and remote ``repo:quant`` references are
out of scope and fail fast at the arch allowlist or in preflight.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import gguf
import mlx.core as mx
import mlx.nn as nn
from mlx.utils import tree_flatten
from mlx_lm.utils import load_config, load_model, load_tokenizer

from vllm_metal.gguf.adapter import GGUFLoadError, GGUFModelAdapter
from vllm_metal.gguf.mlx_native import MLX_NATIVE_GGUF_TYPES, GGUFMLXQuantizedTensor
from vllm_metal.gguf.wrappers import GGUFEmbedding, GGUFLinear

__all__ = ["GGUFLoadError", "GGUFModelLoader", "load_gguf_model"]

_WEIGHT_SUFFIX = ".weight"
_BIAS_SUFFIX = ".bias"
_NUM_LAYERS_KEYS = ("num_hidden_layers", "n_layers", "num_layers", "n_layer")


class GGUFModelLoader:
    """Load a dense GGUF checkpoint into its mlx-lm model.

    Args:
        gguf_path: Path to a local ``.gguf`` file (dense decoder, Q8_0/Q4_0).
        config_dir: Directory with the companion HF ``config.json`` + tokenizer
            that define the mlx-lm skeleton (GGUF files carry weights only).
        target_dtype: Compute dtype for dequantized embedding rows / activations.
        tokenizer_config: Extra kwargs forwarded to mlx_lm tokenizer loading.
    """

    def __init__(
        self,
        gguf_path: str,
        *,
        config_dir: str,
        target_dtype: mx.Dtype,
        tokenizer_config: dict[str, Any] | None = None,
    ) -> None:
        self._gguf_path = Path(gguf_path)
        self._config_dir = Path(config_dir)
        self._target_dtype = target_dtype
        self._tokenizer_config = tokenizer_config or {}

        self._reader: Any = None
        self._config: dict[str, Any] = {}
        self._arch: str = ""
        self._model: nn.Module | None = None
        self._adapter: GGUFModelAdapter | None = None
        self._quant: dict[str, GGUFMLXQuantizedTensor] = {}
        self._plain: dict[str, mx.array] = {}
        self._biases: dict[str, mx.array] = {}
        self._installed: list[tuple[str, GGUFMLXQuantizedTensor, nn.Module]] = []
        self._tokenizer: Any = None

    def load(self) -> tuple[nn.Module, Any]:
        """Run the full load workflow and return ``(model, tokenizer)``.

        Raises:
            GGUFLoadError: On any unsupported file, arch, qtype, shape mismatch, or
                an incompletely populated model (fail fast, never silent).
        """
        self._open_inputs()
        self._preflight()
        self._build_skeleton()
        self._partition()
        self._install_quant_modules()
        self._load_plain_weights()
        self._load_tokenizer()
        return cast("nn.Module", self._model), self._tokenizer

    # -- workflow steps -----------------------------------------------------

    def _open_inputs(self) -> None:
        if self._gguf_path.suffix != ".gguf" or not self._gguf_path.is_file():
            raise GGUFLoadError(f"Not a local .gguf file: {str(self._gguf_path)!r}")
        if not (self._config_dir / "config.json").is_file():
            raise GGUFLoadError(
                f"No config.json in config_dir {str(self._config_dir)!r}"
            )
        try:
            self._reader = gguf.GGUFReader(str(self._gguf_path))
        except (ValueError, OSError) as exc:
            raise GGUFLoadError(
                f"Could not read GGUF file {str(self._gguf_path)!r}: {exc}"
            ) from exc
        self._config = load_config(self._config_dir)
        # The .gguf is the source of truth for its own architecture; gate it
        # against the dense allowlist and require the config to agree.
        self._arch = GGUFModelAdapter.resolve_arch(
            gguf_arch=self._gguf_arch(),
            config_model_type=str(self._config.get("model_type", "")),
        )

    def _preflight(self) -> None:
        """Reject out-of-scope tensors / qtypes from the reader, before ``mx.load``."""
        plain_types = {
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.BF16,
        }
        allowed_weight_types = set(MLX_NATIVE_GGUF_TYPES) | plain_types
        for tensor in self._reader.tensors:
            name = tensor.name
            if any(
                sub in name for sub in GGUFModelAdapter.OUT_OF_SCOPE_TENSOR_SUBSTRINGS
            ):
                raise GGUFLoadError(
                    f"Out-of-scope GGUF tensor {name!r} (fused-QKV/SSM/MoE/vision) "
                    "is not supported by the dense GGUF loader."
                )
            if (
                name.endswith(_WEIGHT_SUFFIX)
                and tensor.tensor_type not in allowed_weight_types
            ):
                raise GGUFLoadError(
                    f"Unsupported qtype {tensor.tensor_type.name} on mapped weight "
                    f"{name!r}; only Q8_0/Q4_0 (and plain F32/F16/BF16) are supported."
                )
            if name.endswith(_BIAS_SUFFIX) and tensor.tensor_type not in plain_types:
                raise GGUFLoadError(
                    f"Unsupported qtype {tensor.tensor_type.name} on bias {name!r}; "
                    "additive biases must be plain F32/F16/BF16."
                )

    def _build_skeleton(self) -> None:
        # Public upstream path: with strict=False it builds the config-only model
        # (no safetensors required) and owns get_classes + its own sanitize. The
        # GGUF plain weights are loaded separately below and are NOT re-sanitized:
        # the partition already excludes exactly what a dense sanitize would strip
        # (the tie-redundant output->lm_head, and rope_freqs via the known-skip).
        self._model, _ = load_model(
            self._config_dir, strict=False, model_config=self._config
        )
        self._adapter = GGUFModelAdapter.from_model(
            self._model,
            gguf=gguf,
            arch=self._arch,
            num_hidden_layers=self._num_hidden_layers(),
        )

    def _partition(self) -> None:
        """Route each GGUF tensor to quant-install / plain-load / bias-side-map."""
        adapter = cast("GGUFModelAdapter", self._adapter)
        tied = bool(self._config.get("tie_word_embeddings", False))
        arrays = mx.load(str(self._gguf_path))
        for tensor in self._reader.tensors:
            name = tensor.name
            if adapter.is_known_skip(name):
                continue
            translated = adapter.translate(name)
            if translated is None:
                raise GGUFLoadError(f"Unmapped GGUF tensor {name!r}.")

            suffix = _BIAS_SUFFIX if name.endswith(_BIAS_SUFFIX) else _WEIGHT_SUFFIX
            module_path = translated[: -len(suffix)]

            if not self._has_module(module_path):
                if tied and translated == "lm_head.weight":
                    continue  # tie-redundant output table; the tied head uses as_linear
                raise GGUFLoadError(
                    f"GGUF tensor {name!r} maps to {translated!r} but module "
                    f"{module_path!r} is absent from the model."
                )

            if suffix == _BIAS_SUFFIX:
                self._validate_plain_shape(translated, arrays[name], name)
                self._biases[module_path] = arrays[name]
            elif tensor.tensor_type in MLX_NATIVE_GGUF_TYPES:
                qt = GGUFMLXQuantizedTensor.from_mx_load(
                    arrays, name, tensor.tensor_type
                )
                self._validate_quant_shape(module_path, qt, name)
                self._quant[module_path] = qt
            else:  # plain F32/F16/BF16
                self._validate_plain_shape(translated, arrays[name], name)
                self._plain[translated] = arrays[name]

        # A bias whose weight is plain (not quantized) loads through the normal
        # path; only biases paired with a quant weight go to GGUFLinear.
        for module_path in [p for p in self._biases if p not in self._quant]:
            self._plain[f"{module_path}{_BIAS_SUFFIX}"] = self._biases.pop(module_path)

    def _install_quant_modules(self) -> None:
        """Replace each quantized module with a GGUF wrapper on the live tree."""
        for module_path, qt in self._quant.items():
            current = self._get_module(module_path)
            bias = self._biases.get(module_path)
            if isinstance(current, nn.Embedding):
                wrapper: nn.Module = GGUFEmbedding(qt, self._target_dtype)
            elif isinstance(current, nn.Linear):
                # Installing the wrapper drops the original bias leaf, so a bias the
                # model expects but the GGUF omitted would vanish before the
                # completeness check could catch it. Fail fast here instead.
                if bias is None and "bias" in current:
                    raise GGUFLoadError(
                        f"Model module {module_path!r} expects a bias but the GGUF "
                        "has no matching .bias tensor."
                    )
                wrapper = GGUFLinear(qt, bias=bias)
            else:
                raise GGUFLoadError(
                    f"Unsupported quantized target module "
                    f"{type(current).__name__!r} at {module_path!r}; expected "
                    "nn.Linear or nn.Embedding."
                )
            self._set_module(module_path, wrapper)
            self._installed.append((module_path, qt, wrapper))

    def _load_plain_weights(self) -> None:
        model = cast("nn.Module", self._model)
        model.eval()
        model.load_weights(list(self._plain.items()), strict=False)
        self._assert_complete()
        mx.eval(model.parameters())
        for _, qt, _ in self._installed:
            mx.eval(qt.qweight, qt.scales, qt.biases)

    def _load_tokenizer(self) -> None:
        self._tokenizer = load_tokenizer(
            self._config_dir,
            tokenizer_config_extra=self._tokenizer_config,
            eos_token_ids=self._config.get("eos_token_id"),
        )

    # -- checks -------------------------------------------------------------

    def _assert_complete(self) -> None:
        """Fail fast unless every live parameter was populated and no key orphaned.

        ``load_weights(strict=False)`` silently tolerates missing keys (left at
        random init), orphan/typo keys, and shape mismatches, so this replaces the
        coverage half of ``strict=True`` (shapes are validated per-tensor in
        ``_partition``). A wrapper's bias leaf counts as owned; the wrappers' hidden
        quant arrays are not parameter leaves.
        """
        model = cast("nn.Module", self._model)
        owned = set(self._plain)
        for module_path, _, wrapper in self._installed:
            wrapper_leaves = cast(
                "list[tuple[str, Any]]", tree_flatten(wrapper.parameters())
            )
            for leaf_name, _ in wrapper_leaves:
                owned.add(f"{module_path}.{leaf_name}")

        model_leaves = cast("list[tuple[str, Any]]", tree_flatten(model.parameters()))
        live = {name for name, _ in model_leaves}
        unfed = sorted(live - owned)
        orphan = sorted(set(self._plain) - live)
        if unfed or orphan:
            raise GGUFLoadError(
                f"Incomplete GGUF load: {len(unfed)} uninitialized "
                f"{unfed[:6]}; {len(orphan)} orphan keys {orphan[:6]}."
            )

    def _validate_quant_shape(
        self, module_path: str, qt: GGUFMLXQuantizedTensor, name: str
    ) -> None:
        weight = self._get_module(module_path).weight
        expected = tuple(int(d) for d in weight.shape)
        actual = (qt.out_features, qt.in_features)
        if actual != expected:
            raise GGUFLoadError(
                f"GGUF tensor {name!r} dims {actual} != model {module_path!r} "
                f"{expected}."
            )

    def _validate_plain_shape(
        self, param_path: str, array: mx.array, name: str
    ) -> None:
        expected = tuple(int(d) for d in self._get_param(param_path).shape)
        actual = tuple(int(d) for d in array.shape)
        if actual != expected:
            raise GGUFLoadError(
                f"GGUF tensor {name!r} shape {actual} != model {param_path!r} "
                f"{expected}."
            )

    # -- reader / model-tree helpers ----------------------------------------

    def _gguf_arch(self) -> str:
        field = self._reader.get_field("general.architecture")
        if field is None:
            raise GGUFLoadError("GGUF file has no general.architecture metadata.")
        return str(field.contents())

    def _num_hidden_layers(self) -> int:
        for source in (self._config, self._config.get("text_config")):
            if isinstance(source, dict):
                for key in _NUM_LAYERS_KEYS:
                    value = source.get(key)
                    if value is not None:
                        return int(value)
        raise GGUFLoadError("Could not determine num_hidden_layers from config.")

    def _get_module(self, path: str) -> Any:
        parent, leaf = self._resolve_parent(path)
        return self._step(parent, leaf)

    def _set_module(self, path: str, value: Any) -> None:
        parent, leaf = self._resolve_parent(path)
        if leaf.isdigit() and isinstance(parent, list):
            parent[int(leaf)] = value
        elif isinstance(parent, dict):
            parent[leaf] = value
        else:
            setattr(parent, leaf, value)

    def _get_param(self, param_path: str) -> mx.array:
        module_path, _, param_name = param_path.rpartition(".")
        return getattr(self._get_module(module_path), param_name)

    def _has_module(self, path: str) -> bool:
        try:
            self._get_module(path)
        except (AttributeError, KeyError, IndexError, ValueError):
            return False
        return True

    def _resolve_parent(self, path: str) -> tuple[Any, str]:
        obj: Any = self._model
        parts = path.split(".")
        for part in parts[:-1]:
            obj = self._step(obj, part)
        return obj, parts[-1]

    @staticmethod
    def _step(obj: Any, part: str) -> Any:
        """Descend one path component: list index, dict key, or module attribute."""
        if part.isdigit() and isinstance(obj, list):
            return obj[int(part)]
        if isinstance(obj, dict):
            return obj[part]
        return getattr(obj, part)


def load_gguf_model(
    gguf_path: str,
    *,
    config_dir: str,
    target_dtype: mx.Dtype,
    tokenizer_config: dict[str, Any] | None = None,
) -> tuple[nn.Module, Any]:
    """Load a dense GGUF checkpoint into ``(model, tokenizer)`` (thin wrapper)."""
    return GGUFModelLoader(
        gguf_path,
        config_dir=config_dir,
        target_dtype=target_dtype,
        tokenizer_config=tokenizer_config,
    ).load()
