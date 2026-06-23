# SPDX-License-Identifier: Apache-2.0
"""MLX-native local GGUF loader (dense decoder-only, Q8_0/Q4_0).

Joins PR 1's ``GGUFMLXQuantizedTensor`` (MLX-native repack) and PR 2's
``GGUFLinear``/``GGUFEmbedding`` wrappers into a working mlx-lm model. The model
skeleton is built through the public ``mlx_lm.utils.load_model(..., strict=False)``
path; this loader layers the GGUF-specific orchestration on top: preflight, read,
partition, install the quantized wrappers, load the plain tensors, and verify
every parameter ended up populated.

``GGUFModelLoader`` is the owner. Model-family name/scope policy lives in
:class:`vllm_metal.gguf.adapter.GGUFModelAdapter`. Design rationale is in
``codex/GGUF_PR3_DESIGN.md`` §5.

Scope: dense decoder-only, Q8_0/Q4_0 per tensor. Linear-attention/SSM hybrids,
fused-QKV, MoE, K-quants, Q4_1, vision and remote ``repo:quant`` references are
out of scope and fail fast at the arch allowlist or in preflight.
"""

from __future__ import annotations

from dataclasses import dataclass
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

__all__ = ["GGUFLoadError", "GGUFModelLoader"]

_GGUF_SUFFIX = ".gguf"
_WEIGHT_SUFFIX = ".weight"
_BIAS_SUFFIX = ".bias"
_NUM_LAYERS_KEYS = ("num_hidden_layers", "n_layers", "num_layers", "n_layer")


GGUFWrapper = GGUFLinear | GGUFEmbedding


@dataclass(frozen=True)
class _PartitionedTensors:
    """GGUF tensors split by how the loader installs them."""

    quant: dict[str, GGUFMLXQuantizedTensor]
    plain: dict[str, mx.array]
    biases: dict[str, mx.array]


@dataclass(frozen=True)
class _InstalledWrapper:
    """A quantized wrapper installed onto the live model tree."""

    module_path: str
    wrapper: GGUFWrapper


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
        self._tokenizer_config = dict(tokenizer_config or {})

    @classmethod
    def for_model(
        cls,
        model_name: str,
        *,
        config_dir: str,
        target_dtype: mx.Dtype,
        tokenizer_config: dict[str, Any] | None = None,
    ) -> GGUFModelLoader:
        """Build a loader for a GGUF model the lifecycle has already routed here.

        Called only after the lifecycle confirmed ``model_config.quantization ==
        "gguf"`` and lazily imported this module (which imports the optional
        ``gguf`` package) — mirroring how the AWQ branch routes on detected
        config. This factory does the GGUF-specific input validation before
        constructing the loader.

        It raises rather than returning ``None`` so a confirmed-GGUF checkpoint
        can never silently fall through to the generic loader and bypass this
        owner — the same contract :meth:`AWQQuantLoader.for_model` uses for GPTQ.

        Args:
            model_name: Already-resolved model path
                (``get_model_download_path(model_config.model)``); for a local
                GGUF this is the ``.gguf`` file.
            config_dir: Companion config/tokenizer directory the user passes via
                ``--tokenizer`` (``model_config.tokenizer``). A ``.gguf`` carries
                weights only, so mlx-lm builds the model from this directory.
            target_dtype: Compute dtype for dequantized embedding rows /
                activations, threaded to the constructed loader.
            tokenizer_config: Extra kwargs forwarded to mlx-lm tokenizer loading
                (e.g. ``trust_remote_code``).

        Raises:
            GGUFLoadError: ``model_name`` is not a local ``.gguf`` file (remote
                ``repo:quant`` / Hub references are not supported yet), or
                ``config_dir`` has no ``config.json`` (``--tokenizer`` omitted or
                pointing at a directory without the companion config).
        """
        gguf_path = Path(model_name)
        if gguf_path.suffix != _GGUF_SUFFIX or not gguf_path.is_file():
            raise GGUFLoadError(
                f"{model_name!r} is a GGUF model, but it is not a local .gguf "
                "file. Remote GGUF (repo:quant / Hub download) is not supported "
                "yet; pass a path to a local .gguf file."
            )
        if not (Path(config_dir) / "config.json").is_file():
            raise GGUFLoadError(
                "Serving a GGUF model needs --tokenizer pointing at a directory "
                f"with config.json and the tokenizer files; got {config_dir!r}. A "
                ".gguf carries weights only, so mlx-lm builds the model from that "
                "directory."
            )
        return cls(
            model_name,
            config_dir=config_dir,
            target_dtype=target_dtype,
            tokenizer_config=tokenizer_config,
        )

    @staticmethod
    def cache_key(
        model_name: str, config_dir: str, *, target_dtype: mx.Dtype
    ) -> tuple[str, str]:
        """Process-cache key for a GGUF load.

        A ``.gguf`` carries weights only; the model skeleton and tokenizer are
        built from ``config_dir`` (the ``--tokenizer`` dir), so the key MUST
        include it — two LLMs for the same ``.gguf`` with different config dirs
        are different models and must not collide (else the second is served the
        first's tokenizer/skeleton, and the config-dir fail-fast is bypassed once
        the cache is warm). ``target_dtype`` is encoded too, dtype-scoped like the
        AWQ owner's: a model first served as bf16 must not be returned when fp16
        is later requested. Static so the lifecycle can compute the key and probe
        the shared cache before building a loader; the 2-tuple shape matches
        ``_generation_cache_key``.
        """
        return (model_name, f"gguf:{config_dir}:{target_dtype}")

    def load(self) -> tuple[nn.Module, Any]:
        """Run the full load workflow and return ``(model, tokenizer)``.

        Raises:
            GGUFLoadError: On any unsupported file, arch, qtype, shape mismatch, or
                an incompletely populated model (fail fast, never silent).
        """
        # Open the local GGUF + companion config and validate they describe one dense arch.
        reader, config = self._open_inputs()
        arch = GGUFModelAdapter.resolve_arch(
            gguf_arch=self._gguf_arch(reader),
            config_model_type=str(config.get("model_type", "")),
        )
        self._preflight(reader)

        # Build the mlx-lm skeleton through the public upstream loader path.
        model, _ = load_model(self._config_dir, strict=False, model_config=config)
        tied = bool(getattr(model.args, "tie_word_embeddings", False))

        # Derive the GGUF->MLX name translation from the live model structure.
        adapter = GGUFModelAdapter.from_model(
            model,
            gguf=gguf,
            arch=arch,
            num_hidden_layers=self._num_hidden_layers(config),
        )

        # Split tensors into wrapper installs, plain weights, and side biases.
        partition = self._partition(reader, model, tied, adapter)
        installed = self._install_quant_modules(
            model, partition.quant, partition.biases
        )
        self._load_plain_weights(model, partition.plain, installed)

        # Build the tokenizer from the companion config directory and return both.
        tokenizer = load_tokenizer(
            self._config_dir,
            tokenizer_config_extra=self._tokenizer_config,
            eos_token_ids=config.get("eos_token_id"),
        )
        return model, tokenizer

    def _open_inputs(self) -> tuple[Any, dict[str, Any]]:
        if self._gguf_path.suffix != _GGUF_SUFFIX or not self._gguf_path.is_file():
            raise GGUFLoadError(f"Not a local .gguf file: {str(self._gguf_path)!r}")
        if not (self._config_dir / "config.json").is_file():
            raise GGUFLoadError(
                f"No config.json in config_dir {str(self._config_dir)!r}"
            )
        try:
            reader = gguf.GGUFReader(str(self._gguf_path))
        except (ValueError, OSError) as exc:
            raise GGUFLoadError(
                f"Could not read GGUF file {str(self._gguf_path)!r}: {exc}"
            ) from exc
        return reader, load_config(self._config_dir)

    def _preflight(self, reader: Any) -> None:
        """Reject out-of-scope tensors / qtypes from the reader, before ``mx.load``."""
        plain_types = {
            gguf.GGMLQuantizationType.F32,
            gguf.GGMLQuantizationType.F16,
            gguf.GGMLQuantizationType.BF16,
        }
        allowed_weight_types = set(MLX_NATIVE_GGUF_TYPES) | plain_types
        for tensor in reader.tensors:
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

    def _partition(
        self,
        reader: Any,
        model: nn.Module,
        tied: bool,
        adapter: GGUFModelAdapter,
    ) -> _PartitionedTensors:
        """Route each GGUF tensor to quant-install / plain-load / bias-side-map."""
        arrays = mx.load(str(self._gguf_path))
        quant: dict[str, GGUFMLXQuantizedTensor] = {}
        plain: dict[str, mx.array] = {}
        biases: dict[str, mx.array] = {}
        for tensor in reader.tensors:
            name = tensor.name
            translated = adapter.translate(name)
            if translated is None:
                continue

            suffix = _BIAS_SUFFIX if name.endswith(_BIAS_SUFFIX) else _WEIGHT_SUFFIX
            module_path = translated[: -len(suffix)]

            try:
                current = self._resolve_path(model, module_path)
            except (AttributeError, KeyError, IndexError) as exc:
                if tied and translated == "lm_head.weight":
                    continue  # tie-redundant output table; the tied head uses as_linear
                raise GGUFLoadError(
                    f"GGUF tensor {name!r} maps to {translated!r} but module "
                    f"{module_path!r} is absent from the model."
                ) from exc

            if suffix == _BIAS_SUFFIX:
                self._validate_plain_shape(model, translated, arrays[name], name)
                biases[module_path] = arrays[name]
            elif tensor.tensor_type in MLX_NATIVE_GGUF_TYPES:
                qt = GGUFMLXQuantizedTensor.from_mx_load(
                    arrays, name, tensor.tensor_type
                )
                self._validate_quant_shape(module_path, current, qt, name)
                quant[module_path] = qt
            else:  # plain F32/F16/BF16
                self._validate_plain_shape(model, translated, arrays[name], name)
                plain[translated] = arrays[name]

        # A bias whose weight is plain (not quantized) loads through the normal
        # path; only biases paired with a quant weight go to GGUFLinear.
        for module_path in [path for path in biases if path not in quant]:
            plain[f"{module_path}{_BIAS_SUFFIX}"] = biases.pop(module_path)
        return _PartitionedTensors(quant=quant, plain=plain, biases=biases)

    def _install_quant_modules(
        self,
        model: nn.Module,
        quant: dict[str, GGUFMLXQuantizedTensor],
        biases: dict[str, mx.array],
    ) -> list[_InstalledWrapper]:
        """Replace each quantized module with a GGUF wrapper on the live tree."""
        installed: list[_InstalledWrapper] = []
        for module_path, qt in quant.items():
            current = self._resolve_path(model, module_path)
            bias = biases.get(module_path)
            if isinstance(current, nn.Embedding):
                wrapper: GGUFWrapper = GGUFEmbedding(qt, self._target_dtype)
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
            self._assign_path(model, module_path, wrapper)
            installed.append(
                _InstalledWrapper(module_path=module_path, wrapper=wrapper)
            )
        return installed

    def _load_plain_weights(
        self,
        model: nn.Module,
        plain: dict[str, mx.array],
        installed: list[_InstalledWrapper],
    ) -> None:
        model.eval()
        model.load_weights(list(plain.items()), strict=False)
        self._assert_complete(model, plain, installed)
        mx.eval(model.parameters())
        for installed_wrapper in installed:
            installed_wrapper.wrapper.eval_arrays()

    def _assert_complete(
        self,
        model: nn.Module,
        plain: dict[str, mx.array],
        installed: list[_InstalledWrapper],
    ) -> None:
        """Fail fast unless every live parameter was populated and no key orphaned.

        ``load_weights(strict=False)`` silently tolerates missing keys (left at
        random init), orphan/typo keys, and shape mismatches, so this replaces the
        coverage half of ``strict=True`` (shapes are validated per-tensor in
        ``_partition``). This exists because ``strict=True`` cannot see the wrapper-
        installed leaves that replace the original quantized ``.weight`` parameters.
        A wrapper's bias leaf counts as owned; the wrappers' hidden quant arrays are
        not parameter leaves.
        """
        owned = set(plain)
        for installed_wrapper in installed:
            wrapper_leaves = cast(
                "list[tuple[str, Any]]",
                tree_flatten(installed_wrapper.wrapper.parameters()),
            )
            for leaf_name, _ in wrapper_leaves:
                owned.add(f"{installed_wrapper.module_path}.{leaf_name}")

        model_leaves = cast("list[tuple[str, Any]]", tree_flatten(model.parameters()))
        live = {name for name, _ in model_leaves}
        unfed = sorted(live - owned)
        orphan = sorted(set(plain) - live)
        if unfed or orphan:
            raise GGUFLoadError(
                f"Incomplete GGUF load: {len(unfed)} uninitialized "
                f"{unfed[:6]}; {len(orphan)} orphan keys {orphan[:6]}."
            )

    def _validate_quant_shape(
        self,
        module_path: str,
        module: Any,
        qt: GGUFMLXQuantizedTensor,
        name: str,
    ) -> None:
        weight = module.weight
        expected = tuple(int(d) for d in weight.shape)
        actual = (qt.out_features, qt.in_features)
        if actual != expected:
            raise GGUFLoadError(
                f"GGUF tensor {name!r} dims {actual} != model {module_path!r} "
                f"{expected}."
            )

    def _validate_plain_shape(
        self, model: nn.Module, param_path: str, array: mx.array, name: str
    ) -> None:
        module_path, _, param_name = param_path.rpartition(".")
        expected = tuple(
            int(d)
            for d in getattr(self._resolve_path(model, module_path), param_name).shape
        )
        actual = tuple(int(d) for d in array.shape)
        if actual != expected:
            raise GGUFLoadError(
                f"GGUF tensor {name!r} shape {actual} != model {param_path!r} "
                f"{expected}."
            )

    @staticmethod
    def _gguf_arch(reader: Any) -> str:
        field = reader.get_field("general.architecture")
        if field is None:
            raise GGUFLoadError("GGUF file has no general.architecture metadata.")
        return str(field.contents())

    @staticmethod
    def _num_hidden_layers(config: dict[str, Any]) -> int:
        for source in (config, config.get("text_config")):
            if isinstance(source, dict):
                for key in _NUM_LAYERS_KEYS:
                    value = source.get(key)
                    if value is not None:
                        return int(value)
        raise GGUFLoadError("Could not determine num_hidden_layers from config.")

    @staticmethod
    def _resolve_path(root: Any, path: str) -> Any:
        obj = root
        for part in path.split("."):
            if part.isdigit() and isinstance(obj, list):
                obj = obj[int(part)]
            elif isinstance(obj, dict):
                obj = obj[part]
            else:
                obj = getattr(obj, part)
        return obj

    @staticmethod
    def _assign_path(root: Any, path: str, value: Any) -> None:
        parent_path, _, leaf = path.rpartition(".")
        parent = (
            GGUFModelLoader._resolve_path(root, parent_path) if parent_path else root
        )
        if leaf.isdigit() and isinstance(parent, list):
            parent[int(leaf)] = value
        elif isinstance(parent, dict):
            parent[leaf] = value
        else:
            setattr(parent, leaf, value)
