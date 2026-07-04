# SPDX-License-Identifier: Apache-2.0
"""vLLM engine integration for GGUF references (post vLLM 0.24 removal).

vLLM 0.24 migrated in-tree GGUF support to the CUDA/ROCm-only
``vllm-project/vllm-gguf-plugin`` (RFC vllm#39583, PR vllm#39612), so nothing
sets ``quantization="gguf"`` for a ``.gguf`` model anymore and vllm-metal's
GGUF routing went dead. This module restores the engine-facing layer with the
same semantics as the official plugin for the fields the Metal path consumes:
a marker quantization config so ``quantization="gguf"`` validates, and two
narrow wraps that detect a local GGUF file, carry it in
``model_config.model_weights``, and point ``model`` at the config source.

Mounted through the ``vllm.general_plugins`` entry point (the official
plugin's own mechanism), which vLLM loads in ``EngineArgs.__post_init__`` and
again inside the EngineCore/worker subprocesses, so the registration is
spawn-safe without import-time patching.

This module must stay importable WITHOUT the optional ``gguf`` package: a
default install loads it on every vLLM run via the entry point.

# SCAFFOLDING: remove when vllm-project/vllm-gguf-plugin installs and imports
# on macOS; then depend on it for this layer (register() already no-ops when
# that plugin is importable).
"""

from __future__ import annotations

import importlib
import importlib.util
import re
from functools import wraps
from pathlib import Path

import torch
from vllm.model_executor.layers.quantization import (
    QUANTIZATION_METHODS,
    get_quantization_config,
    register_quantization_config,
)
from vllm.model_executor.layers.quantization.base_config import (
    QuantizationConfig,
)

__all__ = ["GGUFEngineIntegration", "MetalGGUFConfig", "register"]


class MetalGGUFConfig(QuantizationConfig):
    """Marker quantization config that lets ``quantization="gguf"`` validate.

    vllm-metal never runs vLLM's quantized layers (the Metal runner loads
    through mlx-lm), but ``VllmConfig.__post_init__`` bare-constructs the
    registered config and calls its classmethods on every serve, and the
    instance is pickled into the spawned EngineCore/worker processes — so the
    class must live at module top level and every method must return a real
    value.
    """

    # The dtypes the Metal runtime actually serves.
    _SUPPORTED_ACT_DTYPES = [torch.float16, torch.bfloat16, torch.float32]
    # Metal has no CUDA compute capability; -1 passes any >= comparison.
    _NO_CAPABILITY_GATE = -1

    @classmethod
    def get_name(cls) -> str:
        return "gguf"

    @classmethod
    def get_supported_act_dtypes(cls) -> list[torch.dtype]:
        return cls._SUPPORTED_ACT_DTYPES

    @classmethod
    def get_min_capability(cls) -> int:
        return cls._NO_CAPABILITY_GATE

    @classmethod
    def get_config_filenames(cls) -> list[str]:
        # No sidecar quant config file: vLLM bare-constructs the marker.
        return []

    @classmethod
    def from_config(cls, config: dict) -> MetalGGUFConfig:
        return cls()

    def get_quant_method(self, layer: torch.nn.Module, prefix: str) -> None:
        # Never exercised: vllm-metal bypasses vLLM's model executor.
        return None


class GGUFEngineIntegration:
    """Owner of the GGUF engine-integration policy.

    Detection, config-source precedence, and the fail-fast rules live here as
    class constants and private helpers; :meth:`register` is the single
    entrypoint the ``vllm.general_plugins`` hook calls.
    """

    _GGUF_SUFFIX = ".gguf"
    _GGUF_MAGIC = b"GGUF"

    # Shape-level recognition of the official plugin's remote reference syntax
    # (``repo_id:quant``). Deliberately enum-free: the GGML enums live in the
    # optional ``gguf`` package this module must not import, and on this
    # fail-fast-only path over-recognition is harmless. The tag must read like
    # a GGUF quant name (Q4_K_M, IQ2_M, UD-Q4_K_XL, F16, ...), which keeps
    # ordinary ``repo:tag`` strings out.
    _REMOTE_REF_RE = re.compile(
        r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*:[A-Za-z0-9_+-]+$"
    )
    _QUANT_TAG_RE = re.compile(
        r"(?:^|-)(?:I?Q\d[A-Za-z0-9_]*|F16|F32|BF16|MXFP\d[A-Za-z0-9_]*)$",
        re.IGNORECASE,
    )

    _REMOTE_UNSUPPORTED_MSG = (
        "Remote GGUF references (repo_id:quant) are not supported yet by "
        "vllm-metal; download the .gguf and pass its local path, with "
        "--tokenizer pointing at the model's config/tokenizer directory."
    )

    @classmethod
    def register(cls) -> None:
        """Register the GGUF engine integration (idempotent).

        No-ops when the official ``vllm_gguf_plugin`` actually IMPORTS — once
        it works on macOS it owns this layer (see the module removal
        condition). Discoverable-but-unimportable (today's macOS state: its
        import pulls triton/CUDA) must NOT disable this integration.
        """
        if cls._official_plugin_imports():
            return
        if (
            "gguf" not in QUANTIZATION_METHODS
            or get_quantization_config("gguf") is not MetalGGUFConfig
        ):
            register_quantization_config("gguf")(MetalGGUFConfig)
        cls._patch_engine_args()
        cls._patch_speculator_probe()

    @staticmethod
    def _official_plugin_imports() -> bool:
        if importlib.util.find_spec("vllm_gguf_plugin") is None:
            return False
        try:
            importlib.import_module("vllm_gguf_plugin")
        except ImportError:
            return False
        return True

    @staticmethod
    def _is_local_gguf(model: str | None) -> bool:
        if not model:
            return False
        path = Path(model)
        if not path.is_file():
            return False
        if path.suffix == GGUFEngineIntegration._GGUF_SUFFIX:
            return True
        try:
            with path.open("rb") as f:
                return f.read(4) == GGUFEngineIntegration._GGUF_MAGIC
        except OSError:
            return False

    @classmethod
    def is_remote_gguf_reference(cls, model: str | None) -> bool:
        """Whether ``model`` reads as the plugin's ``repo_id:quant`` syntax.

        Mirrors the official plugin's public ``is_remote_gguf`` at the shape
        level (see the class-constant regexes for the deliberate enum-free
        divergence). Recognized references fail fast in the EngineArgs wrap.
        """
        if not model or not cls._REMOTE_REF_RE.fullmatch(model):
            return False
        _, tag = model.rsplit(":", 1)
        return cls._QUANT_TAG_RE.search(tag) is not None

    @classmethod
    def _resolve_config_source(
        cls, gguf_model: str, tokenizer: str | None, hf_config_path: str | None
    ) -> str:
        """Config source for a local GGUF, with the official plugin's
        precedence: ``hf_config_path > tokenizer > parent dir``.

        A ``.gguf`` carries weights only, so the resolved LOCAL directory must
        hold the model's ``config.json``; failing fast here names the actual
        inputs instead of letting transformers error on a path the user never
        typed. A non-local source (a Hub repo id) is passed through for vLLM
        to resolve.
        """
        if hf_config_path is not None:
            source = hf_config_path
        elif tokenizer and not cls._is_local_gguf(tokenizer):
            source = tokenizer
        else:
            source = str(Path(gguf_model).parent)
        source_path = Path(source)
        if source_path.is_dir() and not (source_path / "config.json").is_file():
            raise ValueError(
                f"Serving GGUF model {gguf_model!r} needs a config source: "
                f"{source!r} has no config.json. A .gguf carries weights only; "
                "pass --tokenizer <dir> pointing at the model's "
                "config/tokenizer directory."
            )
        return source

    @classmethod
    def _patch_engine_args(cls) -> None:
        from vllm.engine.arg_utils import EngineArgs

        if getattr(EngineArgs, "_metal_gguf_patched", False):
            return
        original = EngineArgs.create_model_config

        @wraps(original)
        def create_model_config(self, *args, **kwargs):
            if cls.is_remote_gguf_reference(self.model):
                raise ValueError(cls._REMOTE_UNSUPPORTED_MSG)
            if cls._is_local_gguf(self.model):
                gguf_model = self.model
                if Path(gguf_model).suffix != cls._GGUF_SUFFIX:
                    # Detected by magic bytes only: MLX's loader dispatches on
                    # the file extension, so a suffix-less GGUF would die much
                    # later in the worker with a misleading error.
                    raise ValueError(
                        f"{gguf_model!r} is a GGUF file (magic bytes) but "
                        "vllm-metal requires the .gguf extension; add the "
                        f"extension (e.g. {Path(gguf_model).name + '.gguf'!r})."
                    )
                # Resolve (and validate) the config source BEFORE touching any
                # EngineArgs field, so a fail-fast never leaves the args
                # half-rewritten.
                config_source = cls._resolve_config_source(
                    gguf_model,
                    self.tokenizer if isinstance(self.tokenizer, str) else None,
                    self.hf_config_path,
                )
                if self.quantization not in (None, "gguf"):
                    raise ValueError(
                        f"Cannot serve GGUF model with quantization={self.quantization!r}; "
                        "leave quantization unset or use 'gguf'."
                    )
                self.quantization = "gguf"
                if not self.model_weights:
                    self.model_weights = gguf_model
                if self.served_model_name is None:
                    self.served_model_name = [gguf_model]
                self.model = config_source
            return original(self, *args, **kwargs)

        EngineArgs.create_model_config = create_model_config
        EngineArgs._metal_gguf_patched = True

    @classmethod
    def _patch_speculator_probe(cls) -> None:
        # create_engine_config probes the model with
        # maybe_override_with_speculators BEFORE create_model_config runs, so
        # the probe must skip raw GGUF references too (the official plugin
        # patches the same two namespaces).
        import vllm.engine.arg_utils as arg_utils_module
        import vllm.transformers_utils.config as config_module

        if getattr(arg_utils_module, "_metal_gguf_probe_patched", False):
            return
        original = arg_utils_module.maybe_override_with_speculators

        @wraps(original)
        def maybe_override_with_speculators(
            model,
            tokenizer,
            trust_remote_code,
            revision=None,
            vllm_speculative_config=None,
            hf_token=None,
            **kwargs,
        ):
            # Mirrors the upstream signature so a positionally-passed
            # speculative config is never silently dropped.
            if cls._is_local_gguf(model) or cls.is_remote_gguf_reference(model):
                return model, tokenizer, vllm_speculative_config
            return original(
                model,
                tokenizer,
                trust_remote_code,
                revision=revision,
                vllm_speculative_config=vllm_speculative_config,
                hf_token=hf_token,
                **kwargs,
            )

        arg_utils_module.maybe_override_with_speculators = (
            maybe_override_with_speculators
        )
        config_module.maybe_override_with_speculators = maybe_override_with_speculators
        arg_utils_module._metal_gguf_probe_patched = True


def register() -> None:
    """``vllm.general_plugins`` entry point."""
    GGUFEngineIntegration.register()
