# SPDX-License-Identifier: Apache-2.0
"""GLM-4.7-Flash nextn (MTP) head loading, validation, and deployment metadata."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from json import JSONDecodeError, loads
from numbers import Integral, Real
from pathlib import Path
from threading import Lock
from typing import Any, ClassVar

from vllm.logger import init_logger

from vllm_metal.v1.mlx_lm_paths import mlx_lm_compatible_model_path

logger = init_logger(__name__)

GLM4_MOE_LITE_MTP_MODEL_TYPE = "glm4_moe_lite_mtp"
GLM4_MOE_LITE_MTP_NUM_NEXTN = 1
GLM4_MOE_LITE_TARGET_MODEL_TYPE = "glm4_moe_lite"

# The head ships as a hosted, revision-pinned drafter checkpoint; error messages
# point users at those repos (or at the mlx-vlm split tool for a custom source
# revision or quantization).
HOSTED_HEAD_REPOS = (
    "samithaj/GLM-4.7-Flash-MTP-4bit",
    "samithaj/GLM-4.7-Flash-MTP-bf16",
)
SPLIT_TOOL = "python -m mlx_vlm.speculative.drafters.glm4_moe_lite_mtp.split"
HEAD_SOURCE_HINT = (
    f"Use a hosted GLM-4.7-Flash MTP head ({HOSTED_HEAD_REPOS[0]} or "
    f"{HOSTED_HEAD_REPOS[1]}), or split your own with '{SPLIT_TOOL}'."
)

_HEAD_DOWNLOAD_ALLOW_PATTERNS = [
    "config.json",
    "*.safetensors",
    "*.safetensors.index.json",
]

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


@dataclass(frozen=True, slots=True)
class Glm4MoeLiteMTPHeadMetadata:
    """Validated shape of a GLM-4.7-Flash nextn (MTP) head checkpoint."""

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
    num_nextn_predict_layers: int

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> Glm4MoeLiteMTPHeadMetadata:
        model_type = _config_value(config, "model_type")
        if model_type != GLM4_MOE_LITE_MTP_MODEL_TYPE:
            raise ValueError(
                "Glm4MoeLiteMTP head requires "
                f"model_type={GLM4_MOE_LITE_MTP_MODEL_TYPE!r}, got "
                f"{model_type!r}. {HEAD_SOURCE_HINT}"
            )

        # The drafter split nests the source model config under ``text_config``;
        # the head's shape fields live there. The top-level ``model_type`` above is
        # the guard that this is the head, not the raw target repo
        # (``glm4_moe_lite``). Require the nested schema rather than tolerating a
        # flat checkpoint, so a malformed head is rejected at its own boundary.
        text = _config_value(config, "text_config")
        if not isinstance(text, Mapping):
            raise ValueError(
                "Glm4MoeLiteMTP head config must nest the source model config "
                f"under 'text_config' (the drafter-split schema). {HEAD_SOURCE_HINT}"
            )

        num_nextn = _required_int(text, "num_nextn_predict_layers")
        if num_nextn != GLM4_MOE_LITE_MTP_NUM_NEXTN:
            raise ValueError(
                "Glm4MoeLiteMTP head only supports "
                f"num_nextn_predict_layers={GLM4_MOE_LITE_MTP_NUM_NEXTN}, got "
                f"{num_nextn}"
            )

        block_size = _required_int(config, "block_size")
        if block_size != num_nextn + 1:
            raise ValueError(
                "Glm4MoeLiteMTP head block_size must be "
                f"num_nextn_predict_layers+1={num_nextn + 1}, got {block_size}"
            )

        return cls(
            model_type=str(model_type),
            architectures=_architectures(text),
            hidden_size=_required_positive_int(text, "hidden_size"),
            vocab_size=_required_positive_int(text, "vocab_size"),
            kv_lora_rank=_required_positive_int(text, "kv_lora_rank"),
            qk_rope_head_dim=_required_positive_int(text, "qk_rope_head_dim"),
            qk_nope_head_dim=_required_positive_int(text, "qk_nope_head_dim"),
            v_head_dim=_required_positive_int(text, "v_head_dim"),
            num_attention_heads=_required_positive_int(text, "num_attention_heads"),
            rope_theta=_required_positive_float(text, "rope_theta"),
            rms_norm_eps=_required_positive_float(text, "rms_norm_eps"),
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
                f"{model_path}. {HEAD_SOURCE_HINT}"
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
        from mlx.utils import tree_flatten

        params = dict(tree_flatten(model.parameters()))
        for key, source in (
            ("lm_head.weight", "shared_head.head"),
            ("model.embed_tokens.weight", "the dedicated nextn embedding"),
        ):
            if key not in params:
                raise ValueError(
                    f"GLM4 MTP head checkpoint is missing {key!r} (from {source}). "
                    f"{HEAD_SOURCE_HINT}"
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
                f"{HEAD_SOURCE_HINT}"
            )
        from vllm_metal.v1.mtp_heads.glm4_moe_lite_mtp_model import (
            Glm4MoeLiteMTPArgs,
            Glm4MoeLiteMTPModel,
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
