# SPDX-License-Identifier: Apache-2.0
"""Gemma4 MTP assistant loading and validation."""

from __future__ import annotations

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from json import JSONDecodeError, loads
from numbers import Integral
from pathlib import Path
from threading import Lock
from typing import Any

from vllm.logger import init_logger

from vllm_metal.v1.gemma4_mtp_constants import (
    GEMMA4_MTP_DEFAULT_CENTROID_TOP_K,
    GEMMA4_MTP_DEFAULT_NUM_CENTROIDS,
    GEMMA4_MTP_N_PREDICT,
)
from vllm_metal.v1.mlx_lm_paths import mlx_lm_compatible_model_path

logger = init_logger(__name__)

GEMMA4_MTP_DRAFT_MODEL_TYPES = frozenset({"gemma4_assistant", "gemma4_mtp"})
GEMMA4_MTP_DRAFT_ARCHITECTURES = frozenset(
    {"Gemma4AssistantForCausalLM", "Gemma4MTPModel"}
)
GEMMA4_TARGET_MODEL_TYPES = frozenset({"gemma4", "gemma4_text"})
GEMMA4_MTP_TEXT_MODEL_TYPE = "gemma4_text"
GEMMA4_MTP_VALID_LAYER_TYPES = frozenset({"sliding_attention", "full_attention"})

_ASSISTANT_DOWNLOAD_ALLOW_PATTERNS = [
    "config.json",
    "generation_config.json",
    "model*.safetensors.index.json",
    "*.safetensors",
]

_ASSISTANT_CACHE: dict[tuple[str, str | None], Gemma4MTPAssistantRuntime] = {}
_ASSISTANT_CACHE_LOCK = Lock()


def _identity_model_path(model_name: str) -> str:
    return model_name


@dataclass(frozen=True, slots=True)
class Gemma4MTPAssistantRuntime:
    """Loaded assistant runtime owned by the model lifecycle."""

    model_name: str
    model: Any
    metadata: Gemma4MTPAssistantMetadata
    forward_ready: bool = False


class Gemma4MTPAssistantLoader:
    """Loads and validates a Gemma4 MTP assistant runtime."""

    def __init__(
        self,
        *,
        load_model_fn: Callable[..., tuple[Any, dict[str, Any]]] | None = None,
        download_fn: Callable[[str, str | None], Path] | None = None,
        model_path_resolver: Callable[[str], str] = _identity_model_path,
    ) -> None:
        self._load_model = load_model_fn
        self._download = download_fn
        self._model_path_resolver = model_path_resolver

    def load_if_needed(
        self,
        *,
        speculative_config: Any | None,
        target_hf_config: Any | None,
        target_model_args: Mapping[str, Any],
    ) -> Gemma4MTPAssistantRuntime | None:
        """Load the Gemma4 MTP assistant configured for this target model."""
        if not Gemma4MTPAssistantSource.is_gemma4_mtp(speculative_config):
            return None

        source = Gemma4MTPAssistantSource.from_speculative_config(
            speculative_config
        ).resolve(self._model_path_resolver)
        target_metadata = Gemma4MTPTargetMetadata.from_configs(
            target_hf_config=target_hf_config,
            target_model_args=target_model_args,
        )

        cached = self._cached_runtime(source)
        if cached is not None:
            cached.metadata.validate_compatible_with(target_metadata)
            logger.info("Gemma4 MTP assistant loaded from cache: %s", source.model_name)
            return cached

        return self._load_uncached(source, target_metadata)

    def _load_uncached(
        self,
        source: Gemma4MTPAssistantSource,
        target_metadata: Gemma4MTPTargetMetadata,
    ) -> Gemma4MTPAssistantRuntime:
        logger.info("Loading Gemma4 MTP assistant: %s", source.model_name)
        start_time = time.time()
        model_path = self._download_model(source)
        self._preflight_config(model_path, target_metadata)
        model, assistant_config = self._load_assistant_model(model_path)
        metadata = self._metadata_from_config(assistant_config, target_metadata)
        runtime = Gemma4MTPAssistantRuntime(
            model_name=source.model_name,
            model=model,
            metadata=metadata,
        )
        with _ASSISTANT_CACHE_LOCK:
            _ASSISTANT_CACHE[source.cache_key] = runtime
        logger.info(
            "Gemma4 MTP assistant loaded in %.2fs: %s",
            time.time() - start_time,
            source.model_name,
        )
        return runtime

    def _cached_runtime(
        self,
        source: Gemma4MTPAssistantSource,
    ) -> Gemma4MTPAssistantRuntime | None:
        with _ASSISTANT_CACHE_LOCK:
            return _ASSISTANT_CACHE.get(source.cache_key)

    def _preflight_config(
        self,
        model_path: Path,
        target_metadata: Gemma4MTPTargetMetadata,
    ) -> None:
        assistant_config = self._read_config_file(model_path)
        if assistant_config is None and self._load_model is None:
            raise ValueError(
                "Gemma4 MTP assistant model path must contain config.json: "
                f"{model_path}"
            )
        if assistant_config is not None:
            self._metadata_from_config(assistant_config, target_metadata)

    def _metadata_from_config(
        self,
        assistant_config: Mapping[str, Any],
        target_metadata: Gemma4MTPTargetMetadata,
    ) -> Gemma4MTPAssistantMetadata:
        self._reject_custom_model_file(assistant_config)
        metadata = Gemma4MTPAssistantMetadata.from_config(assistant_config)
        metadata.validate_compatible_with(target_metadata)
        return metadata

    def _load_assistant_model(
        self,
        model_path: Path,
    ) -> tuple[Any, dict[str, Any]]:
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

    def _download_model(self, source: Gemma4MTPAssistantSource) -> Path:
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
                allow_patterns=_ASSISTANT_DOWNLOAD_ALLOW_PATTERNS,
            )
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
                f"Gemma4 MTP assistant config.json is not valid JSON: {config_path}"
            ) from exc
        if not isinstance(config, dict):
            raise ValueError("Gemma4 MTP assistant config.json must contain an object")
        return config

    @staticmethod
    def _reject_custom_model_file(config: Mapping[str, Any]) -> None:
        if "model_file" in config:
            model_file = config["model_file"]
            raise ValueError(
                "Gemma4 MTP assistant loader uses built-in Metal model classes and "
                f"does not support custom model_file={model_file!r}"
            )

    @classmethod
    def _get_model_classes(
        cls,
        config: dict[str, Any],
    ) -> tuple[type[Any], type[Any]]:
        if config.get("model_type") not in GEMMA4_MTP_DRAFT_MODEL_TYPES:
            model_type = config.get("model_type")
            architectures = config.get("architectures")
            raise ValueError(
                "Gemma4 MTP assistant loader only supports gemma4_assistant/"
                f"gemma4_mtp configs, got model_type={model_type!r}, "
                f"architectures={architectures!r}"
            )
        return cls._assistant_model_classes()

    @staticmethod
    def _assistant_model_classes() -> tuple[type[Any], type[Any]]:
        # Keep MLX model imports lazy so config detection and spec-decode
        # metadata tests do not construct Metal-backed modules just by
        # importing this owner.
        from vllm_metal.v1.gemma4_mtp_model import (
            Gemma4MTPAssistantModel,
            Gemma4MTPAssistantModelArgs,
        )

        return Gemma4MTPAssistantModel, Gemma4MTPAssistantModelArgs


@dataclass(frozen=True, slots=True)
class Gemma4MTPAssistantSource:
    """Resolved draft assistant source from vLLM speculative config."""

    model_name: str
    revision: str | None

    @classmethod
    def is_gemma4_mtp(cls, speculative_config: Any | None) -> bool:
        if (
            speculative_config is None
            or getattr(speculative_config, "method", None) != "mtp"
        ):
            return False

        draft_model_config = getattr(speculative_config, "draft_model_config", None)
        if draft_model_config is None:
            return False

        hf_config = getattr(draft_model_config, "hf_config", None)
        return Gemma4MTPAssistantMetadata.is_assistant_config(hf_config)

    @classmethod
    def from_speculative_config(
        cls,
        speculative_config: Any,
    ) -> Gemma4MTPAssistantSource:
        draft_model_config = getattr(speculative_config, "draft_model_config", None)
        model_name = getattr(draft_model_config, "model", None) or getattr(
            speculative_config,
            "model",
            None,
        )
        if not model_name:
            raise ValueError("Gemma4 MTP speculative config is missing the draft model")

        revision = getattr(draft_model_config, "revision", None) or getattr(
            speculative_config,
            "revision",
            None,
        )
        return cls(
            model_name=str(model_name),
            revision=str(revision) if revision else None,
        )

    def resolve(
        self,
        model_path_resolver: Callable[[str], str],
    ) -> Gemma4MTPAssistantSource:
        return Gemma4MTPAssistantSource(
            model_name=model_path_resolver(self.model_name),
            revision=self.revision,
        )

    @property
    def cache_key(self) -> tuple[str, str | None]:
        return (self.model_name, self.revision)


@dataclass(frozen=True, slots=True)
class Gemma4MTPTargetMetadata:
    """Validated target model metadata needed by Gemma4 MTP."""

    vocab_size: int
    hidden_size: int
    non_shared_layer_types: tuple[str, ...]

    @classmethod
    def from_configs(
        cls,
        *,
        target_hf_config: Any | None,
        target_model_args: Mapping[str, Any],
    ) -> Gemma4MTPTargetMetadata:
        target_text_config = _text_config(target_hf_config)
        cls._validate_model_types(target_text_config, target_model_args)
        return cls(
            vocab_size=cls._required_positive_int(
                "vocab_size",
                target_text_config,
                target_model_args,
            ),
            hidden_size=cls._required_positive_int(
                "hidden_size",
                target_text_config,
                target_model_args,
            ),
            non_shared_layer_types=cls._non_shared_layer_types(
                target_text_config,
                target_model_args,
            ),
        )

    @staticmethod
    def _validate_model_types(
        target_text_config: Any | None,
        target_model_args: Mapping[str, Any],
    ) -> None:
        model_types = {
            model_type
            for model_type in (
                _field(target_text_config, "model_type"),
                target_model_args.get("model_type"),
            )
            if model_type is not None
        }
        if not model_types:
            raise ValueError(
                "Gemma4 MTP assistant requires a Gemma4 target model, "
                "got model_type=None"
            )
        unknown_model_types = sorted(model_types - GEMMA4_TARGET_MODEL_TYPES)
        if unknown_model_types:
            raise ValueError(
                "Gemma4 MTP assistant requires a Gemma4 target model, got "
                f"model_type={unknown_model_types[0]!r}"
            )

    @classmethod
    def _required_positive_int(
        cls,
        key: str,
        target_text_config: Any | None,
        target_model_args: Mapping[str, Any],
    ) -> int:
        value = cls._matching_optional_int(key, target_text_config, target_model_args)
        if value is None:
            raise ValueError(f"Missing target model {key}")
        if value <= 0:
            raise ValueError(f"target model {key} must be positive, got {value}")
        return value

    @classmethod
    def _non_shared_layer_types(
        cls,
        target_text_config: Any | None,
        target_model_args: Mapping[str, Any],
    ) -> tuple[str, ...]:
        layer_types = cls._matching_sequence(
            "layer_types",
            target_text_config,
            target_model_args,
        )
        if not layer_types:
            raise ValueError("Gemma4 MTP target model must expose layer_types")
        unknown_layer_types = sorted(set(layer_types) - GEMMA4_MTP_VALID_LAYER_TYPES)
        if unknown_layer_types:
            raise ValueError(
                f"Unsupported Gemma4 MTP target layer types: {unknown_layer_types}"
            )

        num_hidden_layers = cls._matching_optional_int(
            "num_hidden_layers",
            target_text_config,
            target_model_args,
        )
        if num_hidden_layers is not None and len(layer_types) != num_hidden_layers:
            raise ValueError(
                "Gemma4 MTP target layer_types must match num_hidden_layers: "
                f"len(layer_types)={len(layer_types)}, "
                f"num_hidden_layers={num_hidden_layers}"
            )

        num_kv_shared_layers = cls._matching_optional_int(
            "num_kv_shared_layers",
            target_text_config,
            target_model_args,
        )
        if num_kv_shared_layers is None:
            num_kv_shared_layers = 0
        if num_kv_shared_layers < 0 or num_kv_shared_layers >= len(layer_types):
            raise ValueError(
                "Gemma4 MTP target num_kv_shared_layers must leave at least one "
                f"non-shared KV layer: num_kv_shared_layers={num_kv_shared_layers}, "
                f"num_layers={len(layer_types)}"
            )

        num_non_shared = len(layer_types) - num_kv_shared_layers
        return layer_types[:num_non_shared]

    @staticmethod
    def _matching_optional_int(
        key: str,
        target_text_config: Any | None,
        target_model_args: Mapping[str, Any],
    ) -> int | None:
        return _matching_target_value(
            key,
            target_text_config,
            target_model_args,
            parse=lambda config, field: _optional_int(
                config,
                field,
                context="target model",
            ),
        )

    @staticmethod
    def _matching_sequence(
        key: str,
        target_text_config: Any | None,
        target_model_args: Mapping[str, Any],
    ) -> tuple[str, ...]:
        return (
            _matching_target_value(
                key,
                target_text_config,
                target_model_args,
                parse=_sequence_field,
            )
            or ()
        )


@dataclass(frozen=True, slots=True)
class Gemma4MTPAssistantMetadata:
    """Validated Gemma4 MTP assistant shape."""

    model_type: str
    architectures: tuple[str, ...]
    vocab_size: int
    hidden_size: int
    backbone_hidden_size: int
    tie_word_embeddings: bool
    num_hidden_layers: int
    layer_types: tuple[str, ...]
    use_ordered_embeddings: bool

    @classmethod
    def is_assistant_config(cls, config: Any | None) -> bool:
        """Accept raw assistant configs and upstream vLLM's wrapper config."""
        if config is None:
            return False
        model_type = _field(config, "model_type")
        if model_type in GEMMA4_MTP_DRAFT_MODEL_TYPES:
            return True
        return any(
            arch in GEMMA4_MTP_DRAFT_ARCHITECTURES
            for arch in _architectures(config, strict=False)
        )

    @classmethod
    def from_config(cls, config: Any) -> Gemma4MTPAssistantMetadata:
        model_type = _config_value(config, "model_type")
        if model_type not in GEMMA4_MTP_DRAFT_MODEL_TYPES:
            raise ValueError(
                "Gemma4 MTP assistant requires a gemma4_assistant or gemma4_mtp "
                f"config, got model_type={model_type!r}"
            )

        text_config = _text_config(config)
        text_model_type = _config_value(text_config, "model_type")
        if text_model_type != GEMMA4_MTP_TEXT_MODEL_TYPE:
            raise ValueError(
                "Gemma4 MTP assistant text_config.model_type must be "
                f"{GEMMA4_MTP_TEXT_MODEL_TYPE!r}, got {text_model_type!r}"
            )

        vocab_size = _required_positive_int(
            text_config,
            "vocab_size",
            context="assistant",
        )
        top_level_vocab_size = _optional_positive_int(
            config,
            "vocab_size",
            context="assistant",
        )
        if top_level_vocab_size is not None and top_level_vocab_size != vocab_size:
            raise ValueError(
                "Gemma4 MTP assistant vocab_size metadata mismatch: "
                f"top-level={top_level_vocab_size}, text_config={vocab_size}"
            )

        hidden_size = _required_positive_int(
            text_config,
            "hidden_size",
            context="assistant",
        )
        backbone_hidden_size = _required_positive_int(
            config,
            "backbone_hidden_size",
            context="assistant",
        )
        num_hidden_layers = _required_positive_int(
            text_config,
            "num_hidden_layers",
            context="assistant",
        )
        layer_types = _required_layer_types(text_config, num_hidden_layers)
        architectures = _architectures(config, strict=True)
        if not any(arch in GEMMA4_MTP_DRAFT_ARCHITECTURES for arch in architectures):
            raise ValueError(
                "Gemma4 MTP assistant requires a Gemma4 MTP architecture, got "
                f"architectures={architectures!r}"
            )

        n_predict = _optional_int(config, "n_predict", context="assistant")
        if n_predict is not None and n_predict != GEMMA4_MTP_N_PREDICT:
            raise ValueError(
                "Gemma4 MTP assistant config must use "
                f"n_predict={GEMMA4_MTP_N_PREDICT}, got {n_predict!r}"
            )
        tie_word_embeddings = _optional_bool(
            config,
            "tie_word_embeddings",
            context="assistant",
            default=True,
        )
        use_ordered_embeddings = _optional_bool(
            config,
            "use_ordered_embeddings",
            context="assistant",
            default=False,
        )
        cls._validate_ordered_embedding_config(
            config,
            vocab_size=vocab_size,
            use_ordered_embeddings=use_ordered_embeddings,
        )

        return cls(
            model_type=str(model_type),
            architectures=architectures,
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            backbone_hidden_size=backbone_hidden_size,
            tie_word_embeddings=tie_word_embeddings,
            num_hidden_layers=num_hidden_layers,
            layer_types=layer_types,
            use_ordered_embeddings=use_ordered_embeddings,
        )

    @staticmethod
    def _validate_ordered_embedding_config(
        config: Any,
        *,
        vocab_size: int,
        use_ordered_embeddings: bool,
    ) -> None:
        num_centroids = _optional_positive_int(
            config,
            "num_centroids",
            context="assistant",
        )
        centroid_intermediate_top_k = _optional_positive_int(
            config,
            "centroid_intermediate_top_k",
            context="assistant",
        )
        if not use_ordered_embeddings:
            return

        num_centroids = (
            GEMMA4_MTP_DEFAULT_NUM_CENTROIDS if num_centroids is None else num_centroids
        )
        centroid_intermediate_top_k = (
            GEMMA4_MTP_DEFAULT_CENTROID_TOP_K
            if centroid_intermediate_top_k is None
            else centroid_intermediate_top_k
        )
        if vocab_size % num_centroids != 0:
            raise ValueError(
                "Gemma4 MTP assistant vocab_size must be divisible by "
                f"num_centroids: vocab_size={vocab_size}, "
                f"num_centroids={num_centroids}"
            )
        if centroid_intermediate_top_k > num_centroids:
            raise ValueError(
                "Gemma4 MTP assistant centroid_intermediate_top_k must be <= "
                f"num_centroids: centroid_intermediate_top_k="
                f"{centroid_intermediate_top_k}, num_centroids={num_centroids}"
            )

    def validate_compatible_with(self, target: Gemma4MTPTargetMetadata) -> None:
        if self.vocab_size != target.vocab_size:
            raise ValueError(
                "Gemma4 MTP assistant vocab size must match target vocab size: "
                f"assistant={self.vocab_size}, target={target.vocab_size}"
            )
        if self.backbone_hidden_size != target.hidden_size:
            raise ValueError(
                "Gemma4 MTP assistant backbone hidden size must match target "
                f"hidden size: assistant={self.backbone_hidden_size}, "
                f"target={target.hidden_size}"
            )

        if len(self.layer_types) > len(target.non_shared_layer_types):
            raise ValueError(
                "Gemma4 MTP assistant has more layers than target non-shared "
                f"KV layers: assistant_layer_types={self.layer_types}, "
                f"target_non_shared_layer_types={target.non_shared_layer_types}"
            )

        target_tail = target.non_shared_layer_types[-len(self.layer_types) :]
        if self.layer_types != target_tail:
            raise ValueError(
                "Gemma4 MTP assistant layer types must tail-match target "
                "non-shared KV layer types: "
                f"assistant_layer_types={self.layer_types}, "
                f"target_tail_layer_types={target_tail}"
            )


def reset_gemma4_mtp_assistant_cache() -> None:
    """Clear the process-level Gemma4 MTP assistant cache."""
    with _ASSISTANT_CACHE_LOCK:
        _ASSISTANT_CACHE.clear()


def uses_gemma4_mtp(speculative_config: Any | None) -> bool:
    """Return whether speculative config points at Gemma4 MTP."""
    return Gemma4MTPAssistantSource.is_gemma4_mtp(speculative_config)


def is_gemma4_mtp_assistant_config(config: Any | None) -> bool:
    """Return whether a config shape looks like a Gemma4 MTP assistant."""
    return Gemma4MTPAssistantMetadata.is_assistant_config(config)


def validate_gemma4_mtp_assistant_config(
    assistant_config: Any,
    *,
    target_hf_config: Any | None,
    target_model_args: Mapping[str, Any],
) -> Gemma4MTPAssistantMetadata:
    """Validate assistant config compatibility with the loaded target."""
    target_metadata = Gemma4MTPTargetMetadata.from_configs(
        target_hf_config=target_hf_config,
        target_model_args=target_model_args,
    )
    if isinstance(assistant_config, Gemma4MTPAssistantMetadata):
        metadata = assistant_config
    else:
        metadata = Gemma4MTPAssistantMetadata.from_config(assistant_config)
    metadata.validate_compatible_with(target_metadata)
    return metadata


def _required_layer_types(config: Any, num_hidden_layers: int) -> tuple[str, ...]:
    layer_types = _sequence_field(config, "layer_types")
    if len(layer_types) != num_hidden_layers:
        raise ValueError(
            "Gemma4 MTP assistant layer_types must match num_hidden_layers: "
            f"len(layer_types)={len(layer_types)}, "
            f"num_hidden_layers={num_hidden_layers}"
        )
    unknown = sorted(set(layer_types) - GEMMA4_MTP_VALID_LAYER_TYPES)
    if unknown:
        raise ValueError(f"Unsupported Gemma4 MTP assistant layer types: {unknown}")
    return layer_types


def _matching_target_value[T](
    key: str,
    target_text_config: Any | None,
    target_model_args: Mapping[str, Any],
    *,
    parse: Callable[[Any, str], T],
) -> T | None:
    sources: list[tuple[str, T]] = []
    if key in target_model_args and target_model_args.get(key) is not None:
        sources.append(("target_model_args", parse(target_model_args, key)))
    if _field(target_text_config, key) is not None:
        sources.append(("target_hf_config.text_config", parse(target_text_config, key)))
    if not sources:
        return None

    _, value = sources[0]
    for label, other in sources[1:]:
        if other != value:
            raise ValueError(
                f"Gemma4 MTP target {key} metadata mismatch: "
                f"{sources[0][0]}={value}, {label}={other}"
            )
    return value


def _required_int(config: Any, key: str, *, context: str) -> int:
    value = _config_value(config, key)
    if value is None:
        raise ValueError(f"Missing {context} {key}")
    return _coerce_int(value, key=key, context=context)


def _optional_int(config: Any, key: str, *, context: str) -> int | None:
    value = _config_value(config, key)
    if value is None:
        return None
    return _coerce_int(value, key=key, context=context)


def _optional_positive_int(config: Any, key: str, *, context: str) -> int | None:
    value = _optional_int(config, key, context=context)
    if value is None:
        return None
    if value <= 0:
        raise ValueError(f"{context} {key} must be positive, got {value}")
    return value


def _optional_bool(config: Any, key: str, *, context: str, default: bool) -> bool:
    value = _config_value(config, key, default)
    if not isinstance(value, bool):
        raise ValueError(f"{context} {key} must be a boolean, got {value!r}")
    return value


def _coerce_int(value: Any, *, key: str, context: str) -> int:
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{context} {key} must be an integer, got {value!r}")
    return int(value)


def _required_positive_int(config: Any, key: str, *, context: str) -> int:
    value = _required_int(config, key, context=context)
    if value <= 0:
        raise ValueError(f"{context} {key} must be positive, got {value}")
    return value


def _text_config(config: Any | None) -> Any | None:
    if config is None:
        return None
    get_text_config = getattr(config, "get_text_config", None)
    if callable(get_text_config):
        return get_text_config()
    return _field(config, "text_config", config)


def _architectures(config: Any, *, strict: bool) -> tuple[str, ...]:
    architectures = _config_value(config, "architectures", ()) or ()
    if isinstance(architectures, str):
        if strict:
            raise ValueError("assistant architectures must be a non-string sequence")
        return (architectures,)
    if not isinstance(architectures, Sequence):
        if strict:
            raise ValueError("assistant architectures must be a sequence")
        return ()
    names: list[str] = []
    for arch in architectures:
        if not isinstance(arch, str):
            if strict:
                raise ValueError("assistant architectures entries must be strings")
            continue
        names.append(arch)
    return tuple(names)


def _sequence_field(config: Any, key: str) -> tuple[str, ...]:
    value = _config_value(config, key, ()) or ()
    if isinstance(value, str):
        raise ValueError(f"{key} must be a non-string sequence")
    if not isinstance(value, Sequence):
        raise ValueError(f"{key} must be a sequence")
    if any(not isinstance(item, str) for item in value):
        raise ValueError(f"{key} entries must be strings")
    return tuple(value)


def _field(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return getattr(config, key, default)


def _config_value(config: Any, key: str, default: Any = None) -> Any:
    if isinstance(config, Mapping):
        return config.get(key, default)
    return _field(config, key, default)
