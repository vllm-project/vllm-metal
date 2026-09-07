# SPDX-License-Identifier: Apache-2.0
"""Resolved GGUF load identity carried from vLLM config to the loader."""

from __future__ import annotations

import itertools
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Self

from huggingface_hub import HfApi, snapshot_download
from huggingface_hub.utils import filter_repo_objects

_GGUF_SUFFIX = ".gguf"
_REMOTE_REF_RE = re.compile(
    r"^[a-zA-Z0-9][a-zA-Z0-9._-]*/[a-zA-Z0-9][a-zA-Z0-9._-]*:[A-Za-z0-9_+-]+$"
)
_QUANT_TAG_RE = re.compile(
    r"(?:^|-)(?:I?Q\d[A-Za-z0-9_]*|F16|F32|BF16|MXFP\d[A-Za-z0-9_]*)$",
    re.IGNORECASE,
)
_REMOTE_PREFIXES = ("*.", "*-")
_REMOTE_SUFFIXES = ("-*", "")
_SHARD_NAME_RE = re.compile(r"-\d+-of-(\d+)\.gguf$")
_SUPPORTED_REMOTE_QTYPES = frozenset({"Q8_0", "Q4_0", "Q4_1"})
_CONFIG_ALLOW_PATTERNS = ("config.json", "generation_config.json")
_TOKENIZER_ALLOW_PATTERNS = (
    "*.json",
    "*.py",
    "tokenizer.model",
    "*.tiktoken",
    "tiktoken.model",
    "*.txt",
    "*.jsonl",
    "*.jinja",
)


@dataclass(frozen=True, slots=True)
class RemoteGGUFReference:
    """Hugging Face ``repo_id:quant`` GGUF weight reference."""

    repo_id: str
    quant_type: str

    @classmethod
    def parse(cls, value: str) -> Self | None:
        if not _REMOTE_REF_RE.fullmatch(value):
            return None
        repo_id, quant_type = value.rsplit(":", 1)
        if _QUANT_TAG_RE.search(quant_type) is None:
            return None
        return cls(repo_id=repo_id, quant_type=quant_type)

    @property
    def value(self) -> str:
        return f"{self.repo_id}:{self.quant_type}"

    @property
    def allow_patterns(self) -> tuple[str, ...]:
        return tuple(
            f"{prefix}{normalized_quant}{suffix}{_GGUF_SUFFIX}"
            for normalized_quant in (self.quant_type.upper(), self.quant_type.lower())
            for prefix, suffix in itertools.product(_REMOTE_PREFIXES, _REMOTE_SUFFIXES)
        )

    def resolve(
        self,
        *,
        cache_dir: str | None,
        revision: str | None,
        ignore_patterns: list[str] | str | None,
        token: bool | str | None,
    ) -> str:
        if self.quant_type.upper() not in _SUPPORTED_REMOTE_QTYPES:
            supported = ", ".join(sorted(_SUPPORTED_REMOTE_QTYPES))
            raise ValueError(
                f"Remote GGUF qtype {self.quant_type!r} is not supported by "
                f"vllm-metal; supported qtypes: {supported}."
            )
        repo_files = HfApi().list_repo_files(
            repo_id=self.repo_id,
            revision=revision,
            token=token,
        )
        filename = self._select_single_filename(
            sorted(
                filter_repo_objects(
                    repo_files,
                    allow_patterns=list(self.allow_patterns),
                    ignore_patterns=ignore_patterns,
                )
            )
        )
        snapshot_dir = Path(
            snapshot_download(
                repo_id=self.repo_id,
                cache_dir=cache_dir,
                allow_patterns=[filename],
                revision=revision,
                token=token,
            )
        )
        return str(snapshot_dir / filename)

    def _select_single_filename(self, filenames: list[str]) -> str:
        if not filenames:
            raise ValueError(
                f"No {self.quant_type!r} GGUF file found in remote repository "
                f"{self.repo_id!r}."
            )
        if any(_SHARD_NAME_RE.search(filename) for filename in filenames):
            raise ValueError(
                f"Remote sharded GGUF files are not supported yet: {self.value!r}."
            )
        if len(filenames) != 1:
            names = ", ".join(filenames)
            raise ValueError(
                f"Remote GGUF reference {self.value!r} matched multiple files: {names}."
            )
        return filenames[0]


@dataclass(frozen=True, slots=True)
class GGUFLoadSource:
    """Local GGUF weights plus companion config/tokenizer sources."""

    weights_path: str
    config_dir: str
    tokenizer_dir: str

    @classmethod
    def from_model_config(
        cls,
        model_config: Any,
        load_config: Any | None = None,
    ) -> GGUFLoadSource | None:
        if model_config.quantization != "gguf":
            return None

        weights_ref = model_config.model_weights
        cache_dir = None if load_config is None else load_config.download_dir
        ignore_patterns = None if load_config is None else load_config.ignore_patterns
        token = model_config.hf_token
        if cls.is_weights_path(weights_ref):
            weights_path = weights_ref
        elif remote_ref := RemoteGGUFReference.parse(weights_ref):
            weights_path = remote_ref.resolve(
                cache_dir=cache_dir,
                revision=model_config.revision,
                ignore_patterns=ignore_patterns,
                token=token,
            )
        else:
            raise ValueError(
                "GGUF model_config must carry a local .gguf path or remote "
                f"repo_id:quant reference in model_weights; got {weights_ref!r}."
            )

        config_dir = cls._resolve_companion_source(
            model_config.model,
            cache_dir=cache_dir,
            revision=model_config.revision,
            token=token,
            allow_patterns=_CONFIG_ALLOW_PATTERNS,
        )
        tokenizer_dir = cls._resolve_companion_source(
            model_config.tokenizer or model_config.model,
            cache_dir=cache_dir,
            revision=model_config.tokenizer_revision or model_config.revision,
            token=token,
            allow_patterns=_TOKENIZER_ALLOW_PATTERNS,
        )
        return cls(
            weights_path=weights_path,
            config_dir=config_dir,
            tokenizer_dir=tokenizer_dir,
        )

    @staticmethod
    def is_weights_path(value: str) -> bool:
        return value.endswith(_GGUF_SUFFIX)

    @staticmethod
    def is_sharded_weights_path(value: str) -> bool:
        # A -00001-of-00001.gguf set is one complete file and loads as such.
        match = _SHARD_NAME_RE.search(value)
        return match is not None and int(match.group(1)) > 1

    @staticmethod
    def _resolve_companion_source(
        source: str,
        *,
        cache_dir: str | None,
        revision: str | None,
        token: bool | str | None,
        allow_patterns: tuple[str, ...],
    ) -> str:
        source_path = Path(source)
        if source_path.exists() or source_path.is_absolute():
            return source
        return snapshot_download(
            repo_id=source,
            cache_dir=cache_dir,
            allow_patterns=list(allow_patterns),
            revision=revision,
            token=token,
        )
