# SPDX-License-Identifier: Apache-2.0
"""GGUF model reference resolution.

The first supported mode is intentionally local-only. It accepts a single
``.gguf`` file, a complete local ``*-NN-of-MM.gguf`` shard group, or a directory
containing exactly one local GGUF file/group, then resolves a paired HF/MLX
config/tokenizer directory.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_QUANT_SUFFIX_RE = re.compile(
    r"(?i)(?:[-_.](?:UD-)?(?:IQ[1-4]_[A-Z0-9]+|Q[2-8](?:_[0-9A-Z]+)*|BF16|F16|F32))+$"
)
_SHARD_RE = re.compile(
    r"^(?P<prefix>.*)-(?P<index>\d+)-of-(?P<count>\d+)\.gguf$",
    re.IGNORECASE,
)
_SHARD_STEM_RE = re.compile(r"^(?P<prefix>.*)-\d+-of-\d+$", re.IGNORECASE)


@dataclass(frozen=True)
class GGUFReference:
    gguf_path: Path
    model_path: Path
    gguf_paths: tuple[Path, ...] = ()

    @property
    def all_gguf_paths(self) -> tuple[Path, ...]:
        return self.gguf_paths or (self.gguf_path,)

    def cache_key(self) -> tuple[str, str]:
        gguf_key = "\x1f".join(str(path) for path in self.all_gguf_paths)
        return (gguf_key, f"gguf-dense:{self.model_path}")


def is_local_gguf(model_name: str | Path) -> bool:
    path = Path(model_name)
    try:
        return bool(_resolve_gguf_files(path))
    except ValueError:
        return False


def resolve_gguf_reference(
    model_name: str | Path,
    *,
    model_config: Any | None = None,
) -> GGUFReference:
    path = Path(model_name)
    gguf_paths = _resolve_gguf_files(path)
    gguf_path = gguf_paths[0]
    model_path = _resolve_paired_model_path(gguf_path, model_config=model_config)
    return GGUFReference(
        gguf_path=gguf_path,
        model_path=model_path,
        gguf_paths=gguf_paths,
    )


def _resolve_gguf_files(path: Path) -> tuple[Path, ...]:
    if path.is_file():
        if not _has_gguf_header(path):
            raise ValueError(f"Not a GGUF file: {path}")
        return _resolve_file_with_shards(path)

    if path.is_dir():
        gguf_files = _gguf_files_in_dir(path)
        if len(gguf_files) == 1:
            return _resolve_file_with_shards(gguf_files[0])
        if not gguf_files:
            raise ValueError(f"No GGUF files found in directory: {path}")

        resolved = _resolve_directory_gguf_group(gguf_files)
        if resolved is not None:
            return resolved
        raise ValueError(
            f"Directory contains multiple GGUF files or shard groups; "
            f"pass one group explicitly: {path}"
        )

    raise ValueError(f"GGUF path does not exist: {path}")


def _resolve_file_with_shards(path: Path) -> tuple[Path, ...]:
    match = _SHARD_RE.match(path.name)
    if match is None:
        return (path,)

    prefix = match.group("prefix")
    count = int(match.group("count"))
    shards = [
        candidate
        for candidate in path.parent.iterdir()
        if candidate.is_file()
        and candidate.name.startswith(f"{prefix}-")
        and candidate.name.endswith(f"-of-{match.group('count')}.gguf")
        and _has_gguf_header(candidate)
    ]
    return _validate_shard_group(shards, expected_count=count, source=path)


def _resolve_directory_gguf_group(files: list[Path]) -> tuple[Path, ...] | None:
    standalone: list[Path] = []
    shard_groups: dict[tuple[str, str], list[Path]] = {}
    expected_counts: dict[tuple[str, str], int] = {}

    for file in files:
        match = _SHARD_RE.match(file.name)
        if match is None:
            standalone.append(file)
            continue
        key = (match.group("prefix"), match.group("count"))
        shard_groups.setdefault(key, []).append(file)
        expected_counts[key] = int(match.group("count"))

    candidates: list[tuple[Path, ...]] = []
    if len(standalone) == 1:
        candidates.append((standalone[0],))
    elif len(standalone) > 1:
        return None

    for key, group in shard_groups.items():
        try:
            candidates.append(
                _validate_shard_group(
                    group,
                    expected_count=expected_counts[key],
                    source=group[0],
                )
            )
        except ValueError:
            return None

    if len(candidates) == 1:
        return candidates[0]
    return None


def _validate_shard_group(
    shards: list[Path],
    *,
    expected_count: int,
    source: Path,
) -> tuple[Path, ...]:
    indexed: dict[int, Path] = {}
    for shard in shards:
        match = _SHARD_RE.match(shard.name)
        if match is None:
            continue
        index = int(match.group("index"))
        count = int(match.group("count"))
        if count != expected_count or index < 1 or index > expected_count:
            continue
        indexed[index] = shard

    expected = set(range(1, expected_count + 1))
    if set(indexed) != expected:
        missing = ", ".join(str(i) for i in sorted(expected - set(indexed)))
        raise ValueError(
            f"Incomplete GGUF shard group for {source}; missing shard index: {missing}"
        )
    return tuple(indexed[index] for index in sorted(indexed))


def _resolve_paired_model_path(
    gguf_path: Path,
    *,
    model_config: Any | None,
) -> Path:
    for candidate in _paired_model_candidates(gguf_path, model_config):
        if (candidate / "config.json").is_file():
            return candidate

    candidates = ", ".join(
        str(p) for p in _paired_model_candidates(gguf_path, model_config)
    )
    raise ValueError(
        "GGUF loading needs a paired HF/MLX config and tokenizer directory. "
        f"Tried: {candidates}"
    )


def _paired_model_candidates(
    gguf_path: Path,
    model_config: Any | None,
) -> list[Path]:
    candidates: list[Path] = []

    if model_config is not None:
        for attr in ("hf_config_path", "tokenizer", "served_model_name"):
            value = getattr(model_config, attr, None)
            if isinstance(value, (str, Path)):
                candidates.append(Path(value))

    candidates.append(gguf_path.parent)

    stem = _strip_quant_suffix(_strip_shard_suffix(gguf_path.stem))
    for base in (gguf_path.parent, gguf_path.parent.parent):
        if base:
            candidates.append(base / stem)

    deduped: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        expanded = candidate.expanduser()
        if expanded not in seen:
            seen.add(expanded)
            deduped.append(expanded)
    return deduped


def _strip_quant_suffix(stem: str) -> str:
    stripped = _QUANT_SUFFIX_RE.sub("", stem)
    return stripped or stem


def _strip_shard_suffix(stem: str) -> str:
    match = _SHARD_STEM_RE.match(stem)
    return match.group("prefix") if match is not None else stem


def _gguf_files_in_dir(path: Path) -> list[Path]:
    return sorted(p for p in path.glob("*.gguf") if _has_gguf_header(p))


def _has_gguf_header(path: Path) -> bool:
    if path.suffix.lower() != ".gguf":
        return False
    try:
        with path.open("rb") as f:
            return f.read(4) == b"GGUF"
    except OSError:
        return False
