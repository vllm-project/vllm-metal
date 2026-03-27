# SPDX-License-Identifier: Apache-2.0
"""Model-type detection helpers for STT boundary decisions."""

from __future__ import annotations

import json
import logging
from pathlib import Path

try:
    from huggingface_hub import hf_hub_download
except ImportError:  # pragma: no cover
    hf_hub_download = None

logger = logging.getLogger(__name__)

# STT model types that can be auto-detected from config.json.
_STT_MODEL_TYPES = frozenset({"whisper", "qwen3_asr"})


def _resolve_config_file(model_path: str) -> Path | None:
    """Return local or downloaded config.json path for *model_path*."""
    p = Path(model_path)
    if p.is_dir():
        config_file = p / "config.json"
        if config_file.exists():
            return config_file
        logger.debug("No config.json found in local model directory: %s", model_path)

    if hf_hub_download is None:
        logger.debug(
            "huggingface_hub not installed; cannot resolve remote model config: %s",
            model_path,
        )
        return None

    try:
        return Path(hf_hub_download(repo_id=model_path, filename="config.json"))
    except (OSError, ValueError) as exc:
        logger.debug("Failed to download config.json for %s: %s", model_path, exc)
        return None


def _read_model_type(config_file: Path) -> str | None:
    """Read model_type from config.json, returning None on parse/read failure."""
    try:
        with open(config_file) as f:
            cfg = json.load(f)
    except (OSError, json.JSONDecodeError) as exc:
        logger.debug("Failed reading model config %s: %s", config_file, exc)
        return None

    model_type = cfg.get("model_type")
    if not isinstance(model_type, str):
        logger.debug("Config %s missing string model_type", config_file)
        return None
    return model_type.lower()


def is_stt_model(model_path: str) -> bool:
    """Return True when *model_path* resolves to a known STT model type.

    Detection is based on ``model_type`` in the model's ``config.json``.
    Falls back to ``False`` if the config cannot be read.
    """
    config_file = _resolve_config_file(model_path)
    if config_file is None:
        return False
    model_type = _read_model_type(config_file)
    return model_type in _STT_MODEL_TYPES
