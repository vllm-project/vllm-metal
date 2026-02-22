# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text configuration and language constants."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

logger = logging.getLogger(__name__)

# STT model types that can be auto-detected from config.json
_STT_MODEL_TYPES = {"whisper"}

# Officially supported languages (OpenAI docs subset).
# fmt: off
_OFFICIALLY_SUPPORTED = {
    "af", "ar", "hy", "az", "be", "bs", "bg", "ca", "zh", "hr", "cs", "da",
    "nl", "en", "et", "fi", "fr", "gl", "de", "el", "he", "hi", "hu", "is",
    "id", "it", "ja", "kn", "kk", "ko", "lv", "lt", "mk", "ms", "mr", "mi",
    "ne", "no", "fa", "pl", "pt", "ro", "ru", "sr", "sk", "sl", "es", "sw",
    "sv", "tl", "ta", "th", "tr", "uk", "ur", "vi", "cy",
}
# fmt: on


@lru_cache(maxsize=1)
def _get_whisper_languages() -> dict[str, str]:
    """Get Whisper language map from transformers."""
    try:
        from transformers.models.whisper.tokenization_whisper import LANGUAGES

        return dict(LANGUAGES)
    except ImportError:
        logger.warning("transformers not available, using fallback language list")
        return {"en": "english"}


def get_whisper_languages() -> dict[str, str]:
    """Get full Whisper language map (100 languages)."""
    return _get_whisper_languages()


def get_supported_languages() -> set[str]:
    """Get officially supported language codes."""
    return _OFFICIALLY_SUPPORTED.copy()


@dataclass
class SpeechToTextConfig:
    """Runtime configuration for STT processing.

    Controls audio chunking and energy-based splitting parameters.
    """

    max_audio_clip_s: float = 30.0
    overlap_chunk_second: float = 1.0
    min_energy_split_window_size: int = 1600
    # Deprecated: Whisper requires 16kHz; this field is ignored.
    sample_rate: int = 16000


def is_stt_model(model_path: str) -> bool:
    """Return ``True`` if *model_path* points to a Speech-to-Text model.

    Detection is based on ``model_type`` in the model's ``config.json``.
    Falls back to ``False`` if the config cannot be read.
    """
    p = Path(model_path)
    config_file = p / "config.json" if p.is_dir() else None

    # For HuggingFace hub IDs, try downloading config.json
    if config_file is None or not config_file.exists():
        try:
            from huggingface_hub import hf_hub_download

            config_file = Path(
                hf_hub_download(repo_id=model_path, filename="config.json")
            )
        except (ImportError, OSError, ValueError):
            return False

    if config_file is None or not config_file.exists():
        return False

    try:
        with open(config_file) as f:
            cfg = json.load(f)
        return cfg.get("model_type", "").lower() in _STT_MODEL_TYPES
    except (OSError, json.JSONDecodeError, KeyError):
        return False


def validate_language(
    code: str | None,
    *,
    default: str | None = "en",
) -> str | None:
    """Validate and normalise an ISO 639-1 language code.

    Three-tier validation:
    1. Officially supported codes — accepted silently.
    2. Known Whisper codes but not official — accepted with debug log.
    3. Unknown codes — raises ValueError.

    Returns *default* when *code* is None.
    """
    if code is None:
        return default

    code = code.strip().lower()
    whisper_langs = get_whisper_languages()

    if code in _OFFICIALLY_SUPPORTED:
        return code

    if code in whisper_langs:
        logger.debug("Language %r is not officially supported", code)
        return code

    raise ValueError(
        f"Unsupported language: {code!r}. Must be one of "
        f"{', '.join(sorted(_OFFICIALLY_SUPPORTED))}."
    )
