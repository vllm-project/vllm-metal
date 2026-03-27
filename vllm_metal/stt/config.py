# SPDX-License-Identifier: Apache-2.0
"""Whisper language validation helpers shared by STT code."""

from __future__ import annotations

import logging
from functools import lru_cache

logger = logging.getLogger(__name__)

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
def get_whisper_languages() -> dict[str, str]:
    """Get full Whisper language map (100 languages)."""
    try:
        from transformers.models.whisper.tokenization_whisper import LANGUAGES

        return dict(LANGUAGES)
    except ImportError:
        logger.warning("transformers not available, using fallback language list")
        return {"en": "english"}


def get_supported_languages() -> set[str]:
    """Get officially supported language codes."""
    return _OFFICIALLY_SUPPORTED.copy()


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
