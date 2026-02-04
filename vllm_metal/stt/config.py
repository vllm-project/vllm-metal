# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text configuration and language constants."""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path

logger = logging.getLogger(__name__)

# STT model types that can be auto-detected from config.json
_STT_MODEL_TYPES = {"whisper"}

# Full Whisper language map (100 languages recognised by the tokenizer).
# fmt: off
WHISPER_LANGUAGES: dict[str, str] = {
    "en": "english", "zh": "chinese", "de": "german", "es": "spanish",
    "ru": "russian", "ko": "korean", "fr": "french", "ja": "japanese",
    "pt": "portuguese", "tr": "turkish", "pl": "polish", "ca": "catalan",
    "nl": "dutch", "ar": "arabic", "sv": "swedish", "it": "italian",
    "id": "indonesian", "hi": "hindi", "fi": "finnish", "vi": "vietnamese",
    "he": "hebrew", "uk": "ukrainian", "el": "greek", "ms": "malay",
    "cs": "czech", "ro": "romanian", "da": "danish", "hu": "hungarian",
    "ta": "tamil", "no": "norwegian", "th": "thai", "ur": "urdu",
    "hr": "croatian", "bg": "bulgarian", "lt": "lithuanian", "la": "latin",
    "mi": "maori", "ml": "malayalam", "cy": "welsh", "sk": "slovak",
    "te": "telugu", "fa": "persian", "lv": "latvian", "bn": "bengali",
    "sr": "serbian", "az": "azerbaijani", "sl": "slovenian", "kn": "kannada",
    "et": "estonian", "mk": "macedonian", "br": "breton", "eu": "basque",
    "is": "icelandic", "hy": "armenian", "ne": "nepali", "mn": "mongolian",
    "bs": "bosnian", "kk": "kazakh", "sq": "albanian", "sw": "swahili",
    "gl": "galician", "mr": "marathi", "pa": "punjabi", "si": "sinhala",
    "km": "khmer", "sn": "shona", "yo": "yoruba", "so": "somali",
    "af": "afrikaans", "oc": "occitan", "ka": "georgian", "be": "belarusian",
    "tg": "tajik", "sd": "sindhi", "gu": "gujarati", "am": "amharic",
    "yi": "yiddish", "lo": "lao", "uz": "uzbek", "fo": "faroese",
    "ht": "haitian creole", "ps": "pashto", "tk": "turkmen", "nn": "nynorsk",
    "mt": "maltese", "sa": "sanskrit", "lb": "luxembourgish",
    "my": "myanmar", "bo": "tibetan", "tl": "tagalog", "mg": "malagasy",
    "as": "assamese", "tt": "tatar", "haw": "hawaiian", "ln": "lingala",
    "ha": "hausa", "ba": "bashkir", "jw": "javanese", "su": "sundanese",
    "yue": "cantonese",
}

# Officially supported languages (from OpenAI docs) — a subset of the above.
# Languages in WHISPER_LANGUAGES but not here are accepted with a warning.
# Reference: https://platform.openai.com/docs/guides/speech-to-text/supported-languages
ISO639_1_SUPPORTED_LANGS: dict[str, str] = {
    "af": "afrikaans", "ar": "arabic", "hy": "armenian", "az": "azerbaijani",
    "be": "belarusian", "bs": "bosnian", "bg": "bulgarian", "ca": "catalan",
    "zh": "chinese", "hr": "croatian", "cs": "czech", "da": "danish",
    "nl": "dutch", "en": "english", "et": "estonian", "fi": "finnish",
    "fr": "french", "gl": "galician", "de": "german", "el": "greek",
    "he": "hebrew", "hi": "hindi", "hu": "hungarian", "is": "icelandic",
    "id": "indonesian", "it": "italian", "ja": "japanese", "kn": "kannada",
    "kk": "kazakh", "ko": "korean", "lv": "latvian", "lt": "lithuanian",
    "mk": "macedonian", "ms": "malay", "mr": "marathi", "mi": "maori",
    "ne": "nepali", "no": "norwegian", "fa": "persian", "pl": "polish",
    "pt": "portuguese", "ro": "romanian", "ru": "russian", "sr": "serbian",
    "sk": "slovak", "sl": "slovenian", "es": "spanish", "sw": "swahili",
    "sv": "swedish", "tl": "tagalog", "ta": "tamil", "th": "thai",
    "tr": "turkish", "uk": "ukrainian", "ur": "urdu", "vi": "vietnamese",
    "cy": "welsh",
}
# fmt: on


@dataclass
class SpeechToTextConfig:
    """Runtime configuration for STT processing.

    Controls audio chunking, energy-based splitting, and general
    processing parameters.
    """

    sample_rate: int = 16000
    max_audio_clip_s: float = 30.0
    overlap_chunk_second: float = 1.0
    min_energy_split_window_size: int = 1600


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

            config_file = Path(hf_hub_download(repo_id=model_path, filename="config.json"))
        except Exception:
            return False

    if config_file is None or not config_file.exists():
        return False

    try:
        with open(config_file) as f:
            cfg = json.load(f)
        return cfg.get("model_type", "").lower() in _STT_MODEL_TYPES
    except Exception:
        return False


def validate_language(
    code: str | None,
    *,
    default: str | None = "en",
) -> str | None:
    """Validate and normalise an ISO 639-1 language code.

    Three-tier validation matching upstream vLLM behaviour:

    1. Code is in ``ISO639_1_SUPPORTED_LANGS`` — accepted silently.
    2. Code is in ``WHISPER_LANGUAGES`` but **not** officially supported —
       accepted with a warning (results may be less accurate).
    3. Code is unknown — raises ``ValueError``.

    When *code* is ``None``, returns *default* (``"en"`` by default,
    matching upstream Whisper behaviour).
    """
    if code is None:
        if default is not None:
            logger.warning(
                "Defaulting to language=%r. Pass the `language` field "
                "to transcribe audio in a different language.",
                default,
            )
        return default

    code = code.strip().lower()

    if code in ISO639_1_SUPPORTED_LANGS:
        return code

    if code in WHISPER_LANGUAGES:
        logger.warning(
            "Language %r is not officially supported; results may be "
            "less accurate. Supported languages: %s",
            code,
            ", ".join(sorted(ISO639_1_SUPPORTED_LANGS)),
        )
        return code

    raise ValueError(
        f"Unsupported language: {code!r}. Must be one of "
        f"{', '.join(sorted(ISO639_1_SUPPORTED_LANGS))}."
    )
