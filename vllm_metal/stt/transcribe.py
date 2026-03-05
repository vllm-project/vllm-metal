# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text transcription orchestration.

Provides :class:`WhisperTranscriber` — the single owning class for Whisper
inference — plus a convenience :func:`transcribe` entrypoint and
:func:`load_model` for checkpoint loading.
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import numpy as np

from vllm_metal.stt.audio import (
    N_FRAMES,
    SAMPLE_RATE,
    audio_duration,
    load_audio,
    log_mel_spectrogram,
    pad_or_trim,
    split_audio,
)
from vllm_metal.stt.config import SpeechToTextConfig
from vllm_metal.stt.protocol import TranscriptionSegment
from vllm_metal.stt.whisper import WhisperConfig, WhisperModel

logger = logging.getLogger(__name__)

# ===========================================================================
# Constants
# ===========================================================================

# Whisper uses centiseconds for the ``seek`` field in segments.
_SEEK_MULTIPLIER = 100

# Fallback segment duration when the closing timestamp is missing.
_DEFAULT_SEGMENT_DURATION = 30.0

# Maximum prompt tokens to inject via ``<|startofprev|>``.
# Leaves room in the 448-token context window for the actual transcription.
_MAX_PROMPT_TOKENS = 224

# Regex to detect Whisper timestamp tokens like ``<|0.00|>``.
_TIMESTAMP_RE = re.compile(r"<\|(\d+\.\d+)\|>")


# ===========================================================================
# Data types
# ===========================================================================


@dataclass
class TranscriptionResult:
    """Result of a transcription operation.

    Attributes:
        text: Full transcribed text.
        language: Language code used for transcription.
        segments: Timestamped segments (populated only with ``with_timestamps``).
        duration: Total audio duration in seconds.
    """

    text: str
    language: str | None = None
    segments: list[TranscriptionSegment] = field(default_factory=list)
    duration: float = 0.0


# ===========================================================================
# WhisperTranscriber
# ===========================================================================


class WhisperTranscriber:
    """Owns model, tokenizer, and config for Whisper transcription.

    Each instance holds its own tokenizer to avoid cross-instance
    interference when multiple transcribers are active.

    Args:
        model: Loaded :class:`WhisperModel`.
        model_path: Path to model directory (used for tokenizer loading).
        config: Optional :class:`SpeechToTextConfig` overrides.
    """

    def __init__(
        self,
        model: WhisperModel,
        model_path: str | None = None,
        config: SpeechToTextConfig | None = None,
    ):
        self.model = model
        self.model_path = model_path
        self.config = config or SpeechToTextConfig()
        self._tokenizer = None

    @property
    def tokenizer(self):
        """Lazily-loaded per-instance tokenizer."""
        if self._tokenizer is None:
            self._tokenizer = _load_tokenizer(self.model_path)
        return self._tokenizer

    def transcribe(
        self,
        audio: str | np.ndarray | mx.array,
        language: str | None = None,
        task: str = "transcribe",
        prompt: str | None = None,
        with_timestamps: bool = False,
    ) -> TranscriptionResult:
        """Transcribe audio to text.

        Args:
            audio: File path, numpy array, or MLX array of audio samples.
            language: Language code (e.g. ``"en"``, ``"zh"``).
            task: ``"transcribe"`` or ``"translate"``.
            prompt: Optional prompt for guiding proper noun spelling.
            with_timestamps: If True, emit timestamped segments.

        Returns:
            :class:`TranscriptionResult` with text and optional segments.
        """
        if isinstance(audio, str):
            audio = load_audio(audio, sample_rate=SAMPLE_RATE)
        elif isinstance(audio, np.ndarray):
            audio = mx.array(audio, mx.float32)

        total_duration = audio_duration(audio, SAMPLE_RATE)

        chunks = split_audio(
            audio,
            max_clip_s=self.config.max_audio_clip_s,
            overlap_s=self.config.overlap_chunk_second,
            window_size=self.config.min_energy_split_window_size,
            sample_rate=SAMPLE_RATE,
        )

        all_segments: list[TranscriptionSegment] = []
        all_text_parts: list[str] = []
        seg_id_offset = 0

        for chunk_audio, chunk_start in chunks:
            features = self._encode_chunk(chunk_audio)
            output_tokens = self._greedy_decode(
                features, language, task, prompt, with_timestamps=with_timestamps
            )

            if with_timestamps:
                segments = self._extract_segments(
                    output_tokens, chunk_start, seg_id_offset
                )
                for seg in segments:
                    all_segments.append(seg)
                    all_text_parts.append(seg.text)
                seg_id_offset += len(segments)
                # Fallback if no segments extracted
                if not segments:
                    text = self.tokenizer.decode(
                        output_tokens, skip_special_tokens=True
                    )
                    if text.strip():
                        all_text_parts.append(text.strip())
            else:
                text = self.tokenizer.decode(output_tokens, skip_special_tokens=True)
                if text.strip():
                    all_text_parts.append(text.strip())

        return TranscriptionResult(
            text=" ".join(all_text_parts).strip(),
            language=language,
            segments=all_segments if with_timestamps else [],
            duration=total_duration,
        )

    # ---- private helpers ---------------------------------------------------

    def _get_token_id(self, token: str) -> int:
        """Resolve a special token string to its integer ID."""
        return self.tokenizer.convert_tokens_to_ids(token)

    def _encode_prompt(self, prompt: str | None) -> list[int]:
        """Encode a user prompt into ``<|startofprev|>`` prefix tokens.

        Whisper uses ``<|startofprev|> ...tokens...`` before
        ``<|startoftranscript|>`` to condition the decoder on prior
        context (e.g. correct spelling of proper nouns).

        Args:
            prompt: User-supplied prompt string.

        Returns:
            Token IDs to prepend, or empty list if no prompt.
        """
        if not prompt:
            return []
        prompt_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        prompt_ids = prompt_ids[-_MAX_PROMPT_TOKENS:]
        return [self._get_token_id("<|startofprev|>"), *prompt_ids]

    def _greedy_decode(
        self,
        audio_features: mx.array,
        language: str | None = None,
        task: str = "transcribe",
        prompt: str | None = None,
        with_timestamps: bool = False,
        max_tokens: int | None = None,
    ) -> list[int]:
        """Greedy autoregressive decoding.

        A single method handles both timestamp and non-timestamp modes.
        The only difference is whether ``<|notimestamps|>`` is prepended.

        Args:
            audio_features: Encoded audio from the encoder.
            language: Language code.
            task: ``"transcribe"`` or ``"translate"``.
            prompt: Optional context prompt.
            with_timestamps: Allow timestamp token emission.
            max_tokens: Decoding budget (default: 448 with / 224 without).

        Returns:
            List of decoded token IDs (excluding special prefix).
        """
        if max_tokens is None:
            max_tokens = 448 if with_timestamps else _MAX_PROMPT_TOKENS

        prefix = self._encode_prompt(prompt)
        prefix.append(self._get_token_id("<|startoftranscript|>"))
        if self.model.is_multilingual:
            prefix.append(self._get_token_id(f"<|{language or 'en'}|>"))
            prefix.append(self._get_token_id(f"<|{task}|>"))
        if not with_timestamps:
            prefix.append(self._get_token_id("<|notimestamps|>"))

        eot_token = self._get_token_id("<|endoftext|>")
        tokens = mx.array([prefix], dtype=mx.int32)
        kv_cache = None
        output_tokens: list[int] = []

        for _ in range(max_tokens):
            logits, kv_cache = self.model.decode(tokens, audio_features, kv_cache)
            next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
            if next_token == eot_token:
                break
            output_tokens.append(next_token)
            tokens = mx.array([[next_token]], dtype=mx.int32)

        return output_tokens

    def _extract_segments(
        self,
        token_ids: list[int],
        time_offset: float = 0.0,
        segment_id_offset: int = 0,
    ) -> list[TranscriptionSegment]:
        """Parse timestamp token IDs into :class:`TranscriptionSegment` objects.

        Whisper emits pairs of ``<|start|> ... <|end|>`` timestamp tokens
        around each phrase. This groups them into segments.

        Args:
            token_ids: Raw decoded token IDs.
            time_offset: Offset to add (for chunked audio).
            segment_id_offset: Starting segment ID.

        Returns:
            List of :class:`TranscriptionSegment`.
        """
        raw_tokens = [self.tokenizer.convert_ids_to_tokens(tid) for tid in token_ids]

        segments: list[TranscriptionSegment] = []
        seg_start: float | None = None
        seg_tokens: list[int] = []
        seg_id = segment_id_offset

        for tid, text in zip(token_ids, raw_tokens, strict=True):
            m = _TIMESTAMP_RE.match(text)
            if m:
                ts = float(m.group(1))
                if seg_start is None:
                    seg_start = ts
                    seg_tokens = []
                else:
                    seg_text = self.tokenizer.decode(
                        seg_tokens, skip_special_tokens=True
                    )
                    if seg_text.strip():
                        segments.append(
                            TranscriptionSegment(
                                id=seg_id,
                                seek=int(seg_start * _SEEK_MULTIPLIER),
                                start=round(seg_start + time_offset, 2),
                                end=round(ts + time_offset, 2),
                                text=seg_text,
                                tokens=list(seg_tokens),
                            )
                        )
                        seg_id += 1
                    seg_start = None
                    seg_tokens = []
            else:
                seg_tokens.append(tid)

        # Trailing tokens without a closing timestamp
        if seg_start is not None and seg_tokens:
            seg_text = self.tokenizer.decode(seg_tokens, skip_special_tokens=True)
            if seg_text.strip():
                segments.append(
                    TranscriptionSegment(
                        id=seg_id,
                        seek=int(seg_start * _SEEK_MULTIPLIER),
                        start=round(seg_start + time_offset, 2),
                        end=round(
                            seg_start + time_offset + _DEFAULT_SEGMENT_DURATION, 2
                        ),
                        text=seg_text,
                        tokens=list(seg_tokens),
                    )
                )

        return segments

    def _encode_chunk(self, audio_chunk: mx.array) -> mx.array:
        """Encode a single audio chunk into encoder features.

        Args:
            audio_chunk: Raw audio samples.

        Returns:
            Encoder hidden states.
        """
        n_mels = self.model.config.n_mels
        mel = log_mel_spectrogram(audio_chunk, n_mels=n_mels)
        mel = pad_or_trim(mel, N_FRAMES, axis=-1)
        if mel.ndim == 2:
            mel = mel[None, ...]
        mel = mel.transpose(0, 2, 1)
        features = self.model.encode(mel)
        mx.eval(features)
        return features


# ===========================================================================
# Model loading
# ===========================================================================


def load_model(model_path: str | Path, dtype: mx.Dtype = mx.float16) -> WhisperModel:
    """Load a Whisper model from a local directory or HuggingFace repo.

    Args:
        model_path: Local path or HuggingFace repo ID.
        dtype: Model dtype (default: float16).

    Returns:
        Loaded :class:`WhisperModel` ready for inference.

    Raises:
        ValueError: If the model cannot be found or downloaded.
        FileNotFoundError: If config.json or weight files are missing.
    """
    model_path = Path(model_path)

    if not model_path.exists():
        try:
            from huggingface_hub import snapshot_download

            model_path = Path(snapshot_download(repo_id=str(model_path)))
        except ImportError as e:
            raise ValueError(
                f"Could not download model {model_path}: "
                "huggingface_hub is not installed"
            ) from e
        except OSError as e:
            raise ValueError(f"Could not download model: {model_path}") from e

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    config = WhisperConfig.from_dict(config_dict)
    model = WhisperModel(config, dtype)

    # Load weights — prefer safetensors over npz
    weight_files = sorted(model_path.glob("*.safetensors"))
    if not weight_files:
        weight_files = sorted(model_path.glob("*.npz"))
    if not weight_files:
        raise FileNotFoundError(f"No weight files in {model_path}")

    weights: dict[str, mx.array] = {}
    for wf in weight_files:
        weights.update(mx.load(str(wf)))

    # Apply quantization if specified in config
    quantization = config_dict.get("quantization")
    if quantization is not None:

        def class_predicate(p, m):
            return isinstance(m, (nn.Linear, nn.Embedding)) and f"{p}.scales" in weights

        nn.quantize(model, **quantization, class_predicate=class_predicate)

    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model


# ===========================================================================
# Module-level helpers
# ===========================================================================


def _load_tokenizer(model_path: str | None = None):
    """Load a Whisper tokenizer.

    Tries the local model path first (works offline if tokenizer files
    are present), then falls back to ``openai/whisper-small``.

    Args:
        model_path: Local model directory.

    Returns:
        A ``WhisperTokenizer`` instance.
    """
    from transformers import WhisperTokenizer

    if model_path:
        try:
            return WhisperTokenizer.from_pretrained(model_path)
        except (OSError, ValueError) as e:
            logger.debug("Local tokenizer load failed for %s: %s", model_path, e)

    try:
        return WhisperTokenizer.from_pretrained("openai/whisper-small")
    except OSError:
        return WhisperTokenizer.from_pretrained(
            "openai/whisper-small", local_files_only=True
        )


# ===========================================================================
# Convenience entrypoint
# ===========================================================================


def transcribe(
    model: WhisperModel,
    audio: str | np.ndarray | mx.array,
    language: str | None = None,
    task: str = "transcribe",
    prompt: str | None = None,
    with_timestamps: bool = False,
    model_path: str | None = None,
    stt_config: SpeechToTextConfig | None = None,
) -> TranscriptionResult:
    """Transcribe audio to text (convenience wrapper).

    Creates a :class:`WhisperTranscriber` and delegates. For repeated
    calls, prefer constructing a transcriber directly to reuse the
    tokenizer.

    Args:
        model: Loaded :class:`WhisperModel`.
        audio: File path, numpy array, or MLX array.
        language: Language code.
        task: ``"transcribe"`` or ``"translate"``.
        prompt: Optional context prompt.
        with_timestamps: Emit timestamped segments.
        model_path: Path to model directory (for tokenizer loading).
        stt_config: Optional config overrides.

    Returns:
        :class:`TranscriptionResult`.
    """
    transcriber = WhisperTranscriber(model, model_path=model_path, config=stt_config)
    return transcriber.transcribe(audio, language, task, prompt, with_timestamps)
