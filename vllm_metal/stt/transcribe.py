# SPDX-License-Identifier: Apache-2.0
"""Speech-to-Text transcription."""

from __future__ import annotations

import glob
import json
import logging
import re
from dataclasses import dataclass, field
from functools import lru_cache
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


@lru_cache(maxsize=4)
def _get_tokenizer(model_path: str | None = None):
    """Get cached Whisper tokenizer.

    Args:
        model_path: Local model path to load tokenizer from. Falls back to
            openai/whisper-small if not provided or loading fails.
    """
    from transformers import WhisperTokenizer

    # Try local model path first (works offline if tokenizer files present)
    if model_path:
        try:
            return WhisperTokenizer.from_pretrained(model_path)
        except Exception as e:
            logger.debug("Local tokenizer load failed for %s: %s", model_path, e)

    # Fall back to openai/whisper-small (online or cached)
    try:
        return WhisperTokenizer.from_pretrained("openai/whisper-small")
    except OSError:
        return WhisperTokenizer.from_pretrained(
            "openai/whisper-small", local_files_only=True
        )


def _get_token_id(token: str, model_path: str | None = None) -> int:
    """Get token ID from tokenizer."""
    return _get_tokenizer(model_path).convert_tokens_to_ids(token)


@dataclass
class TranscriptionResult:
    """Transcription result."""

    text: str
    language: str | None = None
    segments: list[TranscriptionSegment] = field(default_factory=list)
    duration: float = 0.0


def load_model(model_path: str | Path, dtype: mx.Dtype = mx.float16) -> WhisperModel:
    """Load Whisper model from path or HuggingFace."""
    model_path = Path(model_path)

    if not model_path.exists():
        try:
            from huggingface_hub import snapshot_download

            model_path = Path(snapshot_download(repo_id=str(model_path)))
        except Exception as e:
            raise ValueError(f"Could not load model: {model_path}") from e

    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(f"config.json not found in {model_path}")

    with open(config_path) as f:
        config_dict = json.load(f)

    config = WhisperConfig.from_dict(config_dict)
    model = WhisperModel(config, dtype)

    # Load weights
    weight_files = glob.glob(str(model_path / "*.safetensors"))
    if not weight_files:
        weight_files = glob.glob(str(model_path / "*.npz"))
    if not weight_files:
        raise FileNotFoundError(f"No weight files in {model_path}")

    weights: dict[str, mx.array] = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    # Apply quantization if specified
    quantization = config_dict.get("quantization")
    if quantization is not None:

        def class_predicate(p, m):
            return isinstance(m, (nn.Linear, nn.Embedding)) and f"{p}.scales" in weights

        nn.quantize(model, **quantization, class_predicate=class_predicate)

    weights = model.sanitize(weights)
    model.load_weights(list(weights.items()), strict=False)
    mx.eval(model.parameters())
    return model


def _encode_prompt(prompt: str | None) -> list[int]:
    """Encode a user-supplied prompt into ``<|startofprev|>`` prefix tokens.

    Whisper uses ``<|startofprev|> ...prompt tokens...`` before the
    ``<|startoftranscript|>`` token to condition the decoder on prior
    context (e.g. correct spelling of proper nouns).
    """
    if not prompt:
        return []
    tokenizer = _get_tokenizer()
    prompt_ids = tokenizer.encode(prompt, add_special_tokens=False)
    # Truncate to leave room in the 448-token context window.
    # Whisper's text context is typically 448 tokens; reserve ~224 for output.
    max_prompt_len = 224
    prompt_ids = prompt_ids[-max_prompt_len:]
    return [_get_token_id("<|startofprev|>"), *prompt_ids]


def _greedy_decode(
    model: WhisperModel,
    audio_features: mx.array,
    language: str | None = None,
    task: str = "transcribe",
    prompt: str | None = None,
    max_tokens: int = 224,
) -> list[int]:
    """Greedy decoding without timestamps."""
    # Build initial tokens: optional prompt prefix + task header
    prefix = _encode_prompt(prompt)
    prefix.append(_get_token_id("<|startoftranscript|>"))
    if model.is_multilingual:
        prefix.append(_get_token_id(f"<|{language or 'en'}|>"))
        prefix.append(_get_token_id(f"<|{task}|>"))
    prefix.append(_get_token_id("<|notimestamps|>"))

    eot_token = _get_token_id("<|endoftext|>")
    tokens = mx.array([prefix], dtype=mx.int32)
    kv_cache = None
    output_tokens = []

    for _ in range(max_tokens):
        logits, kv_cache = model.decode(tokens, audio_features, kv_cache)
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        if next_token == eot_token:
            break
        output_tokens.append(next_token)
        tokens = mx.array([[next_token]], dtype=mx.int32)

    return output_tokens


def _decode_with_timestamps(
    model: WhisperModel,
    audio_features: mx.array,
    language: str | None = None,
    task: str = "transcribe",
    prompt: str | None = None,
    max_tokens: int = 448,
) -> list[int]:
    """Greedy decoding *with* timestamp tokens.

    Unlike ``_greedy_decode`` this does **not** prepend
    ``<|notimestamps|>``, allowing the model to emit ``<|0.00|>`` style
    timestamp tokens interleaved with text tokens.
    """
    prefix = _encode_prompt(prompt)
    prefix.append(_get_token_id("<|startoftranscript|>"))
    if model.is_multilingual:
        prefix.append(_get_token_id(f"<|{language or 'en'}|>"))
        prefix.append(_get_token_id(f"<|{task}|>"))

    eot_token = _get_token_id("<|endoftext|>")
    tokens = mx.array([prefix], dtype=mx.int32)
    kv_cache = None
    output_tokens: list[int] = []

    for _ in range(max_tokens):
        logits, kv_cache = model.decode(tokens, audio_features, kv_cache)
        next_token = int(mx.argmax(logits[:, -1, :], axis=-1).item())
        if next_token == eot_token:
            break
        output_tokens.append(next_token)
        tokens = mx.array([[next_token]], dtype=mx.int32)

    return output_tokens


# Regex to detect Whisper timestamp tokens like ``<|0.00|>``
_TIMESTAMP_RE = re.compile(r"<\|(\d+\.\d+)\|>")


def _extract_segments(
    token_ids: list[int],
    time_offset: float = 0.0,
    segment_id_offset: int = 0,
) -> list[TranscriptionSegment]:
    """Parse timestamp token IDs into ``TranscriptionSegment`` objects.

    Whisper emits pairs of ``<|start|>`` ... ``<|end|>`` timestamp
    tokens around each phrase.  This function groups them into segments.
    """
    tokenizer = _get_tokenizer()
    # Use convert_ids_to_tokens to preserve special token strings;
    # decode() strips them.
    raw_tokens = [tokenizer.convert_ids_to_tokens(tid) for tid in token_ids]

    segments: list[TranscriptionSegment] = []
    seg_start: float | None = None
    seg_tokens: list[int] = []
    seg_id = segment_id_offset

    for tid, text in zip(token_ids, raw_tokens, strict=True):
        m = _TIMESTAMP_RE.match(text)
        if m:
            ts = float(m.group(1))
            if seg_start is None:
                # Opening timestamp
                seg_start = ts
                seg_tokens = []
            else:
                # Closing timestamp — emit segment
                seg_text = tokenizer.decode(seg_tokens, skip_special_tokens=True)
                if seg_text.strip():
                    segments.append(
                        TranscriptionSegment(
                            id=seg_id,
                            seek=int(seg_start * 100),
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

    # Handle trailing tokens without a closing timestamp
    if seg_start is not None and seg_tokens:
        seg_text = tokenizer.decode(seg_tokens, skip_special_tokens=True)
        if seg_text.strip():
            segments.append(
                TranscriptionSegment(
                    id=seg_id,
                    seek=int(seg_start * 100),
                    start=round(seg_start + time_offset, 2),
                    end=round(seg_start + time_offset + 30.0, 2),
                    text=seg_text,
                    tokens=list(seg_tokens),
                )
            )

    return segments


def _encode_chunk(model: WhisperModel, audio_chunk: mx.array) -> mx.array:
    """Encode a single audio chunk into features."""
    mel = log_mel_spectrogram(audio_chunk)
    mel = pad_or_trim(mel, N_FRAMES, axis=-1)
    if mel.ndim == 2:
        mel = mel[None, ...]
    mel = mel.transpose(0, 2, 1)
    features = model.encode(mel)
    mx.eval(features)
    return features


def transcribe(
    model: WhisperModel,
    audio: str | np.ndarray | mx.array,
    language: str | None = None,
    task: str = "transcribe",
    prompt: str | None = None,
    with_timestamps: bool = False,
    stt_config: SpeechToTextConfig | None = None,
) -> TranscriptionResult:
    """Transcribe audio to text.

    File paths are resampled to 16kHz; array inputs must already be 16kHz.
    """
    if stt_config is None:
        stt_config = SpeechToTextConfig()

    # Whisper requires 16kHz audio
    if isinstance(audio, str):
        audio = load_audio(audio, sample_rate=SAMPLE_RATE)
    elif isinstance(audio, np.ndarray):
        audio = mx.array(audio, mx.float32)

    total_duration = audio_duration(audio, SAMPLE_RATE)

    if not with_timestamps:
        # Fast path: single chunk, no timestamp tokens
        features = _encode_chunk(model, audio)
        output_tokens = _greedy_decode(model, features, language, task, prompt)
        text = _get_tokenizer().decode(output_tokens, skip_special_tokens=True)
        return TranscriptionResult(
            text=text.strip(),
            language=language,
            duration=total_duration,
        )

    # Timestamp path: split long audio and decode with timestamps
    chunks = split_audio(
        audio,
        max_clip_s=stt_config.max_audio_clip_s,
        overlap_s=stt_config.overlap_chunk_second,
        window_size=stt_config.min_energy_split_window_size,
        sample_rate=SAMPLE_RATE,
    )

    all_segments: list[TranscriptionSegment] = []
    all_text_parts: list[str] = []
    seg_id_offset = 0

    for chunk_audio, chunk_start in chunks:
        features = _encode_chunk(model, chunk_audio)
        output_tokens = _decode_with_timestamps(model, features, language, task, prompt)
        segments = _extract_segments(output_tokens, chunk_start, seg_id_offset)

        for seg in segments:
            all_segments.append(seg)
            all_text_parts.append(seg.text)
        seg_id_offset += len(segments)

        # Fallback: if no segments extracted, decode plainly
        if not segments:
            text = _get_tokenizer().decode(output_tokens, skip_special_tokens=True)
            if text.strip():
                all_text_parts.append(text.strip())

    full_text = " ".join(all_text_parts).strip()
    return TranscriptionResult(
        text=full_text,
        language=language,
        segments=all_segments,
        duration=total_duration,
    )
