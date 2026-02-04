# SPDX-License-Identifier: Apache-2.0
"""OpenAI-compatible transcription API endpoints."""

from __future__ import annotations

import logging
import os
import tempfile
import time
from typing import Annotated

from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import PlainTextResponse

from vllm_metal.stt.config import SpeechToTextConfig, validate_language
from vllm_metal.stt.formatting import format_as_srt, format_as_vtt
from vllm_metal.stt.protocol import (
    ResponseFormat,
    TranscriptionResponse,
    TranscriptionResponseVerbose,
    TranscriptionUsage,
)
from vllm_metal.stt.transcribe import TranscriptionResult, load_model, transcribe
from vllm_metal.stt.whisper import WhisperModel

logger = logging.getLogger(__name__)

router = APIRouter()

# Global model state
_model: WhisperModel | None = None
_model_name: str = ""
_stt_config: SpeechToTextConfig = SpeechToTextConfig()


def init_stt_model(
    model_path: str,
    stt_config: SpeechToTextConfig | None = None,
) -> None:
    """Initialize STT model and optional runtime config."""
    global _model, _model_name, _stt_config
    _model = load_model(model_path)
    _model_name = model_path
    if stt_config is not None:
        _stt_config = stt_config


def _build_response(
    result: TranscriptionResult,
    response_format: ResponseFormat,
    task: str,
    start_time: float,
) -> TranscriptionResponse | TranscriptionResponseVerbose | PlainTextResponse:
    """Build the appropriate response object from a transcription result."""
    elapsed = time.monotonic() - start_time
    usage = TranscriptionUsage(
        prompt_duration_seconds=round(result.duration, 2),
        total_duration_seconds=round(elapsed, 2),
    )
    logger.debug("STT usage: %s", usage.model_dump())

    if response_format == "text":
        return PlainTextResponse(result.text)
    if response_format == "srt":
        return PlainTextResponse(format_as_srt(result.segments))
    if response_format == "vtt":
        return PlainTextResponse(format_as_vtt(result.segments))
    if response_format == "verbose_json":
        return TranscriptionResponseVerbose(
            task=task,
            language=result.language,
            duration=result.duration,
            text=result.text,
            segments=result.segments or None,
        )
    # default: json
    return TranscriptionResponse(text=result.text)


@router.post("/v1/audio/transcriptions")
async def create_transcription(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = "whisper",
    language: Annotated[str | None, Form()] = None,
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = "json",
    temperature: Annotated[float, Form()] = 0.0,
):
    """Create transcription from audio file."""
    if _model is None:
        raise HTTPException(status_code=503, detail="STT model not loaded")

    try:
        language = validate_language(language)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e

    with_timestamps = response_format in ("verbose_json", "srt", "vtt")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    start_time = time.monotonic()
    try:
        result = transcribe(
            _model,
            tmp_path,
            language=language,
            prompt=prompt,
            with_timestamps=with_timestamps,
            stt_config=_stt_config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        os.unlink(tmp_path)

    return _build_response(result, response_format, "transcribe", start_time)


@router.post("/v1/audio/translations")
async def create_translation(
    file: Annotated[UploadFile, File()],
    model: Annotated[str, Form()] = "whisper",
    prompt: Annotated[str | None, Form()] = None,
    response_format: Annotated[ResponseFormat, Form()] = "json",
    temperature: Annotated[float, Form()] = 0.0,
):
    """Translate audio to English text."""
    if _model is None:
        raise HTTPException(status_code=503, detail="STT model not loaded")

    with_timestamps = response_format in ("verbose_json", "srt", "vtt")

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    start_time = time.monotonic()
    try:
        result = transcribe(
            _model,
            tmp_path,
            task="translate",
            prompt=prompt,
            with_timestamps=with_timestamps,
            stt_config=_stt_config,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e)) from e
    finally:
        os.unlink(tmp_path)

    return _build_response(result, response_format, "translate", start_time)
