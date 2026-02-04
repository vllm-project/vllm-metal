# Speech-to-Text (STT)

vllm-metal includes an OpenAI-compatible Speech-to-Text module built on Whisper. It supports transcription, translation, timestamps, and subtitle output — all running natively on Apple Silicon via MLX.

## Quick Start

```bash
vllm-metal --model openai/whisper-small
```

The server auto-detects Whisper models from `config.json` and mounts the `/v1/audio/*` endpoints.

## Supported Models

Any OpenAI Whisper checkpoint (HuggingFace or MLX format):

| Model | Parameters | HuggingFace ID |
|-------|-----------|----------------|
| Whisper Tiny | 39M | `openai/whisper-tiny` |
| Whisper Base | 74M | `openai/whisper-base` |
| Whisper Small | 244M | `openai/whisper-small` |
| Whisper Medium | 769M | `openai/whisper-medium` |
| Whisper Large V3 | 1.5B | `openai/whisper-large-v3` |

MLX-format weights (e.g. from `mlx-community`) are also supported.

## API Endpoints

### `POST /v1/audio/transcriptions`

Transcribe audio to text. Accepts multipart form data.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file (wav, mp3, m4a, etc.) |
| `model` | string | `"whisper"` | Model identifier |
| `language` | string | `null` | ISO 639-1 language code (e.g. `en`, `zh`, `ja`) |
| `prompt` | string | `null` | Initial context to guide transcription (e.g. proper nouns) |
| `response_format` | string | `"json"` | `json`, `text`, `verbose_json`, `srt`, `vtt` |
| `temperature` | float | `0.0` | Sampling temperature |

### `POST /v1/audio/translations`

Translate audio to English. Same parameters as transcriptions (except `language`).

## Response Formats

**`json`** (default):
```json
{"text": "Hello, world."}
```

**`verbose_json`** — includes segments with timestamps:
```json
{
  "task": "transcribe",
  "language": "en",
  "duration": 5.0,
  "text": "Hello, world.",
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.5, "text": " Hello, world.", ...}
  ]
}
```

**`srt`** — SubRip subtitle format:
```
1
00:00:00,000 --> 00:00:02,500
Hello, world.
```

**`vtt`** — WebVTT subtitle format:
```
WEBVTT

00:00:00.000 --> 00:00:02.500
Hello, world.
```

**`text`** — plain text, no JSON wrapping.

## Examples

```bash
# Basic transcription
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@recording.wav"

# Chinese transcription with SRT output
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@recording.wav" -F "language=zh" -F "response_format=srt"

# Guide spelling with prompt
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@meeting.wav" -F "prompt=Kubernetes, MLX, vLLM"

# Translate to English
curl -X POST http://localhost:8000/v1/audio/translations \
  -F "file=@japanese_audio.wav"

# List models
curl http://localhost:8000/v1/models
```

## Supported Languages

100 languages including: English, Chinese, Japanese, Korean, French, German, Spanish, Portuguese, Russian, Arabic, Hindi, and [many more](../vllm_metal/stt/config.py).

## Requirements

- **ffmpeg** is required for audio decoding. Install with `brew install ffmpeg`.
