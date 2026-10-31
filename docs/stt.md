# Speech-to-Text (STT)

vllm-metal supports OpenAI-compatible Speech-to-Text using Whisper and Qwen3-ASR models, running natively on Apple Silicon via MLX.

## Quick Start

```bash
# Start server with a Whisper model
vllm serve openai/whisper-small --port 8000

# Transcribe audio
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@recording.wav"
```

Some audio formats require FFmpeg (`brew install ffmpeg`).

## Supported Models

### Whisper

Any OpenAI Whisper checkpoint (HuggingFace or MLX format):

| Model | Parameters | HuggingFace ID |
|-------|-----------|----------------|
| Whisper Tiny | 39M | `openai/whisper-tiny` |
| Whisper Base | 74M | `openai/whisper-base` |
| Whisper Small | 244M | `openai/whisper-small` |
| Whisper Medium | 769M | `openai/whisper-medium` |
| Whisper Large V3 | 1.5B | `openai/whisper-large-v3` |
| Whisper Large V3 Turbo | 809M | `openai/whisper-large-v3-turbo` |

MLX-format weights (e.g. from `mlx-community`) are also supported.

### Qwen3-ASR

| Model | Parameters | HuggingFace ID |
|-------|-----------|----------------|
| Qwen3-ASR-0.6B | 0.6B | `Qwen/Qwen3-ASR-0.6B` |

> Qwen3-ASR is transcription-only: translation (`/v1/audio/translations`) is not supported.
> The model auto-detects language; `language` and `prompt` parameters are ignored.

## API Endpoints

### `POST /v1/audio/transcriptions`

Transcribe audio to text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file (wav, mp3, m4a, etc.) |
| `model` | string | `"whisper"` | Model identifier |
| `language` | string | `null` | ISO 639-1 language code (e.g. `en`, `zh`) |
| `prompt` | string | `null` | Guide transcription (e.g. proper nouns) |
| `response_format` | string | `"json"` | `json`, `text`, or `verbose_json` |
| `temperature` | float | `0.0` | Sampling temperature; `0` decodes greedily |
| `top_p` / `top_k` / `min_p` | float / int / float | disabled | Restrict the candidate tokens sampled at each step |
| `seed` | int | `null` | Makes a sampled transcription reproducible |
| `repetition_penalty` | float | `1.0` | Penalize tokens already in the prompt or the transcript |
| `frequency_penalty` / `presence_penalty` | float | `0.0` | Penalize tokens already in the transcript |

`max_completion_tokens` caps the transcript. `logit_bias` and `min_tokens` are
rejected for every vllm-metal request.

### `POST /v1/audio/translations`

Translate audio to English. Same parameters as transcriptions (except `language`).

## Response Formats

**`json`** (default):
```json
{"text": "Hello, world."}
```

**`verbose_json`** (Whisper only; Qwen3-ASR has no segment timestamps):
```json
{
  "text": "Hello, world.",
  "language": "en",
  "duration": 2.5,
  "segments": [
    {"id": 0, "start": 0.0, "end": 2.5, "text": "Hello, world.", "avg_logprob": -0.12, "...": "..."}
  ]
}
```

**`text`**: Plain text output.
