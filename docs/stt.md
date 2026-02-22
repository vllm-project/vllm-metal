# Speech-to-Text (STT)

vllm-metal supports OpenAI-compatible Speech-to-Text using Whisper models, running natively on Apple Silicon via MLX.

## Installation

First, install vllm-metal using the install script (see [README](../README.md)):

```bash
./install.sh
```

Then install the optional STT dependencies inside the virtual environment:

```bash
source .venv/bin/activate
pip install vllm-metal[stt]
```

### ffmpeg (Optional)

ffmpeg is only needed for non-WAV audio formats (mp3, m4a, flac, etc.):

```bash
# macOS
brew install ffmpeg

# Not required for WAV files - librosa handles those directly
```

## Quick Start

```bash
# Start server with a Whisper model
vllm serve openai/whisper-small --port 8000

# Transcribe audio
curl -X POST http://localhost:8000/v1/audio/transcriptions \
  -F "file=@recording.wav"
```

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

Transcribe audio to text.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | file | required | Audio file (wav, mp3, m4a, etc.) |
| `model` | string | `"whisper"` | Model identifier |
| `language` | string | `null` | ISO 639-1 language code (e.g. `en`, `zh`) |
| `prompt` | string | `null` | Guide transcription (e.g. proper nouns) |
| `response_format` | string | `"json"` | `json`, `text`, or `verbose_json` |

### `POST /v1/audio/translations`

Translate audio to English. Same parameters as transcriptions (except `language`).

## Response Formats

**`json`** (default):
```json
{"text": "Hello, world."}
```

**`verbose_json`**:
```json
{
  "text": "Hello, world.",
  "language": "en",
  "duration": 2.5
}
```

**`text`**: Plain text output.
