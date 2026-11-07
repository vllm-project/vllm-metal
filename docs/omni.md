# Omni on Apple Silicon

Use the optional vLLM-Omni integration for speech generation and duplex
audio/video conversations through the PyTorch-MPS backend.

## Setup

After the [development setup](CONTRIBUTING.md#development-setup), install:

```bash
VLLM_OMNI_TARGET_DEVICE=cpu uv pip install -e '.[omni]'
```

Run the commands below from the repository root. Quantized models and
distributed serving are not supported.

## Qwen3-TTS

```bash
VLLM_METAL_BACKEND=mps PYTORCH_ENABLE_MPS_FALLBACK=0 vllm-omni serve \
  Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni \
  --deploy-config vllm_metal/pytorch_backend/deploy/qwen3_tts_mps.yaml \
  --host 127.0.0.1 --port 8091
```

Save generated speech to a WAV file:

```bash
curl http://127.0.0.1:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello, have a wonderful day.","voice":"Ryan","language":"English","response_format":"wav"}' \
  --output speech.wav
```

The default profile uses greedy decoding. For streaming and sampling options,
see the [Omni Speech API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/).

## MiniCPM-o 4.5

Run native duplex audio and video conversations with vLLM-Omni. The included
profile targets Macs with at least 64 GB of unified memory and up to eight
concurrent sessions, with an 8K context limit.

```bash
VLLM_METAL_BACKEND=mps PYTORCH_ENABLE_MPS_FALLBACK=0 vllm-omni serve \
  openbmb/MiniCPM-o-4_5 --omni \
  --deploy-config vllm_metal/pytorch_backend/deploy/minicpmo_4_5_mps.yaml \
  --host 127.0.0.1 --port 8097
```

Connect to `ws://127.0.0.1:8097/v1/realtime?duplex=1` using
[Omni's DuplexClient](https://github.com/vllm-project/vllm-omni/blob/main/docs/serving/realtime_duplex_api.md#using-duplexclient).
The model's bundled voice is used unless the client supplies reference audio.
