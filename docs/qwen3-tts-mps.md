# Qwen3-TTS on Apple Silicon

Generate speech with the optional vLLM-Omni integration.

> **Development preview:** The optional dependency installs a compatible Omni preview.

After the [development setup](CONTRIBUTING.md#development-setup), install:

```bash
VLLM_OMNI_TARGET_DEVICE=cpu uv pip install -e '.[omni]' 'torch==2.13.0'
```

Start the server:

```bash
VLLM_METAL_BACKEND=mps vllm-omni serve \
  Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice --omni \
  --host 127.0.0.1 --port 8091
```

Save generated speech to a WAV file:

```bash
curl http://127.0.0.1:8091/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"input":"Hello, have a wonderful day.","voice":"Ryan","language":"English","response_format":"wav"}' \
  --output speech.wav
```

For streaming audio, add `"stream": true`, `"stream_format": "audio"`, and
`"response_format": "pcm"` to the request. See the
[Omni Speech API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/speech_api/)
for request formats and examples.

This preview uses greedy decoding for both the talker and code predictor.
Random sampling, quantized models, and distributed serving are not supported.
