# Omni on Apple Silicon

Use the optional vLLM-Omni integration for speech and image generation through
the PyTorch-MPS backend.

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

## Text-to-image

Generate images with SDXL:

```bash
VLLM_METAL_BACKEND=mps PYTORCH_ENABLE_MPS_FALLBACK=0 vllm-omni serve \
  stabilityai/stable-diffusion-xl-base-1.0 --omni --dtype bfloat16 \
  --host 127.0.0.1 --port 8092
```

Save the generated image to a PNG file:

```bash
curl --fail --show-error http://127.0.0.1:8092/v1/images/generations \
  -H 'Content-Type: application/json' \
  -d '{
    "prompt": "A cat is sitting inside a bowl.",
    "size": "1024x1024",
    "num_inference_steps": 20,
    "guidance_scale": 5.0,
    "seed": 42,
    "response_format": "file"
  }' --output image.png
```

For request options, see the
[Omni Image Generation API](https://docs.vllm.ai/projects/vllm-omni/en/latest/serving/image_generation_api/).
