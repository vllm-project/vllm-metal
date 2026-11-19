# Omni on Apple Silicon

Use the optional vLLM-Omni integration for speech and image generation and
duplex audio/video conversations through the PyTorch-MPS backend.

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

## MiniCPM-o 4.5

Run native duplex audio and video conversations with vLLM-Omni. The included
profile uses the original BF16 checkpoint, an 8K Thinker context, and a limit
of two active sessions. One- and two-session speech workloads were validated
on an M2 Max with 64 GB of unified memory. Playback can have gaps, especially
with two sessions; functional duplex support does not guarantee gapless
realtime conversation.

```bash
VLLM_METAL_BACKEND=mps PYTORCH_ENABLE_MPS_FALLBACK=0 vllm-omni serve \
  openbmb/MiniCPM-o-4_5 --omni \
  --deploy-config vllm_metal/pytorch_backend/deploy/minicpmo_4_5_mps.yaml \
  --host 127.0.0.1 --port 8097
```

Leave the server running. In another terminal with the same environment,
send a short question from a mono, 16 kHz, PCM16 WAV file named `question.wav`.
Save the following as `client.py` beside it and run `python client.py`.
This one-turn check saves the first spoken response to `reply.wav` at 24 kHz:

```python
import asyncio
import wave
from pathlib import Path

from vllm_omni.clients.duplex import DuplexClient, read_pcm16_wav
from vllm_omni.clients.minicpmo_4_5 import create_duplex_session_config


async def main():
    async with (
        asyncio.timeout(60),
        DuplexClient(
            "ws://127.0.0.1:8097/v1/realtime?duplex=1",
            model="openbmb/MiniCPM-o-4_5",
            config=create_duplex_session_config(),
        ) as client,
    ):
        await client.stream_pcm(read_pcm16_wav(Path("question.wav")), chunk_ms=200)
        await client.commit(create_response=False)
        async for response in client.responses():
            if response.decision == "listen":
                continue
            with wave.open("reply.wav", "wb") as output:
                output.setnchannels(1)
                output.setsampwidth(2)
                output.setframerate(24000)
                async for chunk in response.audio():
                    output.writeframes(chunk)
            await response.wait()
            print(response.transcript)
            break


asyncio.run(main())
```

The model's bundled voice is used unless the client supplies reference audio.
For continuous input, video, and playback acknowledgements, see
[Omni's DuplexClient guide](https://github.com/vllm-project/vllm-omni/blob/main/docs/serving/realtime_duplex_api.md#using-duplexclient).

## LFM3-Omni-2B development profile

This single-request profile connects the upstream LFM3 temporal model and
LiquidCodec on one Apple GPU. It targets turn-based speech. The pinned Omni
model still expects Model Runner V1 request metadata. End-to-end serving
requires the [upstream V2 port](https://github.com/vllm-project/vllm-omni/pull/10675).
Duplex input streaming needs separate upstream model support.

To exercise the speech endpoint once the V2 port is available, set
`LFM3_OMNI_TEST_MODEL` to a local copy of `LiquidAI/LFM3-Omni-2B`, including
the `audio_tokenizer/` subdirectory. The following test starts the server,
requests speech, and checks the returned 24 kHz WAV:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 VLLM_METAL_BACKEND=mps \
  python -m pytest -q tests/e2e/test_mps_omni_serving.py -k lfm3
```

To test just the codec on a smaller Mac, the same variable can point to a
local directory with the checkpoint's Python files and `audio_tokenizer/`.
This check compares a streamed waveform with a single decode of the same codes:

```bash
PYTORCH_ENABLE_MPS_FALLBACK=0 VLLM_METAL_BACKEND=mps \
  python -m pytest -q tests/test_mps_backend.py -k lfm3_streaming_codec
```
