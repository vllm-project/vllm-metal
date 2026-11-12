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
