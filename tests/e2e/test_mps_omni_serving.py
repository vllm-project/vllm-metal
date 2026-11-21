# SPDX-License-Identifier: Apache-2.0
"""Optional real HTTP speech tests using local Omni checkpoints."""

import asyncio
import base64
import importlib.util
import io
import json
import os
import socket
import subprocess
import sys
import time
import wave
from pathlib import Path

import httpx
import pytest
import torch

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not torch.backends.mps.is_available()
        or importlib.util.find_spec("vllm_omni") is None,
        reason="Requires MPS and the Omni extra",
    ),
]


@pytest.fixture(scope="module")
def speech_server(tmp_path_factory, request):
    model_env, profile = request.param
    model = os.environ.get(model_env)
    if not model:
        pytest.skip(f"Set {model_env} to the local checkpoint directory")
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    directory = tmp_path_factory.mktemp("mps-omni")
    log = directory / "server.log"
    env = dict(
        os.environ,
        VLLM_METAL_BACKEND="mps",
        PYTORCH_ENABLE_MPS_FALLBACK="0",
        VLLM_NO_USAGE_STATS="1",
        HF_HUB_OFFLINE="1",
    )
    with log.open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm_omni.entrypoints.cli.main",
                "serve",
                model,
                "--omni",
                "--deploy-config",
                str(
                    Path(__file__).resolve().parents[2]
                    / "vllm_metal/pytorch_backend/deploy"
                    / profile
                ),
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
            ],
            env=env,
            stdout=output,
            stderr=subprocess.STDOUT,
        )
        try:
            url = f"http://127.0.0.1:{port}"
            with httpx.Client(timeout=2, trust_env=False) as client:
                for _ in range(120):
                    assert process.poll() is None, log.read_text()
                    try:
                        if client.get(f"{url}/health").status_code == 200:
                            break
                    except (httpx.ConnectError, httpx.TimeoutException):
                        pass
                    time.sleep(1)
                else:
                    pytest.fail(f"Omni server did not become ready:\n{log.read_text()}")
            yield url
        finally:
            process.terminate()
            try:
                process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait(timeout=10)


def speech_request(index):
    return {
        "input": [
            "Hello, have a wonderful day.",
            "Please remember to bring your ticket. The library closes at six.",
        ][index % 2],
        "task_type": "CustomVoice",
        "voice": "Ryan",
        "language": "English",
        "max_new_tokens": 128,
        "response_format": "pcm",
        "stream": True,
    }


@pytest.mark.parametrize(
    "speech_server", [("QWEN3_TTS_TEST_MODEL", "qwen3_tts_mps.yaml")], indirect=True
)
@pytest.mark.parametrize("concurrency,temperature", [(1, 0), (4, 0), (8, 0), (1, 0.9)])
async def test_speech_streams_complete_for_concurrent_requests(
    speech_server, concurrency, temperature
):
    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:

        async def generate(index):
            events, audio = [], bytearray()
            request = speech_request(index)
            request["extra_params"] = {"temperature": temperature}
            async with client.stream(
                "POST", f"{speech_server}/v1/audio/speech", json=request
            ) as response:
                assert response.status_code == 200
                async for line in response.aiter_lines():
                    if not line.startswith("data: "):
                        continue
                    event = json.loads(line[6:])
                    assert event["type"] != "speech.audio.error", event
                    events.append(event["type"])
                    if event["type"] == "speech.audio.delta":
                        audio.extend(base64.b64decode(event["audio"]))
                    elif event["type"] == "speech.audio.done":
                        assert event["usage"]["output_tokens"] < 128
            assert audio
            assert events.count("speech.audio.delta") > 1
            assert events[-1] == "speech.audio.done"

        await asyncio.gather(*(generate(index) for index in range(concurrency)))


@pytest.mark.parametrize(
    "speech_server", [("LFM3_OMNI_TEST_MODEL", "lfm3_omni_mps.yaml")], indirect=True
)
def test_lfm3_speech_returns_nonempty_waveform(speech_server):
    response = httpx.post(
        f"{speech_server}/v1/audio/speech",
        json={
            "input": "Hello from the Mac.",
            "response_format": "wav",
            "seed": 42,
            "max_new_tokens": 96,
            "extra_params": {"audio_temperature": 0.0},
        },
        timeout=240,
    )
    response.raise_for_status()
    with wave.open(io.BytesIO(response.content)) as wav:
        assert wav.getframerate() == 24000
        assert wav.getnchannels() == 1
        assert wav.getsampwidth() == 2
        samples = wav.readframes(wav.getnframes())
        assert len(samples) >= 1920 * 2  # One LiquidCodec frame.
        assert any(samples)
