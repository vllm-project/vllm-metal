# SPDX-License-Identifier: Apache-2.0
"""Optional real HTTP regression using a local Qwen3-TTS CustomVoice checkpoint."""

import asyncio
import base64
import importlib.util
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path

import httpx
import pytest
import torch

MODEL = os.environ.get("QWEN3_TTS_TEST_MODEL")
pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(
        not MODEL
        or not torch.backends.mps.is_available()
        or importlib.util.find_spec("vllm_omni") is None,
        reason="Requires MPS, the Omni extra, and QWEN3_TTS_TEST_MODEL (CustomVoice)",
    ),
]


@pytest.fixture(scope="module")
def speech_server(tmp_path_factory):
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
        VLLM_USE_V2_MODEL_RUNNER="1",
        HF_HUB_OFFLINE="1",
    )
    with log.open("w") as output:
        process = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm_omni.entrypoints.cli.main",
                "serve",
                MODEL,
                "--omni",
                "--deploy-config",
                str(
                    Path(__file__).resolve().parents[1]
                    / "vllm_metal/pytorch_backend/deploy/qwen3_tts_mps.yaml"
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


@pytest.mark.parametrize("concurrency", [1, 4, 8])
async def test_speech_streams_complete_for_concurrent_requests(
    speech_server, concurrency
):
    async with httpx.AsyncClient(timeout=90, trust_env=False) as client:

        async def generate(index):
            events, audio = [], bytearray()
            async with client.stream(
                "POST", f"{speech_server}/v1/audio/speech", json=speech_request(index)
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
            assert len(audio) > 4800
            assert events.count("speech.audio.delta") > 1
            assert events[0] == "speech.audio.delta"
            assert events[-1] == "speech.audio.done"

        await asyncio.gather(*(generate(index) for index in range(concurrency)))


async def test_speech_rejects_random_sampling(speech_server):
    request = speech_request(0)
    request["extra_params"] = {"temperature": 0.9}
    async with httpx.AsyncClient(timeout=30, trust_env=False) as client:
        response = await client.post(f"{speech_server}/v1/audio/speech", json=request)
    events = [
        json.loads(line[6:])
        for line in response.text.splitlines()
        if line.startswith("data: ")
    ]
    assert [event["type"] for event in events] == ["speech.audio.error"]
    assert "temperature" in events[0]["error"]["message"]
