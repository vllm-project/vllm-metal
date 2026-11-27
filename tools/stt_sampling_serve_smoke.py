#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live serve smoke for request sampling on the one-shot STT decode.

Starts ``vllm serve <whisper-model>``, verifies /health and the served model,
then transcribes one speech clip under several sampling params and one
synthetic non-speech clip.
It checks the contracts the decode owes a request: a greedy transcription is
stable, a sampled one is served, a seeded one repeats, and language
auto-detection (which vLLM runs as a hidden
``allowed_token_ids`` request) answers instead of failing the call.

Kept as a maintainer script rather than a slow unit test: it starts the full
engine and needs a local checkpoint.

Run one vLLM process at a time:

    python tools/stt_sampling_serve_smoke.py --audio /path/speech.wav
"""

from __future__ import annotations

import argparse
import json
import math
import socket
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path
from typing import Any

_HEALTH_TIMEOUT_S = 300
_MAX_PORT_ATTEMPTS = 3
_SAMPLE_RATE = 16000
_NON_SPEECH_SECONDS = 5


def _probe_port(requested_port: int) -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", requested_port))
        return sock.getsockname()[1]


def _wait_for_health(
    base_url: str, timeout_s: float, serve: subprocess.Popen, served_model: str
) -> bool:
    deadline = time.monotonic() + timeout_s
    while (remaining := deadline - time.monotonic()) > 0:
        if serve.poll() is not None:
            return False
        try:
            with urllib.request.urlopen(
                f"{base_url}/health", timeout=min(5, remaining)
            ) as resp:
                healthy = resp.status == 200
            if healthy:
                if serve.poll() is not None:
                    return False
                remaining = deadline - time.monotonic()
                if remaining <= 0:
                    return False
                with urllib.request.urlopen(
                    f"{base_url}/v1/models", timeout=min(5, remaining)
                ) as models_resp:
                    models = json.load(models_resp)
                if isinstance(models, dict) and isinstance(models.get("data"), list):
                    if any(
                        model.get("id") == served_model
                        for model in models["data"]
                        if isinstance(model, dict)
                    ):
                        return serve.poll() is None
        except (urllib.error.URLError, ConnectionError, OSError, ValueError):
            pass
        if serve.poll() is not None:
            return False
        time.sleep(min(2, max(0, deadline - time.monotonic())))
    return False


def _write_silence(path: Path) -> None:
    """Write a clip with no speech, so language detection has nothing to match."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(_SAMPLE_RATE)
        handle.writeframes(b"\x00\x00" * (_SAMPLE_RATE * _NON_SPEECH_SECONDS))


def _transcribe(
    base_url: str, audio: Path, **form: str
) -> tuple[int, dict[str, Any] | str]:
    """POST one transcription request; returns the status and the JSON body or error."""
    boundary = uuid.uuid4().hex
    parts: list[bytes] = []
    for name, value in form.items():
        parts.append(
            f'--{boundary}\r\nContent-Disposition: form-data; name="{name}"\r\n'
            f"\r\n{value}\r\n".encode()
        )
    parts.append(
        f'--{boundary}\r\nContent-Disposition: form-data; name="file"; '
        f'filename="{audio.name}"\r\nContent-Type: audio/wav\r\n\r\n'.encode()
    )
    parts.append(audio.read_bytes())
    parts.append(f"\r\n--{boundary}--\r\n".encode())
    request = urllib.request.Request(
        f"{base_url}/v1/audio/transcriptions",
        data=b"".join(parts),
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
    )
    try:
        with urllib.request.urlopen(request, timeout=300) as resp:
            return resp.status, json.load(resp)
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")


def _check_verbose_json(base_url: str, speech: Path, served_model: str) -> bool:
    status, body = _transcribe(
        base_url,
        speech,
        model=served_model,
        language="en",
        response_format="verbose_json",
    )
    if status != 200 or not isinstance(body, dict):
        print(f"[FAIL] verbose_json returned HTTP {status}: {body}")
        return False
    segments = body.get("segments")
    if not isinstance(segments, list) or not segments:
        print("[FAIL] expected transcription segments for the speech clip")
        return False
    for segment in segments:
        score = segment.get("avg_logprob") if isinstance(segment, dict) else None
        if type(score) not in (float, int) or not math.isfinite(score):
            print(f"[FAIL] segment has invalid avg_logprob: {score!r}")
            return False
    print(f"[PASS] {len(segments)} segments with finite avg_logprob")
    return True


def _run_checks(base_url: str, speech: Path, silence: Path, served_model: str) -> bool:
    timed: dict[str, float] = {}

    def timed_transcribe(label: str, audio: Path, **form: str) -> tuple[int, str]:
        started = time.monotonic()
        status, body = _transcribe(base_url, audio, model=served_model, **form)
        timed[label] = time.monotonic() - started
        return status, body.get("text", "") if isinstance(body, dict) else body

    greedy_first = timed_transcribe("greedy", speech)
    greedy_again = timed_transcribe("greedy (repeat)", speech)
    seeded_first = timed_transcribe(
        "temperature=1.0 seed=7", speech, temperature="1.0", seed="7"
    )
    seeded_again = timed_transcribe(
        "temperature=1.0 seed=7 (repeat)", speech, temperature="1.0", seed="7"
    )
    other_seed = timed_transcribe(
        "temperature=1.0 seed=8", speech, temperature="1.0", seed="8"
    )
    top_k_one = timed_transcribe(
        "temperature=1.0 top_k=1", speech, temperature="1.0", top_k="1"
    )
    # These two differ only in whether vLLM runs its hidden language-detection
    # request first, so their times bracket what that request costs.
    pinned_language = timed_transcribe("language=en", speech, language="en")
    auto_language = timed_transcribe("language auto-detected", speech)
    detected = timed_transcribe("non-speech, language auto-detected", silence)

    for label, (status, text) in (
        ("greedy", greedy_first),
        ("greedy (repeat)", greedy_again),
        ("temperature=1.0 seed=7", seeded_first),
        ("temperature=1.0 seed=7 (repeat)", seeded_again),
        ("temperature=1.0 seed=8", other_seed),
        ("temperature=1.0 top_k=1", top_k_one),
        ("language=en", pinned_language),
        ("language auto-detected", auto_language),
        ("non-speech, language auto-detected", detected),
    ):
        print(f"[{status}] {timed[label]:6.3f}s {label}: {text!r}")

    sampled = (seeded_first, seeded_again, other_seed, top_k_one)
    checks = (
        ("greedy is stable", greedy_first[0] == 200 and greedy_first == greedy_again),
        ("sampled requests are served", all(s == 200 for s, _ in sampled)),
        ("seeded sampling repeats", seeded_first == seeded_again),
        ("top_k=1 collapses onto greedy", top_k_one == greedy_first),
        ("non-speech audio is transcribed", detected[0] == 200),
        (
            "language detection does not change the transcript",
            auto_language == pinned_language,
        ),
    )
    print()
    for label, ok in checks:
        print(f"[{'PASS' if ok else 'FAIL'}] {label}")
    return all(ok for _, ok in checks)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", required=True, help="path to a speech clip")
    parser.add_argument("--model", default="openai/whisper-tiny")
    parser.add_argument(
        "--port",
        type=int,
        default=0,
        help="Server port (default: choose an available port)",
    )
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    parser.add_argument(
        "--check-verbose-json",
        action="store_true",
        help="Check verbose transcription segments using an English speech clip",
    )
    args = parser.parse_args()

    attempts = _MAX_PORT_ATTEMPTS if args.port == 0 else 1
    for attempt in range(attempts):
        try:
            port = _probe_port(args.port)
        except OSError as exc:
            print(
                f"FAIL: cannot bind localhost port {args.port}: {exc}", file=sys.stderr
            )
            return 1

        base_url = f"http://127.0.0.1:{port}"
        served_model = f"stt-smoke-{uuid.uuid4().hex}"
        serve = subprocess.Popen(
            [
                "vllm",
                "serve",
                args.model,
                "--served-model-name",
                served_model,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--gpu-memory-utilization",
                str(args.gpu_memory_utilization),
            ]
        )
        try:
            print(f"Waiting for {base_url}/health ...", flush=True)
            if not _wait_for_health(base_url, _HEALTH_TIMEOUT_S, serve, served_model):
                exit_code = serve.poll()
                if args.port == 0 and exit_code is not None and attempt + 1 < attempts:
                    try:
                        _probe_port(port)
                    except OSError:
                        print(f"Port {port} was claimed; retrying on a new port")
                        continue
                message = (
                    f"server exited during startup (exit code {exit_code})"
                    if exit_code is not None
                    else "server did not become healthy"
                )
                print(f"FAIL: {message}", file=sys.stderr)
                return 1
            if args.check_verbose_json:
                return (
                    0
                    if _check_verbose_json(base_url, Path(args.audio), served_model)
                    else 1
                )
            with tempfile.TemporaryDirectory() as workdir:
                silence = Path(workdir) / "silence.wav"
                _write_silence(silence)
                return (
                    0
                    if _run_checks(base_url, Path(args.audio), silence, served_model)
                    else 1
                )
        finally:
            if serve.poll() is None:
                serve.terminate()
            try:
                serve.wait(timeout=30)
            except subprocess.TimeoutExpired:
                serve.kill()
                serve.wait()
    return 1


if __name__ == "__main__":
    sys.exit(main())
