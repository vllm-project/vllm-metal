#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live serve smoke for request sampling on the one-shot STT decode.

Starts ``vllm serve <whisper-model>``, waits for /health, then transcribes one
speech clip under several sampling params and one synthetic non-speech clip.
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
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request
import uuid
import wave
from pathlib import Path

_HEALTH_TIMEOUT_S = 300
_SAMPLE_RATE = 16000
_NON_SPEECH_SECONDS = 5


def _wait_for_health(base_url: str, timeout_s: int) -> bool:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(f"{base_url}/health", timeout=5) as resp:
                if resp.status == 200:
                    return True
        except (urllib.error.URLError, ConnectionError, OSError):
            pass
        time.sleep(2)
    return False


def _write_silence(path: Path) -> None:
    """Write a clip with no speech, so language detection has nothing to match."""
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(_SAMPLE_RATE)
        handle.writeframes(b"\x00\x00" * (_SAMPLE_RATE * _NON_SPEECH_SECONDS))


def _transcribe(base_url: str, audio: Path, **form: str) -> tuple[int, str]:
    """POST one transcription request; returns the status and the text or error."""
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
            return resp.status, json.load(resp).get("text", "")
    except urllib.error.HTTPError as exc:
        return exc.code, exc.read().decode(errors="replace")


def _run_checks(base_url: str, speech: Path, silence: Path) -> bool:
    timed: dict[str, float] = {}

    def timed_transcribe(label: str, audio: Path, **form: str) -> tuple[int, str]:
        started = time.monotonic()
        result = _transcribe(base_url, audio, **form)
        timed[label] = time.monotonic() - started
        return result

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
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    serve = subprocess.Popen(
        [
            "vllm",
            "serve",
            args.model,
            "--port",
            str(args.port),
            "--gpu-memory-utilization",
            str(args.gpu_memory_utilization),
        ]
    )
    try:
        print(f"Waiting for {base_url}/health ...", flush=True)
        if not _wait_for_health(base_url, _HEALTH_TIMEOUT_S):
            print("FAIL: server did not become healthy", file=sys.stderr)
            return 1
        with tempfile.TemporaryDirectory() as workdir:
            silence = Path(workdir) / "silence.wav"
            _write_silence(silence)
            return 0 if _run_checks(base_url, Path(args.audio), silence) else 1
    finally:
        serve.terminate()
        try:
            serve.wait(timeout=30)
        except subprocess.TimeoutExpired:
            serve.kill()


if __name__ == "__main__":
    sys.exit(main())
