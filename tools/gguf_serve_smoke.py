#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Live serve smoke for the local GGUF path (#415 stack).

Starts ``vllm serve <model.gguf> --tokenizer <config-dir>``, waits for /health,
then greedy-decodes through /v1/completions and checks the continuation. This
proves the user-facing GGUF serve workflow end to end through the Metal paged
runner (the path that surfaced the head_dim bug); kept as a maintainer script
rather than a skipped slow unit test.

Run one vLLM process at a time; set the memory fraction for this machine:

    VLLM_METAL_MEMORY_FRACTION=0.5 python tools/gguf_serve_smoke.py \\
        /path/Qwen3-0.6B-Q8_0.gguf --tokenizer /path/Qwen3-0.6B
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import urllib.error
import urllib.request

_HEALTH_TIMEOUT_S = 300
_DEFAULT_PROMPT = "The capital of France is"
_DEFAULT_EXPECT = "Paris"


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


def _greedy_completion(base_url: str, model: str, prompt: str, max_tokens: int) -> str:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": max_tokens,
            "temperature": 0,
        }
    ).encode()
    req = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        body = json.load(resp)
    return body["choices"][0]["text"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("model", help="path to a local .gguf file")
    parser.add_argument(
        "--tokenizer", required=True, help="companion config/tokenizer directory"
    )
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--max-model-len", type=int, default=2048)
    parser.add_argument("--max-tokens", type=int, default=24)
    parser.add_argument("--prompt", default=_DEFAULT_PROMPT)
    parser.add_argument("--expect", default=_DEFAULT_EXPECT)
    args = parser.parse_args()

    base_url = f"http://127.0.0.1:{args.port}"
    serve = subprocess.Popen(
        [
            "vllm",
            "serve",
            args.model,
            "--tokenizer",
            args.tokenizer,
            "--port",
            str(args.port),
            "--max-model-len",
            str(args.max_model_len),
        ]
    )
    try:
        print(f"Waiting for {base_url}/health ...", flush=True)
        if not _wait_for_health(base_url, _HEALTH_TIMEOUT_S):
            print("FAIL: server did not become healthy", file=sys.stderr)
            return 1
        text = _greedy_completion(base_url, args.model, args.prompt, args.max_tokens)
        ok = args.expect in text
        print(f"prompt:     {args.prompt!r}")
        print(f"completion: {text!r}")
        print(
            f"[{'PASS' if ok else 'FAIL'}] expected {args.expect!r} in the continuation"
        )
        return 0 if ok else 1
    finally:
        serve.terminate()
        try:
            serve.wait(timeout=30)
        except subprocess.TimeoutExpired:
            serve.kill()


if __name__ == "__main__":
    sys.exit(main())
