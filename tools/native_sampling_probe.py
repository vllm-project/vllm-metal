# SPDX-License-Identifier: Apache-2.0
"""Corruption probe for the MLX-native sampling path (issue #622 class).

Replays a tool-calling chat request shape against a running ``vllm serve``
instance N times at the given temperature and counts completions containing
any stray CJK-range character — the visible signature of a sampler picking
an out-of-candidate vocab id on Qwen-family vocabularies.

Usage (against a server started with VLLM_METAL_NATIVE_SAMPLING=1):

    vllm serve mlx-community/Qwen3.8-27B-4bit --max-num-seqs 8 &
    python tools/native_sampling_probe.py 60 0.7

The request uses plain temperature/top-k/top-p sampling so it stays on the
native path; pass ``--repetition-penalty`` to push it onto the torch
fallback path instead.
"""

from __future__ import annotations

import argparse
import json
import urllib.request

# Stray-token signature ranges: CJK unified/extensions, CJK punctuation,
# Hiragana/Katakana, Hangul, and Thai — the scripts observed in #622 reports.
_STRAY_SCRIPT_RANGES = (
    (0x0E00, 0x0E7F),  # Thai
    (0x2E80, 0x303F),  # CJK radicals + punctuation
    (0x3040, 0x30FF),  # Hiragana + Katakana
    (0x3400, 0x9FFF),  # CJK unified + extension A
    (0xAC00, 0xD7AF),  # Hangul syllables
    (0xF900, 0xFAFF),  # CJK compatibility ideographs
    (0x20000, 0x2FA1F),  # CJK extensions B+
)


def _build_payload(args: argparse.Namespace) -> dict:
    payload: dict = {
        "model": args.model,
        "messages": [
            {
                "role": "user",
                "content": (
                    "Check the docker logs for the api service and summarize "
                    "any errors you find."
                ),
            }
        ],
        "tools": [
            {
                "type": "function",
                "function": {
                    "name": "docker_logs",
                    "description": "Fetch recent logs for a docker compose service.",
                    "parameters": {
                        "type": "object",
                        "properties": {
                            "service": {"type": "string"},
                            "tail": {"type": "integer"},
                        },
                        "required": ["service"],
                    },
                },
            }
        ],
        "temperature": args.temperature,
        "top_p": args.top_p,
        "top_k": args.top_k,
        "max_tokens": args.max_tokens,
    }
    if args.repetition_penalty != 1.0:
        payload["repetition_penalty"] = args.repetition_penalty
    return payload


def _has_stray_cjk(text: str | None) -> bool:
    return any(
        start <= ord(char) <= end
        for char in text or ""
        for start, end in _STRAY_SCRIPT_RANGES
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("count", type=int)
    parser.add_argument("temperature", type=float)
    parser.add_argument("--model", default="mlx-community/Qwen3.8-27B-4bit")
    parser.add_argument("--endpoint", default="http://localhost:8000")
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-tokens", type=int, default=400)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    args = parser.parse_args()

    payload = _build_payload(args)
    corrupted = 0
    for index in range(args.count):
        request = urllib.request.Request(
            args.endpoint + "/v1/chat/completions",
            json.dumps(payload).encode(),
            {"Content-Type": "application/json"},
        )
        response = json.load(urllib.request.urlopen(request, timeout=300))
        message = response["choices"][0]["message"]
        pieces = [
            message.get("content") or "",
            message.get("reasoning_content") or "",
        ]
        for call in message.get("tool_calls") or []:
            pieces.append(call.get("function", {}).get("arguments") or "")
        if any(_has_stray_cjk(piece) for piece in pieces):
            corrupted += 1
            print(f"[{index}] CORRUPT")
    print(f"{corrupted}/{args.count} corrupted at temperature={args.temperature}")


if __name__ == "__main__":
    main()
