#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare greedy paged serving with the environment's native mlx-lm.

Generate one native reference, then reuse one HTTP server for individual
and batched prompt requests. Both backends use the same checkpoint and input IDs.
See docs/tools.md or --help for usage.
"""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import tempfile
import time
import urllib.request
from contextlib import contextmanager, redirect_stdout, suppress
from itertools import zip_longest
from pathlib import Path

if __package__:
    from .parity_prompts import PROMPTS
else:
    from parity_prompts import PROMPTS


def mlx_generate(
    model_path: str, prompts: list[str], max_tokens: int, top_k: int | None = None
) -> list[dict]:
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    model, tokenizer = load(model_path)
    results = []
    for prompt in prompts:
        input_ids = tokenizer.encode(prompt)
        if not input_ids:
            raise ValueError("The prompt must encode to at least one token.")
        tokens = []
        top_logprobs = []
        for token, logprobs in generate_step(
            mx.array(input_ids),
            model,
            max_tokens=max_tokens,
            sampler=make_sampler(temp=0),
        ):
            tokens.append(int(token))
            if top_k is not None:
                scores = np.array(logprobs.astype(mx.float32))
                indices = np.argsort(-scores, kind="stable")[:top_k]
                top_logprobs.append(
                    [
                        {
                            "id": int(i),
                            "text": tokenizer.decode([int(i)]),
                            "logprob": float(scores[i]),
                            "rank": rank,
                        }
                        for rank, i in enumerate(indices, start=1)
                    ]
                )
        results.append(
            {
                "prompt": prompt,
                "input_ids": input_ids,
                "tokens": tokens,
                "text": tokenizer.decode(tokens),
                "top_logprobs": top_logprobs,
            }
        )
    return results


@contextmanager
def serving(
    model: str, max_model_len: int, max_num_seqs: int, log_path: Path, env: dict
):
    """Start one local server and clean up its process group on every exit."""
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    base_url = f"http://127.0.0.1:{port}"
    with log_path.open("w") as log:
        server = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "vllm.entrypoints.cli.main",
                "serve",
                model,
                "--host",
                "127.0.0.1",
                "--port",
                str(port),
                "--max-model-len",
                str(max_model_len),
                "--max-num-seqs",
                str(max_num_seqs),
                "--no-enable-prefix-caching",
                "--generation-config",
                "vllm",
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            start_new_session=True,
            env=env,
        )
        try:
            deadline = time.monotonic() + 300
            while time.monotonic() < deadline:
                if server.poll() is not None:
                    raise RuntimeError(f"Server exited during startup; see {log_path}")
                try:
                    with urllib.request.urlopen(
                        f"{base_url}/health", timeout=2
                    ) as response:
                        if response.status == 200:
                            break
                except OSError:
                    pass
                time.sleep(1)
            else:
                raise TimeoutError(f"Server did not become healthy; see {log_path}")
            yield f"{base_url}/v1"
        finally:
            with suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGTERM)
            with suppress(subprocess.TimeoutExpired):
                server.wait(timeout=10)
            with suppress(ProcessLookupError):
                os.killpg(server.pid, signal.SIGKILL)
            server.wait()


def check_parity(
    model: str,
    prompts: list[str],
    max_tokens: int,
    top_k: int | None = None,
    batch_sizes: tuple[int, ...] = (1, 2),
    output_dir: Path | None = None,
) -> bool:
    """Generate one reference, then compare every request batch size on one server."""
    if not prompts or not batch_sizes or min(batch_sizes) < 1:
        raise ValueError("Prompts and positive request batch sizes are required.")
    output_dir = output_dir or Path(tempfile.mkdtemp(prefix="metal-parity-"))
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Artifacts: {output_dir.resolve()}", flush=True)
    summary = output_dir / "summary.md"
    summary.write_text(
        "| Prompts/request | EXACT | TOP_K_MATCH | FAIL | Exit |\n|---|---:|---:|---:|---:|\n"
    )
    if not Path(model).is_dir():
        from huggingface_hub import snapshot_download

        model = snapshot_download(model)
    model = str(Path(model).resolve())
    env = os.environ.copy()
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH")])
    )
    env.setdefault("VLLM_METAL_MEMORY_FRACTION", "0.3")
    env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    reference_path = output_dir / "reference.json"
    reference_path.write_text(json.dumps(prompts))
    print("Generating native MLX reference...", flush=True)
    with (output_dir / "reference.log").open("w") as log:
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--model",
                model,
                "--max-tokens",
                str(max_tokens),
                "--generate-reference",
                str(reference_path),
            ]
            + (["--top-k", str(top_k)] if top_k is not None else []),
            check=True,
            env=env,
            stdout=log,
            stderr=subprocess.STDOUT,
        )
    reference = json.loads(reference_path.read_text())
    max_model_len = max(len(row["input_ids"]) for row in reference) + max_tokens
    passed = True
    print("Starting vLLM server...", flush=True)
    with serving(
        model, max_model_len, max(batch_sizes), output_dir / "serve.log", env
    ) as base_url:
        for size in batch_sizes:
            print(f"Prompts/request: {size}", flush=True)
            outputs = http_generate(base_url, model, reference, max_tokens, top_k, size)
            log_path = output_dir / f"batch-{size}.log"
            with log_path.open("w") as log, redirect_stdout(log):
                matched = compare_results(
                    reference, outputs, max_tokens=max_tokens, top_k=top_k
                )
            report = log_path.read_text()
            print(report, end="")
            counts = [
                sum(line.startswith(status + " ") for line in report.splitlines())
                for status in ("EXACT", "TOP_K_MATCH", "FAIL")
            ]
            with summary.open("a") as output:
                print(
                    f"| {size} | {counts[0]} | {counts[1]} | {counts[2]} | {int(not matched)} |",
                    file=output,
                )
            passed &= matched
    return passed


def http_generate(
    base_url: str,
    model: str,
    reference: list[dict],
    max_tokens: int,
    top_k: int | None = None,
    batch_size: int = 1,
) -> list[dict]:
    """Submit prompt batches sequentially to one server, preserving input order."""
    if not reference or batch_size < 1:
        raise ValueError(
            "Reference prompts and a positive request batch size are required."
        )
    results = []
    for start in range(0, len(reference), batch_size):
        batch = reference[start : start + batch_size]
        input_ids = [row["input_ids"] for row in batch]
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/completions",
            data=json.dumps(
                {
                    "model": model,
                    "prompt": input_ids[0] if len(batch) == 1 else input_ids,
                    "temperature": 0,
                    "max_tokens": max_tokens,
                    "ignore_eos": True,
                    "logprobs": top_k,
                    "return_token_ids": True,
                    "return_tokens_as_token_ids": True,
                }
            ).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=120) as response:
            choices = sorted(
                json.load(response)["choices"], key=lambda choice: choice["index"]
            )
        for index, (row, choice) in enumerate(zip(batch, choices, strict=True)):
            if (
                choice["index"] != index
                or choice["prompt_token_ids"] != row["input_ids"]
            ):
                raise ValueError("Server changed prompt indices or input token IDs.")
            results.append(http_result(choice, top_k))
    return results


def http_result(choice: dict, top_k: int | None) -> dict:
    """Convert completion logprobs to the comparator's token-ID format."""
    tokens = choice["token_ids"]
    top_logprobs = []
    if top_k is not None:
        for sampled, step in zip(
            tokens, choice["logprobs"]["top_logprobs"], strict=True
        ):
            candidates = {
                int(token.removeprefix("token_id:")): score
                for token, score in step.items()
            }
            # The API returns the sampled token plus top-K. An extra entry
            # means the sampled token is outside top-K.
            if len(candidates) > top_k:
                candidates.pop(sampled)
            top_logprobs.append(
                [
                    {
                        "id": token,
                        "text": f"token_id:{token}",
                        "logprob": score,
                        "rank": rank,
                    }
                    for rank, (token, score) in enumerate(
                        sorted(candidates.items(), key=lambda item: -item[1]), start=1
                    )
                ]
            )
    return {"tokens": tokens, "text": choice["text"], "top_logprobs": top_logprobs}


def compare_results(
    reference: list[dict],
    outputs: list[dict],
    *,
    max_tokens: int,
    top_k: int | None = None,
) -> bool:
    """Compare exact tokens or mutual top-k membership at the first divergence."""
    passed = True
    for ref, got in zip(reference, outputs, strict=True):
        mismatch = next(
            (
                i
                for i, (a, b) in enumerate(zip_longest(ref["tokens"], got["tokens"]))
                if a != b
            ),
            None,
        )
        complete = len(ref["tokens"]) == len(got["tokens"]) == max_tokens
        exact = complete and mismatch is None
        compatible = False
        if complete and mismatch is not None and top_k is not None:
            # vLLM can include the sampled token outside its requested top-k.
            ref_top, got_top = (
                {
                    c["id"]: c
                    for c in row["top_logprobs"][mismatch]
                    if c["rank"] is not None and c["rank"] <= top_k
                }
                for row in (ref, got)
            )
            ref_token, got_token = ref["tokens"][mismatch], got["tokens"][mismatch]
            compatible = ref_token in got_top and got_token in ref_top
        passed &= exact or compatible
        status = "EXACT" if exact else "TOP_K_MATCH" if compatible else "FAIL"
        print(f"{status} {ref['prompt']!r}")
        if not exact:
            if mismatch is None:
                print(f"  Expected {max_tokens} tokens, got {len(ref['tokens'])}")
            else:
                print(f"  First differing token (0-based): {mismatch}")
            print(f"  mlx-lm: {ref['tokens']}\n          {ref['text']!r}")
            print(f"  metal:  {got['tokens']}\n          {got['text']!r}")
            if complete and mismatch is not None and top_k is not None:
                candidates = (ref["tokens"][mismatch], got["tokens"][mismatch])
                for label, top in (("mlx-lm", ref_top), ("metal", got_top)):
                    for token in candidates:
                        entry = top.get(token)
                        if entry is None:
                            print(f"  {label}: token {token} is outside top-{top_k}")
                        else:
                            print(
                                f"  {label}: token {token} {entry['text']!r}, "
                                f"rank {entry['rank']}, logprob {entry['logprob']:.6f}"
                            )
                print(
                    f"  Matching prefix length: {mismatch}; "
                    "remaining continuation not compared."
                )
    return passed


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", default="Qwen/Qwen3-0.6B")
    parser.add_argument(
        "--prompt",
        action="append",
        help="Plain text prompt; repeat for multiple prompts",
    )
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument(
        "--top-k",
        type=int,
        help="Accept mutual top-k membership at the first divergence",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Prompts per HTTP request (default: 1 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory (default: a new temporary directory)",
    )
    parser.add_argument("--generate-reference", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be positive")
    if min(args.batch_size) < 1:
        parser.error("--batch-size values must be positive")
    if args.generate_reference:
        prompts = json.loads(args.generate_reference.read_text())
        reference = mlx_generate(args.model, prompts, args.max_tokens, args.top_k)
        args.generate_reference.write_text(json.dumps(reference))
        return
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(1))
    passed = check_parity(
        args.model,
        args.prompt or PROMPTS,
        args.max_tokens,
        args.top_k,
        tuple(args.batch_size),
        args.output_dir,
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
