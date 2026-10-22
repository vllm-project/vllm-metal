#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare greedy serving with mlx-lm.

Generate one native reference, then compare individual and batched requests.
Speculative modes observe target verification in isolated engine workers;
ordinary parity uses HTTP serving. Both use the same checkpoint and input IDs.
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
from pathlib import Path

if __package__:
    from .parity_prompts import PROMPTS
    from .parity_utils import compare_results as compare_results
    from .parity_utils import parity_report
else:
    from parity_prompts import PROMPTS
    from parity_utils import compare_results as compare_results
    from parity_utils import parity_report


def mlx_generate(
    model_path: str, prompts: list[str], max_tokens: int, top_k: int | None = None
) -> list[dict]:
    import mlx.core as mx
    import numpy as np
    from mlx_lm import load
    from mlx_lm.generate import generate_step
    from mlx_lm.sample_utils import make_sampler

    from vllm_metal.compat import _patch_transformers_exaone4_config

    _patch_transformers_exaone4_config()
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
    model: str,
    max_model_len: int,
    max_num_seqs: int,
    gpu_memory_utilization: float,
    log_path: Path,
    env: dict,
    speculative_config: dict | None = None,
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
                "--gpu-memory-utilization",
                str(gpu_memory_utilization),
                "--no-enable-prefix-caching",
                "--generation-config",
                "vllm",
            ]
            + (
                ["--speculative-config", json.dumps(speculative_config)]
                if speculative_config is not None
                else []
            ),
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
    gpu_memory_utilization: float = 0.65,
    speculative_config: dict | None = None,
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
    if env.get("VLLM_METAL_BACKEND", "mlx") == "mps":
        env["PYTORCH_ENABLE_MPS_FALLBACK"] = "0"
    env["PYTHONPATH"] = os.pathsep.join(
        filter(None, [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH")])
    )
    env.setdefault("VLLM_METAL_MLA_KERNEL", "1")
    env.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    reference_path = output_dir / "reference.json"
    reference_path.write_text(json.dumps(prompts))
    print("Generating native mlx-lm reference...", flush=True)
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
        model,
        max_model_len,
        max(batch_sizes),
        gpu_memory_utilization,
        output_dir / "serve.log",
        env,
        speculative_config,
    ) as base_url:
        for size in batch_sizes:
            print(f"Prompts/request: {size}", flush=True)
            outputs = http_generate(base_url, model, reference, max_tokens, top_k, size)
            log_path = output_dir / f"batch-{size}.log"
            with log_path.open("w") as log, redirect_stdout(log):
                comparison = parity_report(
                    reference, outputs, max_tokens=max_tokens, top_k=top_k
                )
            report = log_path.read_text()
            print(report, end="")
            counts = comparison["counts"]
            matched = comparison["passed"]
            (output_dir / f"batch-{size}.json").write_text(
                json.dumps({"outputs": outputs, "comparison": comparison}, indent=2)
            )
            with summary.open("a") as output:
                print(
                    f"| {size} | {counts['EXACT']} | {counts['TOP_K_MATCH']} | {counts['FAIL']} | {int(not matched)} |",
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


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--model", required=True)
    parser.add_argument(
        "--prompt",
        action="append",
        help="Plain text prompt; repeat for multiple prompts",
    )
    parser.add_argument("--max-tokens", type=int, default=10)
    parser.add_argument(
        "--top-k",
        type=int,
        help="Accept mutual top-k membership at the first divergence (default: 5 with --method, exact otherwise)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        nargs="+",
        default=[1, 2],
        help="Prompts per request batch (default: 1 2)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Artifact directory (default: a new temporary directory)",
    )
    parser.add_argument(
        "--gpu-memory-utilization",
        type=float,
        default=0.65,
        help="KV budget fraction passed to the vLLM server (default: 0.65)",
    )
    parser.add_argument(
        "--method",
        choices=["dflash", "dspark", "draft_model"],
        help="Compare speculative serving with native MLX and target-only serving",
    )
    parser.add_argument(
        "--speculative-config",
        type=json.loads,
        help="JSON speculative config for HTTP parity (including MPS EAGLE3)",
    )
    parser.add_argument("--draft", help="Draft checkpoint for --method")
    parser.add_argument("--num-draft-tokens", type=int, default=3)
    parser.add_argument("--dspark-draft-topk", type=int)
    parser.add_argument("--dspark-draft-quantization", choices=["q4"])
    parser.add_argument("--draft-schedule", type=json.loads)
    parser.add_argument(
        "--audit-continuations",
        action="store_true",
        help="With --method, strictly replay every emitted prefix through native MLX",
    )
    parser.add_argument(
        "--worker",
        choices=["native", "target", "dflash", "dspark", "draft_model", "replay"],
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--generate-reference", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be positive")
    if min(args.batch_size) < 1 or len(set(args.batch_size)) != len(args.batch_size):
        parser.error("--batch-size values must be positive and unique")
    if args.speculative_config is not None and not isinstance(
        args.speculative_config, dict
    ):
        parser.error("--speculative-config must be a JSON object")
    if args.method is not None:
        if args.speculative_config is not None:
            parser.error("--method cannot be combined with --speculative-config")
        if os.environ.get("VLLM_METAL_BACKEND", "mlx") != "mlx":
            parser.error("--method requires the MLX backend")
        if args.generate_reference:
            parser.error("--generate-reference cannot be combined with --method")
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from tools import speculative_parity

        args.target = args.model
        args.top_k = 5 if args.top_k is None else args.top_k
        if args.output_dir is None:
            args.output_dir = Path(tempfile.mkdtemp(prefix="metal-parity-")) / "results"
        args.output_dir = args.output_dir.resolve()
        speculative_parity.validate_args(args, parser)
        print(f"Artifacts: {args.output_dir}", flush=True)
        raise SystemExit(0 if speculative_parity.run(args) else 1)
    if (
        args.draft
        or args.dspark_draft_topk is not None
        or args.dspark_draft_quantization is not None
        or args.draft_schedule is not None
        or args.audit_continuations
        or args.worker
        or args.num_draft_tokens != 3
    ):
        parser.error("Draft and continuation-audit options require --method")
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
        args.gpu_memory_utilization,
        args.speculative_config,
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
