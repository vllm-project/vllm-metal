#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare greedy paged serving with the environment's native mlx-lm.

Both backends use the same checkpoint and input IDs. CI can reuse a saved
native reference and an HTTP server across request concurrency levels.
See docs/tools.md or --help for usage.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import urllib.request
from concurrent.futures import ThreadPoolExecutor
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


def metal_generate(
    model_path: str,
    reference: list[dict],
    max_tokens: int,
    top_k: int | None = None,
    batch_size: int = 1,
) -> list[dict]:
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_METAL_USE_PAGED_ATTENTION"] = "1"
    os.environ.setdefault("VLLM_METAL_MEMORY_FRACTION", "0.3")
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=model_path,
        max_model_len=max(len(row["input_ids"]) for row in reference) + max_tokens,
        max_num_seqs=batch_size,
        enable_prefix_caching=False,
        disable_log_stats=True,
        max_logprobs=top_k or 20,
    )
    outputs = llm.generate(
        [{"prompt_token_ids": row["input_ids"]} for row in reference],
        SamplingParams(
            temperature=0, max_tokens=max_tokens, ignore_eos=True, logprobs=top_k
        ),
        use_tqdm=False,
    )
    results = []
    for out in outputs:
        completion = out.outputs[0]
        top_logprobs = []
        if top_k is not None:
            assert completion.logprobs is not None
            for step in completion.logprobs:
                assert step is not None
                top_logprobs.append(
                    [
                        {
                            "id": token,
                            "text": value.decoded_token,
                            "logprob": value.logprob,
                            "rank": value.rank,
                        }
                        for token, value in step.items()
                    ]
                )
        results.append(
            {
                "tokens": list(completion.token_ids),
                "text": completion.text,
                "top_logprobs": top_logprobs,
            }
        )
    return results


def check_parity(
    model: str,
    prompts: list[str],
    max_tokens: int,
    top_k: int | None = None,
    batch_size: int = 1,
) -> bool:
    """Run both backends and report the first differing token for each prompt."""
    if not prompts:
        raise ValueError("At least one prompt is required.")
    if not Path(model).is_dir():
        from huggingface_hub import snapshot_download

        model = snapshot_download(model)
    model = str(Path(model).resolve())
    reference = run_backend("mlx", model, prompts, max_tokens, top_k)
    outputs = run_backend("metal", model, reference, max_tokens, top_k, batch_size)
    return compare_results(reference, outputs, max_tokens=max_tokens, top_k=top_k)


def http_generate(
    base_url: str,
    model: str,
    reference: list[dict],
    max_tokens: int,
    top_k: int | None = None,
    concurrency: int = 1,
) -> list[dict]:
    """Generate through one running server, preserving input order."""
    if not reference:
        raise ValueError("At least one reference prompt is required.")

    def generate(row: dict) -> dict:
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/completions",
            data=json.dumps(
                {
                    "model": model,
                    "prompt": row["input_ids"],
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
            [choice] = json.load(response)["choices"]
        if choice["prompt_token_ids"] != row["input_ids"]:
            raise ValueError("Server changed the reference input token IDs.")
        return http_result(choice, top_k)

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        return list(pool.map(generate, reference))


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
            # means the sampled token is outside top-K, just as offline.
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


def run_backend(
    backend: str,
    model: str,
    inputs: list,
    max_tokens: int,
    top_k: int | None = None,
    batch_size: int = 1,
) -> list[dict]:
    """Run one backend in a fresh process, releasing its model before returning."""
    with tempfile.TemporaryDirectory(prefix="metal-parity-") as directory:
        data = Path(directory) / "tokens.json"
        data.write_text(json.dumps(inputs))
        env = os.environ.copy()
        env["PYTHONPATH"] = os.pathsep.join(
            filter(
                None, [str(Path(__file__).resolve().parents[1]), env.get("PYTHONPATH")]
            )
        )
        subprocess.run(
            [
                sys.executable,
                str(Path(__file__).resolve()),
                "--model",
                model,
                "--max-tokens",
                str(max_tokens),
                "--batch-size",
                str(batch_size),
                "--backend",
                backend,
                "--data",
                str(data),
            ]
            + (["--top-k", str(top_k)] if top_k is not None else []),
            check=True,
            env=env,
        )
        return json.loads(data.read_text())


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
        "--batch-size",
        type=int,
        default=1,
        help="Maximum concurrent Metal requests (default: 1); MLX runs sequentially",
    )
    parser.add_argument(
        "--top-k",
        type=int,
        help="Accept mutual top-k membership at the first divergence",
    )
    parser.add_argument("--base-url", help="Running vLLM server URL, including /v1")
    parser.add_argument(
        "--reference", type=Path, help="Saved native reference JSON for HTTP comparison"
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=1,
        help="Concurrent HTTP requests (default: 1)",
    )
    parser.add_argument("--backend", choices=("mlx", "metal"), help=argparse.SUPPRESS)
    parser.add_argument("--data", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.max_tokens < 1:
        parser.error("--max-tokens must be positive")
    if args.top_k is not None and args.top_k < 1:
        parser.error("--top-k must be positive")
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.concurrency < 1:
        parser.error("--concurrency must be positive")
    if bool(args.base_url) != bool(args.reference):
        parser.error("--base-url and --reference must be used together")
    if args.reference:
        reference = json.loads(args.reference.read_text())
        outputs = http_generate(
            args.base_url,
            reference["model"],
            reference["results"],
            reference["max_tokens"],
            reference["top_k"],
            args.concurrency,
        )
        passed = compare_results(
            reference["results"],
            outputs,
            max_tokens=reference["max_tokens"],
            top_k=reference["top_k"],
        )
        raise SystemExit(0 if passed else 1)
    if args.backend:
        data = json.loads(args.data.read_text())
        if args.backend == "mlx":
            outputs = mlx_generate(args.model, data, args.max_tokens, args.top_k)
        else:
            outputs = metal_generate(
                args.model, data, args.max_tokens, args.top_k, args.batch_size
            )
        args.data.write_text(json.dumps(outputs))
    else:
        passed = check_parity(
            args.model,
            args.prompt or PROMPTS,
            args.max_tokens,
            args.top_k,
            args.batch_size,
        )
        raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
