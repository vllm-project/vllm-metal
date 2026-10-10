#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Compare FP32 Laya pooling with original Laya 0.4.1 using identical token IDs.

Run --backend reference in an environment with laya==0.4.1, then use metal
or http in the vllm-metal environment. Reference instrumentation uses Laya's
private _infer method and is intentionally pinned to that package version.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import urllib.request
from pathlib import Path

import numpy as np


def checkpoint_hash(model: Path) -> str:
    digest = hashlib.sha256()
    with (model / "model.safetensors").open("rb") as weights:
        for chunk in iter(lambda: weights.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def cases():
    labels = [f"department {i}" for i in range(11)]
    for length, state in (
        ("short", "The customer was billed twice and requests a refund."),
        ("long", "Routine account activity. " * 700 + "Please refund the payment."),
    ):
        for kind in ("choice", "score"):
            for count in (2, 3, 5, 6, 10, 11):
                question = {
                    "type": kind,
                    "instructions": "Which department should handle this?"
                    if kind == "choice"
                    else "How urgent is this request?",
                    "criteria": dict.fromkeys(
                        labels[:count], "Department responsibilities"
                    )
                    if kind == "choice"
                    else [f"urgency level {i}" for i in range(count)],
                }
                yield f"{length}-{kind}-{count}", state, question
        yield (
            f"{length}-noul",
            state,
            {
                "type": "noul",
                "instructions": "Does the customer request a refund?",
            },
        )


def reference(args):
    import laya
    import torch
    from laya.common import temp_bucket

    version = importlib.metadata.version("laya")
    if version != "0.4.1":
        raise ValueError(
            f"Reference instrumentation requires laya==0.4.1, got {version}"
        )
    torch.set_num_threads(8)
    agent = laya.load(str(args.model), device="cpu", fast=False, compile=False)
    agent.amp_enabled = False
    agent.model.float().eval()
    original_infer = agent._infer
    captured = {}

    def capture(batch):
        logits, actions = original_infer(batch)
        captured.update(
            batch={
                k: v.detach().cpu().tolist()
                for k, v in batch.items()
                if torch.is_tensor(v)
            },
            logits=logits.detach().float().cpu().numpy(),
            actions=actions.detach().float().cpu().numpy(),
        )
        return logits, actions

    agent._infer = capture
    rows = []
    with torch.inference_mode():
        for name, state, question in cases():
            agent.system_one(state, {"q": question})
            batch = captured["batch"]
            count = sum(batch["marker_mask"][0])
            logits = captured["logits"][0, :count]
            actions = captured["actions"][0]
            qtype = batch["qtype"][0]
            temperature = agent.temperature_by_options.get(
                temp_bucket(qtype, count), agent.temperature[qtype]
            )
            p = np.exp(logits / temperature - np.max(logits / temperature))
            action_p = np.exp(actions - np.max(actions))
            rows.append(
                {
                    "id": name,
                    "token_ids": batch["input_ids"][0][
                        : sum(batch["attention_mask"][0])
                    ],
                    "option_logits": logits.tolist(),
                    "action_logits": actions.tolist(),
                    "option_probabilities": (p / p.sum()).tolist(),
                    "action_probabilities": (action_p / action_p.sum()).tolist(),
                }
            )
    data = {
        "schema": 1,
        "weights_sha256": checkpoint_hash(args.model),
        "laya_version": version,
        "torch_version": torch.__version__,
        "description": "Synthetic implementation-parity probes, not labeled quality data",
        "rows": rows,
    }
    args.reference.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
    print(f"Saved {len(rows)} CPU FP32 reference questions to {args.reference}")


def compare(args):
    data = json.loads(args.reference.read_text())
    if data["schema"] != 1 or not data["rows"]:
        raise ValueError("Expected a nonempty schema=1 Laya reference")
    if checkpoint_hash(args.model) != data["weights_sha256"]:
        raise ValueError("Model weights differ from the reference checkpoint")
    rows = data["rows"]
    if args.backend == "metal":
        from vllm import LLM, PoolingParams

        llm = LLM(
            model=str(args.model),
            runner="pooling",
            dtype="float32",
            max_model_len=max(len(row["token_ids"]) for row in rows),
            enforce_eager=True,
            enable_prefix_caching=False,
            max_num_batched_tokens=4096,
            max_num_seqs=8,
            gpu_memory_utilization=0.5,
        )

        def run(activated):
            outputs = llm.encode(
                [{"prompt_token_ids": row["token_ids"]} for row in rows],
                pooling_task="token_classify",
                pooling_params=PoolingParams(
                    task="token_classify", use_activation=activated
                ),
                use_tqdm=False,
            )
            return [out.outputs.data.float().cpu().numpy() for out in outputs]
    else:
        with urllib.request.urlopen(
            args.base_url + "/v1/models", timeout=30
        ) as response:
            model_name = json.load(response)["data"][0]["id"]

        def run(activated):
            payload = {
                "model": model_name,
                "task": "token_classify",
                "input": [row["token_ids"] for row in rows],
                "use_activation": activated,
            }
            request = urllib.request.Request(
                args.base_url + "/pooling",
                data=json.dumps(payload).encode(),
                headers={"Content-Type": "application/json"},
            )
            with urllib.request.urlopen(request, timeout=120) as response:
                return [np.array(out["data"]) for out in json.load(response)["data"]]

    results = []
    for activated in (True, False):
        for row, actual in zip(rows, run(activated), strict=True):
            suffix = "probabilities" if activated else "logits"
            options = np.array(row[f"option_{suffix}"])
            actions = np.array(row[f"action_{suffix}"])
            expected = np.column_stack([options, np.tile(actions, (len(options), 1))])
            matches = (
                actual.shape == expected.shape
                and np.isfinite(actual).all()
                and np.allclose(actual, expected, atol=args.atol, rtol=args.rtol)
            )
            rank_matches = (
                actual.shape == expected.shape
                and actual[:, 0].argmax() == options.argmax()
            )
            results.append(
                {
                    "id": row["id"],
                    "activation": activated,
                    "allclose": bool(matches),
                    "argmax_equal": bool(rank_matches),
                    "max_abs_error": float(np.max(np.abs(actual - expected)))
                    if actual.shape == expected.shape and np.isfinite(actual).all()
                    else None,
                }
            )
    report = {
        "backend": args.backend,
        "questions": len(rows),
        "atol": args.atol,
        "rtol": args.rtol,
        "weights_sha256": data["weights_sha256"],
        "results": results,
    }
    if args.output:
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    for activated in (True, False):
        subset = [row for row in results if row["activation"] == activated]
        print(
            f"activation={activated}: {sum(r['allclose'] for r in subset)}/{len(subset)} allclose; "
            f"{sum(r['argmax_equal'] for r in subset)}/{len(subset)} argmax; "
            f"max_abs_error={max((r['max_abs_error'] for r in subset if r['max_abs_error'] is not None), default=None)}"
        )
    if not all(row["allclose"] and row["argmax_equal"] for row in results):
        raise SystemExit(1)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--backend", choices=("reference", "metal", "http"), required=True
    )
    parser.add_argument(
        "--model", type=Path, required=True, help="Local original/converted checkpoint"
    )
    parser.add_argument(
        "--reference", type=Path, required=True, help="Reference JSON to write/read"
    )
    parser.add_argument("--output", type=Path, help="Comparison report destination")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--atol", type=float, default=2e-5)
    parser.add_argument("--rtol", type=float, default=2e-5)
    args = parser.parse_args()
    if args.atol < 0 or args.rtol < 0:
        parser.error("Tolerances must be nonnegative")
    if args.backend == "reference":
        reference(args)
    else:
        compare(args)


if __name__ == "__main__":
    main()
