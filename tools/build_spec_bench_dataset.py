#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Build the speculative-decoding eval set used to benchmark draft-model SD.

Downloads ``RedHatAI/speculator_benchmarks`` (the dataset the vLLM ``speculators``
repo benchmarks with — natural prompts across 9 task categories, shipped as one
JSONL per category with a ``prompt`` column) and writes a single
``spec_bench``-format JSONL (a ``turns`` column) that vLLM's
``--dataset-name spec_bench`` loader reads directly. Prompts that don't fit the
context budget are dropped, then a fixed number per category is sampled.

Why a natural dataset: spec-decode acceptance rate — and therefore speedup — is
much lower on synthetic sets like ``sonnet``/``random``, so they understate SD.

Run it (writes ./spec_bench_sample.jsonl), then benchmark with greedy sampling
(required for drafting to engage):

    python tools/build_spec_bench_dataset.py

    vllm bench serve --backend vllm --base-url http://127.0.0.1:8000 \
        --model Qwen/Qwen3-8B --endpoint /v1/completions \
        --dataset-name spec_bench --dataset-path spec_bench_sample.jsonl \
        --spec-bench-output-len 128 --num-prompts 100 --max-concurrency 32 \
        --temperature 0 --ignore-eos --seed 0

To change the sample, edit the constants below.
"""

from __future__ import annotations

import glob
import json
import os
import random

from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

# --- knobs ------------------------------------------------------------------
DATASET = "RedHatAI/speculator_benchmarks"  # HF dataset: one *.jsonl per category
TOKENIZER = "Qwen/Qwen3-8B"  # tokenizer for length filtering (use the target model)
PER_CATEGORY = 22  # prompts sampled per category (9 categories -> ~198)
MAX_PROMPT_TOKENS = 1900  # drop prompts longer than this (fits 2048 ctx)
SEED = 1234
OUTPUT = "spec_bench_sample.jsonl"
# ----------------------------------------------------------------------------


def main() -> None:
    snapshot = snapshot_download(repo_id=DATASET, repo_type="dataset")
    tok = AutoTokenizer.from_pretrained(TOKENIZER)
    rng = random.Random(SEED)

    out = []
    for path in sorted(glob.glob(os.path.join(snapshot, "*.jsonl"))):
        category = os.path.basename(path).replace(".jsonl", "")
        rows = [json.loads(line) for line in open(path)]
        fit = [r for r in rows if len(tok(r["prompt"]).input_ids) <= MAX_PROMPT_TOKENS]
        rng.shuffle(fit)
        for r in fit[:PER_CATEGORY]:
            record = {
                "question_id": r.get("question_id"),
                "category": category,
                "turns": [r["prompt"]],
            }
            out.append(record)
    rng.shuffle(out)

    with open(OUTPUT, "w") as fh:
        for r in out:
            fh.write(json.dumps(r) + "\n")
    print(f"wrote {len(out)} prompts -> {OUTPUT}")


if __name__ == "__main__":
    main()
