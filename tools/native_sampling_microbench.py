# SPDX-License-Identifier: Apache-2.0
"""Sampling-only microbench: torch CPU round-trip vs the native MLX graph.

Times one sampling step over a full Qwen-sized vocabulary for both paths at
decode-like batch sizes. The torch arm replicates the vLLM sampler math the
Metal runner uses today (``apply_top_k_top_p`` + softmax + exponential +
argmax on CPU); the native arm times ``mlx_random_tokens`` synchronously —
in the decode pipeline the same graph defers with the step and overlaps the
next forward, so its effective cost is lower than reported here.

Usage:

    python tools/native_sampling_microbench.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

from vllm_metal.v1.sampling_batch import mlx_random_tokens

VOCAB_SIZE = 151936
TOP_K = 20
TOP_P = 0.95
WARMUP_ITERS = 5
TIMED_ITERS = 50


def _time_torch(logits: torch.Tensor, batch: int) -> float:
    def step() -> torch.Tensor:
        filtered = apply_top_k_top_p(
            logits.clone(),
            torch.full((batch,), TOP_K),
            torch.full((batch,), TOP_P),
        )
        probs = filtered.softmax(dim=-1, dtype=torch.float32)
        noise = torch.empty_like(probs)
        noise.exponential_()
        return probs.div(noise).argmax(dim=-1)

    for _ in range(WARMUP_ITERS):
        step()
    started = time.perf_counter()
    for _ in range(TIMED_ITERS):
        step()
    return (time.perf_counter() - started) / TIMED_ITERS * 1000


def _time_native(logits: mx.array, params: list[SamplingParams]) -> float:
    key = mx.random.key(1)

    def step(step_key: mx.array) -> None:
        tokens = mlx_random_tokens(logits, params, step_key)
        mx.eval(tokens)

    for _ in range(WARMUP_ITERS):
        key, subkey = mx.random.split(key)
        step(subkey)
    started = time.perf_counter()
    for _ in range(TIMED_ITERS):
        key, subkey = mx.random.split(key)
        step(subkey)
    return (time.perf_counter() - started) / TIMED_ITERS * 1000


def main() -> None:
    sampling_params = SamplingParams(temperature=0.7, top_k=TOP_K, top_p=TOP_P)
    for batch in (1, 8):
        logits_mx = mx.random.normal((batch, VOCAB_SIZE), key=mx.random.key(0))
        mx.eval(logits_mx)
        logits_torch = torch.tensor(logits_mx.tolist())

        torch_ms = _time_torch(logits_torch, batch)
        native_ms = _time_native(logits_mx, [sampling_params] * batch)
        print(
            f"batch={batch} vocab={VOCAB_SIZE}: "
            f"torch CPU {torch_ms:.2f} ms, native MLX {native_ms:.2f} ms (sync)"
        )


if __name__ == "__main__":
    main()
