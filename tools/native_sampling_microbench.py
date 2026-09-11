# SPDX-License-Identifier: Apache-2.0
"""Sampling-step microbench for the torch path and native MLX graph.

Times one non-greedy sampling step over a Qwen-sized vocabulary. The torch arm
includes MLX evaluation, torch bridging, min-p, top-k/top-p, and categorical
sampling. The native arm times the equivalent ``SamplingBatch`` graph.

Usage:

    python tools/native_sampling_microbench.py
"""

from __future__ import annotations

import time

import mlx.core as mx
import torch
from vllm.sampling_params import SamplingParams
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p

from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch
from vllm_metal.v1.logits_processors import BatchMinPLogitsProcessor
from vllm_metal.v1.sampling_batch import SamplingBatch

VOCAB_SIZE = 151936
TOP_K = 20
TOP_P = 0.95
MIN_P = 0.05
WARMUP_ITERS = 5
TIMED_ITERS = 50


def _time_torch(logits_mx: mx.array, batch: int) -> float:
    min_p_proc = BatchMinPLogitsProcessor(torch.full((batch,), MIN_P))

    def step() -> torch.Tensor:
        logits_f32 = logits_mx.astype(mx.float32)
        mx.eval(logits_f32)
        logits = mlx_to_torch(logits_f32, device="cpu").clone()
        filtered = apply_top_k_top_p(
            min_p_proc.apply(logits),
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
        tokens = SamplingBatch._native_random_tokens(logits, params, step_key)
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
    sampling_params = SamplingParams(
        temperature=0.7, top_k=TOP_K, top_p=TOP_P, min_p=MIN_P
    )
    for batch in (1, 8):
        logits_mx = mx.random.normal((batch, VOCAB_SIZE), key=mx.random.key(0))
        mx.eval(logits_mx)

        torch_ms = _time_torch(logits_mx, batch)
        native_ms = _time_native(logits_mx, [sampling_params] * batch)
        print(
            f"batch={batch} vocab={VOCAB_SIZE}: "
            f"torch CPU {torch_ms:.2f} ms, native MLX {native_ms:.2f} ms (sync)"
        )


if __name__ == "__main__":
    main()
