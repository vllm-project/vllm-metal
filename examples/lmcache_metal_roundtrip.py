# ruff: noqa: N806
# SPDX-License-Identifier: Apache-2.0
"""End-to-end demo: route the live vllm-metal MLX KV cache through LMCache.

Runs Qwen3-0.6B on Apple Metal/MLX, then inside the worker:
  1. stores the live MLX paged KV into LMCache,
  2. zeroes those slots in the live MLX cache (proves data really leaves),
  3. retrieves from LMCache back into the live cache,
  4. asserts the restored KV is bit-exact.

Usage:
  python examples/lmcache_metal_roundtrip.py
"""

import os

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"

from vllm_metal.bootstrap import bootstrap_metal_platform

bootstrap_metal_platform(require_metal=True)


def _roundtrip(worker):
    import time

    import torch

    from vllm_metal.lmcache_integration import MetalLMCacheConnector

    cache = worker.model_runner._paged_attention_runtime.kv_cache
    conn = MetalLMCacheConnector(instance_id="demo")

    N = 500
    token_ids = list(range(1000, 1000 + N))
    block_size = cache.block_size
    # Contiguous slots 0..N-1 => block_ids 0..ceil(N/bs)-1
    n_blocks = (N + block_size - 1) // block_size
    block_ids = list(range(n_blocks))

    # Snapshot ground-truth KV at those slots (bridged, bidirectional views).
    from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch

    def snap():
        out = []
        for i in range(cache.num_layers):
            k = mlx_to_torch(cache.key_caches[i], device="cpu")
            v = mlx_to_torch(cache.value_caches[i], device="cpu")
            nb, bs, nh, hd = k.shape
            f = torch.stack([k, v], dim=0).reshape(2, nb * bs, nh, hd)
            out.append(f[:, :N].clone())
        return out

    truth = snap()
    conn.store(cache, token_ids, block_ids)
    time.sleep(0.3)
    matched = conn.lookup(token_ids)

    # Corrupt the live MLX cache at those slots.
    for i in range(cache.num_layers):
        k = mlx_to_torch(cache.key_caches[i], device="cpu")
        v = mlx_to_torch(cache.value_caches[i], device="cpu")
        nb, bs, nh, hd = k.shape
        torch.stack([k, v], dim=0).reshape(2, nb * bs, nh, hd)[:, :N] = 0

    hits = conn.retrieve(cache, token_ids, block_ids)
    restored = snap()
    ok = all(
        torch.equal(restored[i][:, :hits], truth[i][:, :hits])
        for i in range(cache.num_layers)
    )
    return {
        "lookup_matched": matched,
        "retrieve_hits": hits,
        "restored_bit_exact": ok,
        "stats": conn.stats(),
    }


def main():
    from vllm import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen3-0.6B", max_model_len=1024, gpu_memory_utilization=0.15)
    llm.generate(
        ["The quick brown fox jumps over the lazy dog. " * 8],
        SamplingParams(temperature=0.0, max_tokens=8),
    )
    print("ROUNDTRIP:", llm.llm_engine.collective_rpc(_roundtrip))


if __name__ == "__main__":
    main()
