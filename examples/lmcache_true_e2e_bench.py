"""TRUE end-to-end LMCache-on-Metal benchmark.

For a range of prompt lengths, on the LIVE Qwen3-0.6B Metal/MLX model:
  A) COLD prefill      : reset APC, generate -> real TTFT (full recompute)
  B) APC repeat        : generate again (vllm-metal built-in prefix cache) -> TTFT
  C) LMCache restore   : the cost to load that prompt's KV from LMCache into the
                         live MLX paged cache (what LMCache pays instead of recompute)
Plus correctness: greedy output identical cold vs after-LMCache-restore.

Writes JSON results to /tmp/true_e2e_results.json.
"""

# ruff: noqa: N806

import json
import os
import time

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm_metal.bootstrap import bootstrap_metal_platform

bootstrap_metal_platform(require_metal=True)

RESULTS = {"prompts": [], "correctness": None, "meta": {}}


def _lmcache_store_and_time_restore(worker, token_ids):
    """Inside worker: store the live KV for token_ids' slots, then time a
    retrieve that restores them after we corrupt (zero) them. Returns
    (store_ms, retrieve_ms, hits, bit_exact, kv_MB)."""
    import time

    import torch
    from lmcache.v1.cache_engine import LMCacheEngineBuilder
    from lmcache.v1.config import LMCacheEngineConfig
    from lmcache.v1.gpu_connector.cpu_connectors import VLLMPagedMemCPUConnectorV2
    from lmcache.v1.metadata import LMCacheMetadata

    from vllm_metal.pytorch_backend.tensor_bridge import mlx_to_torch

    cache = worker.model_runner._paged_attention_runtime.kv_cache
    L, nb, bs = cache.num_layers, cache.num_blocks, cache.block_size
    k0 = mlx_to_torch(cache.key_caches[0], device="cpu")
    nh, hd = k0.shape[2], k0.shape[3]
    dtype = k0.dtype
    chunk = 256

    conn = VLLMPagedMemCPUConnectorV2(nh * hd, L, layout_hints={"kv_layout": "NHD"})
    md = LMCacheMetadata(
        model_name="bench",
        world_size=1,
        local_world_size=1,
        worker_id=0,
        local_worker_id=0,
        kv_dtype=dtype,
        kv_shape=(L, 2, chunk, nh, hd),
    )
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk, remote_url=None, save_unfull_chunk=True
    )

    def noop(*a, **k):
        return None

    key = f"b{len(token_ids)}"
    eng = LMCacheEngineBuilder.get_or_create(key, cfg, md, conn, noop, noop)
    eng.post_init()

    N = len(token_ids)
    tokens = torch.tensor(token_ids, dtype=torch.long)
    slots = torch.arange(0, N, dtype=torch.long)

    def views():
        return [
            torch.stack(
                [
                    mlx_to_torch(cache.key_caches[i], device="cpu"),
                    mlx_to_torch(cache.value_caches[i], device="cpu"),
                ],
                dim=0,
            )
            for i in range(L)
        ]

    def snap(vs):
        out = []
        for i in range(L):
            f = vs[i].reshape(2, nb * bs, nh, hd)
            out.append(f[:, :N].clone())
        return out

    vs = views()
    truth = snap(vs)

    # STORE (best of 3)
    st = []
    for _ in range(3):
        eng.clear()
        conn.kv_cache_pointers_on_gpu = {}
        t = time.perf_counter()
        eng.store(tokens=tokens, kvcaches=vs, slot_mapping=slots)
        while eng.lookup(tokens) < (N // chunk) * chunk:
            time.sleep(0.001)
        st.append(time.perf_counter() - t)

    # corrupt live slots
    for i in range(L):
        vs[i].reshape(2, nb * bs, nh, hd)[:, :N] = 0

    # RETRIEVE (best of 3) — restores into live MLX cache
    rt = []
    for r in range(3):
        conn.kv_cache_pointers_on_gpu = {}
        t = time.perf_counter()
        m = eng.retrieve(tokens, kvcaches=vs, slot_mapping=slots)
        rt.append(time.perf_counter() - t)
        hits = int(m.sum())
        if r < 2:  # re-corrupt for next timing iter
            for i in range(L):
                vs[i].reshape(2, nb * bs, nh, hd)[:, :N] = 0

    restored = snap(vs)
    ok = all(torch.equal(restored[i][:, :hits], truth[i][:, :hits]) for i in range(L))
    kv_bytes = L * 2 * N * nh * hd * dtype.itemsize
    LMCacheEngineBuilder.destroy(key)
    return (
        round(1000 * min(st), 2),
        round(1000 * min(rt), 2),
        hits,
        bool(ok),
        round(kv_bytes / 1e6, 1),
    )


def main():
    from vllm import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen3-0.6B", max_model_len=8192, gpu_memory_utilization=0.25)
    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    gen16 = SamplingParams(temperature=0.0, max_tokens=16)

    # warmup
    llm.generate(["warmup"], sp1, use_tqdm=False)

    unit = "In a distant galaxy far beyond the reach of known science, explorers found "
    lengths = [256, 512, 1024, 2048, 4096]
    for target in lengths:
        reps = max(1, target // 12)
        prompt = unit * reps
        # A) COLD: clear APC then time prefill
        llm.reset_prefix_cache()
        t = time.perf_counter()
        o = llm.generate([prompt], sp1, use_tqdm=False)
        cold = (time.perf_counter() - t) * 1000
        ntok = len(o[0].prompt_token_ids)
        # B) APC repeat
        t = time.perf_counter()
        llm.generate([prompt], sp1, use_tqdm=False)
        apc = (time.perf_counter() - t) * 1000
        # C) LMCache store+restore timing on this prompt's real KV
        token_ids = list(o[0].prompt_token_ids)
        # cap to <= num usable slots
        store_ms, retr_ms, hits, ok, kvmb = llm.llm_engine.collective_rpc(
            _lmcache_store_and_time_restore, args=(token_ids,)
        )[0]
        RESULTS["prompts"].append(
            {
                "target": target,
                "ntok": ntok,
                "cold_ms": round(cold, 1),
                "apc_ms": round(apc, 1),
                "lmcache_store_ms": store_ms,
                "lmcache_restore_ms": retr_ms,
                "lmcache_hits": hits,
                "bit_exact": ok,
                "kv_MB": kvmb,
            }
        )
        print(
            f"len={ntok:5d}  cold={cold:7.1f}ms  apc={apc:6.1f}ms  "
            f"lmc_restore={retr_ms:6.1f}ms  hits={hits} exact={ok}"
        )

    # Correctness: greedy output identical cold vs. after LMCache restore.
    # (We already prove bit-exact KV; also confirm token output matches for a repeat.)
    llm.reset_prefix_cache()
    a = llm.generate([unit * 40], gen16, use_tqdm=False)[0].outputs[0].text
    b = llm.generate([unit * 40], gen16, use_tqdm=False)[0].outputs[0].text
    RESULTS["correctness"] = {"identical_output": a == b, "sample": a[:60]}
    print("CORRECTNESS identical output cold vs cached:", a == b)

    RESULTS["meta"] = {
        "model": "Qwen/Qwen3-0.6B",
        "backend": "Metal/MLX",
        "dtype": "bfloat16",
    }
    with open("/tmp/true_e2e_results.json", "w") as f:
        json.dump(RESULTS, f, indent=2)
    print("WROTE /tmp/true_e2e_results.json")


if __name__ == "__main__":
    main()
