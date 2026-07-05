"""TTFT: with the correct connector, does APC-reset + LMCache retrieve beat cold recompute?
Longer prompt => more tokens skipped => bigger win. Median of 3 per condition."""

import os
import statistics
import time

os.environ["VLLM_METAL_ENABLE_LMCACHE"] = "1"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm_metal.bootstrap import bootstrap_metal_platform

bootstrap_metal_platform(require_metal=True)


def _stats(w):
    c = w.model_runner._lmcache_connector
    return None if c is None else dict(c.stats())


def ttft(llm, prompt, sp):
    t = time.perf_counter()
    o = llm.generate([prompt], sp, use_tqdm=False)
    return (time.perf_counter() - t) * 1000, o[0].outputs[0].text


def main():
    from vllm import LLM, SamplingParams
    from vllm.config import KVTransferConfig

    kvt = KVTransferConfig(
        kv_connector="MetalLMCacheKVConnector",
        kv_connector_module_path="vllm_metal.lmcache_kv_connector",
        kv_role="kv_both",
    )
    llm = LLM(
        model="Qwen/Qwen3-0.6B",
        max_model_len=4096,
        gpu_memory_utilization=0.25,
        kv_transfer_config=kvt,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    unit = "In a distant galaxy far beyond the reach of known science, explorers found "
    prompt = unit * 80  # ~1200 tokens
    llm.generate(["warmup"], sp, use_tqdm=False)

    # COLD baseline: distinct prompt each trial + APC reset => true recompute, no LMCache hit
    colds = []
    for i in range(3):
        llm.reset_prefix_cache()
        salt = f"[cold {i}] "
        ms, _ = ttft(llm, salt + prompt, sp)
        colds.append(ms)
    cold = statistics.median(colds)
    print(f"COLD median={cold:.1f}ms  all={[round(x) for x in colds]}")

    # Prime LMCache with the target prompt (store), then reset APC, then measure retrieve-served TTFT
    llm.reset_prefix_cache()
    ms0, out0 = ttft(llm, prompt, sp)
    print(
        f"PRIME (cold+store)={ms0:.1f}ms out={out0!r} stats={llm.llm_engine.collective_rpc(_stats)[0]}"
    )
    lmc = []
    outs = []
    for _ in range(3):
        llm.reset_prefix_cache()  # kill APC so only LMCache can serve the prefix
        ms, out = ttft(llm, prompt, sp)
        lmc.append(ms)
        outs.append(out)
    lm = statistics.median(lmc)
    print(f"LMCACHE-served median={lm:.1f}ms  all={[round(x) for x in lmc]}")
    print(f"stats={llm.llm_engine.collective_rpc(_stats)[0]}")
    print(f"OUTPUT match cold-vs-lmcache: {out0 == outs[-1]}  (out={outs[-1]!r})")
    print(f"TTFT SPEEDUP cold/lmcache = {cold / lm:.2f}x")


if __name__ == "__main__":
    main()
