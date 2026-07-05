import os

os.environ["VLLM_METAL_ENABLE_LMCACHE"] = "1"
os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm_metal.bootstrap import bootstrap_metal_platform

bootstrap_metal_platform(require_metal=True)


def _stats(w):
    c = w.model_runner._lmcache_connector
    return None if c is None else dict(c.stats())


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
        max_model_len=2048,
        gpu_memory_utilization=0.25,
        kv_transfer_config=kvt,
    )
    sp = SamplingParams(temperature=0.0, max_tokens=8)
    prompt = "The number seven is a prime number and also " * 30
    llm.reset_prefix_cache()
    o1 = llm.generate([prompt], sp, use_tqdm=False)
    out1 = o1[0].outputs[0].text
    n = len(o1[0].prompt_token_ids)
    print(
        f"CALL1 ntok={n} stats={llm.llm_engine.collective_rpc(_stats)[0]} out1={out1!r}"
    )
    llm.reset_prefix_cache()
    o2 = llm.generate([prompt], sp, use_tqdm=False)
    out2 = o2[0].outputs[0].text
    print(f"CALL2 stats={llm.llm_engine.collective_rpc(_stats)[0]} out2={out2!r}")
    print("MATCH:", out1 == out2)


if __name__ == "__main__":
    main()
