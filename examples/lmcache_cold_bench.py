"""Clean cold-prefill TTFT measurement: median of 5 trials per length, APC reset
before each cold trial. Pairs with the stable LMCache restore numbers."""

import json
import os
import statistics
import time

os.environ["VLLM_ALLOW_INSECURE_SERIALIZATION"] = "1"
os.environ.setdefault("VLLM_LOGGING_LEVEL", "WARNING")
from vllm_metal.bootstrap import bootstrap_metal_platform

bootstrap_metal_platform(require_metal=True)


def main():
    from vllm import LLM, SamplingParams

    llm = LLM(model="Qwen/Qwen3-0.6B", max_model_len=8192, gpu_memory_utilization=0.25)
    sp1 = SamplingParams(temperature=0.0, max_tokens=1)
    llm.generate(["warmup warmup warmup"], sp1, use_tqdm=False)

    unit = "In a distant galaxy far beyond the reach of known science, explorers found "
    lengths = [256, 512, 1024, 2048, 4096]
    out = []
    for target in lengths:
        reps = max(1, target // 12)
        # use DISTINCT prompt per trial so APC-reset truly gives a cold prefill each time
        colds = []
        ntok = None
        for trial in range(5):
            prompt = f"[trial {trial} salt {target}] " + unit * reps
            llm.reset_prefix_cache()
            t = time.perf_counter()
            o = llm.generate([prompt], sp1, use_tqdm=False)
            colds.append((time.perf_counter() - t) * 1000)
            ntok = len(o[0].prompt_token_ids)
        med = statistics.median(colds)
        out.append(
            {
                "ntok": ntok,
                "cold_ms_median": round(med, 1),
                "cold_ms_min": round(min(colds), 1),
                "cold_ms_all": [round(c, 1) for c in colds],
            }
        )
        print(
            f"len={ntok:5d}  cold median={med:7.1f}ms  min={min(colds):7.1f}  all={[round(c) for c in colds]}"
        )
    json.dump(out, open("/tmp/cold_results.json", "w"), indent=2)
    print("WROTE /tmp/cold_results.json")


if __name__ == "__main__":
    main()
