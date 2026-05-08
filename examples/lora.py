# SPDX-License-Identifier: Apache-2.0
"""LoRA quickstart on vllm-metal.

MODEL_ID="Qwen/Qwen3.5-0.8B" LORA_PATH="Oysiyl/qwen3.5-0.8b-unslop-good-lora-v1" LORA_NAME="unslop-v1" VLLM_ENABLE_V1_MULTIPROCESSING=0 python examples/lora.py
"""

from __future__ import annotations

import os
import sys

from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest

os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")

MODEL_ID = os.environ.get("MODEL_ID", "Qwen/Qwen3.5-0.8B")
LORA_PATH = os.environ.get("LORA_PATH")
LORA_NAME = os.environ.get("LORA_NAME", "demo-lora")
LORA_RANK = int(os.environ.get("LORA_RANK", "16"))
PROMPT = os.environ.get("PROMPT", "Hi, tell me about yourself.")

if not LORA_PATH:
    sys.exit(
        "LORA_PATH is not set. Point it at a PEFT adapter directory "
        "must contain adapter_config.json + adapter_model.safetensors"
        "or an HF repo id. See the docstring at the top of this file."
    )


if __name__ == "__main__":
    print(f"Loading base model: {MODEL_ID}")
    llm = LLM(
        model=MODEL_ID,
        enable_lora=True,
        max_loras=1,
        max_lora_rank=LORA_RANK,
        max_cpu_loras=2,
        max_model_len=512,
        max_num_seqs=1,
    )
    sp = SamplingParams(temperature=0, max_tokens=80)

    print("WITHOUT LoRA")
    base_out = llm.generate([PROMPT], sp)[0].outputs[0].text
    print(base_out)

    print(f"Attaching LoRA: name={LORA_NAME!r} path={LORA_PATH!r}")
    lora_req = LoRARequest(lora_name=LORA_NAME, lora_int_id=1, lora_path=LORA_PATH)
    llm.llm_engine.add_lora(lora_req)

    print("WITH LoRA")
    lora_out = llm.generate([PROMPT], sp, lora_request=lora_req)[0].outputs[0].text
    print(lora_out)

    print("DIFF")
    if base_out.strip() == lora_out.strip():
        print("Outputs are identical.")
    else:
        print("Outputs differ.")
