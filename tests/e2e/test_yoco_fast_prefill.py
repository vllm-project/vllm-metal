# SPDX-License-Identifier: Apache-2.0
"""Real-model parity for YOCO fast prefill."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_REAL_GEMMA4_MODEL_ENV = "GEMMA4_FAST_PREFILL_MODEL_PATH"
_REAL_GEMMA4_FALLBACK_MODEL_ENV = "GEMMA4_MODEL_PATH"
_REAL_GEMMA4_PROMPTS = [
    "The capital of France is",
    "One plus one equals",
]
_REAL_GEMMA4_MAX_MODEL_LEN = 1024
_REAL_GEMMA4_MAX_TOKENS = 8
_FAST_PREFILL_ENABLED_ATTR = "_vllm_metal_yoco_fast_prefill_enabled"
_REAL_RESULT_MARKER = "VLLM_METAL_YOCO_FAST_PREFILL_RESULT="


def _run_real_gemma4_paged_path(
    model_path: str,
    *,
    skip_fast_prefill_hook: bool = False,
) -> tuple[dict[str, list[int]], dict[str, list[float]], bool]:
    gpu_memory_utilization = float(
        os.environ.get("GEMMA4_FAST_PREFILL_MEMORY_FRACTION", "0.5")
    )
    env = os.environ.copy()
    env["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"

    script = f"""
import json
from contextlib import nullcontext, suppress
from unittest.mock import patch

from vllm import LLM, SamplingParams

model_path = {model_path!r}
prompts = {json.dumps(_REAL_GEMMA4_PROMPTS)}
enabled_attr = {_FAST_PREFILL_ENABLED_ATTR!r}
fast_prefill_hook = (
    patch(
        "vllm_metal.v1.cache_policy.try_enable_gemma4_yoco_fast_prefill",
        return_value=False,
    )
    if {skip_fast_prefill_hook!r}
    else nullcontext()
)
with fast_prefill_hook:
    llm = LLM(
        model=model_path,
        max_model_len={_REAL_GEMMA4_MAX_MODEL_LEN},
        max_num_seqs={len(_REAL_GEMMA4_PROMPTS)},
        enable_prefix_caching=False,
        gpu_memory_utilization={gpu_memory_utilization!r},
    )
runner = llm.llm_engine.model_executor.driver_worker.model_runner
model = runner.model
language_model = getattr(model, "language_model", None)
candidates = [
    model,
    getattr(model, "model", None),
    language_model,
    getattr(language_model, "model", None),
]
fast_prefill_enabled = any(
    bool(getattr(candidate, enabled_attr, False))
    for candidate in candidates
    if candidate is not None
)
sp = SamplingParams(
    temperature=0,
    max_tokens={_REAL_GEMMA4_MAX_TOKENS},
    ignore_eos=True,
)
outputs = llm.generate(prompts, sp)
tokens = {{o.prompt: list(o.outputs[0].token_ids) for o in outputs}}
prompt_outputs = llm.generate(
    prompts, SamplingParams(temperature=0, max_tokens=1, prompt_logprobs=0)
)
prompt_logprobs = {{
    o.prompt: [
        entry[token_id].logprob
        for token_id, entry in zip(o.prompt_token_ids[1:], o.prompt_logprobs[1:])
    ]
    for o in prompt_outputs
}}
print(
    {_REAL_RESULT_MARKER!r}
    + json.dumps(
        {{
            "tokens": tokens,
            "prompt_logprobs": prompt_logprobs,
            "enabled": fast_prefill_enabled,
        }}
    ),
    flush=True,
)
engine = getattr(llm, "llm_engine", None)
engine_core = getattr(engine, "engine_core", None)
shutdown = getattr(engine_core, "shutdown", None)
if callable(shutdown):
    with suppress(Exception):
        shutdown()
"""
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=Path(__file__).resolve().parents[2],
        env=env,
        text=True,
        capture_output=True,
        timeout=300,
        check=False,
    )
    output = f"{result.stdout}\n{result.stderr}"
    if result.returncode != 0:
        raise AssertionError(
            "real Gemma4 paged-path subprocess failed "
            f"(skip_fast_prefill_hook={skip_fast_prefill_hook}, "
            f"code={result.returncode})\n"
            f"{output[-8000:]}"
        )
    for line in output.splitlines():
        if line.startswith(_REAL_RESULT_MARKER):
            payload = json.loads(line[len(_REAL_RESULT_MARKER) :])
            return (
                payload["tokens"],
                payload["prompt_logprobs"],
                bool(payload["enabled"]),
            )
    raise AssertionError(
        f"real Gemma4 paged-path subprocess did not report result\n{output[-8000:]}"
    )


@pytest.mark.slow
def test_real_gemma4_paged_path_default_fast_prefill_matches_original() -> None:
    model_path = os.environ.get(_REAL_GEMMA4_MODEL_ENV) or os.environ.get(
        _REAL_GEMMA4_FALLBACK_MODEL_ENV
    )
    if not model_path:
        pytest.skip(
            f"Set {_REAL_GEMMA4_MODEL_ENV} or {_REAL_GEMMA4_FALLBACK_MODEL_ENV} "
            "to run the real Gemma4 fast-prefill parity test"
        )
    if not os.path.isdir(model_path):
        pytest.skip(f"{model_path} is not a directory")

    original_tokens, original_prompt_logprobs, original_enabled = (
        _run_real_gemma4_paged_path(model_path, skip_fast_prefill_hook=True)
    )
    fast_tokens, fast_prompt_logprobs, fast_enabled = _run_real_gemma4_paged_path(
        model_path,
    )

    assert not original_enabled
    assert fast_enabled
    assert fast_tokens == original_tokens
    assert fast_prompt_logprobs == original_prompt_logprobs
