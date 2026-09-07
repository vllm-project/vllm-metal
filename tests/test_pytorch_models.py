# SPDX-License-Identifier: Apache-2.0
"""Tiny end-to-end checks for upstream PyTorch models on Apple MPS.

The model weights are created locally, and raw token IDs are used throughout,
so these tests never need a tokenizer or network access.  Every case runs in a
new interpreter because vLLM selects its platform while importing plugins.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

_CASES = ("llama", "qwen2", "qwen3")


def _run_case(case: str, model_dir: Path, *, scheduler_stress: bool = False) -> None:
    env = os.environ.copy()
    env.update(
        {
            "VLLM_METAL_MODEL_BACKEND": "torch",
            "VLLM_ENABLE_V1_MULTIPROCESSING": "0",
        }
    )
    command = [sys.executable, str(Path(__file__).resolve()), "--child", case]
    if scheduler_stress:
        command.append("--scheduler-stress")
    command.append(str(model_dir))
    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        env=env,
        timeout=300,
    )
    if result.returncode == 77:
        pytest.skip("PyTorch MPS is unavailable")
    assert result.returncode == 0, (
        f"{case} PyTorch/MPS parity failed (exit {result.returncode}):\n"
        f"{result.stdout[-4000:]}\n{result.stderr[-4000:]}"
    )


@pytest.mark.parametrize("case", _CASES)
def test_tiny_upstream_model_matches_hf_and_mlx(case: str, tmp_path: Path) -> None:
    _run_case(case, tmp_path / case)


def test_chunked_prefill_and_prefix_cache_match_hf_cpu(tmp_path: Path) -> None:
    _run_case("llama", tmp_path / "llama-scheduler", scheduler_stress=True)


def _make_model(case: str):
    import torch
    from transformers import (
        AutoModelForCausalLM,
        LlamaConfig,
        Qwen2Config,
        Qwen3Config,
    )

    common = {
        "vocab_size": 128,
        "hidden_size": 64,
        "intermediate_size": 128,
        "num_hidden_layers": 2,
        "num_attention_heads": 2,
        "num_key_value_heads": 2,
        "max_position_embeddings": 128,
        "bos_token_id": 1,
        "eos_token_id": 2,
        "pad_token_id": 0,
        "tie_word_embeddings": False,
    }
    config_classes = {
        "llama": LlamaConfig,
        "qwen2": Qwen2Config,
        "qwen3": Qwen3Config,
    }
    if case == "qwen3":
        common["head_dim"] = 32
    torch.manual_seed(20260907)
    model = AutoModelForCausalLM.from_config(config_classes[case](**common))
    return model.eval()


def _hf_greedy(model, prompts: list[list[int]], max_tokens: int) -> list[list[int]]:
    import torch

    generated: list[list[int]] = []
    with torch.inference_mode():
        for prompt in prompts:
            sequence = list(prompt)
            new_tokens: list[int] = []
            for _ in range(max_tokens):
                input_ids = torch.tensor([sequence], dtype=torch.long)
                token = int(model(input_ids=input_ids).logits[0, -1].argmax())
                sequence.append(token)
                new_tokens.append(token)
            generated.append(new_tokens)
    return generated


def _mlx_greedy(
    model_dir: Path, prompts: list[list[int]], max_tokens: int
) -> list[list[int]]:
    import mlx.core as mx
    from mlx_lm.utils import load_model

    config = json.loads((model_dir / "config.json").read_text())
    rope_config = config.get("rope_parameters") or config
    rope_theta = rope_config["rope_theta"]
    model, _ = load_model(
        model_dir, lazy=False, model_config={"rope_theta": rope_theta}
    )
    generated: list[list[int]] = []
    try:
        for prompt in prompts:
            sequence = list(prompt)
            new_tokens: list[int] = []
            for _ in range(max_tokens):
                logits = model(mx.array([sequence]))
                token = int(mx.argmax(logits[0, -1]).item())
                sequence.append(token)
                new_tokens.append(token)
            generated.append(new_tokens)
    finally:
        del model
        mx.clear_cache()
    return generated


def _child(case: str, model_dir: Path, scheduler_stress: bool) -> None:
    import torch

    if not torch.backends.mps.is_available():
        raise SystemExit(77)

    if scheduler_stress:
        prefix = list(range(10, 42))
        prompts = [
            prefix + [51, 52, 53, 54, 55],
            prefix + [61, 62, 63],
            [7, 9, 11, 13, 15, 17, 19],
        ]
        max_tokens = 6
    else:
        prompts = [
            [1, 7, 11, 19, 23],
            [1, 31, 29],
            [1, 5, 8, 13, 21, 34, 55, 89],
        ]
        max_tokens = 4

    model = _make_model(case)
    expected = _hf_greedy(model, prompts, max_tokens)
    model.save_pretrained(model_dir, safe_serialization=True)
    del model
    mlx_expected = _mlx_greedy(model_dir, prompts, max_tokens)
    assert mlx_expected == expected, f"HF CPU: {expected}; MLX: {mlx_expected}"

    # Import only after the backend selector has been set by the parent.
    from vllm import LLM, SamplingParams

    llm = LLM(
        model=str(model_dir),
        skip_tokenizer_init=True,
        max_model_len=128 if scheduler_stress else 64,
        max_num_seqs=4,
        max_num_batched_tokens=32 if scheduler_stress else 64,
        kv_cache_memory_bytes=8 * 1024**2,
        enable_chunked_prefill=scheduler_stress,
        enable_prefix_caching=scheduler_stress,
        enforce_eager=True,
        dtype="float32",
        seed=0,
    )
    placement = llm.apply_model(
        lambda model: (
            type(model).__module__.startswith("vllm."),
            all(parameter.device.type == "mps" for parameter in model.parameters()),
        )
    )
    assert placement == [(True, True)]
    params = SamplingParams(
        temperature=0,
        max_tokens=max_tokens,
        ignore_eos=True,
    )
    raw_prompts = [{"prompt_token_ids": prompt} for prompt in prompts]

    def generate() -> list[list[int]]:
        outputs = llm.generate(raw_prompts, params, use_tqdm=False)
        return [list(output.outputs[0].token_ids) for output in outputs]

    actual = generate()
    assert actual == expected, f"HF CPU: {expected}; vLLM MPS: {actual}"
    if scheduler_stress:
        cached = generate()
        assert cached == expected, f"HF CPU: {expected}; cached MPS: {cached}"

        controls = SamplingParams(
            logit_bias={7: 100.0},
            temperature=0,
            max_tokens=2,
            ignore_eos=True,
            logprobs=1,
            prompt_logprobs=1,
        )
        controlled = llm.generate([raw_prompts[-1]], controls, use_tqdm=False)[0]
        assert list(controlled.outputs[0].token_ids) == [7, 7]
        assert controlled.outputs[0].logprobs is not None
        assert controlled.prompt_logprobs is not None


if __name__ == "__main__":
    if len(sys.argv) not in (4, 5) or sys.argv[1] != "--child":
        raise SystemExit(
            "usage: test_pytorch_models.py --child CASE [--scheduler-stress] DIR"
        )
    stress = sys.argv[3] == "--scheduler-stress"
    directory = Path(sys.argv[4] if stress else sys.argv[3])
    _child(sys.argv[2], directory, stress)
