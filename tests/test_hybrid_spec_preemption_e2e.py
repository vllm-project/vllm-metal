# SPDX-License-Identifier: Apache-2.0
"""Hybrid n-gram speculation across V2 preemption with local Qwen3.5-0.8B.

Divergent prompts check structure; repeated prompts require spec-on/off token
identity. Each arm owns one engine in a spawned process; not run in CI.
"""

import json
import os
from itertools import zip_longest
from pathlib import Path

import pytest
from huggingface_hub import try_to_load_from_cache


def _local_model_path() -> str | None:
    model = os.environ.get("QWEN35_MODEL_PATH")
    if not model:
        config = try_to_load_from_cache("Qwen/Qwen3.5-0.8B", "config.json")
        if isinstance(config, str):
            model = str(Path(config).parent)
    for candidate in (model, "~/.cache/modelscope/hub/models/Qwen/Qwen3.5-0.8B"):
        if not candidate:
            continue
        path = Path(candidate).expanduser()
        if (
            path.is_dir()
            and (path / "config.json").is_file()
            and any(
                (path / name).is_file()
                for name in ("tokenizer.json", "tokenizer_config.json")
            )
            and any(weight.is_file() for weight in path.glob("*.safetensors"))
        ):
            return str(path.resolve())
    return None


MODEL_PATH = _local_model_path()
pytestmark = pytest.mark.skipif(
    MODEL_PATH is None,
    reason="Qwen/Qwen3.5-0.8B not in local cache (set QWEN35_MODEL_PATH)",
)


def _run_arm(case: str, enable_spec: bool, out_path: str, model: str):
    os.environ["VLLM_ENABLE_V1_MULTIPROCESSING"] = "0"
    os.environ["VLLM_USE_V2_MODEL_RUNNER"] = "1"
    from vllm import LLM, SamplingParams

    from tools.hybrid_spec_parity import CORPUS_A, CORPUS_B, SUFFIXES
    from vllm_metal.v1.model_runner import MetalModelRunner
    from vllm_metal.v1.spec_decode import SpeculativeDecodeController

    readmitted, records = [], []
    original_readmit = MetalModelRunner._readmit_request
    original_verify = SpeculativeDecodeController.verify_greedy

    def readmit(self, batch, new_req, scheduler_output):
        original_readmit(self, batch, new_req, scheduler_output)
        readmitted.append(new_req.req_id)

    def verify(self, logits, decode_reqs, decode_segments):
        result = original_verify(self, logits, decode_reqs, decode_segments)
        for seg, ids in zip(decode_segments, result, strict=True):
            records.append(
                (seg.req_id, seg.num_query_tokens, len(ids), seg.req_id in readmitted)
            )
        return result

    MetalModelRunner._readmit_request = readmit
    SpeculativeDecodeController.verify_greedy = verify
    llm = LLM(
        model=model,
        max_model_len=1024,
        max_num_seqs=8,
        block_size=160,
        num_gpu_blocks_override=30,
        enable_prefix_caching=False,
        async_scheduling=False,
        gpu_memory_utilization=0.5,
        seed=0,
        speculative_config=(
            {
                "method": "ngram",
                "num_speculative_tokens": 3,
                "prompt_lookup_min": 2,
                "prompt_lookup_max": 3,
            }
            if enable_spec
            else None
        ),
    )
    try:
        assert llm.llm_engine.vllm_config.use_v2_model_runner
        tokenizer = llm.get_tokenizer()
        if case == "divergent":
            prompts = []
            for corpus in (CORPUS_A, CORPUS_B):
                prefix = tokenizer.encode(corpus)
                for suffix in SUFFIXES:
                    tail = tokenizer.encode(suffix, add_special_tokens=False)
                    token_ids = prefix[: 600 - len(tail)] + tail
                    prompts.append({"prompt_token_ids": token_ids})
        else:
            assert case == "repeated", case
            base = (
                "The vLLM Metal backend executes transformer inference on Apple "
                "silicon GPUs using MLX. "
                "Paged attention divides the key value cache into fixed size blocks. "
            ) * 60
            topics = [
                "oceans",
                "volcanoes",
                "Rust",
                "Rome",
                "plants",
                "stars",
                "jazz",
                "TCP",
            ]
            prompts = [
                {"prompt_token_ids": tokenizer(f"{t}: " + base)["input_ids"][:600]}
                for t in topics
            ]
        assert len(prompts) == 8
        assert all(len(prompt["prompt_token_ids"]) == 600 for prompt in prompts)
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0, max_tokens=160, ignore_eos=True),
            use_tqdm=False,
        )
        result = {
            "tokens": [list(out.outputs[0].token_ids) for out in outputs],
            "readmitted": readmitted,
            "verify": records,
            "use_v2": llm.llm_engine.vllm_config.use_v2_model_runner,
            "async_scheduling": (
                llm.llm_engine.vllm_config.scheduler_config.async_scheduling
            ),
        }
        Path(out_path).write_text(json.dumps(result), encoding="utf-8")
    finally:
        llm.llm_engine.engine_core.shutdown()


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("VLLM_METAL_E2E", "1") != "1", reason="e2e disabled")
def test_ngram_spec_divergent_hybrid_preemption_structure_v2(
    run_in_spawn_process, tmp_path
):
    # Spec-on/off greedy differs only at exact top-2 logit ties / batch-shape
    # numerics here, even without preemption and on V1; identity is not an oracle.
    path = tmp_path / "divergent.json"
    run_in_spawn_process(_run_arm, "divergent", True, str(path), MODEL_PATH)
    arm = json.loads(path.read_text(encoding="utf-8"))
    assert arm["use_v2"]
    assert arm["async_scheduling"] is False
    assert arm["readmitted"], "No request was readmitted"
    resumed_records = [record for record in arm["verify"] if record[3]]
    assert any(span > 1 for _, span, _, _ in resumed_records), (
        "No drafts were verified after readmission"
    )
    assert any(committed == span > 1 for _, span, committed, _ in resumed_records), (
        "No draft span was fully accepted after readmission"
    )
    assert any(committed < span for _, span, committed, _ in resumed_records), (
        "No draft span was partially accepted after readmission"
    )
    assert len(arm["tokens"]) == 8
    for i, tokens in enumerate(arm["tokens"]):
        assert len(tokens) == 160, f"Prompt {i}: expected 160 generated tokens"


@pytest.mark.slow
@pytest.mark.skipif(os.environ.get("VLLM_METAL_E2E", "1") != "1", reason="e2e disabled")
def test_ngram_spec_repeated_is_lossless_across_hybrid_preemption_v2(
    run_in_spawn_process, tmp_path
):
    arms = []
    for enable_spec in (False, True):
        label = "spec-on" if enable_spec else "spec-off"
        path = tmp_path / f"{label}.json"
        run_in_spawn_process(
            _run_arm, "repeated", enable_spec, str(path), MODEL_PATH, label=label
        )
        arm = json.loads(path.read_text(encoding="utf-8"))
        assert arm["use_v2"], label
        assert arm["async_scheduling"] is False, label
        assert arm["readmitted"], f"{label}: no request was readmitted"
        assert len(arm["tokens"]) == 8, label
        arms.append(arm)
    assert any(span > 1 and resumed for _, span, _, resumed in arms[1]["verify"]), (
        "No drafts were verified after readmission"
    )
    for i, (plain, spec) in enumerate(
        zip(arms[0]["tokens"], arms[1]["tokens"], strict=True)
    ):
        first_diff = next(
            (pos for pos, (a, b) in enumerate(zip_longest(plain, spec)) if a != b), None
        )
        assert first_diff is None, f"Prompt {i}: first differing position {first_diff}"
        assert len(plain) == len(spec) == 160, (
            f"Prompt {i}: expected 160 generated tokens per arm"
        )
