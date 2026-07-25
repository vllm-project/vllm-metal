# SPDX-License-Identifier: Apache-2.0
"""Engine-level tests for the GLM native MTP drafter on a tiny generated checkpoint.

Builds a randomly initialised ``glm4_moe_lite`` target and a matching MTP head on
disk (~1 MB, no checkpoint download) from the real model classes, then drives a
real ``LLM`` through the normal front door so the drafter runs the served path.

Run with: ``pytest -m slow tests/test_glm4_moe_lite_mtp_e2e.py``
"""

from __future__ import annotations

import gc
import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest

pytestmark = pytest.mark.slow

# Shapes are the smallest that keep every structural feature the head depends on:
# an MoE layer past first_k_dense_replace, absorbed MLA with a rope split, and a
# trained nextn depth of one.
_TARGET_CONFIG: dict[str, Any] = {
    "architectures": ["Glm4MoeLiteForCausalLM"],
    "model_type": "glm4_moe_lite",
    "attention_bias": False,
    "attention_dropout": 0.0,
    "dtype": "bfloat16",
    "first_k_dense_replace": 1,
    "hidden_act": "silu",
    "norm_topk_prob": True,
    "num_nextn_predict_layers": 1,
    "partial_rotary_factor": 1.0,
    "rms_norm_eps": 1e-05,
    "rope_scaling": None,
    "rope_theta": 1000000,
    "routed_scaling_factor": 1.8,
    "tie_word_embeddings": False,
    "topk_method": "noaux_tc",
    "hidden_size": 64,
    "intermediate_size": 128,
    "moe_intermediate_size": 32,
    "num_hidden_layers": 2,
    "num_attention_heads": 4,
    "num_key_value_heads": 4,
    "n_routed_experts": 8,
    "num_experts_per_tok": 2,
    "n_shared_experts": 1,
    "n_group": 1,
    "topk_group": 1,
    "kv_lora_rank": 32,
    "q_lora_rank": 32,
    "qk_nope_head_dim": 16,
    "qk_rope_head_dim": 16,
    "v_head_dim": 16,
    "vocab_size": 512,
    "max_position_embeddings": 1024,
    "eos_token_id": 2,
    "pad_token_id": 3,
}

_PROMPTS = [
    "<t5> <t6> <t7> <t8>",
    "<t100> <t101> <t102>",
    "<t200> <t201> <t202> <t203> <t204>",
    "<t300> <t301>",
]
# Eight concurrent streams over a 32-block budget cannot be resident at once, so
# the scheduler must preempt and resume mid-generation.
_PRESSURE_PROMPTS = [f"<t{i}> <t{i + 1}> <t{i + 2}>" for i in range(0, 128, 16)]
_PRESSURE_BLOCKS = 32


@dataclass(frozen=True)
class _Checkpoints:
    """Filesystem paths of the generated target/head pairs."""

    target: Path
    head: Path
    accept_all_target: Path
    accept_all_head: Path


@dataclass(frozen=True)
class _RunResult:
    """Generated token ids plus the engine counters for one engine run."""

    tokens: list[list[int]]
    counters: dict[str, Any]

    @property
    def drafted(self) -> int:
        return self.counters.get("spec_decode_num_draft_tokens") or 0

    @property
    def accepted(self) -> int:
        return self.counters.get("spec_decode_num_accepted_tokens") or 0

    @property
    def preemptions(self) -> int:
        return self.counters.get("num_preemptions") or 0


def _write_tokenizer(directory: Path) -> None:
    from tokenizers import Tokenizer, models, pre_tokenizers

    specials = ["<unk>", "<bos>", "<eos>", "<pad>"]
    vocab = {token: index for index, token in enumerate(specials)}
    for index in range(_TARGET_CONFIG["vocab_size"] - len(specials)):
        vocab[f"<t{index}>"] = len(vocab)
    tokenizer = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    tokenizer.save(str(directory / "tokenizer.json"))
    (directory / "tokenizer_config.json").write_text(
        json.dumps(
            {
                "tokenizer_class": "PreTrainedTokenizerFast",
                "unk_token": "<unk>",
                "bos_token": "<bos>",
                "eos_token": "<eos>",
                "pad_token": "<pad>",
                "model_max_length": _TARGET_CONFIG["max_position_embeddings"],
            }
        ),
        encoding="utf-8",
    )


def _save_random_weights(model: Any, directory: Path) -> None:
    import mlx.core as mx
    from mlx.utils import tree_flatten

    weights = {
        name: value.astype(mx.bfloat16)
        for name, value in dict(tree_flatten(model.parameters())).items()
    }
    mx.save_safetensors(str(directory / "model.safetensors"), weights)


def _build_target(directory: Path) -> None:
    from mlx_lm.utils import _get_classes

    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps(_TARGET_CONFIG), encoding="utf-8")
    _write_tokenizer(directory)
    model_class, args_class = _get_classes(_TARGET_CONFIG)
    _save_random_weights(model_class(args_class.from_dict(_TARGET_CONFIG)), directory)


def _build_head(directory: Path, target: Path) -> None:
    """Write the head in the hosted drafter-split schema (shape under text_config)."""
    from vllm_metal.v1.mtp_heads.glm4_moe_lite_mtp_model import (
        Glm4MoeLiteMTPArgs,
        Glm4MoeLiteMTPModel,
    )

    directory.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "glm4_moe_lite_mtp",
        "block_size": 2,
        "tie_word_embeddings": False,
        "text_config": _TARGET_CONFIG,
    }
    (directory / "config.json").write_text(json.dumps(config), encoding="utf-8")
    for name in ("tokenizer.json", "tokenizer_config.json"):
        shutil.copy(target / name, directory / name)
    _save_random_weights(
        Glm4MoeLiteMTPModel(Glm4MoeLiteMTPArgs.from_dict(config)), directory
    )


def _copy_with_zeroed_lm_head(source: Path, destination: Path) -> None:
    """Copy a checkpoint with a zeroed ``lm_head`` so greedy decoding always emits token 0.

    Target and head then agree on every position, which pins acceptance at 1.0 and
    makes the accept/bonus-token path deterministic instead of weight-dependent.
    """
    import mlx.core as mx

    shutil.copytree(source, destination)
    weights = mx.load(str(destination / "model.safetensors"))
    weights["lm_head.weight"] = mx.zeros_like(weights["lm_head.weight"])
    mx.save_safetensors(str(destination / "model.safetensors"), weights)


@pytest.fixture(scope="session")
def checkpoints(tmp_path_factory: pytest.TempPathFactory) -> _Checkpoints:
    root = tmp_path_factory.mktemp("tiny_glm4_moe_lite")
    target, head = root / "target", root / "head"
    _build_target(target)
    _build_head(head, target)
    accept_all_target, accept_all_head = (
        root / "target_accept_all",
        root / "head_accept_all",
    )
    _copy_with_zeroed_lm_head(target, accept_all_target)
    _copy_with_zeroed_lm_head(head, accept_all_head)
    return _Checkpoints(target, head, accept_all_target, accept_all_head)


def _generate(
    target: Path,
    head: Path | None,
    prompts: list[str],
    *,
    max_tokens: int,
    num_blocks: int | None = None,
) -> _RunResult:
    """Boot one engine, generate greedily, and return its tokens and counters."""
    from vllm import LLM, SamplingParams

    kwargs: dict[str, Any] = {
        "model": str(target),
        "max_model_len": 256,
        "gpu_memory_utilization": 0.20,
        "enable_prefix_caching": False,
        "async_scheduling": False,
        "disable_log_stats": False,
        "enforce_eager": True,
    }
    if head is not None:
        kwargs["speculative_config"] = {
            "method": "mtp",
            "model": str(head),
            "num_speculative_tokens": 1,
        }
    if num_blocks is not None:
        kwargs["num_gpu_blocks_override"] = num_blocks

    llm = LLM(**kwargs)
    try:
        outputs = llm.generate(
            prompts,
            SamplingParams(temperature=0.0, max_tokens=max_tokens, ignore_eos=True),
        )
        tokens = [list(output.outputs[0].token_ids) for output in outputs]
        counters: dict[str, Any] = {}
        for metric in llm.get_metrics():
            value = getattr(metric, "value", None)
            if value is None:
                value = getattr(metric, "values", None)
            counters[metric.name.split(":")[-1]] = value
        return _RunResult(tokens=tokens, counters=counters)
    finally:
        del llm
        gc.collect()


@pytest.fixture(scope="session")
def baseline(checkpoints: _Checkpoints) -> _RunResult:
    return _generate(checkpoints.target, None, _PROMPTS, max_tokens=32)


@pytest.fixture(scope="session")
def speculative(checkpoints: _Checkpoints) -> _RunResult:
    return _generate(checkpoints.target, checkpoints.head, _PROMPTS, max_tokens=32)


@pytest.fixture(scope="session")
def pressure_baseline(checkpoints: _Checkpoints) -> _RunResult:
    return _generate(checkpoints.target, None, _PRESSURE_PROMPTS, max_tokens=160)


@pytest.fixture(scope="session")
def pressure_speculative(checkpoints: _Checkpoints) -> _RunResult:
    return _generate(
        checkpoints.target,
        checkpoints.head,
        _PRESSURE_PROMPTS,
        max_tokens=160,
        num_blocks=_PRESSURE_BLOCKS,
    )


@pytest.fixture(scope="session")
def accept_all(checkpoints: _Checkpoints) -> _RunResult:
    return _generate(
        checkpoints.accept_all_target,
        checkpoints.accept_all_head,
        _PROMPTS,
        max_tokens=32,
    )


@pytest.fixture(scope="session")
def accept_all_under_pressure(checkpoints: _Checkpoints) -> _RunResult:
    return _generate(
        checkpoints.accept_all_target,
        checkpoints.accept_all_head,
        _PRESSURE_PROMPTS,
        max_tokens=160,
        num_blocks=_PRESSURE_BLOCKS,
    )


def test_speculative_decoding_drafts_and_stays_lossless(
    baseline: _RunResult, speculative: _RunResult
) -> None:
    assert speculative.drafted > 0
    assert speculative.tokens == baseline.tokens


def test_accepted_drafts_are_lossless_and_counted(accept_all: _RunResult) -> None:
    assert accept_all.accepted == accept_all.drafted > 0
    assert all(set(row) == {0} for row in accept_all.tokens)


def test_preemption_keeps_generation_lossless(
    pressure_baseline: _RunResult, pressure_speculative: _RunResult
) -> None:
    assert pressure_speculative.preemptions > 0
    assert pressure_speculative.drafted > 0
    assert pressure_speculative.tokens == pressure_baseline.tokens


def test_preemption_releases_head_state(
    monkeypatch: pytest.MonkeyPatch, checkpoints: _Checkpoints
) -> None:
    """A preempted request hands its slab back instead of holding it while it waits.

    Output equality cannot show this: rejected drafts leave generation unchanged
    either way, so the run is observed through the release seam itself.
    """
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
    from vllm_metal.v1.mtp_proposer import NativeMTPProposer

    released: list[set[str]] = []
    release_requests = NativeMTPProposer.release_requests

    def record(self: NativeMTPProposer, req_ids: set[str]) -> None:
        released.append(set(req_ids))
        return release_requests(self, req_ids)

    monkeypatch.setattr(NativeMTPProposer, "release_requests", record)
    result = _generate(
        checkpoints.target,
        checkpoints.head,
        _PRESSURE_PROMPTS,
        max_tokens=160,
        num_blocks=_PRESSURE_BLOCKS,
    )

    assert result.preemptions > 0
    assert any(released)


def test_accepted_drafts_survive_preemption(
    accept_all_under_pressure: _RunResult,
) -> None:
    assert accept_all_under_pressure.preemptions > 0
    assert accept_all_under_pressure.accepted == accept_all_under_pressure.drafted > 0
    assert all(len(row) == 160 for row in accept_all_under_pressure.tokens)
