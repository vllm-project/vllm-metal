# SPDX-License-Identifier: Apache-2.0
"""Pure unit tests for the Gemma4 MTP benchmark helper."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from tools.benchmark import gemma4_mtp_benchmark as bench


def test_select_prompts_cycles_to_batch_size() -> None:
    assert bench.select_prompts(batch_size=5, prompts=["a", "b"]) == [
        "a",
        "b",
        "a",
        "b",
        "a",
    ]


def test_select_prompts_rejects_empty_source() -> None:
    with pytest.raises(ValueError, match="at least one prompt"):
        bench.select_prompts(batch_size=1, prompts=[])


def test_summarize_outputs_counts_tokens_without_text() -> None:
    outputs = [
        SimpleNamespace(
            prompt_token_ids=[1, 2],
            outputs=[SimpleNamespace(token_ids=[3, 4, 5], text="hidden")],
        ),
        SimpleNamespace(
            prompt_token_ids=[6],
            outputs=[SimpleNamespace(token_ids=[7], text="hidden")],
        ),
    ]

    prompt_tokens, output_tokens, samples = bench.summarize_outputs(
        outputs,
        include_text=False,
    )

    assert prompt_tokens == 3
    assert output_tokens == 4
    assert samples == [
        {"prompt_tokens": 2, "output_tokens": 3, "token_ids": [3, 4, 5]},
        {"prompt_tokens": 1, "output_tokens": 1, "token_ids": [7]},
    ]


def test_summarize_outputs_can_include_text() -> None:
    outputs = [
        SimpleNamespace(
            prompt_token_ids=[1],
            outputs=[SimpleNamespace(token_ids=[2], text="hello")],
        )
    ]

    _, _, samples = bench.summarize_outputs(outputs, include_text=True)

    assert samples == [
        {"prompt_tokens": 1, "output_tokens": 1, "token_ids": [2], "text": "hello"}
    ]


def test_parser_defaults_to_sync_scheduling() -> None:
    args = bench.build_parser().parse_args(["--model", "target"])

    assert args.async_scheduling is False
    assert args.num_speculative_tokens == 1


def test_build_llm_kwargs_omits_speculative_config_for_baseline() -> None:
    args = bench.build_parser().parse_args(["--model", "target"])

    kwargs = bench._build_llm_kwargs(args)

    assert kwargs["model"] == "target"
    assert kwargs["async_scheduling"] is False
    assert "gpu_memory_utilization" not in kwargs
    assert "speculative_config" not in kwargs


def test_build_llm_kwargs_adds_mtp_assistant_config() -> None:
    args = bench.build_parser().parse_args(
        [
            "--model",
            "target",
            "--assistant-model",
            "assistant",
            "--assistant-revision",
            "rev",
            "--num-speculative-tokens",
            "1",
            "--gpu-memory-utilization",
            "0.5",
        ]
    )

    kwargs = bench._build_llm_kwargs(args)

    assert kwargs["gpu_memory_utilization"] == 0.5
    assert kwargs["speculative_config"] == {
        "method": "mtp",
        "model": "assistant",
        "num_speculative_tokens": 1,
        "revision": "rev",
    }
