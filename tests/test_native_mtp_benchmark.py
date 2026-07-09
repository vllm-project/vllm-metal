# SPDX-License-Identifier: Apache-2.0
"""Pure unit tests for the native MTP benchmark helper (thin gemma4 sibling)."""

from __future__ import annotations

from types import SimpleNamespace

from tools.benchmark import native_mtp_benchmark as bench


def test_parser_extends_gemma4_parser_with_chat_flag() -> None:
    args = bench.build_parser().parse_args(["--model", "m"])

    assert args.chat is False
    assert args.async_scheduling is False

    args = bench.build_parser().parse_args(["--model", "m", "--chat"])
    assert args.chat is True


def test_spec_decode_metrics_computes_acceptance_rate() -> None:
    class _Llm:
        def get_metrics(self):
            return [
                SimpleNamespace(name="vllm:spec_decode_num_drafts", value=10),
                SimpleNamespace(name="vllm:spec_decode_num_draft_tokens", value=10),
                SimpleNamespace(name="vllm:spec_decode_num_accepted_tokens", value=8),
                SimpleNamespace(name="vllm:other_metric", value=1),
            ]

    metrics = bench.spec_decode_metrics(_Llm())

    assert metrics["spec_decode_num_draft_tokens"] == 10
    assert metrics["spec_decode_num_accepted_tokens"] == 8
    assert metrics["acceptance_rate"] == 0.8
    assert "other_metric" not in metrics


def test_spec_decode_metrics_tolerates_metrics_error() -> None:
    class _Llm:
        def get_metrics(self):
            raise RuntimeError("no metrics")

    metrics = bench.spec_decode_metrics(_Llm())

    assert "metrics_error" in metrics
    assert "acceptance_rate" not in metrics
