# SPDX-License-Identifier: Apache-2.0
"""Whole-continuation qualification must not hide a failure after a near tie."""

import copy
import importlib
import json
from types import SimpleNamespace

import mlx.core as mx
import numpy as np
import pytest

from tools.continuation_audit import (
    audit_outputs,
    decision_kinds,
    replay_tokens,
    score_token,
)


def scores(values, token):
    return score_token(np.array(values, dtype=np.float32), token)


def evidence(tokens):
    return {
        "input_ids": [4, 3],
        "tokens": tokens,
        "decisions": [scores(np.eye(6)[token], token) for token in tokens],
    }


@pytest.mark.parametrize("tokens,argmax", [([4], [1]), ([4, 5, 0], [1, 5, 0])])
def test_replay_uses_emitted_prefix_and_fresh_native_cache(tokens, argmax):
    class Model:
        def __init__(self):
            self.calls = []
            self.caches = 0
            # Free greedy: 0 -> 1 -> 2 -> 3. Observed: 0 -> 4 -> 5 -> 0.
            self.logits = mx.eye(6)[mx.array([1, 2, 3, 0, 5, 0])] * 10

        def make_cache(self):
            self.caches += 1
            return []

        def __call__(self, inputs, cache):
            self.calls.extend(inputs[0].tolist())
            return self.logits[inputs]

    model = Model()
    for _ in range(2):
        rows = replay_tokens(model, [0], tokens)
        assert [row["argmax"] for row in rows] == argmax
        assert [row["token"] for row in rows] == tokens
    assert model.caches == 2
    # mlx-lm also evaluates one final lookahead row, which is never reported.
    assert model.calls == ([0] + tokens) * 2


def test_later_failure_is_retained_after_first_divergent_tie():
    reference = evidence([0, 2, 3])
    output = evidence([1, 4, 5])
    replayed = [
        scores([0, 0, -1, -2, -3, -4], 1),  # tie still fails strict argmax
        scores([-5, -4, -3, -2, 0, -1], 4),  # correct on the divergent prefix
        scores([0, -1, -2, -3, -4, -9], 5),  # later error outside top five
    ]
    prefixes = []

    def replay(prompt, tokens):
        prefixes.append((prompt, tokens))
        return replayed

    report = audit_outputs([reference], [output], replay, max_tokens=3)
    assert prefixes == [([4, 3], [1, 4, 5])]
    assert not report["passed"]
    assert report["tokens"] == 3
    assert report["native_argmax_mismatches"] == 2
    assert report["native_tied_mismatches"] == 1
    assert report["native_outside_top5"] == 1
    assert report["max_native_argmax_gap"] == 9
    assert report["serving_argmax_mismatches"] == 0
    assert [row["prefix_length"] for row in report["results"][0]["rows"]] == [2, 3, 4]


def test_verifier_choice_and_native_choice_are_separate_checks():
    output = evidence([1, 2])
    output["decisions"][1] = scores([0, 9, 8, 0, 0, 0], 2)
    replayed = evidence([1, 2])["decisions"]
    report = audit_outputs([output], [output], lambda *_: replayed, max_tokens=2)
    assert not report["passed"]
    assert report["native_argmax_mismatches"] == 0
    assert report["serving_argmax_mismatches"] == 1


def test_complete_exact_control_passes_without_serving_evidence():
    row = evidence([1, 2])
    control = {key: row[key] for key in ("input_ids", "tokens")}
    report = audit_outputs(
        [control], [control], lambda *_: row["decisions"], max_tokens=2, serving=False
    )
    assert report["passed"] and report["exact_sequences"] == 1
    assert report["serving_argmax_mismatches"] is None


@pytest.mark.parametrize(
    "problem",
    [
        "empty",
        "missing_sequence",
        "prompt",
        "truncated",
        "truncated_reference",
        "missing_decisions",
        "null_decisions",
        "null_serving_row",
        "null_native_row",
        "row_token",
        "bool_row_token",
        "missing_replay",
        "nan",
        "negative_gap",
        "bad_rank",
        "bool_token",
        "negative_token",
    ],
)
def test_incomplete_or_invalid_evidence_cannot_pass(problem):
    ref, got = evidence([1, 2]), evidence([1, 2])
    native = copy.deepcopy(ref["decisions"])
    refs, outputs = [ref], [got]
    if problem == "empty":
        refs, outputs = [], []
    elif problem == "missing_sequence":
        outputs = []
    elif problem == "prompt":
        got["input_ids"] = [99]
    elif problem == "truncated":
        got["tokens"].pop()
    elif problem == "truncated_reference":
        ref["tokens"].pop()
    elif problem == "missing_decisions":
        got["decisions"].pop()
    elif problem == "null_decisions":
        got["decisions"] = None
    elif problem == "null_serving_row":
        got["decisions"][0] = None
    elif problem == "null_native_row":
        native[0] = None
    elif problem == "row_token":
        native[1]["token"] = 3
    elif problem == "bool_row_token":
        native[0]["token"] = True
    elif problem == "missing_replay":
        native.pop()
    elif problem == "nan":
        native[1]["argmax_gap"] = float("nan")
    elif problem == "negative_gap":
        native[1]["argmax_gap"] = -1
    elif problem == "bad_rank":
        native[1]["rank"] = 0
    elif problem == "bool_token":
        got["tokens"][0] = True
    else:
        got["tokens"][0] = -1
    with pytest.raises(ValueError):
        audit_outputs(refs, outputs, lambda *_: native, max_tokens=2)


@pytest.mark.parametrize(
    "values,token",
    [
        ([], 0),
        ([[1.0]], 0),
        ([float("nan")], 0),
        ([float("inf")], 0),
        ([0.0], -1),
        ([0.0], 1),
        ([0.0], True),
    ],
)
def test_scores_require_a_finite_vector_and_valid_token(values, token):
    with pytest.raises(ValueError):
        scores(values, token)


def test_score_ranking_preserves_lowest_token_id_ties():
    row = scores([0, 1, 1, 0], 2)
    assert row["argmax"] == 1 and row["rank"] == 2 and row["argmax_gap"] == 0
    assert [item["id"] for item in row["top_logprobs"]] == [1, 2, 0, 3]


def test_report_normalization_does_not_change_the_verifiers_argmax():
    row = scores([0, 1e-9, 0], 1)
    assert row["argmax"] == 1 and row["rank"] == 1 and row["argmax_gap"] == 0
    assert row["top_logprobs"][0]["id"] == 1


@pytest.mark.parametrize(
    "drafts,emitted,kinds",
    [
        ([], [4], ["decode"]),
        ([1, 2], [4], ["correction"]),
        ([1, 2], [1, 4], ["accepted", "correction"]),
        ([1, 2], [1, 2, 4], ["accepted", "accepted", "bonus"]),
    ],
)
def test_verification_rows_identify_accepted_correction_and_bonus(
    drafts, emitted, kinds
):
    assert decision_kinds(drafts, emitted) == kinds


@pytest.mark.parametrize(
    "drafts,emitted",
    [
        ([], []),
        ([], [1, 2]),
        ([1, 2], [1]),
        ([1, 2], [3, 4]),
    ],
)
def test_invalid_verification_alignment_is_rejected(drafts, emitted):
    with pytest.raises(ValueError):
        decision_kinds(drafts, emitted)


@pytest.mark.parametrize("problem", ["truncated", "wrong_token", "extra_token"])
def test_replay_rejects_broken_generation_and_closes_stream(monkeypatch, problem):
    closed = []

    def generate(*args, **kwargs):
        try:
            yield 1, mx.array([0.0, 1.0, 0.0])
            if problem == "truncated":
                return
            yield (0 if problem == "wrong_token" else 2), mx.array([0.0, 0.0, 1.0])
            if problem == "extra_token":
                yield 0, mx.array([1.0, 0.0, 0.0])
        finally:
            closed.append(True)

    monkeypatch.setattr(
        importlib.import_module("mlx_lm.generate"), "generate_step", generate
    )
    with pytest.raises(ValueError):
        replay_tokens(None, [0], [1, 2])
    assert closed == [True]


@pytest.mark.parametrize(
    "audit_passed,token_status",
    [
        (None, "exact"),
        (None, "top_k"),
        (False, "exact"),
        (True, "fail"),
        (True, "top_k"),
        (True, "exact"),
    ],
)
@pytest.mark.parametrize("hub_models", [False, True])
@pytest.mark.parametrize("method", ["dspark", "suffix"])
def test_cli_requires_both_audits_and_preserves_the_report(
    monkeypatch, tmp_path, audit_passed, token_status, hub_models, method
):
    import tools.speculative_parity as tool
    from tools import check_parity as cli

    target, draft = tmp_path / "target", tmp_path / "draft"
    target.mkdir()
    draft.mkdir()
    output = tmp_path / "results"
    resolved = []

    def snapshot(model):
        resolved.append(model)
        return str(target if model == "test/target" else draft)

    monkeypatch.setattr("huggingface_hub.snapshot_download", snapshot)
    monkeypatch.setattr(
        "sys.argv",
        [
            "parity",
            "--method",
            method,
            "--model",
            "test/target" if hub_models else str(target),
            *(
                ["--draft", "test/draft" if hub_models else str(draft)]
                if method != "suffix"
                else []
            ),
            *(["--audit-continuations"] if audit_passed is not None else []),
            "--output-dir",
            str(output),
        ],
    )
    commands = []
    report = {
        "passed": audit_passed,
        "reports": {
            f"{method}-b1": {
                "tokens": 32,
                "native_argmax_mismatches": int(not audit_passed),
                "serving_argmax_mismatches": 0,
            }
        },
    }
    real_run = tool.subprocess.run

    def run(command, **kwargs):
        if "--worker" not in command:
            # vLLM probes the OS while its schedule utility is imported.
            return real_run(command, **kwargs)
        commands.append(command)
        worker = command[-1]
        assert kwargs["check"]
        assert kwargs["timeout"] == (3000 if worker == "replay" else 600)
        if worker == "native":
            (output / "native.json").write_text(
                json.dumps(
                    [
                        {
                            "prompt": "test",
                            "input_ids": [0],
                            "tokens": [1] * 10,
                            "text": "test",
                            "top_logprobs": [
                                [
                                    {
                                        "id": token,
                                        "text": "",
                                        "rank": rank,
                                        "logprob": -rank,
                                    }
                                    for rank, token in enumerate([1, 2], 1)
                                ]
                            ]
                            * 10,
                        }
                    ]
                )
            )
        elif worker == "replay":
            (output / "continuation-audit.json").write_text(json.dumps(report))
        else:
            for batch in (1, 2):
                (output / f"{worker}-b{batch}.json").write_text(
                    json.dumps(
                        {
                            "stats": {},
                            "outputs": [
                                {
                                    "input_ids": [0],
                                    "tokens": [1] * 10
                                    if worker == "target"
                                    else [
                                        {"exact": 1, "top_k": 2, "fail": 9}[
                                            token_status
                                        ]
                                    ]
                                    + [1] * 9,
                                    "text": "test",
                                    "top_logprobs": [
                                        [
                                            {
                                                "id": token,
                                                "text": "",
                                                "rank": rank,
                                                "logprob": -rank,
                                            }
                                            for rank, token in enumerate(
                                                {
                                                    "exact": [1, 2],
                                                    "top_k": [2, 1],
                                                    "fail": [9, 8],
                                                }[token_status],
                                                1,
                                            )
                                        ]
                                    ]
                                    * 10,
                                }
                            ],
                        }
                    )
                )
        return SimpleNamespace(returncode=0)

    monkeypatch.setattr(tool.subprocess, "run", run)
    # A TOP_K_MATCH success must not override the strict audit result.
    with pytest.raises(SystemExit) as caught:
        cli.main()
    expected = (
        audit_passed is not False
        and token_status != "fail"
        and (method != "suffix" or token_status == "exact")
    )
    assert caught.value.code == (0 if expected else 1)
    summary = json.loads((output / "summary.json").read_text())
    assert summary["passed"] == expected
    if method == "suffix":
        assert summary["target_suffix_exact_passed"] == (token_status == "exact")
    if audit_passed is not None:
        assert summary["continuation_audit_passed"] == audit_passed
        assert json.loads((output / "continuation-audit.json").read_text()) == report
    else:
        assert "continuation_audit_passed" not in summary
        assert not (output / "continuation-audit.json").exists()
    assert [command[-1] for command in commands] == [
        "native",
        "target",
        method,
        *(["replay"] if audit_passed is not None else []),
    ]
    if method == "suffix":
        assert all("--draft" not in command for command in commands)
        assert all(command[-6:-4] == ["--model", str(target)] for command in commands)
    else:
        assert all(
            command[-8:-4] == ["--model", str(target), "--draft", str(draft)]
            for command in commands
        )
    expected_models = ["test/target"]
    if method != "suffix":
        expected_models.append("test/draft")
    assert resolved == (expected_models if hub_models else [])
    metadata = json.loads((output / "metadata.json").read_text())
    assert metadata["target"] == str(target)
    assert metadata["draft"] == (None if method == "suffix" else str(draft))


def test_cli_rejects_duplicate_batch_sizes_before_starting_workers(
    monkeypatch, tmp_path
):
    from tools import check_parity as cli

    monkeypatch.setattr(
        "sys.argv",
        [
            "parity",
            "--model",
            "test/target",
            "--batch-size",
            "1",
            "1",
            "--output-dir",
            str(tmp_path / "results"),
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert not (tmp_path / "results").exists()


def test_cli_rejects_q4_for_other_drafting_methods(monkeypatch, tmp_path):
    from tools import check_parity as cli

    monkeypatch.setattr(
        "sys.argv",
        [
            "parity",
            "--model",
            "test/target",
            "--method",
            "dflash",
            "--dspark-draft-quantization",
            "q4",
            "--output-dir",
            str(tmp_path / "results"),
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("worker", [None, "suffix"])
def test_cli_rejects_draft_checkpoint_for_suffix(monkeypatch, tmp_path, capsys, worker):
    from tools import check_parity as cli
    from tools import speculative_parity as tool

    monkeypatch.setattr(
        tool,
        "run",
        lambda *_: pytest.fail("Invalid arguments reached model resolution"),
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "parity",
            "--model",
            "test/target",
            "--method",
            "suffix",
            "--draft",
            "test/unused-draft",
            "--output-dir",
            str(tmp_path / "results"),
            *(["--worker", worker] if worker is not None else []),
        ],
    )
    with pytest.raises(SystemExit) as caught:
        cli.main()
    assert caught.value.code == 2
    assert "--draft cannot be used with --method suffix" in capsys.readouterr().err
    assert not (tmp_path / "results").exists()


@pytest.mark.parametrize("worker", ["target", "dspark"])
def test_q4_parity_option_only_applies_to_draft_worker(monkeypatch, tmp_path, worker):
    import vllm

    import tools.speculative_parity as tool

    (tmp_path / "native.json").write_text("[]")
    args = SimpleNamespace(
        worker=worker,
        method="dspark",
        target="target",
        draft="draft",
        num_draft_tokens=7,
        draft_schedule=None,
        dspark_draft_topk=64,
        dspark_draft_quantization="q4",
        batch_size=[1, 4],
        output_dir=tmp_path,
        gpu_memory_utilization=0.25,
    )

    class ConfigCapturedError(Exception):
        pass

    def llm(**kwargs):
        assert kwargs["additional_config"] == (
            {"dspark_draft_quantization": "q4"} if worker == "dspark" else {}
        )
        raise ConfigCapturedError

    monkeypatch.setattr(vllm, "LLM", llm)
    # run_engine owns its subprocess environment; restore it in this unit test.
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    with pytest.raises(ConfigCapturedError):
        tool.run_engine(args)


def test_duplicate_prompts_keep_their_own_internal_verification_rows(
    monkeypatch, tmp_path
):
    import mlx.core as mx
    import vllm

    from tools.speculative_parity import run_engine

    refs = [{"input_ids": [8]}] * 2
    (tmp_path / "native.json").write_text(json.dumps(refs))
    # Scheduler order differs from the public request order, and prompts repeat.
    internal_ids = ["core-second", "core-first"]
    state = SimpleNamespace(
        scheduler_output=SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(req_id=key, prompt_token_ids=[8])
                for key in internal_ids
            ]
        ),
        decode_reqs=[],
        decode_segments=[],
        prefill_reqs=[
            SimpleNamespace(req_id=key, token_ids=[8]) for key in internal_ids
        ],
        batch=SimpleNamespace(
            paged_prefill_entries=[SimpleNamespace(result_mode="final")] * 2
        ),
        logits_cu_seqlens=[0, 1, 2],
        logits=mx.array([[[0.0, 0.0, 3.0], [0.0, 3.0, 0.0]]]),
    )
    runner = SimpleNamespace(
        _drafter=None,
        _execute_model_state=state,
        _sample_paged_batch=lambda: None,
        _request_states={
            key: SimpleNamespace(token_ids=[8, token])
            for key, token in zip(internal_ids, [2, 1], strict=True)
        },
    )
    closed = []
    engine = SimpleNamespace(
        model_executor=SimpleNamespace(
            driver_worker=SimpleNamespace(model_runner=runner)
        ),
        output_processor=SimpleNamespace(
            request_states={
                "core-first": SimpleNamespace(external_req_id="0"),
                "core-second": SimpleNamespace(external_req_id="1"),
            }
        ),
        engine_core=SimpleNamespace(shutdown=lambda: closed.append(True)),
    )

    def generate(prompts, params, use_tqdm):
        runner._sample_paged_batch()
        return [
            SimpleNamespace(
                request_id=str(i),
                outputs=[SimpleNamespace(token_ids=[i + 1], text=str(i + 1))],
            )
            for i in range(2)
        ]

    monkeypatch.setattr(
        vllm,
        "LLM",
        lambda **kwargs: SimpleNamespace(llm_engine=engine, generate=generate),
    )
    monkeypatch.setenv("VLLM_ENABLE_V1_MULTIPROCESSING", "1")
    run_engine(
        SimpleNamespace(
            worker="target",
            method="dspark",
            target="target",
            draft="draft",
            batch_size=[2],
            max_tokens=1,
            top_k=2,
            output_dir=tmp_path,
            gpu_memory_utilization=0.25,
            dspark_draft_topk=None,
            dspark_draft_quantization=None,
            audit_continuations=False,
        )
    )
    outputs = json.loads((tmp_path / "target-b2.json").read_text())["outputs"]
    assert [row["tokens"] for row in outputs] == [[1], [2]]
    assert [row["top_logprobs"][0][0]["id"] for row in outputs] == [1, 2]
    assert closed == [True]
