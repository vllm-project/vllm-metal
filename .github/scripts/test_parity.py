# SPDX-License-Identifier: Apache-2.0
import json
import urllib.error
from unittest.mock import Mock

import parity
import pytest


@pytest.fixture
def event(monkeypatch, tmp_path):
    payload = {
        "issue": {"number": 42, "pull_request": {}},
        "comment": {
            "body": "/ci parity",
            "user": {"login": "reviewer"},
            "author_association": "MEMBER",
        },
    }
    event_path = tmp_path / "event.json"
    event_path.write_text(json.dumps(payload))
    for key, value in {
        "GITHUB_REPOSITORY": parity.REPOSITORY,
        "GITHUB_EVENT_NAME": "issue_comment",
        "GITHUB_EVENT_PATH": str(event_path),
        "GITHUB_SHA": "main-sha",
        "GITHUB_SERVER_URL": "https://github.com",
        "GITHUB_RUN_ID": "123",
        "GITHUB_STEP_SUMMARY": str(tmp_path / "summary.md"),
        "PARITY_TESTED_SHA": "pr-sha",
    }.items():
        monkeypatch.setenv(key, value)


def test_read_only_user_cannot_launch_pr_code(event, monkeypatch):
    api = Mock(return_value={"permission": "read"})
    monkeypatch.setattr(parity, "github_api", api)
    assert parity.start() is None
    api.assert_called_once_with("collaborators/reviewer/permission")


def test_captured_fork_sha_and_original_check_are_used(event, monkeypatch):
    pr = {
        "state": "open",
        "head": {"sha": "pr-sha", "repo": {"full_name": "contributor/fork"}},
    }
    api = Mock(side_effect=[{"permission": "maintain"}, pr, {"id": 99}, pr, {}])
    monkeypatch.setattr(parity, "github_api", api)
    target = parity.start()
    assert target["repository"] == "contributor/fork"
    assert target["sha"] == "pr-sha"
    assert api.call_args.kwargs["data"]["head_sha"] == "pr-sha"

    pr["head"]["sha"] = "newer-pr-sha"
    parity.finish(target["check_id"], "failure")
    assert api.call_args.args == ("check-runs/99",)
    assert api.call_args.kwargs["data"]["conclusion"] == "failure"


@pytest.mark.parametrize(
    ("head", "expected"),
    [
        ({"head": {"sha": "a" * 40}}, "Up to date at report time."),
        ({"head": {"sha": "b" * 40}}, "Outdated"),
        (urllib.error.URLError("unavailable"), "Freshness unknown"),
    ],
)
def test_report_preserves_result_and_shows_freshness(
    event, monkeypatch, tmp_path, head, expected
):
    monkeypatch.setenv("PARITY_TESTED_SHA", "a" * 40)
    api = Mock(side_effect=[head, {}])
    monkeypatch.setattr(parity, "github_api", api)
    parity.finish("99", "success")
    assert api.call_args_list[0].args == ("pulls/42",)
    assert api.call_args.args == ("check-runs/99",)
    data = api.call_args.kwargs["data"]
    assert data["conclusion"] == "success"
    summary = data["output"]["summary"]
    assert f"Tested commit: `{'a' * 40}`" in summary
    assert expected in summary
    if isinstance(head, dict):
        assert f"Current PR head: `{head['head']['sha']}`" in summary
    assert summary in (tmp_path / "summary.md").read_text()


def test_nightly_report_does_not_fetch_pr_head(event, monkeypatch):
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    monkeypatch.setenv("PARITY_TESTED_SHA", "main-sha")
    api = Mock()
    monkeypatch.setattr(parity, "github_api", api)
    parity.finish("99", "success")
    assert api.call_count == 1
    assert api.call_args.args == ("check-runs/99",)
    assert (
        "Tested commit: `main-sha`" in api.call_args.kwargs["data"]["output"]["summary"]
    )


def test_nightly_uses_event_sha_and_does_not_run_in_forks(event, monkeypatch):
    api = Mock()
    monkeypatch.setattr(parity, "github_api", api)
    monkeypatch.setenv("GITHUB_EVENT_NAME", "schedule")
    assert parity.resolve_target() == {
        "repository": parity.REPOSITORY,
        "sha": "main-sha",
        "key": "nightly",
    }
    monkeypatch.setenv("GITHUB_REPOSITORY", "contributor/fork")
    assert parity.start() is None
    api.assert_not_called()


def test_summary_preserves_parity_counts_and_missing_batches(tmp_path):
    (tmp_path / "batch-1.json").write_text(
        json.dumps(
            {
                "comparison": {
                    "counts": {"EXACT": 30, "TOP_K_MATCH": 9, "FAIL": 1},
                    "passed": False,
                }
            }
        )
    )
    text = parity.summarize(
        tmp_path, model="Qwen/Qwen3.5-0.8B", revision="rev", sha="head"
    )
    assert "Commit: `head`" in text and "Checkpoint revision: `rev`" in text
    assert "| Native → HTTP target | 1 | 30 | 9 | 1 | FAIL |" in text
    assert "| Native → HTTP target | 2 | — | — | — | Missing |" in text
    assert "same-model" not in text


@pytest.mark.parametrize("passed", [True, False])
def test_summary_exposes_acceptance_failure_and_incomplete_suite(tmp_path, passed):
    case = tmp_path / "acceptance/test_same_model0"
    case.mkdir(parents=True)
    (case.parent / "test_same_modelcurrent").symlink_to(case, target_is_directory=True)
    (case / "metrics.json").write_text(
        json.dumps(
            {
                "kind": "same-model",
                "batch_size": 1,
                "phases": {
                    "cold": {
                        "before": {
                            "num_drafts": 2,
                            "num_draft_tokens": 6,
                            "num_accepted_tokens": 6,
                            "per_position": [2, 2, 2],
                        },
                        "after": {
                            "num_drafts": 4,
                            "num_draft_tokens": 12,
                            "num_accepted_tokens": 11,
                            "per_position": [4, 4, 3],
                        },
                        "passed": passed,
                    }
                },
            }
        )
    )
    (tmp_path / "acceptance.xml").write_text(
        '<testsuites><testsuite tests="1" failures="1" errors="0" skipped="0" /></testsuites>'
    )
    text = parity.summarize(
        tmp_path, model="Qwen/Qwen3-0.6B", revision="rev", sha="head"
    )
    status = "PASS" if passed else "FAIL"
    assert (
        f"| same-model | 1 | cold | 2 | 5 / 6 | 83.33% | 3.50 | 100.0 / 100.0 / 50.0 | {status} |"
        in text
    )
    assert "| same-model | 1 | cached | — | — | — | — | — | Missing |" in text
    assert "| controlled | 2 | cached | — | — | — | — | — | Missing |" in text
    assert "Acceptance suite: 1 tests, 1 failures, 0 errors, 0 skipped." in text
    assert "| Native → same-model draft | 1 | — | — | — | Missing |" in text


def test_summary_does_not_choose_between_duplicate_attempts(tmp_path):
    for name in ("first", "retry"):
        case = tmp_path / "acceptance" / name
        case.mkdir(parents=True)
        (case / "metrics.json").write_text(
            json.dumps({"kind": "controlled", "batch_size": 1, "phases": {}})
        )
    with pytest.raises(ValueError, match="Duplicate acceptance results"):
        parity.summarize(tmp_path, model="Qwen/Qwen3-0.6B", revision="rev", sha="head")
